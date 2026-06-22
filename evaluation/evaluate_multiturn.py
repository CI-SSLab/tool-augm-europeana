#!/usr/bin/env python3
"""
evaluate_multiturn.py — Multi-turn evaluation for the short-term memory module.

Three scenario types over 15 representative artists (45 sessions, ~105 turns):

  Scenario A — Co-reference chain (3 turns per session):
    T1: "Who is {artist}?"
    T2: "When did this artist live?"            ← implicit co-reference
    T3: "Name three famous works by this artist."  ← implicit co-reference

  Scenario B — Artist switch (2 turns per session):
    T1: "Tell me about {artist_1}."
    T2: "Tell me about {artist_2}."             ← memory must reset

  Scenario C — Implicit follow-up (2 turns per session):
    T1: "Name some works by {artist}."
    T2: "When was the earliest one created?"    ← context from T1

Metrics (agent only; no baseline — the isolated LLM has no memory module):
  CRR  – Co-reference Resolution Rate: turns A2+A3 where correct artist in Europeana query
  SIR  – Switch Isolation Rate:        turns B2 where artist_2 (not artist_1) in query
  TDR  – Per-turn Tool Discipline:     all turns where tool called exactly once
  SAR  – Per-turn Source Adherence:    all turns where ≥1 Europeana URL in response

Usage:
    cd new_source
    python evaluation/evaluate_multiturn.py
    python evaluation/evaluate_multiturn.py --out evaluation/ --dry-run
"""

import argparse
import csv
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_TRACING", "false")

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from crewai import Crew, Process

from crew.agents.qa_agent import build_qa_agent
from crew.tasks.answer_task import build_answer_task
from crew.tool.europeana_tool import (
    clean_query,
    europeana_search,
    get_last_tool_trace,
    _LAST_TRACE,          # imported to allow per-call reset (avoids stale trace)
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_RECORDS       = 2
INTER_QUERY_DELAY = 1.0   # seconds between consecutive queries

# 15 artists drawn from the main evaluation seed list
MT_ARTIST_SEEDS = [
    "Caravaggio", "Raphael", "Rembrandt", "Vermeer", "Claude Monet",
    "Edgar Degas", "Paul Cézanne", "Albrecht Dürer", "Gustav Klimt",
    "Edvard Munch", "Diego Velázquez", "Francisco Goya",
    "Eugène Delacroix", "Titian", "Botticelli",
]

# Scenario turn templates ──────────────────────────────────────────────────────
SCENARIO_A_TURNS = [
    "Who is {artist}?",
    "When did this artist live?",
    "Name three famous works by this artist.",
]

SCENARIO_B_TURNS = [
    "Tell me about {artist_1}.",
    "Tell me about {artist_2}.",
]

SCENARIO_C_TURNS = [
    "Name some works by {artist}.",
    "When was the earliest one created?",
]

# ── Utility functions — ported from app.py (no Streamlit) ────────────────────
_ANSWER_MARKER = re.compile(r"<<<ANSWER>>>(.*?)<<<END>>>", re.DOTALL | re.IGNORECASE)
_EUROPEANA_URL = re.compile(r"https?://(?:www\.)?europeana\.eu\S+", re.IGNORECASE)

_REASONING_PREFIXES = (
    "the user wants", "we have to", "we need to",
    "provide short answer", "provide answer", "final answer:",
    "use europeana", "let me", "i should", "i will",
    "the assistant", "do not mention",
)
_SUSPICIOUS_PATTERNS = (
    "do not mention the policy", "do not mention the tool",
    "i can't comply", "i cannot comply",
    "i'm sorry, but i can't",
)
_ARTIST_INTROS = (
    "who is ", "who was ",
    "tell me about ", "tell me who is ", "tell me who was ",
    "describe ", "what do you know about ",
    "give me information on ", "give me info on ",
)


def is_suspicious(text: str) -> bool:
    if not text:
        return True
    tl = text.lower()
    return any(p in tl for p in _SUSPICIOUS_PATTERNS)


def extract_answer(raw) -> tuple[str, str]:
    """Extracts the final answer from CrewAI's raw output (same logic as app.py)."""
    raw_str = "" if raw is None else (raw if isinstance(raw, str) else str(raw))
    if not raw_str:
        return "", ""
    m = _ANSWER_MARKER.search(raw_str)
    if m:
        return m.group(1).strip(), raw_str
    cleaned, started = [], False
    for line in raw_str.splitlines():
        low = line.strip().lower()
        if not low and not started:
            continue
        if any(low.startswith(p) for p in _REASONING_PREFIXES):
            continue
        if line.strip():
            started = True
        if started:
            cleaned.append(line)
    return "\n".join(cleaned).strip(), raw_str


def shorten_for_memory(text: str, max_len: int = 400) -> str:
    """Shortens text for short-term memory (same logic as app.py)."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_dot = truncated.rfind(".")
    return truncated[:last_dot + 1] if last_dot > max_len * 0.5 else truncated


def update_current_artist(question: str, state: dict) -> str:
    """Updates current_artist in session state (exact replica of app.py)."""
    previous = state.get("current_artist", "")
    is_intro = any(question.strip().lower().startswith(p) for p in _ARTIST_INTROS)
    if is_intro:
        new_artist = clean_query(question)
        if new_artist:
            if previous and previous.lower() != new_artist.lower():
                state["last_user_question"] = ""
                state["last_answer_summary"] = ""
            state["current_artist"] = new_artist
            return new_artist
    if previous:
        return previous
    candidate = clean_query(question)
    if candidate:
        state["current_artist"] = candidate
    return candidate or ""


def run_agent(question: str, current_artist: str, max_attempts: int = 2) -> tuple[str, str, dict]:
    """Exact replica of run_agent() in app.py.

    Resets _LAST_TRACE before each call so that get_last_tool_trace()
    always reflects the current invocation, not a stale previous one.
    """
    # Reset global trace to avoid stale reads when tool is not called
    _LAST_TRACE.update(query="", result="")
    last_raw, last_trace = "", {"query": "", "result": ""}

    for attempt in range(1, max_attempts + 1):
        try:
            agent = build_qa_agent()
            crew  = Crew(
                agents=[agent],
                tasks=[build_answer_task(agent)],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff(inputs={
                "question":       question,
                "current_artist": current_artist or "(none)",
            })
            last_trace = get_last_tool_trace()
            final_text, last_raw = extract_answer(getattr(result, "raw", result))
            if final_text and not is_suspicious(final_text):
                return final_text, last_raw, last_trace
        except Exception as exc:
            err = str(exc).lower()
            recoverable = any(s in err for s in
                              ("validation error", "string_type", "taskoutput"))
            if attempt >= max_attempts or not recoverable:
                raise

    return ("(invalid response after multiple attempts)", last_raw, last_trace)


# ── Graceful shutdown ─────────────────────────────────────────────────────────
_SHUTDOWN = False

def _sigint_handler(sig, frame):
    global _SHUTDOWN
    if not _SHUTDOWN:
        _SHUTDOWN = True
        print("\n[Ctrl+C — finalising current turn and exiting …]", flush=True)

signal.signal(signal.SIGINT, _sigint_handler)


# ── Artist sampling ───────────────────────────────────────────────────────────
def sample_artists(seeds: list[str]) -> list[dict]:
    """Validate seeds on Europeana (same logic as evaluate.py)."""
    confirmed = []
    log.info("Validating %d artist seeds on Europeana …", len(seeds))
    for seed in seeds:
        if _SHUTDOWN:
            break
        try:
            raw   = europeana_search(seed)
            data  = json.loads(raw)
            items = data.get("items", [])
            if len(items) >= MIN_RECORDS:
                confirmed.append({"name": seed, "records": items})
                log.info("  ✓  %s  (%d works)", seed, len(items))
            else:
                log.warning("  ✗  %s  (%d works — skipped)", seed, len(items))
        except Exception as exc:
            log.warning("  !!  %s  — %s", seed, exc)
        time.sleep(0.5)
    return confirmed


# ── Scoring ───────────────────────────────────────────────────────────────────
def _artist_in_query(trace: dict, artist_name: str) -> bool:
    """True if artist_name appears (case-insensitive) in the Europeana query field."""
    q = (trace.get("query") or "").lower()
    return artist_name.lower() in q


def score_turn(
    answer: str,
    trace: dict,
    expected_artist: str,
    excluded_artist: str | None = None,
) -> dict:
    """Per-turn metrics for multi-turn evaluation.

    artist_resolved: used for CRR (Scenario A turns 2-3)
    switch_isolated: used for SIR (Scenario B turn 2); None when not applicable
    """
    tool_called     = bool(trace.get("result"))   # False when trace was reset and tool not called
    cited_urls      = _EUROPEANA_URL.findall(answer or "")

    # CRR: only meaningful if the tool was actually called in this turn
    artist_resolved = _artist_in_query(trace, expected_artist) if tool_called else False

    # SIR: correct artist present AND wrong artist absent from the query
    switch_isolated = None
    if excluded_artist is not None:
        if tool_called:
            correct_present = _artist_in_query(trace, expected_artist)
            wrong_present   = _artist_in_query(trace, excluded_artist)
            switch_isolated = correct_present and not wrong_present
        else:
            switch_isolated = False   # tool not called = isolation failed

    return {
        "tool_called":      tool_called,
        "source_adherence": len(cited_urls) > 0,
        "artist_resolved":  artist_resolved,
        "switch_isolated":  switch_isolated,
    }


# ── CSV output ────────────────────────────────────────────────────────────────
MT_CSV_FIELDS = [
    "session_id", "scenario", "turn", "artist", "artist_2",
    "question", "answer",
    "tool_called", "source_adherence", "artist_resolved", "switch_isolated",
    "error",
]


def write_csv(rows: list[dict], path: Path) -> None:
    def fmt(v):
        if v is None:       return ""
        if isinstance(v, bool): return "1" if v else "0"
        return str(v)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MT_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(row.get(k)) for k in MT_CSV_FIELDS})
    log.info("CSV saved → %s  (%d rows)", path, len(rows))


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(rows: list[dict]) -> None:
    def pct_subset(subset: list[dict], key: str) -> str:
        vals = [r[key] for r in subset if isinstance(r.get(key), bool)]
        if not vals:
            return "N/A"
        return f"{sum(vals)/len(vals)*100:.1f}%  ({sum(vals)}/{len(vals)})"

    crr_rows = [r for r in rows if r.get("scenario") == "A" and r.get("turn", 1) > 1]
    sir_rows = [r for r in rows if r.get("scenario") == "B" and r.get("turn", 1) == 2]
    all_rows = rows

    print("\n" + "=" * 65)
    print(f"  MULTI-TURN SUMMARY — {len(rows)} turns across 45 sessions")
    print("=" * 65)
    print(f"  {'Metric':<44} {'Score':>16}")
    print("  " + "-" * 62)
    print(f"  {'Co-reference Resolution Rate  (CRR)':<44} {pct_subset(crr_rows, 'artist_resolved'):>16}")
    print(f"  {'Switch Isolation Rate  (SIR)':<44} {pct_subset(sir_rows, 'switch_isolated'):>16}")
    print(f"  {'Per-turn Tool Discipline':<44} {pct_subset(all_rows, 'tool_called'):>16}")
    print(f"  {'Per-turn Source Adherence':<44} {pct_subset(all_rows, 'source_adherence'):>16}")
    print("=" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-turn memory evaluation (CrewAI + Ollama + Europeana).")
    p.add_argument("--out",     type=str, default="evaluation",
                   help="Output directory for CSV results (default: evaluation/)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned turns without calling the agent or Europeana.")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for var in ("EUROPEANA_API_KEY", "OLLAMA_API_KEY"):
        if not os.getenv(var):
            log.error("%s not set. Add it to new_source/.env.", var)
            sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        # Print planned turns and exit
        session_id = 0
        for art in MT_ARTIST_SEEDS:
            session_id += 1
            for t, tpl in enumerate(SCENARIO_A_TURNS, 1):
                print(f"[A/s{session_id:02d}/T{t}] {art:20s} | {tpl.format(artist=art)}")
        for i, art1 in enumerate(MT_ARTIST_SEEDS):
            session_id += 1
            art2 = MT_ARTIST_SEEDS[(i + 1) % len(MT_ARTIST_SEEDS)]
            for t, tpl in enumerate(SCENARIO_B_TURNS, 1):
                print(f"[B/s{session_id:02d}/T{t}] {art1:20s} → {art2:20s} | {tpl.format(artist_1=art1, artist_2=art2)}")
        for art in MT_ARTIST_SEEDS:
            session_id += 1
            for t, tpl in enumerate(SCENARIO_C_TURNS, 1):
                print(f"[C/s{session_id:02d}/T{t}] {art:20s} | {tpl.format(artist=art)}")
        print(f"\nDry-run: {session_id} sessions planned.")
        return

    # Validate artists on Europeana
    artists = sample_artists(MT_ARTIST_SEEDS)
    if len(artists) < 2:
        log.error("Not enough validated artists. Aborting.")
        sys.exit(1)

    log.info("Starting multi-turn evaluation with %d validated artists.", len(artists))
    rows: list[dict] = []
    session_id = 0

    # ── Scenario A: co-reference chain ────────────────────────────────────────
    log.info("\n── Scenario A: co-reference chain ──────────────────────────")
    for art in artists:
        if _SHUTDOWN:
            break
        session_id += 1
        artist = art["name"]
        state  = {"current_artist": "", "last_user_question": "", "last_answer_summary": ""}

        for turn_idx, template in enumerate(SCENARIO_A_TURNS, start=1):
            if _SHUTDOWN:
                break
            question       = template.format(artist=artist)
            current_artist = update_current_artist(question, state)
            log.info("  [A/s%02d/T%d]  %-22s  %s", session_id, turn_idx, artist, question)

            error = ""
            try:
                answer, _, trace = run_agent(question, current_artist)
                s = score_turn(answer, trace, expected_artist=artist)
                state["last_user_question"]  = question
                state["last_answer_summary"] = shorten_for_memory(answer)
            except Exception as exc:
                answer, trace = "", {}
                s = {"tool_called": False, "source_adherence": False,
                     "artist_resolved": False, "switch_isolated": None}
                error = str(exc)
                log.warning("    agent error: %s", exc)

            rows.append({
                "session_id":       session_id,
                "scenario":         "A",
                "turn":             turn_idx,
                "artist":           artist,
                "artist_2":         "",
                "question":         question,
                "answer":           (answer or "")[:400],
                "tool_called":      s["tool_called"],
                "source_adherence": s["source_adherence"],
                # artist_resolved only scored on implicit turns (T2 and T3)
                "artist_resolved":  s["artist_resolved"] if turn_idx > 1 else None,
                "switch_isolated":  None,
                "error":            error,
            })
            time.sleep(INTER_QUERY_DELAY)

    # ── Scenario B: artist switch ─────────────────────────────────────────────
    log.info("\n── Scenario B: artist switch ────────────────────────────────")
    for i, art in enumerate(artists):
        if _SHUTDOWN:
            break
        session_id += 1
        art1  = art["name"]
        art2  = artists[(i + 1) % len(artists)]["name"]
        state = {"current_artist": "", "last_user_question": "", "last_answer_summary": ""}

        for turn_idx, template in enumerate(SCENARIO_B_TURNS, start=1):
            if _SHUTDOWN:
                break
            question       = template.format(artist_1=art1, artist_2=art2)
            current_artist = update_current_artist(question, state)
            expected       = art1 if turn_idx == 1 else art2
            excluded       = None  if turn_idx == 1 else art1

            log.info("  [B/s%02d/T%d]  %-14s → %-14s  %s",
                     session_id, turn_idx, art1, art2, question)

            error = ""
            try:
                answer, _, trace = run_agent(question, current_artist)
                s = score_turn(answer, trace, expected_artist=expected,
                               excluded_artist=excluded)
                state["last_user_question"]  = question
                state["last_answer_summary"] = shorten_for_memory(answer)
            except Exception as exc:
                answer, trace = "", {}
                s = {"tool_called": False, "source_adherence": False,
                     "artist_resolved": False, "switch_isolated": None}
                error = str(exc)
                log.warning("    agent error: %s", exc)

            rows.append({
                "session_id":       session_id,
                "scenario":         "B",
                "turn":             turn_idx,
                "artist":           art1,
                "artist_2":         art2,
                "question":         question,
                "answer":           (answer or "")[:400],
                "tool_called":      s["tool_called"],
                "source_adherence": s["source_adherence"],
                "artist_resolved":  None,
                # switch_isolated only scored on T2
                "switch_isolated":  s["switch_isolated"] if turn_idx == 2 else None,
                "error":            error,
            })
            time.sleep(INTER_QUERY_DELAY)

    # ── Scenario C: implicit follow-up ────────────────────────────────────────
    log.info("\n── Scenario C: implicit follow-up ───────────────────────────")
    for art in artists:
        if _SHUTDOWN:
            break
        session_id += 1
        artist = art["name"]
        state  = {"current_artist": "", "last_user_question": "", "last_answer_summary": ""}

        for turn_idx, template in enumerate(SCENARIO_C_TURNS, start=1):
            if _SHUTDOWN:
                break
            question       = template.format(artist=artist)
            current_artist = update_current_artist(question, state)
            log.info("  [C/s%02d/T%d]  %-22s  %s", session_id, turn_idx, artist, question)

            error = ""
            try:
                answer, _, trace = run_agent(question, current_artist)
                s = score_turn(answer, trace, expected_artist=artist)
                state["last_user_question"]  = question
                state["last_answer_summary"] = shorten_for_memory(answer)
            except Exception as exc:
                answer, trace = "", {}
                s = {"tool_called": False, "source_adherence": False,
                     "artist_resolved": False, "switch_isolated": None}
                error = str(exc)
                log.warning("    agent error: %s", exc)

            rows.append({
                "session_id":       session_id,
                "scenario":         "C",
                "turn":             turn_idx,
                "artist":           artist,
                "artist_2":         "",
                "question":         question,
                "answer":           (answer or "")[:400],
                "tool_called":      s["tool_called"],
                "source_adherence": s["source_adherence"],
                # artist_resolved scored on T2 only (follow-up turn)
                "artist_resolved":  s["artist_resolved"] if turn_idx > 1 else None,
                "switch_isolated":  None,
                "error":            error,
            })
            time.sleep(INTER_QUERY_DELAY)

    # ── Final output ──────────────────────────────────────────────────────────
    csv_path = out_dir / f"results_multiturn_{run_id}.csv"
    write_csv(rows, csv_path)
    print_summary(rows)
    log.info("Multi-turn evaluation complete. Results: %s", csv_path)


if __name__ == "__main__":
    main()
