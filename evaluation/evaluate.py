#!/usr/bin/env python3
"""
evaluate.py — Evaluation protocol for the Europeana tool-augmented AI agent.

Faithfully reproduces the architecture of app.py:
  - Full agent  : build_qa_agent() + build_answer_task() (CrewAI + Ollama + Europeana tool)
  - Baseline    : same Ollama LLM, no tool
  - Short-term memory : current_artist + last_answer_summary propagated across the 3
                        questions for each artist, exactly as app.py does with st.session_state

Protocol: 50 artists × 3 questions = 150 queries.

Usage:
    cd new_source
    python evaluation/evaluate.py
    python evaluation/evaluate.py --resume evaluation/checkpoint_<id>.json
    python evaluation/evaluate.py --artists 10 --out ./results --skip-baseline
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

# Disable CrewAI telemetry before any import
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_TRACING", "false")

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from crewai import Agent, Crew, LLM, Process, Task

from crew.agents.qa_agent import build_qa_agent
from crew.tasks.answer_task import build_answer_task
from crew.tool.europeana_tool import (
    clean_query,
    europeana_search,
    get_last_tool_trace,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
N_ARTISTS    = 50
MIN_RECORDS  = 2
INTER_QUERY_DELAY = 1.0   # seconds between consecutive queries

# Question templates instantiated per artist — same logic as in the app
QUESTION_TEMPLATES = [
    "Who is {artist}?",              # Q1  biographical / intro
    "Name some works by {artist}.",  # Q2  artwork enumeration
    "When did {artist} live?",       # Q3  temporal
]
Q_LABELS = ["biographical", "artwork_enum", "temporal"]

# ── Artist seed list ──────────────────────────────────────────────────────────
ARTIST_SEEDS = [
    "Caravaggio", "Raphael", "Michelangelo", "Leonardo da Vinci", "Botticelli",
    "Titian", "Tintoretto", "Paolo Veronese", "Giorgione", "Andrea Mantegna",
    "Giovanni Bellini", "Annibale Carracci", "Guido Reni", "Guercino",
    "Pellizza da Volpedo", "Giovanni Segantini", "Amedeo Modigliani",
    "Giorgio de Chirico", "Umberto Boccioni", "Giacomo Balla", "Carlo Carrà",
    "Rembrandt", "Vermeer", "Jan van Eyck", "Rogier van der Weyden",
    "Hans Memling", "Hieronymus Bosch", "Pieter Bruegel", "Peter Paul Rubens",
    "Anthony van Dyck", "Frans Hals",
    "Diego Velázquez", "Francisco Goya", "El Greco",
    "Nicolas Poussin", "Antoine Watteau", "Jacques-Louis David",
    "Eugène Delacroix", "Gustave Courbet", "Jean-Baptiste-Camille Corot",
    "Jean-François Millet", "Honoré Daumier",
    "Claude Monet", "Pierre-Auguste Renoir", "Edgar Degas", "Camille Pissarro",
    "Paul Cézanne", "Georges Seurat", "Paul Gauguin", "Henri de Toulouse-Lautrec",
    "Albrecht Dürer", "Hans Holbein", "Lucas Cranach", "Gustav Klimt",
    "Egon Schiele", "Wassily Kandinsky", "Edvard Munch", "Arnold Böcklin",
    "Pablo Picasso", "Georges Braque", "Henri Matisse", "Fernand Léger",
    "Salvador Dalí", "René Magritte", "Joan Miró", "Piet Mondrian",
    "William Hogarth", "Thomas Gainsborough", "J.M.W. Turner", "John Constable",
    "Auguste Rodin", "Édouard Manet", "Berthe Morisot",
    "Alfred Sisley", "Paul Signac", "Henri Rousseau", "Pierre Bonnard",
    "Ferdinand Hodler", "Max Ernst", "Alberto Giacometti", "Constantin Brancusi",
]

# ── Graceful shutdown ─────────────────────────────────────────────────────────
_SHUTDOWN = False

def _sigint_handler(sig, frame):
    global _SHUTDOWN
    if not _SHUTDOWN:
        _SHUTDOWN = True
        print("\n[Ctrl+C received — saving checkpoint and exiting …]", flush=True)

signal.signal(signal.SIGINT, _sigint_handler)


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTIONS PORTED FROM app.py  (no Streamlit dependencies)
# ══════════════════════════════════════════════════════════════════════════════

# Markers used by answer_task.py
_ANSWER_MARKER = re.compile(r"<<<ANSWER>>>(.*?)<<<END>>>", re.DOTALL | re.IGNORECASE)

_REASONING_PREFIXES = (
    "the user wants", "we have to", "we need to",
    "provide short answer", "provide answer", "final answer:",
    "use europeana", "let me", "i should", "i will",
    "the assistant", "do not mention",
)

_SUSPICIOUS_PATTERNS = (
    "do not mention the policy", "do not mention the tool",
    "do not mention the conversation",
    "i can't comply", "i cannot comply",
    "i'm sorry, but i can't", "im sorry, but i cant",
)

# Intro prefixes for update_current_artist
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
    """Extracts the final answer from CrewAI's raw output."""
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
    """Shortens text for short-term memory, cutting at a sentence boundary."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_dot = truncated.rfind(".")
    return truncated[:last_dot + 1] if last_dot > max_len * 0.5 else truncated


def update_current_artist(question: str, state: dict) -> str:
    """Updates current_artist in the session state dict.

    Exact replica of update_current_artist() in app.py, using a plain dict
    instead of st.session_state.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
#  FULL AGENT  (mirrors run_agent in app.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(question: str, current_artist: str, max_attempts: int = 2) -> tuple[str, str, dict]:
    """Exact replica of run_agent() in app.py.

    Builds a fresh agent for every question (as in app.py) and runs the crew
    with retry on known Pydantic/TaskOutput errors.
    Returns (final_text, raw_output, trace).
    """
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


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE (same LLM, no tool)
# ══════════════════════════════════════════════════════════════════════════════

def _build_baseline_llm() -> LLM:
    """Same LLM as build_qa_agent() — identical to qa_agent.py."""
    ollama_url  = os.getenv("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/")
    api_key     = os.getenv("OLLAMA_API_KEY")
    model_name  = os.getenv("OLLAMA_MODEL", "openai/gpt-oss:20b")

    kwargs: dict = {
        "model":       model_name,
        "base_url":    f"{ollama_url}/v1",
        "api_key":     api_key,
        "temperature": 0.1,
    }
    if "gpt-oss" in model_name.lower():
        kwargs["reasoning_effort"] = "low"
    return LLM(**kwargs)


def _build_baseline_agent() -> Agent:
    """Same role/backstory as the full agent, tools=[]."""
    return Agent(
        role="Art expert",
        goal=(
            "Answer questions about artists using exclusively "
            "your own general knowledge."
        ),
        backstory=(
            "You are a scholar of European art history. Answer questions "
            "based on your knowledge, without access to any external tool."
        ),
        llm=_build_baseline_llm(),
        tools=[],
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )


def _build_baseline_task(agent: Agent) -> Task:
    return Task(
        description=(
            "User question: {question}\n"
            "Current artist thread: {current_artist}\n\n"
            "Answer in English in 4-8 sentences using your general knowledge. "
            "You have no access to external tools: use only what you know.\n\n"
            "Output format (mandatory): enclose the answer between <<<ANSWER>>> and <<<END>>>."
        ),
        expected_output="English text enclosed between <<<ANSWER>>> and <<<END>>>.",
        agent=agent,
        async_execution=False,
    )


def run_baseline(question: str, current_artist: str, max_attempts: int = 2) -> tuple[str, str]:
    """Baseline: same run_agent() structure but without the tool.

    Returns (final_text, raw_output).
    """
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        try:
            agent = _build_baseline_agent()
            crew  = Crew(
                agents=[agent],
                tasks=[_build_baseline_task(agent)],
                process=Process.sequential,
                verbose=False,
            )
            result  = crew.kickoff(inputs={
                "question":       question,
                "current_artist": current_artist or "(none)",
            })
            final_text, last_raw = extract_answer(getattr(result, "raw", result))

            if final_text and not is_suspicious(final_text):
                return final_text, last_raw

        except Exception as exc:
            err = str(exc).lower()
            recoverable = any(s in err for s in
                              ("validation error", "string_type", "taskoutput"))
            if attempt >= max_attempts or not recoverable:
                raise

    return ("(invalid baseline response)", last_raw)


# ══════════════════════════════════════════════════════════════════════════════
#  ARTIST SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_artists(n: int) -> list[dict]:
    """Validate seeds on Europeana and return the first n with ≥ MIN_RECORDS works."""
    confirmed = []
    log.info("Artist sampling — target %d from %d seeds …", n, len(ARTIST_SEEDS))

    for seed in ARTIST_SEEDS:
        if len(confirmed) >= n or _SHUTDOWN:
            break
        try:
            raw   = europeana_search(seed)
            data  = json.loads(raw)
            items = data.get("items", [])
            if len(items) >= MIN_RECORDS:
                confirmed.append({"name": seed, "records": items})
                log.info("  [%2d/%2d] ✓  %s  (%d works)", len(confirmed), n, seed, len(items))
            else:
                log.debug("  [ -- ]    %s  (%d works)", seed, len(items))
        except Exception as exc:
            log.warning("  [ !! ]    %s  — %s", seed, exc)
        time.sleep(0.5)

    return confirmed


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════════════

_EUROPEANA_URL = re.compile(r"https?://(?:www\.)?europeana\.eu\S+", re.IGNORECASE)
_YEAR_RE       = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def _gt_years(records: list[dict]) -> set[int]:
    years = set()
    for rec in records:
        for y in _YEAR_RE.findall(str(rec.get("year") or "")):
            years.add(int(y))
    return years


def score(answer: str, trace: dict, gt_records: list[dict], is_agent: bool) -> dict:
    """Compute the 5 binary metrics for a single response."""
    tool_called  = bool(trace.get("result")) if is_agent else None
    cited_urls   = _EUROPEANA_URL.findall(answer or "")
    answer_years = [int(y) for y in _YEAR_RE.findall(answer or "")]
    gt_yr        = _gt_years(gt_records)

    hall = False
    if answer_years and gt_yr:
        for ay in answer_years:
            if not any(abs(ay - gy) <= 2 for gy in gt_yr):
                hall = True
                break

    return {
        "tool_called":      tool_called,           # agent only
        "source_adherence": len(cited_urls) > 0,   # structurally False for baseline
        "year_present":     len(answer_years) > 0,
        "hallucination":    hall,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            ck = json.load(f)
        log.info("Checkpoint loaded: %s (%d rows done)", path, len(ck.get("rows", [])))
        return ck
    except Exception as exc:
        log.error("Cannot load checkpoint: %s", exc)
        return None


def save_checkpoint(path: Path, data: dict) -> None:
    data["last_updated"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)


# ══════════════════════════════════════════════════════════════════════════════
#  CSV OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "q_id", "artist", "q_type", "question",
    "agent_answer", "agent_tool_called", "agent_source_adherence",
    "agent_year_present", "agent_hallucination", "agent_error",
    "baseline_answer", "baseline_year_present", "baseline_hallucination",
    "baseline_error",
]


def _fmt(v) -> str:
    if v is None:  return "N/A"
    if isinstance(v, bool): return "1" if v else "0"
    return str(v)


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in CSV_FIELDS})
    log.info("CSV saved → %s  (%d rows)", path, len(rows))


def print_summary(rows: list[dict]) -> None:
    def pct(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), bool)]
        return f"{sum(vals)/len(vals):.1%}" if vals else "N/A"

    print("\n" + "=" * 60)
    print(f"  SUMMARY — {len(rows)} queries  ({len(rows)//3} artists × 3 Q)")
    print("=" * 60)
    print(f"  {'Metric':<28} {'Agent':>8}  {'Baseline':>10}")
    print("  " + "-" * 50)
    for label, ak, bk in [
        ("tool_called",      "agent_tool_called",     None),
        ("source_adherence", "agent_source_adherence", None),
        ("year_present",     "agent_year_present",     "baseline_year_present"),
        ("hallucination",    "agent_hallucination",    "baseline_hallucination"),
    ]:
        b = pct(bk) if bk else "  —"
        print(f"  {label:<28} {pct(ak):>8}  {b:>10}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the Europeana tool-augmented agent (CrewAI + Ollama)")
    p.add_argument("--artists",       type=int, default=N_ARTISTS)
    p.add_argument("--out",           type=str, default="evaluation")
    p.add_argument("--resume",        type=str, default=None, metavar="CHECKPOINT")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--dry-run",       action="store_true")
    return p.parse_args()


def main():
    global _SHUTDOWN
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for var in ("EUROPEANA_API_KEY", "OLLAMA_API_KEY"):
        if not os.getenv(var):
            log.error("%s not set. Add it to new_source/.env.", var)
            sys.exit(1)

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    ck_path = out_dir / f"checkpoint_{run_id}.json"

    checkpoint: dict | None = None
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume))
        if not checkpoint:
            sys.exit(1)
        run_id  = checkpoint.get("run_id", run_id)
        ck_path = Path(args.resume)

    # ── Artists ───────────────────────────────────────────────────────────────
    if checkpoint:
        artists = checkpoint["artists"]
        log.info("Resuming with %d artists from checkpoint.", len(artists))
    else:
        artists = sample_artists(args.artists)
        if not artists:
            log.error("No artists validated. Aborting.")
            sys.exit(1)

    total = len(artists) * len(QUESTION_TEMPLATES)
    log.info("Plan: %d artists × %d questions = %d total queries",
             len(artists), len(QUESTION_TEMPLATES), total)

    if args.dry_run:
        for i, a in enumerate(artists, 1):
            for tpl in QUESTION_TEMPLATES:
                print(f"  {i:>3}. {tpl.format(artist=a['name'])}")
        return

    # ── State ─────────────────────────────────────────────────────────────────
    rows: list[dict]   = list((checkpoint or {}).get("rows", []))
    done_ids: set[int] = {r["q_id"] for r in rows}

    ck_data = {"run_id": run_id, "artists": artists, "rows": rows}
    save_checkpoint(ck_path, ck_data)
    log.info("Checkpoint: %s", ck_path)

    # ── Main loop ─────────────────────────────────────────────────────────────
    q_id = 0

    for artist_info in artists:
        artist     = artist_info["name"]
        gt_records = artist_info["records"]

        # Short-term memory: shared state across the 3 questions for this artist
        # (replaces st.session_state from app.py)
        session = {"current_artist": "", "last_user_question": "", "last_answer_summary": ""}

        for tpl, qtype in zip(QUESTION_TEMPLATES, Q_LABELS):
            q_id += 1
            question = tpl.format(artist=artist)

            if q_id in done_ids:
                # Advance session state even for skipped questions, for consistency
                update_current_artist(question, session)
                log.info("[%3d/%d]  %-28s  %s  (skip)", q_id, total, artist, qtype)
                continue

            log.info("[%3d/%d]  %-28s  %s", q_id, total, artist, qtype)

            # Update current_artist exactly as app.py does before run_agent()
            current_artist = update_current_artist(question, session)

            # ── Full agent ────────────────────────────────────────────────────
            agent_answer = agent_error = ""
            agent_scores = {}
            try:
                final_text, _, trace = run_agent(question, current_artist)
                agent_answer = final_text
                agent_scores = score(final_text, trace, gt_records, is_agent=True)
                # Update short-term memory exactly as app.py does after run_agent()
                session["last_user_question"]  = question
                session["last_answer_summary"] = shorten_for_memory(final_text)
            except Exception as exc:
                agent_error  = str(exc)
                agent_scores = {"tool_called": None, "source_adherence": False,
                                "year_present": False, "hallucination": False}
                log.warning("    agent error: %s", exc)

            # ── Baseline ──────────────────────────────────────────────────────
            baseline_answer = baseline_error = ""
            baseline_scores = {}
            if not args.skip_baseline:
                try:
                    bl_text, _ = run_baseline(question, current_artist)
                    baseline_answer = bl_text
                    baseline_scores = score(bl_text, {}, gt_records, is_agent=False)
                except Exception as exc:
                    baseline_error  = str(exc)
                    baseline_scores = {"year_present": False, "hallucination": False}
                    log.warning("    baseline error: %s", exc)

            # ── Save row ──────────────────────────────────────────────────────
            row = {
                "q_id":    q_id,
                "artist":  artist,
                "q_type":  qtype,
                "question": question,
                "agent_answer":           (agent_answer or "")[:500],
                "agent_tool_called":       agent_scores.get("tool_called"),
                "agent_source_adherence":  agent_scores.get("source_adherence", False),
                "agent_year_present":      agent_scores.get("year_present", False),
                "agent_hallucination":     agent_scores.get("hallucination", False),
                "agent_error":             agent_error,
                "baseline_answer":         (baseline_answer or "")[:500],
                "baseline_year_present":   baseline_scores.get("year_present", False),
                "baseline_hallucination":  baseline_scores.get("hallucination", False),
                "baseline_error":          baseline_error,
            }
            rows.append(row)
            ck_data["rows"] = rows
            save_checkpoint(ck_path, ck_data)

            if _SHUTDOWN:
                log.info("Checkpoint saved at %s", ck_path)
                log.info("Resume with:  python evaluation/evaluate.py --resume %s", ck_path)
                sys.exit(0)

            time.sleep(INTER_QUERY_DELAY)

    # ── Final output ──────────────────────────────────────────────────────────
    csv_path = out_dir / f"results_{run_id}.csv"
    write_csv(rows, csv_path)
    print_summary(rows)
    ck_data["completed"] = True
    save_checkpoint(ck_path, ck_data)
    log.info("Evaluation complete — final checkpoint: %s", ck_path)


if __name__ == "__main__":
    main()
