import os
import re
import traceback

# Disattivo la telemetria di crewai prima di importarlo, sennò la setta lui
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TRACING"] = "false"

import streamlit as st
from dotenv import load_dotenv
from crewai import Crew, Process

from crew.agents.qa_agent import build_qa_agent
from crew.tasks.answer_task import build_answer_task
from crew.tool.europeana_tool import get_last_tool_trace, clean_query

load_dotenv()
st.set_page_config(page_title="Esperto d'arte", page_icon="🎨", layout="centered")


# ----- ESTRAZIONE E PULIZIA OUTPUT -----

# Il task chiede al modello di racchiudere la risposta tra questi marcatori
_ANSWER_MARKER = re.compile(r"<<<RISPOSTA>>>(.*?)<<<FINE>>>", re.DOTALL | re.IGNORECASE)

# Frasi di "ragionamento interno" del modello, da scartare se mancano i marcatori
_REASONING_PREFIXES = (
    "the user wants", "we have to", "we need to",
    "provide short answer", "provide answer", "final answer:",
    "use europeana", "let me", "i should", "i will",
    "the assistant", "do not mention",
)

# Pattern di output corrotto (gpt-oss a volte fa leakare il ragionamento
# interno o produce dei falsi rifiuti). In quel caso ritento.
_SUSPICIOUS_PATTERNS = (
    "do not mention the policy",
    "do not mention the tool",
    "do not mention the conversation",
    "i can't comply", "i cannot comply",
    "i'm sorry, but i can't", "im sorry, but i cant",
)


def is_suspicious(text: str) -> bool:
    if not text:
        return True
    text_l = text.lower()
    return any(p in text_l for p in _SUSPICIOUS_PATTERNS)


def extract_answer(raw) -> tuple[str, str]:
    # Ritorna (testo finale ripulito, raw_output come stringa).
    # A volte CrewAI restituisce una lista di tool_calls invece di una stringa,
    # quindi forzo str() per non far crashare la UI.
    raw_str = "" if raw is None else (raw if isinstance(raw, str) else str(raw))
    if not raw_str:
        return "", ""

    # Caso normale: trovo i marcatori
    match = _ANSWER_MARKER.search(raw_str)
    if match:
        return match.group(1).strip(), raw_str

    # Fallback: ripulisco riga per riga togliendo il ragionamento interno
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


# ----- GESTIONE ARTISTA CORRENTE -----

# Prefissi che riconoscono le domande introduttive su un nuovo artista
_ARTIST_INTROS = (
    "chi è ", "chi e ", "chi era ",
    "parlami di ", "raccontami di ",
    "dimmi di ", "dimmi chi è ", "dimmi chi e ",
    "who is ", "tell me about ",
)


def update_current_artist(question: str) -> str:
    # Tiene traccia dell'artista corrente per i follow-up tipo
    # "quando è morto?" senza perdere il contesto.
    previous = st.session_state.get("current_artist", "")
    is_intro = any(question.strip().lower().startswith(p) for p in _ARTIST_INTROS)

    if is_intro:
        new_artist = clean_query(question)
        if new_artist:
            st.session_state.current_artist = new_artist
            return new_artist

    if previous:
        return previous

    # Caso limite: nessun pattern e nessuno stato precedente
    candidate = clean_query(question)
    if candidate:
        st.session_state.current_artist = candidate
    return candidate or ""


# ----- ESECUZIONE AGENTE CON RETRY -----

def run_agent(question: str, current_artist: str, max_attempts: int = 2):
    # Costruisco un agente nuovo per ogni domanda: riusare lo stesso agente
    # in fila destabilizza gpt-oss e fa entrare il modello in loop di tool calls.
    last_raw, last_trace = "", {"query": "", "result": ""}

    for attempt in range(1, max_attempts + 1):
        try:
            agent = build_qa_agent()
            crew = Crew(
                agents=[agent],
                tasks=[build_answer_task(agent)],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff(inputs={
                "question": question,
                "current_artist": current_artist or "(nessuno)",
            })

            last_trace = get_last_tool_trace()
            final_text, last_raw = extract_answer(getattr(result, "raw", result))

            if final_text and not is_suspicious(final_text):
                return final_text, last_raw, last_trace
            # Output non valido: se ho ancora tentativi, riprovo

        except Exception as e:
            # Errori "noti" del modello (Pydantic / TaskOutput): vale la pena riprovare
            err = str(e).lower()
            recoverable = any(s in err for s in ("validation error", "string_type", "taskoutput"))
            if attempt >= max_attempts or not recoverable:
                raise

    fallback = (
        "Il modello ha prodotto una risposta non valida dopo più tentativi. "
        "Prova a riformulare la domanda o a pulire la chat."
    )
    return fallback, last_raw, last_trace


# ----- UI: PANNELLO DETTAGLI E UTILITY -----

def render_trace(question: str, raw_output: str, trace: dict):
    with st.expander("🔎 Dettagli esecuzione", expanded=False):
        st.caption("Traccia tecnica della risposta generata")
        tab_in, tab_api, tab_llm = st.tabs(["Input", "API Europeana", "LLM / CrewAI"])

        with tab_in:
            st.markdown("**Domanda dell'utente**")
            st.code(question or "", language="text")
            st.markdown("**Query inviata al tool**")
            st.code(trace.get("query", ""), language="text")
            st.markdown("**Artista corrente del thread**")
            st.code(st.session_state.get("current_artist", "") or "(nessuno)", language="text")

        with tab_api:
            st.markdown("**Output restituito da Europeana**")
            st.code(trace.get("result", ""), language="json")

        with tab_llm:
            st.markdown("**Output grezzo di CrewAI**")
            st.code(raw_output or "", language="text")


# ----- SIDEBAR -----

with st.sidebar:
    st.header("ℹ️ Informazioni")
    st.write(f"**Modello AI:** {os.getenv('OLLAMA_MODEL', 'gpt-oss:20b')}")
    st.write("**Tool:** Europeana API")
    st.write("**Framework:** CrewAI + Streamlit")

    current = st.session_state.get("current_artist", "")
    if current:
        st.write(f"**Artista corrente:** {current}")

    st.divider()
    st.write("### Esempi di domande")
    for example in ("Chi è Caravaggio?", "Chi è Picasso?", "Chi è Van Gogh?"):
        if st.button(example):
            st.session_state.example_question = example


# ----- HEADER E STATO -----

st.title("🎨 Esperto d'arte")
st.caption("Assistente AI che risponde a domande su artisti utilizzando Europeana.")

# Inizializzo le chiavi di session_state al primo giro
for key, default in (("messages", []), ("current_artist", "")):
    st.session_state.setdefault(key, default)

if not st.session_state.messages:
    st.info("Prova a chiedere qualcosa su un artista europeo.")

if st.button("Pulisci chat"):
    st.session_state.messages = []
    st.session_state.current_artist = ""
    st.rerun()


# ----- STORICO MESSAGGI -----

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "tool_trace" in msg:
            render_trace(
                question=msg.get("user_question", ""),
                raw_output=msg.get("raw_output", ""),
                trace=msg["tool_trace"],
            )


# ----- INPUT -----

question = st.chat_input("Scrivi la tua domanda, ad esempio: Chi è Caravaggio?")
# Se l'utente ha cliccato un pulsante di esempio uso quella domanda
if "example_question" in st.session_state:
    question = st.session_state.pop("example_question")


# ----- GESTIONE DOMANDA -----

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Aggiorno l'artista corrente prima di lanciare l'agente
    current_artist = update_current_artist(question)

    with st.chat_message("assistant"):
        with st.spinner("Sto cercando la risposta..."):
            try:
                final_text, raw_output, trace = run_agent(question, current_artist)

                if not final_text:
                    final_text = (
                        "Non sono riuscito a generare una risposta valida. "
                        "Prova a riformulare la domanda."
                    )

                st.markdown(final_text)
                render_trace(question, raw_output, trace)
                st.divider()

                # Salvo tutto nello storico, compresi i dati per i dettagli tecnici
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_text,
                    "tool_trace": trace,
                    "raw_output": raw_output,
                    "user_question": question,
                })

            except Exception as e:
                error_msg = f"Errore: {e}"
                st.error(error_msg)
                with st.expander("Dettagli errore"):
                    st.code(traceback.format_exc(), language="text")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "user_question": question,
                })