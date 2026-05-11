import re
import os
import json
import time
import requests
from dotenv import load_dotenv
from crewai.tools import tool

EUROPEANA_URL = "https://api.europeana.eu/record/v2/search.json"

# Tengo traccia dell'ultima chiamata per il pannello "Dettagli esecuzione"
# di Streamlit. Stato globale = single-user, ma per il prototipo va bene.
_LAST_TRACE = {"query": "", "result": ""}

# Cache delle risposte in memoria, vive finché vive il processo
_RESULT_CACHE = {}


def get_last_tool_trace() -> dict:
    return dict(_LAST_TRACE)


# ----- PULIZIA DELLA QUERY -----

# Riconosce i prefissi tipici delle domande per isolare il nome dell'artista
# (es. "Chi è Caravaggio?" -> "Caravaggio")
_QUESTION_PREFIX = re.compile(
    r"^\s*(?:"
    r"chi\s*(?:è|e|era)\s+|"
    r"parlami\s+di\s+|raccontami\s+di\s+|"
    r"dimmi\s+(?:chi\s+è\s+|qualcosa\s+su\s+|di\s+)|"
    r"who\s+is\s+|tell\s+me\s+about\s+"
    r")",
    flags=re.IGNORECASE,
)


def clean_query(question: str) -> str:
    if not question:
        return question
    q = question.strip().strip('"\'')
    q = _QUESTION_PREFIX.sub("", q).strip()
    q = q.rstrip(" ?!.,;:").strip()
    return q or question


# ----- ESTRAZIONE DAI RECORD EUROPEANA -----

# Campi possibili per l'autore. NON uso `dataProvider` (è il museo che ha
# digitalizzato l'opera, non l'autore: finivo con creator = "KU Leuven").
_CREATOR_FIELDS = ("dcCreator", "creator", "edmAgentLabel")


def _first(value):
    # Europeana ritorna spesso liste: prendo il primo elemento utile
    if isinstance(value, list) and value:
        return value[0]
    return value if isinstance(value, str) else ""


def _build_url(item: dict) -> str:
    item_id = item.get("id")
    if isinstance(item_id, str) and item_id.strip():
        prefix = "https://www.europeana.eu/it/item"
        return f"{prefix}{item_id}" if item_id.startswith("/") else f"{prefix}/{item_id}"

    guid = item.get("guid")
    if isinstance(guid, str) and "europeana.eu" in guid:
        return guid

    return ""


def _extract_creator(item: dict) -> str:
    for field in _CREATOR_FIELDS:
        value = _first(item.get(field))
        if value:
            return value
    return ""


def _creator_matches(creator: str, artist: str) -> bool:
    # Verifica che il creator corrisponda davvero all'artista cercato.
    # Senza filtro arrivano oggetti dove l'artista è solo il SOGGETTO
    # (es. foto di Picasso scattate da altri), non l'autore.
    if not creator or not artist:
        return False

    creator_l = creator.lower()
    artist_l = artist.lower().strip()

    if artist_l in creator_l:
        return True

    # Match sul cognome, ma solo se è abbastanza lungo da non essere ambiguo
    parts = artist_l.split()
    surname = parts[-1] if parts else ""
    return len(surname) >= 4 and surname in creator_l


def _normalize_items(results: list, artist_name: str, max_rows: int = 5) -> list:
    # Filtra per creatore e ordina per anno + titolo, così l'output è stabile
    normalized = []
    for item in results:
        creator = _extract_creator(item)
        url = _build_url(item)

        if artist_name and not _creator_matches(creator, artist_name):
            continue
        # Senza url l'item è inutile, non potrei nemmeno citarlo nelle FONTI
        if not url:
            continue

        normalized.append({
            "title": _first(item.get("title")) or "(senza titolo)",
            "creator": creator,
            "year": _first(item.get("year")),
            "url": url,
        })

    def sort_key(it):
        match = re.search(r"\d{4}", str(it.get("year", "") or ""))
        year_int = int(match.group()) if match else 9999
        return (year_int, it.get("title", ""))

    normalized.sort(key=sort_key)
    return normalized[:max_rows]


# ----- CHIAMATA HTTP A EUROPEANA -----

def _http_search(api_key: str, name: str, use_who: bool, only_images: bool) -> list:
    # Combina il filtro `who` (autore) con `TYPE:IMAGE` per ottenere opere
    # visive e non documentari/libri. Retry esponenziale sui 5xx.
    qf = []
    if use_who:
        qf.append(f'who:"{name}"')
    if only_images:
        qf.append("TYPE:IMAGE")

    params = {"wskey": api_key, "query": name, "rows": 30, "profile": "standard"}
    if qf:
        # Passando una lista, requests serializza come qf=...&qf=...
        params["qf"] = qf

    headers = {"User-Agent": "TesiCrewAI/1.0"}

    for attempt in range(3):
        try:
            r = requests.get(EUROPEANA_URL, params=params, headers=headers, timeout=20)
            # Gli errori 5xx sono transienti, conviene ritentare
            if 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return (r.json() or {}).get("items", []) or []
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)  # backoff esponenziale: 1s, 2s


# ----- ENTRY POINT: RICERCA A CASCATA -----

# Tre strategie dalla più stringente alla più permissiva. Mi fermo alla
# prima che restituisce almeno 2 risultati validi.
_STRATEGIES = (
    {"use_who": True,  "only_images": True},   # 1) opere visive attribuite
    {"use_who": True,  "only_images": False},  # 2) include video / 3D
    {"use_who": False, "only_images": True},   # 3) ultima spiaggia
)


def _payload(query: str, items: list = None, error: str = None) -> str:
    out = {"query": query, "items": items or []}
    if error:
        out["error"] = error
    return json.dumps(out, ensure_ascii=False, indent=2)


def europeana_search(question: str) -> str:
    load_dotenv()
    api_key = os.getenv("EUROPEANA_API_KEY")
    if not api_key:
        return _payload(question, error="EUROPEANA_API_KEY non trovata nel file .env")

    name = clean_query(question) or question
    cache_key = name.lower().strip()
    if cache_key in _RESULT_CACHE:
        return _RESULT_CACHE[cache_key]

    items, last_error = [], None
    for strat in _STRATEGIES:
        try:
            raw = _http_search(api_key, name, **strat)
            items = _normalize_items(raw, artist_name=name)
            if len(items) >= 2:
                break
        except Exception as e:
            last_error = e

    if items:
        result = _payload(name, items=items)
    else:
        error = (
            f"Errore durante la chiamata a Europeana: {last_error}"
            if last_error
            else "Nessuna opera attribuita a questo artista trovata su Europeana"
        )
        result = _payload(name, error=error)

    _RESULT_CACHE[cache_key] = result
    return result


@tool("europeana_search")
def europeana_search_tool(question: str) -> str:
    """
    Cerca opere d'arte di un artista su Europeana.
    IMPORTANTE: passa SOLO il nome dell'artista (es. "Caravaggio", "Pablo Picasso"),
    NON l'intera domanda dell'utente.
    Restituisce un JSON con al massimo 5 opere effettivamente attribuite all'artista,
    ognuna con title, creator, year, url. Se non trova nulla, restituisce un JSON con
    items=[] e un campo error: in quel caso rispondi con la conoscenza generale
    e segnala l'assenza di fonti Europeana.
    """
    result = europeana_search(question)
    _LAST_TRACE.update(query=question, result=result)
    return result