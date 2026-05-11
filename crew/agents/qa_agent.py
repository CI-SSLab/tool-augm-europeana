import os
from crewai import Agent, LLM

from crew.tool.europeana_tool import europeana_search_tool


def build_qa_agent() -> Agent:
    #Costruisce l'agente che risponde alle domande sugli artisti

    #Configurazione del modello dal file .env, con default su gpt-oss:20b
    ollama_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/")
    api_key = os.getenv("OLLAMA_API_KEY")
    model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

    #Costruisco i parametri dell'LLM in un dizionario perché reasoning_effort
    #va passato solo per gpt-oss, gli altri modelli non lo accettano
    llm_kwargs = {
        "model": f"openai/{model_name}",
        "base_url": f"{ollama_url}/v1",
        "api_key": api_key,
        "temperature": 0.1,  # bassa per risposte più stabili
    }
    if "gpt-oss" in model_name.lower():
        #Con effort di default il modello a volte chiama il tool più volte
        #del necessario, mettendolo a low il problema si riduce
        llm_kwargs["reasoning_effort"] = "low"

    llm_model = LLM(**llm_kwargs)

    art_expert_agent = Agent(
        role="Esperto d'arte",
        goal=(
            "Rispondere a domande su artisti unendo la propria conoscenza generale "
            "con i dati specifici e le fonti recuperate da Europeana."
        ),
        #Nel backstory specifico che Europeana va usata SOLO per le fonti,
        #altrimenti il modello prova a tirare fuori biografie dai dati del tool
        backstory=(
            "Sei uno studioso di storia dell'arte europea. Quando rispondi su un "
            "artista usi la tua conoscenza per inquadrarlo (chi è, periodo, stile, "
            "importanza) e consulti Europeana esclusivamente per recuperare opere "
            "reali e fonti verificabili da citare. Non inventi mai titoli, date o "
            "link: tutti i dati specifici devono provenire dal tool. "
            "Chiami il tool una sola volta, poi rispondi."
        ),
        llm=llm_model,
        tools=[europeana_search_tool],
        allow_delegation=False,
        verbose=False,
        #Flusso ideale: 1 chiamata al tool + 1 risposta finale.
        #Se metto max_iter troppo alto e il modello sbaglia, entra in loop
        max_iter=4,
    )

    return art_expert_agent