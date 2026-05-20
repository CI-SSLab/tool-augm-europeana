# Agente AI per la consultazione del patrimonio culturale europeo

Questo progetto nasce nell’ambito della mia tesi di laurea e ha come obiettivo la realizzazione di un assistente conversazionale intelligente dedicato al patrimonio culturale europeo.

L’applicazione permette all’utente di porre domande in linguaggio naturale su artisti e opere d’arte, sfruttando un modello linguistico di grandi dimensioni (LLM) integrato con Europeana, una delle principali piattaforme europee per la consultazione di contenuti culturali digitali.

Il sistema è stato sviluppato utilizzando CrewAI, Streamlit e Ollama, con l’obiettivo di combinare le capacità di comprensione del linguaggio naturale degli LLM con dati affidabili provenienti da fonti esterne.

---

# Funzionalità principali

- Interfaccia conversazionale simile a ChatGPT
- Recupero di informazioni artistiche tramite Europeana API
- Generazione di risposte in linguaggio naturale
- Memoria conversazionale di breve termine
- Visualizzazione dei dettagli tecnici dell’esecuzione
- Controllo del comportamento del modello tramite task dedicati

---

# Struttura del sistema

L’applicazione è composta da diverse componenti che collaborano tra loro:

- **Streamlit** → gestione dell’interfaccia utente;
- **CrewAI** → orchestrazione dell’agente intelligente;
- **GPT-OSS tramite Ollama** → comprensione e generazione del testo;
- **Tool Europeana** → recupero delle informazioni dal database culturale;
- **Memoria conversazionale** → supporto alle interazioni multi-turno.

L’intero sistema segue un approccio *tool-augmented*, in cui il modello linguistico utilizza fonti esterne per migliorare l’affidabilità delle risposte generate.

---

# Tecnologie utilizzate

- Python
- Streamlit
- CrewAI
- Ollama
- GPT-OSS
- Europeana API
- Requests
- python-dotenv

---

# Struttura del progetto

```bash
project/
│
├── app.py
├── qa_agent.py
├── answer_task.py
├── europeana_tool.py
├── .env
├── requirements.txt
└── README.md
```

---

# Descrizione dei file

## `app.py`

Gestisce l’interfaccia Streamlit e il flusso principale dell’applicazione.

Si occupa di:

- ricevere l’input dell’utente;
- mantenere lo storico della conversazione;
- costruire la memoria breve;
- creare ed eseguire l’agente;
- mostrare i risultati e i dettagli tecnici.

---

## `qa_agent.py`

Contiene la configurazione dell’agente intelligente.

L’agente viene definito come un esperto d’arte e utilizza il modello `gpt-oss` tramite Ollama.  
Può utilizzare esclusivamente il tool Europeana per recuperare informazioni.

---

## `answer_task.py`

Definisce il task assegnato all’agente.

Include le istruzioni operative e le regole da rispettare durante la generazione della risposta, ad esempio:

- utilizzo controllato della memoria;
- una sola chiamata al tool;
- obbligo di usare dati provenienti da Europeana;
- generazione della risposta in italiano.

---

## `europeana_tool.py`

Implementa il tool personalizzato per l’accesso all’API Europeana.

Tra le principali funzionalità:

- pulizia delle query;
- gestione delle richieste HTTP;
- retry automatici in caso di errore;
- normalizzazione dei risultati ottenuti;
- estrazione di titolo, autore, anno e link delle opere.

---

# Flusso di esecuzione

Quando l’utente inserisce una domanda:

1. il sistema acquisisce l’input;
2. costruisce il contesto conversazionale;
3. crea un nuovo agente CrewAI;
4. esegue il task associato;
5. interroga Europeana tramite il tool dedicato;
6. genera la risposta finale;
7. pulisce l’output prodotto dal modello;
8. aggiorna la memoria della conversazione.

---

# Memoria conversazionale

Per mantenere continuità nelle interazioni, il sistema utilizza una memoria di breve termine composta da:

- ultima domanda dell’utente;
- riassunto dell’ultima risposta generata.

La memoria viene utilizzata solo quando realmente utile al contesto della nuova richiesta.

---

# Trasparenza del sistema

L’applicazione include un pannello tecnico opzionale che consente di visualizzare:

- la query inviata a Europeana;
- i risultati restituiti dall’API;
- l’output grezzo generato dal modello linguistico.

Questa scelta è stata adottata per migliorare la trasparenza del sistema e facilitare eventuali attività di analisi o debugging.

---

# Installazione

## 1. Clonare il repository

```bash
git clone https://github.com/USERNAME/NOME_REPOSITORY.git
cd NOME_REPOSITORY
```

---

## 2. Creare un ambiente virtuale

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Installare le dipendenze

```bash
pip install -r requirements.txt
```

---

# Configurazione del file `.env`

Creare un file `.env` nella root del progetto:

```env
EUROPEANA_API_KEY=your_api_key

OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_ollama_api_key
OLLAMA_MODEL=gpt-oss:20b
```

---

# Avvio dell’applicazione

```bash
streamlit run app.py
```

---

# Esempi di domande

- Chi è Caravaggio?
- Parlami di Van Gogh
- Quali opere di Picasso sono presenti su Europeana?
- Confronta Monet e Renoir
- Descrivi un’opera di Leonardo da Vinci

---

# Possibili sviluppi futuri

Tra i possibili miglioramenti futuri:

- integrazione di più fonti culturali;
- ricerca semantica tramite embeddings;
- memoria conversazionale avanzata;
- supporto multimodale;
- ranking intelligente dei risultati;
- generazione automatica di suggerimenti per l’utente.

---

# Contesto accademico

Progetto sviluppato nell’ambito della tesi di laurea:

**“Agente AI per un sistema intelligente per la comprensione e la consultazione del patrimonio culturale europeo”**

Corso di Laurea in Informatica.
