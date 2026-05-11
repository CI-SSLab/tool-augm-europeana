from crewai import Task


def build_answer_task(agent):
    #Costruisce il task con il prompt che guida la risposta dell'agente.
    #I segnaposto {question} e {current_artist} vengono sostituiti da CrewAI
    #con i valori passati a crew.kickoff(inputs={...}).
    return Task(
        description=(
            "Domanda dell'utente: {question}\n"
            "Artista corrente del thread: {current_artist}\n\n"

            #La "REGOLA CRITICA" è in cima al prompt apposta: senza, gpt-oss
            #tende a chiamare il tool più volte ed entrare in loop
            "REGOLA CRITICA (rispettala sempre):\n"
            "Chiami il tool europeana_search ESATTAMENTE UNA volta. "
            "DOPO aver ricevuto la risposta del tool (anche se è vuota o contiene "
            "un campo 'error'), NON chiami più nessun tool e procedi SUBITO "
            "alla risposta finale racchiusa tra <<<RISPOSTA>>> e <<<FINE>>>. "
            "Non ripetere mai la stessa chiamata al tool.\n\n"

            "Procedura:\n"
            "1) Chiama una sola volta europeana_search passando come argomento "
            "SOLO il nome dell'artista (es. 'Caravaggio', 'Pablo Picasso'). "
            "Se la domanda è un follow-up (es. 'quando è morto?', 'dove è nato?'), "
            "usa il nome che trovi in 'Artista corrente del thread'.\n"
            "2) Subito dopo la chiamata, scrivi una risposta in italiano di 4-8 frasi "
            "sull'artista. Usa la tua conoscenza generale per biografia, periodo, "
            "stile e importanza. Cita 1-3 opere prese dai risultati del tool come esempi.\n"
            "3) Se i risultati del tool sono vuoti o contengono un campo 'error', "
            "rispondi comunque con la conoscenza generale e segnala che Europeana "
            "non ha restituito opere certe.\n"
            "4) Non inventare titoli, anni o url. Tutti i dati specifici devono "
            "venire dal tool.\n\n"

            #I marcatori <<<RISPOSTA>>> e <<<FINE>>> mi servono in app.py per
            #estrarre solo il testo finale ed evitare che il ragionamento
            #interno del modello finisca nella chat
            "Formato dell'output (obbligatorio):\n"
            "Racchiudi la risposta finale tra i marcatori <<<RISPOSTA>>> e <<<FINE>>>. "
            "Termina la risposta con una sezione 'FONTI:' seguita dagli url del tool, "
            "uno per riga. Se non hai url, scrivi 'FONTI: nessuna fonte Europeana disponibile'."
        ),
        expected_output=(
            "Testo italiano racchiuso tra <<<RISPOSTA>>> e <<<FINE>>> "
            "con sezione FONTI conclusiva."
        ),
        agent=agent,
        async_execution=False,
    )