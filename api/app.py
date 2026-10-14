"""API Flask dei due casi d'uso: POST /check-spam, POST /search-simil e GET /messaggi (database della ricerca)."""
import os
import time
from pathlib import Path

from flask import Flask, jsonify, request

from spamsearch import SemanticSearch, SpamClassifier
from spamsearch.search import METODI
from spamsearch.spam import MODELLI

TOP_K_MAX = 50
PER_PAGINA_MAX = 100
LUNGHEZZA_MAX_TESTO = 2000


def crea_app(cartella_artefatti=None):
    cartella_artefatti = Path(cartella_artefatti or os.environ.get("ARTEFATTI", "artefatti"))
    app = Flask(__name__, static_folder=None)
    app.json.ensure_ascii = False
    app.json.sort_keys = False

    # I modelli vengono caricati una volta sola, all'avvio
    spam = SpamClassifier.carica(cartella_artefatti / "spam")
    ricerca = SemanticSearch.carica(cartella_artefatti / "search")

    def errore(messaggio, codice=400):
        return jsonify({"errore": messaggio}), codice

    def leggi_testo(corpo, campo):
        valore = corpo.get(campo)
        if not isinstance(valore, str) or not valore.strip():
            return None, errore(f"Il campo '{campo}' è obbligatorio e deve essere una stringa non vuota")
        if len(valore) > LUNGHEZZA_MAX_TESTO:
            return None, errore(f"Il campo '{campo}' supera i {LUNGHEZZA_MAX_TESTO} caratteri")
        return valore.strip(), None

    def intero(nome, predefinito, minimo, massimo):
        """Legge un parametro intero dalla query string, o None se non è valido."""
        try:
            valore = int(request.args.get(nome, predefinito))
        except ValueError:
            return None
        return valore if minimo <= valore <= massimo else None

    @app.get("/")
    def indice():
        return jsonify({"servizio": "spamsearch-api",
                        "endpoint": ["POST /check-spam", "POST /search-simil", "GET /messaggi", "GET /health"]})

    @app.get("/health")
    def health():
        return jsonify({
            "stato": "ok",
            "check-spam": {"modelli": list(MODELLI), "metriche_test": spam.metriche},
            "search-simil": {"metodi": list(METODI), "documenti": len(ricerca.documenti)},
        })

    @app.post("/check-spam")
    def check_spam():
        corpo = request.get_json(silent=True) or {}
        testo, risposta_errore = leggi_testo(corpo, "testo")
        if risposta_errore:
            return risposta_errore
        modello = corpo.get("modello", "naive-bayes")
        if modello not in MODELLI:
            return errore(f"Modello sconosciuto '{modello}': scegli tra {', '.join(MODELLI)}")

        inizio = time.perf_counter()
        risultato = spam.predici(testo, modello)
        return jsonify({"testo": testo, **risultato, "tempo_ms": round((time.perf_counter() - inizio) * 1000, 2)})

    @app.post("/search-simil")
    def search_simil():
        corpo = request.get_json(silent=True) or {}
        query, risposta_errore = leggi_testo(corpo, "query")
        if risposta_errore:
            return risposta_errore
        metodo = corpo.get("metodo", "sentence-transformer")
        if metodo not in METODI:
            return errore(f"Metodo sconosciuto '{metodo}': scegli tra {', '.join(METODI)}")
        top_k = corpo.get("top_k", 5)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= TOP_K_MAX:
            return errore(f"'top_k' deve essere un intero tra 1 e {TOP_K_MAX}")

        inizio = time.perf_counter()
        risultati = ricerca.cerca(query, top_k, metodo)
        return jsonify({"query": query, "metodo": metodo, "top_k": top_k, "risultati": risultati,
                        "tempo_ms": round((time.perf_counter() - inizio) * 1000, 2)})

    @app.get("/messaggi")
    def messaggi():
        """Elenco a pagine dei messaggi del database della ricerca semantica."""
        totale = len(ricerca.documenti)
        per_pagina = intero("per_pagina", 20, 1, PER_PAGINA_MAX)
        if per_pagina is None:
            return errore(f"'per_pagina' deve essere un intero tra 1 e {PER_PAGINA_MAX}")
        pagine = max(1, -(-totale // per_pagina))  # divisione arrotondata per eccesso
        pagina = intero("pagina", 1, 1, pagine)
        if pagina is None:
            return errore(f"'pagina' deve essere un intero tra 1 e {pagine}")

        inizio = (pagina - 1) * per_pagina
        return jsonify({
            "totale": totale, "pagina": pagina, "pagine": pagine, "per_pagina": per_pagina,
            "messaggi": [{"doc_id": i, "documento": ricerca.documenti[i]}
                         for i in range(inizio, min(inizio + per_pagina, totale))],
        })

    return app


if __name__ == "__main__":
    crea_app().run(host="0.0.0.0", port=int(os.environ.get("PORTA", 8000)))
