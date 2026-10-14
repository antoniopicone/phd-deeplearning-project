"""Estrae dai notebook le definizioni riusabili e genera i moduli della libreria `spamsearch`.

Il codice dei modelli vive nei notebook: questo script lo copia nella libreria, così API e
notebook usano esattamente le stesse classi e funzioni. Da ogni cella di codice vengono prese
solo le definizioni elencate in ESTRAZIONI (funzioni, classi, costanti), con i commenti che le
precedono; tutto il resto (training, grafici, stampe) viene ignorato.

Uso:
    python estrai_da_notebook.py
"""
import ast
import json
from pathlib import Path

RADICE = Path(__file__).resolve().parent

ESTRAZIONI = [
    {
        "notebook": "spam-filter.ipynb",
        "modulo": "spamsearch/_estratto_spam.py",
        "import": ["from collections import Counter", "", "import torch", "import torch.nn as nn"],
        "nomi": ["conta_parole_ripetute", "text_to_indices", "SpamRNN", "SpamTransformer"],
    },
    {
        "notebook": "semantic-search.ipynb",
        "modulo": "spamsearch/_estratto_search.py",
        "import": [],
        "nomi": ["STOP_WORDS_IT", "NOME_MODELLO"],
    },
]


def celle_di_codice(percorso_notebook):
    """Restituisce il sorgente di ogni cella di codice, senza i comandi speciali di Jupyter (%, !)."""
    notebook = json.loads(percorso_notebook.read_text(encoding="utf-8"))
    for cella in notebook["cells"]:
        if cella["cell_type"] != "code":
            continue
        righe = "".join(cella["source"]).splitlines()
        # Le righe magiche vengono svuotate (non rimosse) per mantenere i numeri di riga
        yield "\n".join("" if r.lstrip().startswith(("%", "!")) else r for r in righe)


def nome_definito(nodo):
    """Nome definito da un nodo di primo livello (funzione, classe o assegnazione semplice)."""
    if isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
        return nodo.name
    if isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 and isinstance(nodo.targets[0], ast.Name):
        return nodo.targets[0].id
    return None


def sorgente_con_commenti(sorgente, nodo):
    """Sorgente del nodo, inclusi i commenti immediatamente sopra di esso."""
    righe = sorgente.splitlines()
    inizio = nodo.lineno - 1
    while inizio > 0 and righe[inizio - 1].lstrip().startswith("#") and not righe[inizio - 1].lstrip().startswith("# ="):
        inizio -= 1
    return "\n".join(righe[inizio:nodo.end_lineno])


def estrai(spec):
    percorso_notebook = RADICE / spec["notebook"]
    trovati = {}
    for sorgente in celle_di_codice(percorso_notebook):
        for nodo in ast.parse(sorgente).body:
            nome = nome_definito(nodo)
            if nome in spec["nomi"]:
                if nome in trovati:
                    raise ValueError(f"'{nome}' è definito più volte in {spec['notebook']}")
                trovati[nome] = sorgente_con_commenti(sorgente, nodo)

    mancanti = [n for n in spec["nomi"] if n not in trovati]
    if mancanti:
        raise ValueError(f"Definizioni non trovate in {spec['notebook']}: {mancanti}")

    intestazione = [
        f'"""GENERATO AUTOMATICAMENTE da {spec["notebook"]} con estrai_da_notebook.py: non modificare a mano."""',
        *spec["import"],
    ]
    contenuto = "\n".join(intestazione) + "\n\n\n" + "\n\n\n".join(trovati[n] for n in spec["nomi"]) + "\n"
    percorso_modulo = RADICE / spec["modulo"]
    percorso_modulo.write_text(contenuto, encoding="utf-8")
    print(f"{spec['notebook']} -> {spec['modulo']}: {', '.join(spec['nomi'])}")


if __name__ == "__main__":
    for spec in ESTRAZIONI:
        estrai(spec)
