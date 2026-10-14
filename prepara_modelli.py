"""Addestra il filtro spam e indicizza il database della ricerca semantica, salvando gli artefatti.

Viene eseguito una volta sola (in Docker, durante la build dell'immagine), così l'API all'avvio
deve solo caricare i modelli pronti.

Uso:
    python prepara_modelli.py [--dataset datasets/italian.csv] [--artefatti artefatti]
"""
import argparse
import time

from spamsearch import SemanticSearch, SpamClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="datasets/italian.csv")
    parser.add_argument("--artefatti", default="artefatti")
    args = parser.parse_args()

    print("--- Filtro spam: addestramento LSTM e TF-IDF + Naive Bayes ---")
    inizio = time.perf_counter()
    spam = SpamClassifier.addestra(args.dataset)
    spam.salva(f"{args.artefatti}/spam")
    for modello in ("lstm", "naive-bayes"):
        m = spam.metriche[modello]
        print(f"{modello}: accuratezza {m['test_accuracy']:.2%} | F1 {m['test_f1']:.4f}")
    print(f"Completato in {time.perf_counter() - inizio:.1f}s\n")

    print("--- Ricerca semantica: embedding del database e indice TF-IDF ---")
    inizio = time.perf_counter()
    ricerca = SemanticSearch.costruisci(args.dataset)
    ricerca.salva(f"{args.artefatti}/search")
    print(f"{len(ricerca.documenti)} documenti indicizzati in {time.perf_counter() - inizio:.1f}s")
