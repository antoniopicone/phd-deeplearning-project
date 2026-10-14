"""Caso d'uso 2: ricerca semantica (Sentence-Transformer e baseline TF-IDF), come in semantic-search.ipynb."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ._estratto_search import NOME_MODELLO, STOP_WORDS_IT

METODI = ("sentence-transformer", "tfidf")


def carica_sentence_transformer(nome_modello=NOME_MODELLO):
    from sentence_transformers import SentenceTransformer  # import lento: solo quando serve
    return SentenceTransformer(nome_modello, device="cpu")


class SemanticSearch:
    """Motore di ricerca sulle frasi ham del dataset, con embedding pre-calcolati per tutto il database."""

    def __init__(self, documenti, modello_st, embedding_db, vectorizer, matrice_tfidf):
        self.documenti = documenti
        self.modello_st = modello_st
        self.embedding_db = embedding_db  # [N, 384], normalizzati
        self.vectorizer = vectorizer
        self.matrice_tfidf = matrice_tfidf

    # ------------------------------------------------------------------ indicizzazione
    @classmethod
    def costruisci(cls, percorso_csv, nome_modello=NOME_MODELLO):
        """Database = frasi ham (labels == 0) senza duplicati, come nel notebook."""
        df = pd.read_csv(percorso_csv)
        documenti = df.loc[df["labels"] == 0, "text"].drop_duplicates().tolist()

        modello_st = carica_sentence_transformer(nome_modello)
        embedding_db = modello_st.encode(documenti, batch_size=128, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False)

        vectorizer = TfidfVectorizer(lowercase=True, stop_words=STOP_WORDS_IT)
        matrice_tfidf = vectorizer.fit_transform(documenti)
        return cls(documenti, modello_st, embedding_db, vectorizer, matrice_tfidf)

    # ------------------------------------------------------------------ salvataggio
    def salva(self, cartella):
        cartella = Path(cartella)
        cartella.mkdir(parents=True, exist_ok=True)
        (cartella / "documenti.json").write_text(json.dumps(self.documenti, ensure_ascii=False))
        np.save(cartella / "embedding_db.npy", self.embedding_db)
        joblib.dump({"vectorizer": self.vectorizer, "matrice_tfidf": self.matrice_tfidf}, cartella / "tfidf.joblib")

    @classmethod
    def carica(cls, cartella, nome_modello=NOME_MODELLO):
        cartella = Path(cartella)
        documenti = json.loads((cartella / "documenti.json").read_text())
        tfidf = joblib.load(cartella / "tfidf.joblib")
        return cls(documenti, carica_sentence_transformer(nome_modello), np.load(cartella / "embedding_db.npy"),
                   tfidf["vectorizer"], tfidf["matrice_tfidf"])

    # ------------------------------------------------------------------ ricerca
    def cerca(self, query, top_k=5, metodo="sentence-transformer"):
        """Restituisce i `top_k` documenti più simili a `query` (similarità del coseno)."""
        if metodo not in METODI:
            raise ValueError(f"Metodo sconosciuto '{metodo}': scegli tra {', '.join(METODI)}")
        if metodo == "sentence-transformer":
            vettore_query = self.modello_st.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            scores = self.embedding_db @ vettore_query  # vettori normalizzati: prodotto scalare = coseno
        else:
            scores = cosine_similarity(self.vectorizer.transform([query]), self.matrice_tfidf)[0]
        migliori = np.argsort(scores)[::-1][:top_k]
        return [
            {"rank": rank, "score": round(float(scores[i]), 4), "doc_id": int(i), "documento": self.documenti[i]}
            for rank, i in enumerate(migliori, start=1)
        ]
