"""Caso d'uso 1: filtro spam (LSTM e baseline TF-IDF + Naive Bayes), come in spam-filter.ipynb."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import DataLoader, TensorDataset

from ._estratto_spam import SpamRNN, conta_parole_ripetute, text_to_indices

MODELLI = ("naive-bayes", "lstm")

# Iperparametri del notebook
SEED = 42
EMBEDDING_DIM = 8
HIDDEN_DIM = 8
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHE = 15


class SpamClassifier:
    """Classificatore spam/ham con due modelli: TF-IDF + Naive Bayes (predefinito: il più rapido) e LSTM (il più accurato nel notebook)."""

    def __init__(self, rete, vocabolario, vectorizer, classificatore_nb, metriche=None):
        self.rete = rete.eval()
        self.vocabolario = vocabolario
        self.vectorizer = vectorizer
        self.classificatore_nb = classificatore_nb
        self.metriche = metriche or {}

    # ------------------------------------------------------------------ addestramento
    @classmethod
    def addestra(cls, percorso_csv, num_epoche=NUM_EPOCHE, log=print):
        """Addestra i due modelli come nel notebook (split 70/15/15, pesi dell'epoca con validation loss minima)."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        df = pd.read_csv(percorso_csv)
        testi, labels = df["text"].tolist(), df["labels"].tolist()
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            testi, labels, test_size=0.30, stratify=labels, random_state=SEED)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

        # Baseline: TF-IDF + Naive Bayes, addestrati solo sul train
        max_features = conta_parole_ripetute(X_train, TfidfVectorizer().build_analyzer())
        vectorizer = TfidfVectorizer(max_features=max_features)
        classificatore_nb = MultinomialNB().fit(vectorizer.fit_transform(X_train), y_train)
        pred_nb = classificatore_nb.predict(vectorizer.transform(X_test))

        # Vocabolario costruito solo sul train (0 = padding, 1 = parola sconosciuta)
        parole = sorted(set(" ".join(X_train).lower().split()))
        vocabolario = {parola: i + 2 for i, parola in enumerate(parole)}
        vocabolario["<PAD>"] = 0
        vocabolario["<UNK>"] = 1

        def tensori(t, y):
            return torch.tensor([text_to_indices(x, vocabolario) for x in t]), torch.tensor(y, dtype=torch.float32)

        X_train_t, y_train_t = tensori(X_train, y_train)
        X_val_t, y_val_t = tensori(X_val, y_val)
        X_test_t, y_test_t = tensori(X_test, y_test)
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

        rete = SpamRNN(len(vocabolario), EMBEDDING_DIM, HIDDEN_DIM, rnn_type="LSTM")
        optimizer = optim.Adam(rete.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        miglior_val_loss, miglior_stato, miglior_epoca = float("inf"), None, 0
        for epoca in range(1, num_epoche + 1):
            rete.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(rete(batch_X), batch_y)
                loss.backward()
                optimizer.step()
            rete.eval()
            with torch.no_grad():
                val_loss = criterion(rete(X_val_t), y_val_t).item()
            log(f"  epoca {epoca:2d}/{num_epoche} | validation loss {val_loss:.4f}")
            if val_loss < miglior_val_loss:
                miglior_val_loss, miglior_epoca = val_loss, epoca
                miglior_stato = copy.deepcopy(rete.state_dict())

        rete.load_state_dict(miglior_stato)
        rete.eval()
        with torch.no_grad():
            pred_lstm = (rete(X_test_t) > 0).float().numpy()

        metriche = {
            "lstm": {"test_accuracy": accuracy_score(y_test, pred_lstm), "test_f1": f1_score(y_test, pred_lstm),
                     "miglior_epoca": miglior_epoca},
            "naive-bayes": {"test_accuracy": accuracy_score(y_test, pred_nb), "test_f1": f1_score(y_test, pred_nb),
                            "max_features": max_features},
            "dimensioni_split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        }
        return cls(rete, vocabolario, vectorizer, classificatore_nb, metriche)

    # ------------------------------------------------------------------ salvataggio
    def salva(self, cartella):
        cartella = Path(cartella)
        cartella.mkdir(parents=True, exist_ok=True)
        torch.save(self.rete.state_dict(), cartella / "lstm.pt")
        (cartella / "vocabolario.json").write_text(json.dumps(self.vocabolario, ensure_ascii=False))
        joblib.dump({"vectorizer": self.vectorizer, "classificatore_nb": self.classificatore_nb},
                    cartella / "naive_bayes.joblib")
        (cartella / "metriche.json").write_text(json.dumps(self.metriche, indent=2))

    @classmethod
    def carica(cls, cartella):
        cartella = Path(cartella)
        vocabolario = json.loads((cartella / "vocabolario.json").read_text())
        rete = SpamRNN(len(vocabolario), EMBEDDING_DIM, HIDDEN_DIM, rnn_type="LSTM")
        rete.load_state_dict(torch.load(cartella / "lstm.pt", map_location="cpu"))
        baseline = joblib.load(cartella / "naive_bayes.joblib")
        metriche = json.loads((cartella / "metriche.json").read_text())
        return cls(rete, vocabolario, baseline["vectorizer"], baseline["classificatore_nb"], metriche)

    # ------------------------------------------------------------------ predizione
    def predici(self, testo, modello="naive-bayes"):
        """Restituisce la probabilità che `testo` sia spam secondo il modello scelto."""
        if modello not in MODELLI:
            raise ValueError(f"Modello sconosciuto '{modello}': scegli tra {', '.join(MODELLI)}")
        if modello == "lstm":
            X = torch.tensor([text_to_indices(testo, self.vocabolario)])
            with torch.no_grad():
                probabilita = torch.sigmoid(self.rete(X)).item()
        else:
            probabilita = float(self.classificatore_nb.predict_proba(self.vectorizer.transform([testo]))[0, 1])
        return {
            "modello": modello,
            "spam": probabilita >= 0.5,
            "etichetta": "spam" if probabilita >= 0.5 else "ham",
            "probabilita_spam": round(probabilita, 4),
        }
