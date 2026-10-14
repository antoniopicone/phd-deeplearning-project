"""GENERATO AUTOMATICAMENTE da spam-filter.ipynb con estrai_da_notebook.py: non modificare a mano."""
import torch
import torch.nn as nn


def text_to_indices(testo, vocabolario, max_len=10):
    indici = [vocabolario.get(p, vocabolario["<UNK>"]) for p in testo.lower().split()]
    # Padding o troncamento per allineare le lunghezze dei messaggi
    if len(indici) < max_len:
        indici += [0] * (max_len - len(indici))
    else:
        indici = indici[:max_len]
    return indici


# Modello Ricorrente Unificato (Gestisce sia RNN classica, che GRU, che LSTM)
class SpamRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, rnn_type="LSTM"):
        super().__init__()
        # padding_idx=0 dice alla rete di ignorare i vettori di riempimento
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if rnn_type == "RNN":
            self.core = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == "GRU":
            self.core = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == "LSTM":
            self.core = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1) # Spara 1 singolo Logit grezzo [0.5]

    def forward(self, text):
        embedded = self.embedding(text)
        if isinstance(self.core, nn.LSTM):
            output, (hidden, cell) = self.core(embedded)
        else:
            output, hidden = self.core(embedded)
        # Esaminiamo l'ultimo hidden state emesso
        return self.fc(hidden[-1, :, :]).squeeze(1)


# Modello Parallelo basato su Transformer (Self-Attention)
class SpamTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, hidden_dim, num_layers, max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, text):
        batch_size, seq_len = text.size()
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(text.device)

        # Somma Semantica + Posizionale [0.5]
        x = self.embedding(text) + self.pos_embedding(pos)
        features = self.transformer(x)
        # Pooling: facciamo la media delle caratteristiche di tutte le parole del messaggio
        pooled = features.mean(dim=1)
        return self.fc(pooled).squeeze(1)
