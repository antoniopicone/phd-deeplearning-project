# spam-guard

Classificatore di **sospetto spam** per il mio client di posta (Rust): un servizio REST locale
che, a ogni nuova email, restituisce `P(spam)`.

Progetto d'esame del corso *Advanced Deep Learning*: confronta un **Transformer piccolo**
con una baseline **TF-IDF + regressione logistica** e verifica se la differenza di accuracy
è **statisticamente significativa** (McNemar, bootstrap appaiato, 5×2cv F-test, Cochran's Q).
Nel banco di prova entrano anche un BERT pre-addestrato, **Jev** (TypeSafe AI) e,
opzionalmente, un SLM locale.

```
nuova email ──► client Rust ──POST /v1/classify──► spam-guard (Docker, 127.0.0.1:8088)
                     ▲                                   │  TF-IDF │ mini-Transformer │ BERT-tiny
                     └──────── {spam_probability, is_spam} ◄──┘
```

## Risultati

SpamAssassin Public Corpus, 5855 email dopo la deduplicazione (30% spam), split stratificato
70/15/15, test set di 879 email. Training su CPU (2 core); su GPU i tempi scendono di 10–30×.

| Modello | Parametri | Accuracy (IC95%) | MCC | FPR | Training |
|---|---|---|---|---|---|
| TF-IDF + LogReg | 100k feature | **98,75%** [97,8; 99,3] | **0,970** | **1,1%** | 9 s |
| Mini-Transformer da zero | 1,37 M | 98,18% [97,1; 98,9] | 0,957 | 2,0% | 6,5 min |
| BERT-tiny pre-addestrato | 4,39 M | 96,81% [95,4; 97,8] | 0,924 | 2,1% | 5,6 min |

- **TF-IDF vs mini-Transformer:** McNemar p = 0,18, Δacc −0,57 pp con IC bootstrap [−1,25; +0,11]
  → sul singolo run la differenza **non è significativa**. Il **5×2cv F-test** (10 addestramenti)
  dà p = 0,03 con 10 differenze su 10 a favore del TF-IDF: come algoritmo, con ~4000 email,
  il TF-IDF è leggermente ma sistematicamente migliore.
- **Cochran's Q** = 13,5 (p = 0,001); dopo Holm è significativa solo TF-IDF vs BERT-tiny (p = 0,006).
- Le **mappe di attenzione** mostrano che il primo blocco encoder distribuisce l'attenzione in modo
  uniforme (≈ 1/n): fa una media, cioè un bag-of-words. Con pochi dati il transformer non sfrutta
  l'ordine delle parole — lo stesso fenomeno di ViT senza pre-training su grandi dataset.
- Il servizio usa come default il modello con MCC più alto **in validation** (oggi il TF-IDF: ~5 ms/email).

## Struttura

```
spam-guard/
├── notebooks/spam_guard.ipynb   prototipo completo (Jupyter / Colab), già eseguito con output
├── spamguard/                   libreria condivisa da notebook, script e servizio
│   ├── data.py                  download SpamAssassin, parsing RFC 822, normalizzazione, split
│   ├── baseline.py              TF-IDF (parole + n-grammi di caratteri) + LogisticRegression
│   ├── transformer.py           mini-Transformer encoder in Keras 3 + estrazione attivazioni
│   ├── pretrained.py            fine-tuning BERT con KerasHub
│   ├── stats.py                 metriche, McNemar, bootstrap, 5×2cv, Cochran's Q, Holm
│   ├── external.py              client Jev (TypeSafe) e SLM locale OpenAI-compatibile
│   └── plotting.py              stile dei grafici
├── scripts/train.py             pipeline di training riproducibile (CLI, usata anche da Docker)
├── service/app.py               API REST (FastAPI)
├── Dockerfile                   build multi-stage: train → serve
├── docker-compose.yml           servizio su 127.0.0.1:8088
├── clients/rust/                crate `spamguard-client` + esempio
├── models/                      artefatti addestrati (generati)
└── reports/                     predizioni sul test set, diagramma dell'architettura
```

## 1. Prototipo: notebook

**Colab** (consigliato per la GPU): carica `notebooks/spam_guard.ipynb`, imposta
`REPO_URL` nella prima cella con l'URL del repository, *Runtime → Cambia tipo di runtime → GPU*,
poi *Esegui tutto*. Su GPU si può alzare `PRESET` a `bert_small_en_uncased` o
`distil_bert_base_multi` (multilingue, per la posta in italiano).

**In locale:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # serve anche graphviz per il diagramma: apt/brew install graphviz
jupyter lab notebooks/spam_guard.ipynb
```

Il notebook scarica da solo lo [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)
in `data/raw/` e contiene, nell'ordine: dati e leakage, baseline TF-IDF, architettura del
mini-Transformer (diagramma + conteggio parametri), training, **visualizzazione di attivazioni
e mappe di attenzione**, fine-tuning BERT, confronto statistico, Jev/SLM, classificatore vs LLM,
parallelismo e GPU, export.

### Aggiungere la propria posta (italiano)

Non esistono grandi corpora pubblici di spam email in italiano. Esporta *Inbox* e *Junk*
dal client in formato mbox e aggiungili:

```bash
python scripts/train.py --extra-ham ~/Mail/Inbox.mbox --extra-spam ~/Mail/Junk.mbox \
                        --pretrained distil_bert_base_multi      # meglio su GPU
```

## 2. Training da riga di comando

```bash
python scripts/train.py                      # TF-IDF + mini-Transformer + BERT-tiny → models/
python scripts/train.py --pretrained ""      # salta il BERT (più veloce)
```

Il modello di default del servizio viene scelto **sulla validation** (MCC più alto), mai sul test set,
e scritto in `models/metadata.json`.

## 3. Deploy: Docker + API REST

```bash
docker compose up -d --build                 # usa i modelli in ./models (es. scaricati da Colab)
# oppure: addestra DENTRO la build, nessun artefatto locale necessario
docker build --target serve-trained -t spam-guard:latest .
```

| Immagine | Comando | Dimensione | Modelli |
|---|---|---|---|
| completa | `docker build -t spam-guard .` | ~3,9 GB | TF-IDF, Transformer, BERT, ensemble |
| slim | `docker build --build-arg REQUIREMENTS=service/requirements-slim.txt -t spam-guard:slim .` | ~0,8 GB | solo TF-IDF (il default attuale), ~3 ms/email |

Nell'immagine slim TensorFlow non c'è: il servizio salta da solo i file `.keras` e lo scrive nel log.
Se Docker Hub risponde *429 Too Many Requests*:
`docker build --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim ...`

Il container gira come utente non root, con filesystem in sola lettura, e la porta è esposta
**solo su 127.0.0.1**: il servizio è pensato per il client di posta sulla stessa macchina.

### API

| Metodo | Percorso | Descrizione |
|---|---|---|
| GET | `/health` | stato e modelli caricati |
| GET | `/v1/models` | modello di default, data di training, metriche di test |
| POST | `/v1/classify` | una email: `{subject, body}` oppure `{raw}` (RFC 822 completo) |
| POST | `/v1/classify/batch` | `{"emails": [...]}`, fino a 256 |

Campi opzionali: `model` (`tfidf` · `transformer` · `pretrained` · `ensemble`) e `threshold` (0–1).

```bash
curl -s localhost:8088/v1/classify -H 'content-type: application/json' \
  -d '{"subject":"URGENT: you have WON $1,000,000","body":"Click here to claim your prize now!"}'
# {"spam_probability":0.997951,"is_spam":true,"threshold":0.5,"model":"tfidf","latency_ms":4.84}
```

Documentazione interattiva OpenAPI: <http://127.0.0.1:8088/docs>.

## 4. Integrazione nel client Rust

```toml
# Cargo.toml del client di posta
spamguard-client = { path = "../spam-guard/clients/rust" }
```

```rust
use spamguard_client::SpamGuard;
use std::time::Duration;

let guard = SpamGuard::new("http://127.0.0.1:8088", Duration::from_secs(2))?;
// alla ricezione (es. dopo IMAP FETCH BODY[])
if let Some(v) = guard.check_or_allow(&raw_message).await {
    if v.is_spam { move_to_junk(uid).await?; }
}   // None = servizio non raggiungibile → fail-open: la mail resta in Inbox
```

Prova veloce: `cd clients/rust && cargo run --example check -- /percorso/messaggio.eml`.

La politica è **fail-open**: se il servizio è giù o lento, la mail non viene bloccata
(meglio uno spam in Inbox che una mail vera persa).

## 5. Jev e SLM nel banco di prova (opzionale)

```bash
export TYPESAFE_API_KEY=...                          # Jev, https://docs.typesafe.ai
export SLM_BASE_URL=http://127.0.0.1:8080/v1          # llama-swap / mlx_lm.server / llama.cpp / Ollama
export SLM_MODEL=qwen3.5-4b
```

Il notebook (sezione 7) li valuta zero-shot su un sottoinsieme stratificato di 300 email del
test set, con McNemar + Holm contro gli altri modelli, e ne misura la latenza. Le risposte
sono messe in cache in `cache/*.jsonl`. Jev resta **solo** nel banco di prova: il servizio di
produzione è locale e non invia la posta a terzi.

## 6. Benchmark statistico, in breve

| Situazione | Test |
|---|---|
| 2 modelli, addestrati una volta, stesso test set | **McNemar** (+ IC bootstrap della differenza) |
| 2 algoritmi, includendo la variabilità del training | **5×2cv F-test** (Alpaydin) |
| 3+ modelli sullo stesso test set | **Cochran's Q**, poi McNemar a coppie con **Holm** |
| più dataset | Wilcoxon (2 modelli), Friedman + Nemenyi (k modelli) |

Il documento *"Come fare benchmark tra classificatori"* spiega il perché di ciascuna scelta.

## Riferimenti

- Vaswani et al. (2017), *Attention Is All You Need*.
- Devlin et al. (2019), *BERT*; Turc et al. (2019), *Well-Read Students Learn Better* (BERT-tiny/small).
- Dietterich (1998), *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*.
- Alpaydin (1999), *Combined 5×2 cv F Test for Comparing Supervised Classification Learning Algorithms*.
- Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR.
- TypeSafe AI, [documentazione di Jev](https://docs.typesafe.ai/introduction).
