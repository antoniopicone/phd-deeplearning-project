# Progetto per l'Esame di Advanced Deep Learning: Architetture Neurali a Confronto per l'Elaborazione del Linguaggio Naturale

---

## 1. Introduzione e Razionale del Progetto
Il campo del Natural Language Processing (NLP) ha vissuto nell'ultimo decennio una rivoluzione radicale, transitando da modelli statistici tradizionali e architetture sequenziali ricorsive fino al paradigma dominante basato sulla *Self-Attention* (Transformer).

Questo progetto d'esame nasce con l'obiettivo di mappare empiricamente questo percorso storico e ingegneristico, analizzando il trade-off tra **complessità computazionale**, **efficienza hardware** e **accuratezza semantica**. Il cuore dello studio consiste nel dimostrare che l'algoritmo "più moderno" non è necessariamente sempre l'unica scelta corretta, ma che la progettazione di un sistema di Intelligenza Artificiale deve sempre basarsi su un'attenta analisi del caso d'uso specifico.

Per fare ciò, il progetto si articola su **due pilastri applicativi opposti** e complementari:
1. **Caso d'uso 1: Filtro Spam (Classificazione Binaria Leggera)**: un problema guidato da parole chiave e testi brevi, dove si valuta l'efficienza dei modelli computazionalmente leggeri rispetto a quelli pesanti.
2. **Caso d'uso 2: Ricerca Semantica (Semantic Textual Similarity)**: un problema complesso in cui il significato prescinde dalle singole parole utilizzate, dove si analizza la capacità delle reti di mappare i concetti in uno spazio geometrico multidimensionale.

---

## 2. Tecnologie a Confronto
Il progetto implementa e analizza quattro approcci fondamentali, ciascuno rappresentante di un'epoca storica specifica, più un modello pre-addestrato usato come riferimento nel secondo caso d'uso.

### A. TF-IDF (L'approccio Statistico Classico)
*   **Meccanismo:** Trasforma il testo in una matrice di frequenze pesata (*Term Frequency - Inverse Document Frequency*). Nel filtro spam è seguito da un classificatore **Naive Bayes**, che sfrutta il teorema di Bayes ipotizzando l'indipendenza condizionale tra le parole (*Bag of Words*); nella ricerca semantica i vettori TF-IDF vengono confrontati direttamente con la *Cosine Similarity*.
*   **Ruolo nel progetto:** Funge da *baseline*. Dimostra l'efficacia dei metodi deterministici basati sull'occorrenza lessicale pura, privi di nozione di ordine o sintassi.

### B. Recurrent Neural Networks (RNN Classica)
*   **Meccanismo:** Introduce per la prima volta un ciclo di feedback temporale per elaborare sequenze, aggiornando un *Hidden State* passo dopo passo.
*   **Ruolo nel progetto:** Evidenzia storicamente l'introduzione della memoria nel Deep Learning, ma funge anche da esempio didattico per il fenomeno della scomparsa del gradiente, mostrando l'incapacità di preservare informazioni su sequenze lunghe (a causa delle derivate nella backward pass).

### C. LSTM (Long Short-Term Memory) & GRU (Gated Recurrent Unit)
*   **Meccanismo:** Varianti avanzate delle RNN. La **LSTM** introduce il *Cell State* (un nastro trasportatore lineare per l'errore) regolato da tre porte logiche (Forget, Input, Output). La **GRU** semplifica questa struttura fondendo i flussi in un unico *Hidden State* gestito da sole due porte (Update, Reset).
*   **Ruolo nel progetto:** Rappresentano l'apice del Deep Learning sequenziale e mostrano come si risolve il problema della scomparsa del gradiente. In teoria la GRU, con circa il 25% di parametri in meno, dovrebbe addestrarsi più velocemente della LSTM: gli esperimenti mostrano che **sulla GPU Apple (MPS) accade il contrario** (vedi §5), perché la velocità reale dipende anche da quanto è ottimizzata l'implementazione di ciascun layer sull'hardware.

### D. Transformer (Encoder-only Architecture)
*   **Meccanismo:** Elimina completamente i cicli ricorrenti e la natura sequenziale dell'elaborazione. Introduce il meccanismo di *Self-Attention* (Scaled Dot-Product Attention) combinato con i *Positional Embeddings* per analizzare le relazioni tra tutte le parole della frase contemporaneamente (in parallelo).
*   **Ruolo nel progetto:** Rappresenta lo stato dell'arte del Deep Learning parallelo e bidirezionale. Il progetto mostra che l'architettura da sola non basta: un Transformer piccolo e addestrato da zero non supera le reti ricorrenti nello spam filter, mentre lo stesso tipo di architettura, **pre-addestrata** su grandi quantità di testo, domina nella ricerca semantica.

### E. Sentence-Transformer pre-addestrato (solo Caso d'Uso 2)
*   **Modello:** [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2): Transformer a 12 strati, embedding a 384 dimensioni, pre-addestrato su testi in oltre 50 lingue (italiano compreso) e poi addestrato con una loss contrastiva su milioni di coppie di parafrasi.
*   **Ruolo nel progetto:** Termine di confronto per mostrare cosa serve, oltre all'architettura, per ottenere embedding che catturano il significato: pre-training su larga scala e un obiettivo di addestramento semantico.

---

## 3. Dataset
Entrambi i casi d'uso usano `datasets/italian.csv`, *Amazon (Massive) Italian Spam Collection Dataset*: 18.719 messaggi brevi in italiano con etichetta `labels` (1 = spam, 0 = ham). Il dataset è **bilanciato**: 9.604 spam e 9.115 ham.

---

## 4. I Due Casi d'Uso

### 📊 Caso d'Uso 1: Lo Spam Filter: [`spam-filter.ipynb`](spam-filter.ipynb)
*   **Razionale:** Dimostrare che per compiti di classificazione semplici e circoscritti, l'approccio statistico classico (TF-IDF) o le reti leggere raggiungono metriche eccellenti con una frazione del costo computazionale.
*   **Focus per l'esame:** Analisi dell'efficienza e della sostenibilità hardware: accuratezza confrontata con tempo di addestramento e tempo di inferenza.

**Pipeline:**
*   **Split train / validation / test 70/15/15**, stratificato sulla label, eseguito **prima** di costruire TF-IDF e vocabolario per evitare *data leakage* (13.103 / 2.808 / 2.808 messaggi).
*   **Baseline:** TF-IDF (100 feature) + Multinomial Naive Bayes.
*   **Reti neurali:** RNN, GRU, LSTM e Transformer (1 strato, 2 teste di attenzione) con embedding e hidden state a 8 dimensioni, sequenze di 10 token.
*   **Addestramento:** 15 epoche, batch da 64, Adam con learning rate `0.01`, loss `BCEWithLogitsLoss`. A ogni epoca il modello viene valutato sul validation set e si conservano i pesi dell'epoca con la **validation loss minima**; il test set viene usato **una sola volta**, alla fine.
*   **Metriche:** accuratezza e F1-Score sul test set; tempo di addestramento (solo le epoche, con sincronizzazione della GPU prima di ogni misura); tempo di inferenza dal testo grezzo alla predizione, sia in batch (tutto il test set) sia su un singolo messaggio alla volta.
*   **Hardware:** GPU Apple (MPS) se disponibile, altrimenti CUDA o CPU.

### 🎯 Caso d'Uso 2: Il Motore di Ricerca Semantica: [`semantic-search.ipynb`](semantic-search.ipynb)
*   **Database:** le frasi *ham* del dataset, senza duplicati (**7.263 documenti**).
*   **Razionale:** Mostrare i limiti del TF-IDF, che ragiona solo sulle parole in comune e non riconosce i sinonimi ("vettura" ≠ "auto" ≠ "macchina"), e di reti con pesi non addestrati, e mostrare come un Transformer pre-addestrato produca embedding che catturano il significato.
*   **Metodologia:** Estrazione delle caratteristiche dai modelli (*Feature Extraction*) tramite funzioni dedicate `get_embedding()`, seguita dal calcolo geometrico della *Cosine Similarity* tra la query e ogni documento; si restituiscono i 5 documenti più simili.

**Metodi confrontati:**
*   **LSTM, GRU e Transformer** (embedding a 16 dimensioni, sequenze di 16 token) con **pesi casuali, non addestrati**: rappresentano l'architettura "nuda", senza alcuna conoscenza del linguaggio. L'embedding è lo stato nascosto all'ultima parola reale (reti ricorrenti) o la media sulle sole parole reali (Transformer), escludendo il padding.
*   **TF-IDF + Cosine Similarity**, con una lista di *stop words* italiane (articoli, preposizioni, pronomi, interrogativi, dimostrativi).
*   **Sentence-Transformer pre-addestrato** (vedi §2.E).

**Valutazione della qualità della ricerca (LLM come giudice):**
1.  **20 query di valutazione** su argomenti presenti nel database, formulate quando possibile con parole diverse da quelle dei documenti, per non avvantaggiare i metodi lessicali.
2.  **Pooling** (metodo TREC): i primi 5 risultati di tutti i metodi vengono uniti e ogni coppia (query, documento) distinta viene giudicata una sola volta.
3.  **Giudice:** un LLM locale, `gemma3:12b` via [Ollama](https://ollama.com), assegna un voto di pertinenza 0/1/2 (non / parzialmente / pienamente pertinente), con temperatura 0 e output JSON vincolato. I giudizi sono salvati in cache in `datasets/giudizi_llm.csv`.
4.  **Metriche:** **P@5**, la quota dei primi 5 risultati pertinenti, e **MRR**, cioè 1 / posizione del primo risultato pertinente. Entrambe sono mediate sulle 20 query, con **intervallo di confidenza al 95%** calcolato con bootstrap sulle query.
5.  **Validazione del giudice:** su un campione casuale di 50 coppie (`datasets/campione_validazione_giudice.csv`) i voti dell'LLM sono stati confrontati con quelli di un secondo annotatore che non vedeva i voti dell'LLM.

---

## 5. Risultati

### Caso d'Uso 1: Spam Filter
Risultati sul test set (2.808 messaggi), GPU Apple MPS:

| Modello | Accuratezza | F1 | Training | Inferenza batch (per msg) | Inferenza singolo msg |
|---|---|---|---|---|---|
| LSTM | **95.5%** | **0.956** | 4.9 s | 2.9 µs | 0.52 ms |
| GRU | 95.3% | 0.954 | 13.2 s | 3.5 µs | 1.01 ms |
| RNN | 94.8% | 0.950 | 9.8 s | 3.7 µs | 0.91 ms |
| Transformer | 94.6% | 0.948 | 9.7 s | 4.3 µs | 0.76 ms |
| TF-IDF + Naive Bayes | 91.8% | 0.921 | **58 ms** | 3.6 µs | **0.16 ms** |

**Osservazioni principali:**
*   Le reti neurali guadagnano **3–4 punti di accuratezza** sulla baseline, ma con un training **circa 100–200 volte più lungo**.
*   Tutte le reti raggiungono la validation loss minima già alla **2ª–3ª epoca**: le epoche successive portano a overfitting.
*   Sulla GPU Apple la **LSTM è la rete più veloce** e la **GRU la più lenta**, contrariamente alle attese teoriche basate sul numero di parametri.
*   **Inferenza in batch:** tutti i metodi impiegano 3–4 µs per messaggio, perché il costo è dominato dalla tokenizzazione in Python. **Sul singolo messaggio** la baseline è 3–6 volte più veloce, perché le reti pagano l'overhead di trasferimento dei dati verso la GPU.
*   Le differenze tra le reti (0.2–1 punto) sono dello stesso ordine della variabilità tra un'esecuzione e l'altra sulla GPU: per confronti rigorosi servono più run con seed diversi.

Grafico prodotto: scatter **accuratezza vs tempo** su tre pannelli (training, inferenza in batch, inferenza a singolo messaggio), in scala logaritmica.

### Caso d'Uso 2: Ricerca Semantica
Per la query *"Qual è il prezzo di questa vettura"*:
*   **Reti non addestrate:** risultati casuali, senza alcun legame con la query.
*   **TF-IDF:** "vettura" non compare nel database, quindi l'unica parola utile resta "prezzo" e restituisce frasi su prezzi di smartphone, cibo, ecc.
*   **Sentence-Transformer:** restituisce frasi su **auto** e **macchine**, pur senza alcuna parola in comune con la query.

**Qualità della ricerca** (20 query, giudice `gemma3:12b`, intervallo di confidenza al 95%):

| Metodo | P@5 | IC 95% | MRR | IC 95% |
|---|---|---|---|---|
| Sentence-Transformer | **0.81** | 0.68 – 0.93 | **0.917** | 0.78 – 1.00 |
| TF-IDF | 0.58 | 0.44 – 0.71 | 0.788 | 0.64 – 0.93 |
| LSTM (non addestrata) | 0.09 | 0.02 – 0.18 | 0.217 | 0.07 – 0.42 |
| GRU (non addestrata) | 0.07 | 0.02 – 0.13 | 0.146 | 0.03 – 0.30 |
| Transformer (non addestrato) | 0.01 | 0.00 – 0.03 | 0.012 | 0.00 – 0.04 |

*   Il Sentence-Transformer restituisce in media **4 risultati pertinenti su 5**. Nel P@5 il suo vantaggio sul TF-IDF è netto: gli intervalli di confidenza quasi non si sovrappongono. Nell'MRR la differenza è più contenuta: spesso anche il TF-IDF mette un risultato pertinente in cima.
*   **Affidabilità del giudice:** con valutazione umana: accordo esatto 88%, accordo pertinente/non pertinente 92%, kappa di Cohen pesato **0.80**. 

**Tempi di ricerca** (GPU Apple MPS):

| Fase | Sentence-Transformer | TF-IDF |
|---|---|---|
| Caricamento del modello (una volta) | ~3.5 s | — |
| Indicizzazione del database (una volta) | ~3 s | 0.03 s |
| **Ricerca di una query** | **~6 ms** | **~0.5 ms** |

Il modello pre-addestrato è circa 10 volte più lento del TF-IDF a ogni query, ma 6 ms restano ampiamente adeguati a un uso interattivo. Quasi tutto il tempo è speso nel calcolo dell'embedding della query; il confronto con i 7.263 documenti costa meno di 0.2 ms.

Grafici prodotti: istogrammi di **P@5 e MRR** per metodo, con barre di errore dell'intervallo di confidenza al 95%.

---

## 6. Come Eseguire i Notebook

1.  **Ambiente Python** (testato con Python 3.13):
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install torch scikit-learn numpy pandas matplotlib sentence-transformers
    ```
    Le celle iniziali dei notebook installano comunque le dipendenze necessarie.
2.  **Spam filter:** eseguire `spam-filter.ipynb` dall'inizio. Il training completo dei 4 modelli richiede circa 1 minuto su GPU Apple.
3.  **Ricerca semantica:** eseguire `semantic-search.ipynb`. Al primo avvio viene scaricato il Sentence-Transformer (~470 MB).
4.  **Valutazione con LLM come giudice:** richiede [Ollama](https://ollama.com) in esecuzione con il modello scaricato:
    ```bash
    ollama pull gemma3:12b
    ```
    I giudizi già calcolati sono in `datasets/giudizi_llm.csv` e non vengono ricalcolati; il giudice viene interrogato solo per le coppie (query, documento) nuove, circa 1.4 s per coppia.

---

## 7. Servizio API (Flask + Docker) e Interfaccia Web

I due casi d'uso sono esposti come servizio web, costruito a partire dal codice dei notebook, con un'interfaccia web per provarli.

| File | Ruolo |
|---|---|
| `estrai_da_notebook.py` | Legge i notebook ed estrae (via `ast`) le definizioni riusabili, cioè `text_to_indices`, `SpamRNN`, `SpamTransformer`, `STOP_WORDS_IT` e `NOME_MODELLO`, nei moduli generati `spamsearch/_estratto_*.py` |
| `spamsearch/` | Libreria: `SpamClassifier` (TF-IDF/Naive Bayes, predefinito, e LSTM, addestrati come nel notebook) e `SemanticSearch` (Sentence-Transformer + TF-IDF) |
| `prepara_modelli.py` | Addestra il filtro spam e indicizza il database della ricerca, salvando gli artefatti in `artefatti/` |
| `api/app.py` | API Flask |
| `web/index.html`, `web/nginx.conf` | Interfaccia web, servita da un container nginx che inoltra le chiamate `/api/*` all'API |
| `Dockerfile`, `docker-compose.yml` | Due servizi: `api`, un'immagine con PyTorch solo CPU in cui estrazione, addestramento e download del modello avvengono **durante la build**, così all'avvio carica solo modelli già pronti; e `web`, con nginx |

**Avvio:**
```bash
docker compose up --build        # la prima build richiede qualche minuto
```
*   **Interfaccia web:** <http://localhost:8080>
    *   **Controllo spam:** si inserisce un messaggio, si sceglie il modello e si vedono il risultato spam/ham, con la probabilità, e la risposta JSON formattata.
    *   **Messaggi e ricerca:** l'elenco a pagine dei 7.263 messaggi del database; una ricerca semantica sostituisce l'elenco con i messaggi più simili e il loro punteggio.
*   **API:** <http://localhost:8000>

Per cambiare le porte: `PORTA_WEB=9080 PORTA=9000 docker compose up`. La porta 5000 non è usata perché su macOS è occupata da AirPlay.

**Endpoint:**

| Metodo e percorso | Corpo JSON | Risposta |
|---|---|---|
| `POST /check-spam` | `{"testo": "...", "modello": "naive-bayes" \| "lstm"}` (`modello` facoltativo, default `naive-bayes`) | `etichetta` (`spam`/`ham`), `spam`, `probabilita_spam`, `tempo_ms` |
| `POST /search-simil` | `{"query": "...", "metodo": "sentence-transformer" \| "tfidf", "top_k": 5}` (`metodo` e `top_k` facoltativi; `top_k` tra 1 e 50) | `risultati`: lista di `{rank, score, doc_id, documento}`, più `tempo_ms` |
| `GET /messaggi?pagina=1&per_pagina=20` | — (`per_pagina` tra 1 e 100) | `messaggi`: lista di `{doc_id, documento}`, più `totale`, `pagina`, `pagine` |
| `GET /health` | — | stato del servizio e metriche sul test set dei modelli spam |

Esempi:
```bash
curl -X POST http://localhost:8000/check-spam -H 'Content-Type: application/json' \
     -d '{"testo": "Hai vinto un premio di 1000 euro, clicca subito!"}'

curl -X POST http://localhost:8000/search-simil -H 'Content-Type: application/json' \
     -d '{"query": "Qual è il prezzo di questa vettura", "top_k": 3}'
```

**Senza Docker:** `pip install -r requirements.txt`, poi `python estrai_da_notebook.py && python prepara_modelli.py` e infine `python -m api.app`. Se si modificano le classi nei notebook, basta rieseguire `estrai_da_notebook.py` per aggiornare la libreria.

---

## 8. Limiti e Sviluppi Futuri
*   **Più esecuzioni con seed diversi** nello spam filter, per riportare media ± deviazione standard e stabilire se le differenze tra le reti sono significative.
*   **Curve di apprendimento** (loss di training e validation per epoca) e **early stopping**, dato che i modelli si fermano alla 2ª–3ª epoca.
*   **Matrici di confusione** e precision/recall separate: in un filtro antispam un falso positivo (messaggio legittimo finito nello spam) pesa più di un falso negativo.
*   **Addestramento degli embedder** della ricerca semantica su coppie di parafrasi (Cosine Embedding Loss / Contrastive Loss), oppure a partire da word embedding pre-addestrati per l'italiano.
*   **Più query di valutazione** e **validazione umana** del giudice LLM.
