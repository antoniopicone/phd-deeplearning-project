FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    ARTEFATTI=/app/artefatti

WORKDIR /app

# PyTorch solo CPU (l'immagine standard includerebbe CUDA, diversi GB inutili qui)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 1. Estrazione del codice dai notebook nella libreria
# 2. Addestramento del filtro spam e indicizzazione del database (scarica anche il Sentence-Transformer)
COPY spam-filter.ipynb semantic-search.ipynb estrai_da_notebook.py prepara_modelli.py ./
COPY spamsearch/ spamsearch/
COPY datasets/italian.csv datasets/italian.csv
RUN python estrai_da_notebook.py && python prepara_modelli.py --artefatti "$ARTEFATTI"

COPY api/ api/

EXPOSE 8000
# Un solo worker: i modelli restano in memoria una volta sola; i thread gestiscono le richieste concorrenti
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "api.app:crea_app()"]
