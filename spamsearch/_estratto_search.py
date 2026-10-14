"""GENERATO AUTOMATICAMENTE da semantic-search.ipynb con estrai_da_notebook.py: non modificare a mano."""


# Stop words italiane: parole molto comuni (articoli, preposizioni, pronomi...) che non portano significato
STOP_WORDS_IT = [
    'il', 'lo', 'la', 'le', 'gli', 'un', 'uno', 'una',
    'di', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
    'del', 'dello', 'della', 'dei', 'degli', 'delle', 'al', 'allo', 'alla', 'ai', 'agli', 'alle',
    'dal', 'dallo', 'dalla', 'dai', 'dagli', 'dalle', 'nel', 'nello', 'nella', 'nei', 'negli', 'nelle',
    'sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle',
    'ed', 'ma', 'se', 'che', 'chi', 'non', 'come', 'anche', 'più',
    'mi', 'ti', 'ci', 'vi', 'si', 'ne', 'li',
    'qual', 'quale', 'questo', 'questa', 'quello', 'quella',
]


# Transformer (12 layer, embedding a 384 dimensioni) pre-addestrato su testi in 50+ lingue
# e poi addestrato su milioni di coppie di parafrasi con una loss contrastiva:
# frasi con lo stesso significato -> vettori vicini, anche se usano parole diverse
NOME_MODELLO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
