"""Libreria dei due casi d'uso del progetto: filtro spam e ricerca semantica.

I moduli `_estratto_*.py` sono generati dai notebook con `estrai_da_notebook.py`.
"""
from .search import SemanticSearch
from .spam import SpamClassifier

__all__ = ["SpamClassifier", "SemanticSearch"]
