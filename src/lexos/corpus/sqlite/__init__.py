"""__init__.py.

SQLite database support for Corpus.

This submodule provides SQLite-backed storage for corpus data.

Last updated: November 15, 2025
"""

from lexos.corpus.sqlite.database import SQLiteBackend, SQLiteMetadata
from lexos.corpus.sqlite.integration import SQLiteCorpus, create_corpus

__all__ = [
    "SQLiteBackend",
    "SQLiteMetadata",
    "SQLiteCorpus",
    "create_corpus",
]
