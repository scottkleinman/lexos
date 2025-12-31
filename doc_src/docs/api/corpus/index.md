# Corpus

The `corpus` module provides functionality for document management and statistical analysis in the Lexos ecosystem. It provides centralized storage, metadata management, and inter-module communication capabilities that enable seamless integration with analysis modules. By default, it is entirely file-based; however, there is an option to manage a corpus database with SQLite.

---

## Core Classes

### `Corpus` (`corpus.py`)

The [`corpus`](corpus.md) module main container for managing collections of documents. Provides document storage, metadata management, and inter-module communication capabilities.

### `Record` (`record.py`)

The [`record`](record.md) module implements an individual document container with robust metadata and serialization capabilities.

### `CorpusStats` (`corpus_stats.py`)

The [`CorpusStats`](corpus_stats.md) module provides methods for generating statistics about a corpus.

### `LexosModelCache` and `RecordsDict` (`utils.py`)

The [`utils`](utils.md) module provides utility classes for efficient model management and type-safe record storage.

---

## SQLite Database

Database management is implemented in two modules:

### `SQLiteBackend` (database.py)

The [`database`](sqlite/database.md) module provides the main database functionality.

### `SQLiteCorpus` (integration.py)

The [`integration`](sqlite/integration.md) module the handler for integration with the main `corpus` API.

## Corpus Analysis Report

The [`corpus_analysis_report`](corpus_analysis_report.md) module provides a helper function for generating a comprehensive analysis of the contents of a `corpus` instance.
