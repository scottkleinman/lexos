"""conftest.py for tests/corpus/sqlite/.

Provides session-scoped cleanup for the `test_corpus/` directory that several
inline test methods in test_integration.py create by passing the hardcoded
relative path ``corpus_dir="test_corpus"`` to SQLiteCorpus/create_corpus.
All those instances use ``sqlite_path=":memory:"`` so no database file is
written, but the Corpus base class still creates the directory tree
(``test_corpus/``, ``test_corpus/data/``, ``test_corpus/corpus_metadata.json``)
in the current working directory at test runtime.

Also removes ``corpus.db``, ``corpus_metadata.json``, and the ``corpus/``
subdirectory that can appear here when pytest is run from this directory
directly rather than from the project root.
"""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_corpus_dir():
    """Remove stale corpus artifacts from the tests/corpus/sqlite/ working directory."""
    yield
    here = Path(__file__).parent

    # Hard-coded "test_corpus" directory used by several inline test methods
    test_corpus = Path("test_corpus")
    if test_corpus.exists():
        shutil.rmtree(test_corpus, ignore_errors=True)

    # Default corpus_dir="corpus" and auto-generated corpus.db / corpus_metadata.json
    # that appear when tests are invoked from this directory rather than the project root
    for name in ("corpus_metadata.json", "corpus.db"):
        target = here / name
        if target.exists():
            target.unlink()
    corpus_subdir = here / "corpus"
    if corpus_subdir.exists() and corpus_subdir.is_dir():
        shutil.rmtree(corpus_subdir, ignore_errors=True)
