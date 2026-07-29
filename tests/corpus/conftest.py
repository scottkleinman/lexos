"""conftest.py for tests/corpus/.

Cleans up files and directories that corpus tests can leave behind in this
directory when pytest is invoked from here rather than the project root:

- ``corpus_metadata.json``  – written by Corpus.__init__ to ``corpus_dir``
- ``corpus.db``             – written by SQLiteCorpus/create_corpus to ``corpus_dir``
- ``corpus/``               – the default ``corpus_dir="corpus"`` subdirectory,
                              which contains the above two files plus ``data/``
"""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_corpus_artifacts():
    """Remove stale corpus files from the tests/corpus/ working directory."""
    yield
    here = Path(__file__).parent
    for name in ("corpus_metadata.json", "corpus.db"):
        target = here / name
        if target.exists():
            target.unlink()
    corpus_subdir = here / "corpus"
    if corpus_subdir.exists() and corpus_subdir.is_dir():
        shutil.rmtree(corpus_subdir, ignore_errors=True)
