"""__init__.py.

Public API for the `lexos.corpus` package.

The core of the package is the `Corpus` class, which is a collection of
`Record` objects.

Last Updated: 2026-07-22
Last Tested: 2026-07-22
"""

from lexos.corpus.corpus import Corpus
from lexos.corpus.corpus_stats import CorpusStats
from lexos.corpus.record import Record
from lexos.corpus.utils import RecordsDict

__all__ = ["Corpus", "Record", "CorpusStats", "RecordsDict"]
