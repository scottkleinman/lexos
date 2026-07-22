"""Public API for the `lexos.corpus` package.

The core of the package is the `Corpus` class, which is a collection of
`Record` objects.

Phase 1 export surface:
- Corpus
- Record
- CorpusStats
- RecordsDict
"""

from lexos.corpus.corpus import Corpus
from lexos.corpus.corpus_stats import CorpusStats
from lexos.corpus.record import Record
from lexos.corpus.utils import RecordsDict

__all__ = ["Corpus", "Record", "CorpusStats", "RecordsDict"]
