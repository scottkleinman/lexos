"""__init__.py.

Top-level API for the Lexos library.

Last Updated: 2026-07-22
Last Tested: 2026-07-22
"""

from lexos.corpus import Corpus, Record
from lexos.cutter import TextCutter, TokenCutter
from lexos.dtm import DTM
from lexos.kwic import Kwic
from lexos.rolling_windows import Windows
from lexos.scrubber import Scrubber
from lexos.structural_stylometry import StructuralAnalyzer
from lexos.tokenizer import Tokenizer
from lexos.topic_modeling import Mallet
from lexos.visualization import BubbleViz, WordCloud

__all__ = [
    "BubbleViz",
    "Corpus",
    "DTM",
    "Kwic",
    "Mallet",
    "Record",
    "Scrubber",
    "StructuralAnalyzer",
    "TextCutter",
    "TokenCutter",
    "Tokenizer",
    "Windows",
    "WordCloud",
]

__version__ = "0.1.0"
__docs__ = "https://scottkleinman.github.io/lexos/"
__repo__ = "https://github.com/scottkleinman/lexos"
