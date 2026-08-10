"""__init__.py.

Last Update: August 9, 2026
Last Tested: August 9, 2026
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console
from rich.table import Table

__title__ = "Lexos"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2022- The Lexomics Project"
__status__ = "pre-release"
__version__ = "0.2.0-beta"
__version_info__ = (0, 2, 0, "beta")
__docs__ = "https://scottkleinman.github.io/lexos/"
__repo__ = "https://github.com/scottkleinman/lexos"

__info__ = {
    "title": __title__,
    "version": __version__,
    "docs": __docs__,
    "repo": __repo__,
    "copyright": __copyright__,
    "license": __license__,
    "status": __status__,
}


def get_info():
    """Print the package information in a table format."""
    table = Table(title="Lexos Package Info", box=box.ROUNDED)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="white")
    for key, value in __info__.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    Console().print("\n")
    Console().print(table)
    Console().print("\n")


__all__ = [
    "BubbleViz",
    "Corpus",
    "DataLoader",
    "DTM",
    "Kwic",
    "Loader",
    "Mallet",
    "ParallelLoader",
    "Record",
    "Scrubber",
    "StructuralAnalyzer",
    "TextCutter",
    "TokenCutter",
    "Tokenizer",
    "Windows",
    "WordCloud",
]

_LAZY_IMPORTS = {
    "BubbleViz": ("lexos.visualization.bubbleviz", "BubbleViz"),
    "Corpus": ("lexos.corpus.corpus", "Corpus"),
    "DataLoader": ("lexos.io.data_loader", "DataLoader"),
    "DTM": ("lexos.dtm", "DTM"),
    "Kwic": ("lexos.kwic", "Kwic"),
    "Loader": ("lexos.io.loader", "Loader"),
    "Mallet": ("lexos.topic_modeling.mallet.mallet", "Mallet"),
    "ParallelLoader": ("lexos.io.parallel_loader", "ParallelLoader"),
    "Record": ("lexos.corpus.record", "Record"),
    "Scrubber": ("lexos.scrubber.scrubber", "Scrubber"),
    "StructuralAnalyzer": (
        "lexos.structural_stylometry.structural_stylometry",
        "StructuralAnalyzer",
    ),
    "TextCutter": ("lexos.cutter.text_cutter", "TextCutter"),
    "TokenCutter": ("lexos.cutter.token_cutter", "TokenCutter"),
    "Tokenizer": ("lexos.tokenizer", "Tokenizer"),
    "Windows": ("lexos.rolling_windows", "Windows"),
    "WordCloud": ("lexos.visualization.cloud", "WordCloud"),
}


def __getattr__(name: str) -> Any:
    """Lazy import for submodules and classes.

    Args:
        name (str): The name of the attribute to access.

    Returns:
        Any: The imported module or class.
    """
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List available attributes for the module.

    Returns:
        list[str]: A list of attribute names available in the module.
    """
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover
    from lexos.corpus.corpus import Corpus
    from lexos.corpus.record import Record
    from lexos.cutter.text_cutter import TextCutter
    from lexos.cutter.token_cutter import TokenCutter
    from lexos.dtm import DTM
    from lexos.io.data_loader import DataLoader
    from lexos.io.loader import Loader
    from lexos.io.parallel_loader import ParallelLoader
    from lexos.kwic import Kwic
    from lexos.rolling_windows import Windows
    from lexos.scrubber.scrubber import Scrubber
    from lexos.structural_stylometry.structural_stylometry import StructuralAnalyzer
    from lexos.tokenizer import Tokenizer
    from lexos.topic_modeling.mallet.mallet import Mallet
    from lexos.visualization.bubbleviz import BubbleViz
    from lexos.visualization.cloud import WordCloud
