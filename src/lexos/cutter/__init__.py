"""Public API for the `lexos.cutter` package.

Cutter classes are used to split texts into segments of a specified length.

Last Updated: 2026-07-22
Last Tested: 2026-07-22
"""

from lexos.cutter.text_cutter import TextCutter
from lexos.cutter.token_cutter import TokenCutter

__all__ = ["TextCutter", "TokenCutter"]

__version__ = "0.1.0"
__docs__ = "https://scottkleinman.github.io/lexos/"
__repo__ = "https://github.com/scottkleinman/lexos"
