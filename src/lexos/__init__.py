"""__init__.py."""

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
    Console().print(table)
