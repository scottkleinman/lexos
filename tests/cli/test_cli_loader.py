"""test_cli_loader.py.

Test the Loader API for the Lexos project from the CLI.

Usage:

```
poetry run python test_cli_loader.py "test_data/txt/Austen_Pride.txt" "test_data/txt/Austen_Sense.txt"
```
"""
from typing import List

import typer

from lexos.io import basic

LANG = {
    "loading": "Loading...",
    "done": "Done.",
    "help": "A list of paths or urls to files to be loaded.",
}

app = typer.Typer()


@app.command("load")
def cli_load(data: List[str] = typer.Argument(..., help=LANG["help"])):
    """Load data."""
    # Ensure multiple arguments are passed as list items
    data = [x for x in data]
    # Begin loading
    typer.echo(LANG["loading"])
    # Load data
    loader = basic.Loader()
    loader.load(data)
    # End loading
    typer.echo(LANG["done"])
    # Show output
    for i, text in enumerate(loader.texts):
        typer.echo(f"Text {i} preview:")
        typer.echo(text[0:50])
        typer.echo("\n")


if __name__ == "__main__":
    app()
