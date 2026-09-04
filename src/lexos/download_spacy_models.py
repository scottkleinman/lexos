"""download_spacy_models.py.

Last Updated: September 4, 2026
Last Tested: September 4, 2026
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import spacy
from wasabi import msg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for downloading spaCy models.

    Args:
        argv (Sequence[str] | None): Command-line arguments to parse. If None, defaults to sys.argv.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download one or more spaCy language models."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="One or more spaCy model names to download (for example: xx_sent_ud_sm en_core_web_sm).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Download one or more spaCy language models from the command line.

    Args:
        argv (Sequence[str] | None): Command-line arguments to parse. If None, defaults to sys.argv.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    args = parse_args(argv)

    for model_name in args.models:
        msg.info(f"Downloading spaCy model: {model_name}")
        try:
            spacy.cli.download(model_name)
        except SystemExit as exc:
            if exc.code not in (0, None):
                msg.fail(f"Failed to download {model_name}", exits=1)
                return int(exc.code)
        msg.good(f"Finished: {model_name}")

    return 0


if __name__ == "__main__":
    main()
