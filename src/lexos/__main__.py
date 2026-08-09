"""__main__.py.

Command-line entrypoint for the lexos package.
"""

import argparse

from lexos import get_info


def main() -> None:
    """Main function for the command-line interface."""
    parser = argparse.ArgumentParser(description="Lexos command-line interface")
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print package information",
    )
    args = parser.parse_args()

    if args.info:
        get_info()


if __name__ == "__main__":
    main()
