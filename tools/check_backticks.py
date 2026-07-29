"""check_backticks.py.

Last Updated: July 29, 2026
"""

import re
import sys
from pathlib import Path

# Matches double backticks but ignores triple backtick code blocks
DOUBLE_BACKTICK_RE = re.compile(r"(?<!`)``([^`\n]+)``(?!`)")


def fix_file(path: Path) -> bool:
    """Check and fix a file for double backticks.

    Args:
        path (Path): The path to the file to check.

    Returns:
        bool: True if the file was modified, False otherwise.
    """
    content = path.read_text(encoding="utf-8")
    if not DOUBLE_BACKTICK_RE.search(content):
        return False

    # Replace double backticks with single backticks
    new_content = DOUBLE_BACKTICK_RE.sub(r"`\1`", content)

    if new_content != content:
        path.write_text(new_content, encoding="utf-8")
        print(f"Fixed double backticks in {path}")
        return True
    return False


if __name__ == "__main__":
    modified = False
    for arg in sys.argv[1:]:
        if fix_file(Path(arg)):
            modified = True
    # If any file was modified, we exit with 1 to notify pre-commit
    sys.exit(1 if modified else 0)
