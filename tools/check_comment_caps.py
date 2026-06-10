"""check_comment_caps.py.

Precommit hook to make sure that all inline comments start with a capital letter, except for certain special cases (e.g. "noqa", "type: ignore", "pylint:", "eslint-", "todo", "fixme").

Usage:
    python check_comment_caps.py <filepath>

Supports Python and JavaScript files. For Python, it checks comments that start with "#" and are not full-line comments. For JavaScript, it checks comments that start with "//" and are not full-line comments.

Currently set to run only on Python files in the pre-commit configuration, but can be easily extended to check JavaScript files as well by adding the appropriate entry in the pre-commit config and ensuring the script is called with the correct file paths.

Last Update: June 9, 2026
"""
import re
import sys
import tokenize
from io import StringIO
from pathlib import Path

SKIP_PREFIXES = (
    "noqa",
    "type: ignore",
    "pylint:",
    "eslint-",
    "todo",
    "fixme",
)

def first_alpha(s: str) ->  str | None:
    """Return the first alphabetic character in the string, or None if there isn't one.

    Args:
        s (str): The string to search.

    Returns:
        str | None: The first alphabetic character, or None if there isn't one.
    """
    m = re.search(r"[A-Za-z]", s)
    return m.group(0) if m else None

def should_skip(comment_body: str) -> bool:
    """Determine if the comment should be skipped based on its content.

    Args:
        comment_body (str): The content of the comment.

    Returns:
        bool: True if the comment should be skipped, False otherwise.
    """
    text = comment_body.strip().lower()
    return any(text.startswith(p) for p in SKIP_PREFIXES)

def check_python(path: Path) -> list[tuple[int, str]]:
    """Check inline comments in a Python file for capitalization.

    Args:
        path (Path): The path to the Python file to check.

    Returns:
        list[tuple[int, str]]: A list of errors found, where each error is a tuple of the line number and the error message.
    """
    errors = []
    text = path.read_text(encoding="utf-8")
    toks = tokenize.generate_tokens(StringIO(text).readline)
    for tok in toks:
        if tok.type != tokenize.COMMENT:
            continue
        line_no, col = tok.start
        if col == 0:
            continue  # full-line comment, not inline
        body = tok.string.lstrip("#").strip()
        if not body or should_skip(body):
            continue
        ch = first_alpha(body)
        if ch and ch.islower():
            errors.append((line_no, "Inline comment should start with a capital letter"))
    return errors

INLINE_JS_RE = re.compile(r"^(?P<code>.*\S.*?)\s*//\s*(?P<comment>.*)$")

def check_js(path: Path) -> list[tuple[int, str]]:
    """Check inline comments in a JavaScript file for capitalization.

    Args:
        path (Path): The path to the JavaScript file to check.

    Returns:
        list[tuple[int, str]]: A list of errors found, where each error is a tuple of the line number and the error message.
    """
    errors = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m = INLINE_JS_RE.match(line)
        if not m:
            continue
        body = m.group("comment").strip()
        if not body or should_skip(body):
            continue
        ch = first_alpha(body)
        if ch and ch.islower():
            errors.append((i, "Inline comment should start with a capital letter"))
    return errors

def main(paths) -> int:
    """Main function to check comment capitalization in the given files.

    Args:
        paths (list[str]): A list of file paths to check.

    Returns:
        int: 0 if all checks passed, 1 otherwise.
    """
    failed = False
    for p in paths:
        path = Path(p)
        if path.suffix == ".py":
            errs = check_python(path)
        elif path.suffix == ".js":
            errs = check_js(path)
        else:
            continue

        for line_no, msg in errs:
            failed = True
            print(f"{path}:{line_no}: {msg}")

    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
