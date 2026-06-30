"""check_comment_caps.py.

Precommit hook to make sure that all comments (or inline comments) start with a capital letter, except for certain special cases (e.g. "noqa", "type: ignore", "pylint:", "eslint-", "todo", "fixme").

Usage:
    python check_comment_caps.py <filepath>

Supports Python and JavaScript files. For Python, it checks comments that start with "#" (it can be modified to check only comments that are are not full-line comments. For JavaScript, it checks comments that start with "//".

Currently set to run only on Python files in the pre-commit configuration, but can be easily extended to check JavaScript files as well by adding the appropriate entry in the pre-commit config and ensuring the script is called with the correct file paths.

Last Update: June 26, 2026
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
    "pragma",
)


def first_alpha(s: str) -> str | None:
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


CODE_LIKE_RE = re.compile(
    r"""
    ^\s*(?:
        # Block headers that must end with ":"
        (?:if|elif|for|while|with|try|except|finally|def|class|match|case)\b.*:\s*$ |
        else\s*:\s*$ |

        # Simple statements
        (?:return|yield|raise|assert|del|pass|break|continue)\b(?:\s+.+)?$ |

        # Imports
        import\s+.+$ |
        from\s+\S+\s+import\s+.+$ |

        # Decorators
        @[\w.]+(?:\([^#]*\))?$ |

        # Assignments / augmented assignments / walrus
        [A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\[[^\]]+\])?\s*
        (?:=|:=|\+=|-=|\*=|/=|//=|%=|\*\*=|&=|\|=|\^=|>>=|<<=)\s*.+$ |

        # Bare function/method call statements
        [A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\([^#]*\)\s*$
    )
    """,
    re.VERBOSE,
)


def looks_like_commented_code(text: str) -> bool:
    """Heuristic to determine if a comment looks like commented-out code.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the comment looks like commented-out code, False otherwise.
    """
    body = text.strip()
    return bool(CODE_LIKE_RE.match(body))


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
        body = tok.string.lstrip("#").strip()
        if not body or should_skip(body):
            continue

        # Full-line comments that are actually commented-out code
        if col == 0:
            body_l = body.lstrip()
            if body_l and body_l[0].islower() and CODE_LIKE_RE.match(body_l):
                continue

        ch = first_alpha(body)
        if ch and ch.islower():
            errors.append((line_no, "Comment should start with a capital letter"))
    return errors


FULL_LINE_JS_RE = re.compile(r"^\s*//\s*(?P<comment>.*)$")
INLINE_JS_RE = re.compile(r"^(?P<code>.*\S.*?)\s*//\s*(?P<comment>.*)$")

JS_CODE_LIKE_RE = re.compile(
    r"""
    ^\s*(?:
        # Control/block headers
        (?:if|else\s+if|else|for|while|switch|case|default|try|catch|finally)\b.*(?:\{)?\s*$ |

        # Declarations
        (?:const|let|var|function|class|import|export)\b.+$ |

        # Simple statements
        (?:return|throw|break|continue)\b(?:\s+.+)?$ |

        # Assignments / updates
        [A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*|\[[^\]]+\])*\s*
        (?:=|\+=|-=|\*=|/=|%=|\*\*=|&&=|\|\|=|\?\?=|\+\+|--)\s*.+$ |

        # Function/method calls
        [A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\([^#]*\)\s*;?\s*$
    )
    """,
    re.VERBOSE,
)


def looks_like_commented_js_code(text: str) -> bool:
    """Heuristic to determine if a comment looks like commented-out JavaScript code.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the comment looks like commented-out JavaScript code, False otherwise.
    """
    body = text.strip()
    return bool(JS_CODE_LIKE_RE.match(body))


def check_js(path: Path) -> list[tuple[int, str]]:
    """Check JS comments for capitalization, skipping commented-out code.

    Args:
        path (Path): The path to the JS file to check.

    Returns:
        list[tuple[int, str]]: A list of errors found, where each error is a tuple of the line number and the error message.
    """
    errors = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m_full = FULL_LINE_JS_RE.match(line)
        m_inline = None if m_full else INLINE_JS_RE.match(line)

        if not m_full and not m_inline:
            continue

        body = (m_full or m_inline).group("comment").strip()
        if not body or should_skip(body):
            continue

        # Skip full-line comments that are likely commented-out JS code
        if m_full:
            body_l = body.lstrip()
            if body_l and body_l[0].islower() and looks_like_commented_js_code(body_l):
                continue

        ch = first_alpha(body)
        if ch and ch.islower():
            errors.append((i, "Comment should start with a capital letter"))
    return errors


def main(paths) -> int:
    """Main function to check comment capitalization in the given files.

    Args:
        paths (list[str]): A list of file paths to check.

    Returns:
        int: 0 if all checks passed, 1 otherwise.
    """
    if not paths:
        print("check_comment_caps: no filenames were provided.")
        print("Run via pre-commit with staged files, or pass --files/--all-files.")
        return 1

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
