"""normalize.py."""
import re
import unicodedata
from typing import Literal

from . import resources


def bullet_points(text: str) -> str:
    """Normalize bullet points.

    Normalises all "fancy" bullet point symbols in `text` to just the basic
    ASCII "-", provided they are the first non-whitespace characters on a new
    line (like a list of items). Duplicates Textacy's `utils.normalize_bullets`.

    Args:
        text (str): The text to normalize.

    Returns:
        The normalized text.
    """
    return resources.RE_BULLET_POINTS.sub(r"\1-", text)


def hyphenated_words(text: str) -> str:
    """Normalize hyphenated words.

    Normalize words in `text` that have been split across lines by a hyphen
    for visual consistency (aka hyphenated) by joining the pieces back together,
    sans hyphen and whitespace. Duplicates Textacy's `utils.normalize_hyphens`.

    Args:
        text (str): The text to normalize.

    Returns:
        The normalized text.
    """
    return resources.RE_HYPHENATED_WORD.sub(r"\1\2", text)


def lower_case(text: str) -> str:
    """Convert `text` to lower case.

    Args:
        text (str): The text to convert to lower case.

    Returns:
        The converted text.
    """
    return text.lower()


def quotation_marks(text: str) -> str:
    """Normalize quotation marks.

    Normalize all "fancy" single- and double-quotation marks in `text`
    to just the basic ASCII equivalents. Note that this will also normalize fancy
    apostrophes, which are typically represented as single quotation marks.
    Duplicates Textacy's `utils.normalize_quotation_marks`.

    Args:
        text (str): The text to normalize.

    Returns:
        The normalized text.
    """
    return text.translate(resources.QUOTE_TRANSLATION_TABLE)


def repeating_chars(text: str, *, chars: str, maxn: int = 1) -> str:
    """Normalize repeating characters in `text`.

    Truncating their number of consecutive repetitions to `maxn`.
    Duplicates Textacy's `utils.normalize_repeating_chars`.

    Args:
        text (str): The text to normalize.
        chars: One or more characters whose consecutive repetitions are to be
            normalized, e.g. "." or "?!".
        maxn: Maximum number of consecutive repetitions of `chars` to which
            longer repetitions will be truncated.

    Returns:
        str
    """
    return re.sub(r"({}){{{},}}".format(re.escape(chars), maxn + 1), chars * maxn, text)


def unicode(text: str, *, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC") -> str:
    """Normalize unicode characters in `text` into canonical forms.

    Duplicates Textacy's `utils.normalize_unicode`.

    Args:
        text (str): The text to normalize.
        form: Form of normalization applied to unicode characters.
            For example, an "e" with accute accent "´" can be written as "e´"
            (canonical decomposition, "NFD") or "é" (canonical composition, "NFC").
            Unicode can be normalized to NFC form without any change in meaning,
            so it's usually a safe bet. If "NFKC", additional normalizations are applied
            that can change characters' meanings, e.g. ellipsis characters are replaced
            with three periods.

    See Also:
        https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
    """
    return unicodedata.normalize(form, text)


def whitespace(text: str) -> str:
    """Normalize whitespace.

    Replace all contiguous zero-width spaces with an empty string,
    line-breaking spaces with a single newline, and non-breaking spaces
    with a single space, then strip any leading/trailing whitespace.

    Args:
        text (str): The text to normalize.

    Returns:
        The normalized text.
    """
    text = resources.RE_ZWSP.sub("", text)
    text = resources.RE_LINEBREAK.sub(r"\n", text)
    text = resources.RE_NONBREAKING_SPACE.sub(" ", text)
    return text.strip()
