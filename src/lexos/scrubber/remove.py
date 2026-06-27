"""remove.py.

Last Update: 2026-06-26
Tested: 2026-06-26
"""

import os
import re
import sys
import unicodedata
from functools import lru_cache
from typing import Collection, Optional

from pydantic import ConfigDict, validate_call

from lexos.scrubber import resources
from lexos.util import ensure_list, to_collection

validation_config = ConfigDict(arbitrary_types_allowed=True)

try:
    from lxml import etree, html

    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False

from lexos.exceptions import LexosException

# Pre-compile the regex for performance (faster when called repeatedly). \s matches all whitespace characters.
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)

# Pre-compute unicode digit character class and punctuation translation tables once per import
_UNICODE_DIGITS = "".join(
    chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("N")
)

_DIGITS_PATTERN = re.compile(
    r"([+-]?(?:["
    + re.escape(_UNICODE_DIGITS)
    + r"]+))|((?<="
    + re.escape(_UNICODE_DIGITS)
    + r")[\u0027\u002C\u002E\u00B7\u02D9\u066B\u066C\u2396]["
    + re.escape(_UNICODE_DIGITS)
    + r"]+)",
    re.UNICODE,
)

_PUNCTUATION_TABLE = dict.fromkeys(
    (i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")),
    "",
)


@lru_cache(maxsize=64)
def _get_punctuation_translation_table(exclude: tuple[str, ...]) -> dict[int, str]:
    if not exclude:
        return _PUNCTUATION_TABLE

    exclude_set = set(exclude)
    return dict.fromkeys(
        (
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
            and chr(i) not in exclude_set
        ),
        "",
    )


@lru_cache(maxsize=128)
def _compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


# Security constants similar to Django's strip_tags
_LONG_OPEN_TAG_RE = re.compile(r"<[a-zA-Z][^>]{1000,}")
MAX_TAGS_DEPTH = 50

# Pre-compiled marker regexes for Gutenberg header/footer detection
_TEXT_START_RE = re.compile(
    r"^(?:" + "|".join(re.escape(token) for token in resources.TEXT_START_MARKERS) + ")"
)
_TEXT_END_RE = re.compile(
    r"^(?:" + "|".join(re.escape(token) for token in resources.TEXT_END_MARKERS) + ")"
)
_LEGALESE_START_RE = re.compile(
    r"^(?:"
    + "|".join(re.escape(token) for token in resources.LEGALESE_START_MARKERS)
    + ")"
)
_LEGALESE_END_RE = re.compile(
    r"^(?:"
    + "|".join(re.escape(token) for token in resources.LEGALESE_END_MARKERS)
    + ")"
)


@validate_call(config=validation_config)
def accents(
    text: str,
    *,
    fast: Optional[bool] = False,
    accents: Optional[str | tuple[str, ...]] = None,
) -> str:
    """Remove accents from any accented unicode characters in `text`, either by replacing them with ASCII equivalents or removing them entirely.

    Args:
        text (str): The text from which accents will be removed.
        fast (Optional[bool]): If False, accents are removed from any unicode symbol
            with a direct ASCII equivalent; if True, accented chars
            for all unicode symbols are removed, regardless.
        accents (Optional[str | tuple[str, ...]]): An optional string or tuple of strings indicating the
            names of diacritics to be stripped.

    Returns:
        str: The text with accents removed.

    Note: `fast=True` can be significantly faster than `fast=False`,
        but its transformation of `text` is less "safe" and more likely
        to result in changes of meaning, spelling errors, etc.

    See Also:
        - For a chart containing Unicode standard names of diacritics, see
        https://en.wikipedia.org/wiki/Combining_Diacritical_Marks#Character_table
        - For a more powerful (but slower) alternative, check out `unidecode`:
        https://github.com/avian2/unidecode
    """
    if fast is False:
        if accents:
            if isinstance(accents, str):
                accents = set(unicodedata.lookup(accents))
            elif len(accents) == 1:
                accents = set(unicodedata.lookup(accents[0]))
            else:
                accents = set(map(unicodedata.lookup, accents))
            return "".join(
                char
                for char in unicodedata.normalize("NFKD", text)
                if char not in accents
            )
        else:
            return "".join(
                char
                for char in unicodedata.normalize("NFKD", text)
                if not unicodedata.combining(char)
            )
    else:
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("ascii")
        )


def brackets(
    text: str,
    *,
    only: Optional[str | Collection[str]] = ["curly", "square", "round"],
) -> str:
    """Remove text within curly {}, square [], and/or round () brackets, as well as the brackets themselves.

    Args:
        text (str): The text from which brackets will be removed.
        only (Optional[str | Collection[str]]): Remove only those bracketed contents
            as specified here: "curly", "square", and/or "round". For example,
            `"square"` removes only those contents found between square brackets,
            while `["round", "square"]`  removes those contents found between square
            or round brackets, but not curly.

    Returns:
        str: The text with brackets removed.

    Note:
        This function relies on regular expressions, applied sequentially for curly,
        square, then round brackets; as such, it doesn't handle nested brackets of the
        same type and may behave unexpectedly on text with "wild" use of brackets.
        It should be fine removing structured bracketed contents, as is often used,
        for instance, to denote in-text citations.
    """
    only = to_collection(only, val_type=str, col_type=set)
    if only is None or "curly" in only:
        text = resources.RE_BRACKETS_CURLY.sub("", text)
    if only is None or "square" in only:
        text = resources.RE_BRACKETS_SQUARE.sub("", text)
    if only is None or "round" in only:
        text = resources.RE_BRACKETS_ROUND.sub("", text)
    return text


def digits(text: str, *, only: Optional[str | Collection[str]] = None) -> str:
    """Remove digits.

    Remove digits from `text` by replacing all instances of digits
    (or a subset thereof specified by `only`) with whitespace.

    Removes signed/unsigned numbers and decimal/delimiter-separated
    numbers. Does not remove currency symbols. Some tokens containing
    digits will be modified.

    Args:
        text (str): The text from which digits will be removed.
        only (Optional[str | Collection[str]]): Remove only those digits specified here. For example,
            `"9"` removes only 9, while `["1", "2", "3"]` removes 1, 2, 3;
            if None, all unicode digits marks are removed.

    Returns:
        str: The text with digits removed.
    """
    if only:
        if isinstance(only, list):
            pat = _compile_pattern(f"[{''.join(only)}]")
        else:
            pat = _compile_pattern(only)
    else:
        # Using "." to represent any unicode character used to indicate
        # a decimal number, and "***" to represent any sequence of
        # unicode digits, this pattern will match:
        # 1) ***
        # 2) ***.***
        unicode_digits = ""
        for i in range(sys.maxunicode):
            if unicodedata.category(chr(i)).startswith("N"):
                unicode_digits = unicode_digits + chr(i)
        pat = _DIGITS_PATTERN
    return str(re.sub(pat, r"", text))


def project_gutenberg_headers(text: str) -> str:
    """Remove Project Gutenberg headers and footers.

    Args:
        text (str): The text from which headers and footers will be removed.

    Returns:
        str: The text with Project Gutenberg boilerplate removed.

    Notes:
        This function is reproduced from Gutenberg package's `strip_headers()`
        function (https://github.com/c-w/gutenberg), itself a port of the C++ utility
        by Johannes Krugel.
    """
    lines = text.splitlines()
    sep = str(os.linesep)

    out = []
    i = 0
    footer_found = False
    ignore_section = False

    for line in lines:
        reset = False

        if i <= 600:
            # Check if the header ends here
            if _TEXT_START_RE.match(line):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if _TEXT_END_RE.match(line):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                break

        if _LEGALESE_START_RE.match(line):
            ignore_section = True
            continue
        elif _LEGALESE_END_RE.match(line):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1

    return sep.join(out).strip()


def tags(
    text: str, sep: Optional[str] = " ", remove_whitespace: Optional[bool] = True
) -> str:
    """Remove tags from `text`.

    Args:
        text (str): The markup (XML or HTML) to process.
        sep (Optional[str]): String to insert between extracted text segments.
        remove_whitespace (Optional[bool]): If True, collapses multiple whitespace characters into a single separator.

    Returns:
        str: A string containing just the text found between tags.

    Notes:
    - If you want to perfom selective removal of tags, use `replace.tag_map` instead.
    - Uses lxml for speed with a fallback to BeautifulSoup.
    """
    if not text or not text.strip():
        return ""

    # Security check: detect potential DoS from large unclosed tags or excessive nesting
    # Similar to Django's security fix (CVE-2024-53907)
    for long_open_tag in _LONG_OPEN_TAG_RE.finditer(text):
        if long_open_tag.group().count("<") >= MAX_TAGS_DEPTH:
            raise LexosException(
                "Potential security risk: excessive nested or unclosed tags detected."
            )

    sep_clean = sep if sep is not None else ""
    plaintext = None

    # Try using lxml for speed
    if _HAS_LXML:
        try:
            # lxml.html.fromstring is extremely fast and very forgiving of both HTML fragments and full documents
            tree = html.fromstring(text)

            # tree.itertext() is a fast C-level generator. joining with the separator effectively between tag boundaries
            # Good for large documents, as it avoids building a large intermediate list
            plaintext = sep_clean.join(tree.itertext())
        except Exception:
            # If the HTML parser fails, try the recovery XML parser for strict XML
            try:
                # recover=True allows processing of slightly malformed XML
                # etree.fromstring handles encoding declarations best when given bytes
                parser = etree.XMLParser(recover=True, no_network=True)
                tree = etree.fromstring(text.encode("utf-8"), parser=parser)
                plaintext = sep_clean.join(tree.itertext())
            except Exception:
                # Proceed to BeautifulSoup if lxml fails completely
                pass

    # Fallback to BeautifulSoup
    if plaintext is None:
        try:
            from bs4 import BeautifulSoup

            # Use lxml as the internal engine for speed if available
            feature = "lxml" if _HAS_LXML else "html.parser"
            soup = BeautifulSoup(text, feature)
            plaintext = soup.get_text(separator=sep_clean)
        except Exception:
            # Last resort: return text as-is if no parser can handle it
            plaintext = text

    if remove_whitespace:
        # For very large strings, regex (re.sub) is often a bottleneck; string.split() and string
        # join() is significantly faster when the separator is a single space
        if sep_clean == " ":
            plaintext = " ".join(plaintext.split())
        else:
            plaintext = _WHITESPACE_RE.sub(sep_clean, plaintext).strip()

    return plaintext


def new_lines(text: str) -> str:
    """Remove new lines.

    Remove all line-breaking spaces.

    Args:
        text (str): The text from which new lines will be removed.

    Returns:
        str: The text with line-breaking spaces removed.
    """
    return resources.RE_LINEBREAK.sub("", text).strip()


def pattern(text: str, *, pattern: Optional[str | Collection[str]]) -> str:
    """Remove strings from `text` using a regex pattern.

    Args:
        text (str): The text from which patterns will be removed.
        pattern (Optional[str | Collection[str]]): The pattern to match.

    Returns:
        str: The text with the pattern removed.
    """
    if isinstance(pattern, list):
        pattern = "|".join(pattern)
    if pattern is None:
        return text
    pat = _compile_pattern(pattern)
    return re.sub(pat, "", text)


def punctuation(
    text: str,
    *,
    exclude: Optional[str | Collection[str]] = None,
    only: Optional[str | Collection[str]] = None,
) -> str:
    """Remove punctuation from `text`.

    Removes all instances of punctuation (or a subset thereof specified by `only`).

    Args:
        text (str): The text from which punctuation will be removed.
        exclude (Optional[str | Collection[str]]): Remove all punctuation except designated characters.
        only (Optional[str | Collection[str]]): Remove only those punctuation marks specified here.
            For example, `"."` removes only periods, while `[",", ";", ":"]` removes commas,
            semicolons, and colons; if None, all unicode punctuation marks are removed.

    Returns:
        str: The text with punctuation removed.

    Note:
        When `only=None`, Python's built-in `str.translate()` is used;
        otherwise, a regular expression is used. The former's performance
        can be up to an order of magnitude faster.
    """
    if only is not None:
        only = to_collection(only, val_type=str, col_type=set)
        return re.sub("[{}]+".format(re.escape("".join(only))), "", text)
    else:
        if exclude:
            exclude = ensure_list(exclude)
        else:
            exclude = []
        exclude_key = tuple(sorted(exclude))
        translation_table = _get_punctuation_translation_table(exclude_key)
        return text.translate(translation_table)


def tabs(text: str) -> str:
    """Remove tabs.

    If you want to replace tabs with a single space, use
    `normalize.whitespace()` instead.

    Args:
        text (str): The text from which tabs will be removed.

    Returns:
        str: The text with tabs removed.
    """
    return resources.RE_TAB.sub("", text)
