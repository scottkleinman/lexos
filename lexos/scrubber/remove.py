"""remove.py."""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from typing import Collection, Optional, Union

from .. import utils
from . import resources


def accents(text: str, *, fast: bool = False, accents: Union[str, tuple] = None) -> str:
    """Remove accents from any accented unicode characters in `text`, either
    by replacing them with ASCII equivalents or removing them entirely.

    Args:
        text (str): The text from which accents will be removed.
        fast: If False, accents are removed from any unicode symbol
            with a direct ASCII equivalent; if True, accented chars
            for all unicode symbols are removed, regardless.
        accents: An optional string or tuple of strings indicating the
            names of diacritics to be stripped.

    Returns:
        str

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
    only: Optional[str | Collection[str]] = None,
) -> str:
    """Remove text within curly {}, square [], and/or round () brackets, as well as
    the brackets themselves.

    Args:
        text (str): The text from which brackets will be removed.
        only: Remove only those bracketed contents as specified here: "curly", "square",
            and/or "round". For example, `"square"` removes only those contents found
            between square brackets, while `["round", "square"]` removes those contents
            found between square or round brackets, but not curly.

    Returns:
        str

    Note:
        This function relies on regular expressions, applied sequentially for curly,
        square, then round brackets; as such, it doesn't handle nested brackets of the
        same type and may behave unexpectedly on text with "wild" use of brackets.
        It should be fine removing structured bracketed contents, as is often used,
        for instance, to denote in-text citations.
    """
    only = utils.to_collection(only, val_type=str, col_type=set)
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
        only: Remove only those digits specified here. For example,
            `"9"` removes only 9, while `["1", "2", "3"]` removes 1, 2, 3;
            if None, all unicode digits marks are removed.

    Returns:
        str
    """
    if only:
        if isinstance(only, list):
            pattern = re.compile(f'[{"".join(only)}]')
        else:
            pattern = re.compile(only)
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
        pattern = re.compile(
            r"([+-]?["
            + re.escape(unicode_digits)
            + r"])|((?<="
            + re.escape(unicode_digits)
            + r")[\u0027|\u002C|\u002E|\u00B7|"
            r"\u02D9|\u066B|\u066C|\u2396][" + re.escape(unicode_digits) + r"]+)",
            re.UNICODE,
        )
    return str(re.sub(pattern, r"", text))


def project_gutenberg_headers(text: str) -> str:
    """Remove Project Gutenberg headers and footers.

    Args:
        text (str): The text from which headers and footers will be removed.

    Returns:
        str

    Notes:
        This function is reproduced from Gutenberg package's `strip_headers()`
        function (https://github.com/c-w/gutenberg), itself a port ofthe C++ utility
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
            if any(line.startswith(token) for token in resources.TEXT_START_MARKERS):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in resources.TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                break

        if any(line.startswith(token) for token in resources.LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in resources.LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1

    return sep.join(out).strip()


def tags(text: str, sep: str = " ", remove_whitespace: bool = True) -> str:
    """Remove tags from `text`.

    Args:
        text (str): The text from which tags will be removed.
        sep: A string to insert between tags and text found between them.
        remove_whitespace: If True, remove extra whitespace between text
            after tags are removed.

    Returns:
        A string containing just the text found between tags and other
        non-data elements.

    Note:
        - If you want to perfom selective removal of tags,
            use `replace.tag_map` instead.
        - This function relies on the stdlib `html.parser.HTMLParser`.
            It appears to work for stripping tags from both html and xml.
            Using `lxml` or BeautifulSoup might be faster, but this is untested.
        - This function preserves text in comments, as well as tags
    """
    parser = resources.HTMLTextExtractor()
    parser.feed(text)
    text = parser.get_text(sep=sep)
    if remove_whitespace:
        text = re.sub(r"[\n\s\t\v ]+", sep, text, re.UNICODE)
    return text


def new_lines(text: str) -> str:
    """Remove new lines.

    Remove all line-breaking spaces.

    Args:
        text (str): The text from which new lines will be removed.

    Returns:
        The normalized text.
    """
    return resources.RE_LINEBREAK.sub("", text).strip()


def pattern(text: str, *, pattern: Union[str, Collection[str]]) -> str:
    """Remove strings from `text` using a regex pattern.

    Args:
        text (str): The text from which patterns will be removed.
        pattern: The pattern to match.

    Returns:
        str
    """
    if isinstance(pattern, list):
        pattern = "|".join(pattern)
    pat = re.compile(pattern)
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
        exclude: Remove all punctuation except designated characters.
        only: Remove only those punctuation marks specified here. For example,
            `"."` removes only periods, while `[",", ";", ":"]` removes commas,
            semicolons, and colons; if None, all unicode punctuation marks are removed.

    Returns:
        str

    Note:
        When `only=None`, Python's built-in `str.translate()` is used;
        otherwise, a regular expression is used. The former's performance
        can be up to an order of magnitude faster.
    """
    if only is not None:
        only = utils.to_collection(only, val_type=str, col_type=set)
        return re.sub("[{}]+".format(re.escape("".join(only))), "", text)
    else:
        if exclude:
            exclude = utils.ensure_list(exclude)
        else:
            exclude = []
        # Note: We can't use the cached translation table because it replaces
        # the punctuation with whitespace, so we have to build a new one.
        translation_table = dict.fromkeys(
            (
                i
                for i in range(sys.maxunicode)
                if unicodedata.category(chr(i)).startswith("P")
                and chr(i) not in exclude
            ),
            "",
        )
        return text.translate(translation_table)


def tabs(text: str) -> str:
    """Remove tabs.

    If you want to replace tabs with a single space, use
    `normalize.whitespace()` instead.

    Args:
        text (str): The text from which tabs will be removed.

    Returns:
        The stripped text.
    """
    return resources.RE_TAB.sub("", text)
