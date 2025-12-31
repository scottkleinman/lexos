"""resources.py.

Last Update: 2025-01-15
Tested: 2025-01-15
"""

import functools
import html.parser
import re
import sys
import unicodedata
from typing import Any, Optional, Pattern


class HTMLTextExtractor(html.parser.HTMLParser):
    """Simple subclass of :class:`html.parser.HTMLParser`.

    Collects data elements (non-tag, -comment, -pi, etc. elements)
    fed to the parser, then make them available as stripped, concatenated
    text via `HTMLTextExtractor.get_text()`.

    Note:
        Users probably shouldn't deal with this class directly;
        instead, use `:func:`remove.remove_html_tags()`.
    """

    def __init__(self):
        """Initialize the parser."""
        super().__init__()
        self.data = []

    def handle_data(self, data: Any) -> None:
        """Handle data elements.

        Args:
            data (Any): The data element(s) to handle.
        """
        self.data.append(data)

    def get_text(self, sep: Optional[str] = "") -> str:
        """Return the collected text.

        Args:
            sep (Optional[str]): The separator to join the collected text with.

        Returns:
            str: The collected text.
        """
        return sep.join(self.data).strip()


# Compile regexes, so we don't do this on the fly and rely on caching

RE_LINEBREAK: Pattern = re.compile(r"(\r\n|[\n\v])+")
RE_NONBREAKING_SPACE: Pattern = re.compile(r"[^\S\n\v]+")
RE_ZWSP: Pattern = re.compile(r"[\u200B\u2060\uFEFF]+")
RE_TAB: Pattern = re.compile(r"[\t\v]+")

RE_BRACKETS_CURLY = re.compile(r"\{[^{}]*?\}")
RE_BRACKETS_ROUND = re.compile(r"\([^()]*?\)")
RE_BRACKETS_SQUARE = re.compile(r"\[[^\[\]]*?\]")

RE_BULLET_POINTS = re.compile(
    # require bullet points as first non-whitespace char on a new line, like a list
    r"((^|\n)\s*?)"
    r"([\u2022\u2023\u2043\u204C\u204D\u2219\u25aa\u25CF\u25E6\u29BE\u29BF\u30fb])",
)

# source: https://gist.github.com/dperini/729294
RE_URL: Pattern = re.compile(
    r"(?:^|(?<![\w/.]))"
    # protocol identifier
    # r"(?:(?:https?|ftp)://)"  <-- alt?
    r"(?:(?:https?://|ftp://|www\d{0,3}\.))"
    # user:pass authentication
    r"(?:\S+(?::\S*)?@)?"
    r"(?:"
    # IP address exclusion
    # private & local networks
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    # host name
    r"(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
    # domain name
    r"(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
    # TLD identifier
    r"(?:\.(?:[a-z\u00a1-\uffff]{2,}))"
    r")"
    # port number
    r"(?::\d{2,5})?"
    # resource path
    r"(?:/\S*)?"
    r"(?:$|(?![\w?!+&/]))",
    flags=re.IGNORECASE,
)

RE_SHORT_URL: Pattern = re.compile(
    r"(?:^|(?<![\w/.]))"
    # optional scheme
    r"(?:(?:https?://)?)"
    # domain
    r"(?:\w-?)*?\w+(?:\.[a-z]{2,12}){1,3}"
    r"/"
    # hash
    r"[^\s.,?!'\"|+]{2,12}"
    r"(?:$|(?![\w?!+&/]))",
    flags=re.IGNORECASE,
)

RE_EMAIL: Pattern = re.compile(
    r"(?:mailto:)?"
    r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(\.([a-z]{2,})){1,3}"
    r"(?:$|(?=\b))",
    flags=re.IGNORECASE,
)

RE_USER_HANDLE: Pattern = re.compile(
    r"(?:^|(?<![\w@.]))@\w+",
    flags=re.IGNORECASE,
)

RE_HASHTAG: Pattern = re.compile(
    r"(?:^|(?<![\w#＃.]))(#|＃)(?!\d)\w+",
    flags=re.IGNORECASE,
)

RE_PHONE_NUMBER: Pattern = re.compile(
    # core components of a phone number
    r"(?:^|(?<=[^\w)]))(\+?1[ .-]?)?(\(?\d{3}\)?[ .-]?)?(\d{3}[ .-]?\d{4})"
    # extensions, etc.
    r"(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W))",
    flags=re.IGNORECASE,
)

RE_NUMBER: Pattern = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?"
    r"(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)"
    r"(?:$|(?=\b))"
)

RE_CURRENCY_SYMBOL: Pattern = re.compile(
    r"[$¢£¤¥ƒ֏؋৲৳૱௹฿៛ℳ元円圆圓﷼\u20A0-\u20C0]",
)

RE_EMOJI: Pattern
if sys.maxunicode < 0x10FFFF:
    RE_EMOJI = re.compile(
        r"[\u2600-\u26FF\u2700-\u27BF]",
        flags=re.IGNORECASE,
    )
else:
    RE_EMOJI = re.compile(
        r"[\u2600-\u26FF\u2700-\u27BF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]",
        flags=re.IGNORECASE,
    )

RE_HYPHENATED_WORD: Pattern = re.compile(
    r"(\w{2,}(?<!\d))\-\s+((?!\d)\w{2,})",
    flags=re.IGNORECASE,
)


# build mapping of unicode punctuation symbol ordinals to their replacements
# and lazy-load the big one, since it's relatively expensive to compute

QUOTE_TRANSLATION_TABLE: dict[int, int] = {
    ord(x): ord(y)
    for x, y in [
        ("ʼ", "'"),
        ("‘", "'"),
        ("’", "'"),
        ("´", "'"),
        ("`", "'"),
        ("“", '"'),
        ("”", '"'),
    ]
}


@functools.lru_cache(maxsize=None)
def _get_punct_translation_table():
    """Get the punctuation translation table."""
    return dict.fromkeys(
        (
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        ),
        " ",
    )


# Hello, PEP 562: https://www.python.org/dev/peps/pep-0562
def __getattr__(name: str) -> Any:
    """Call an attribute lookup from a table."""
    if name == "PUNCT_TRANSLATION_TABLE":
        return _get_punct_translation_table()
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# For Stripping Project Gutenberg Boilerplate

TEXT_START_MARKERS = frozenset(
    (
        "*END*THE SMALL PRINT",
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "This etext was prepared by",
        "E-text prepared by",
        "Produced by",
        "Distributed Proofreading Team",
        "Proofreading Team at http://www.pgdp.net",
        "http://gallica.bnf.fr)",
        "      http://archive.org/details/",
        "http://www.pgdp.net",
        "by The Internet Archive)",
        "by The Internet Archive/Canadian Libraries",
        "by The Internet Archive/American Libraries",
        "public domain material from the Internet Archive",
        "Internet Archive)",
        "Internet Archive/Canadian Libraries",
        "Internet Archive/American Libraries",
        "material from the Google Print project",
        "*END THE SMALL PRINT",
        "***START OF THE PROJECT GUTENBERG",
        "This etext was produced by",
        "*** START OF THE COPYRIGHTED",
        "The Project Gutenberg",
        "http://gutenberg.spiegel.de/ erreichbar.",
        "Project Runeberg publishes",
        "Beginning of this Project Gutenberg",
        "Project Gutenberg Online Distributed",
        "Gutenberg Online Distributed",
        "the Project Gutenberg Online Distributed",
        "Project Gutenberg TEI",
        "This eBook was prepared by",
        "http://gutenberg2000.de erreichbar.",
        "This Etext was prepared by",
        "This Project Gutenberg Etext was prepared by",
        "Gutenberg Distributed Proofreaders",
        "Project Gutenberg Distributed Proofreaders",
        "the Project Gutenberg Online Distributed Proofreading Team",
        "**The Project Gutenberg",
        "*SMALL PRINT!",
        "More information about this book is at the top of this file.",
        "tells you about restrictions in how the file may be used.",
        "l'authorization à les utilizer pour preparer ce texte.",
        "of the etext through OCR.",
        "*****These eBooks Were Prepared By Thousands of Volunteers!*****",
        "We need your donations more than ever!",
        " *** START OF THIS PROJECT GUTENBERG",
        "****     SMALL PRINT!",
        '["Small Print" V.',
        "      (http://www.ibiblio.org/gutenberg/",
        "and the Project Gutenberg Online Distributed Proofreading Team",
        "Mary Meehan, and the Project Gutenberg Online Distributed Proofreading",
        "                this Project Gutenberg edition.",
    )
)


TEXT_END_MARKERS = frozenset(
    (
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of The Project Gutenberg",
        "Ende dieses Project Gutenberg",
        "by Project Gutenberg",
        "End of Project Gutenberg",
        "End of this Project Gutenberg",
        "Ende dieses Projekt Gutenberg",
        "        ***END OF THE PROJECT GUTENBERG",
        "*** END OF THE COPYRIGHTED",
        "End of this is COPYRIGHTED",
        "Ende dieses Etextes ",
        "Ende dieses Project Gutenber",
        "Ende diese Project Gutenberg",
        "**This is a COPYRIGHTED Project Gutenberg Etext, Details Above**",
        "Fin de Project Gutenberg",
        "The Project Gutenberg Etext of ",
        "Ce document fut presente en lecture",
        "Ce document fut présenté en lecture",
        "More information about this book is at the top of this file.",
        "We need your donations more than ever!",
        "END OF PROJECT GUTENBERG",
        " End of the Project Gutenberg",
        " *** END OF THIS PROJECT GUTENBERG",
    )
)


LEGALESE_START_MARKERS = frozenset(("<<THIS ELECTRONIC VERSION OF",))


LEGALESE_END_MARKERS = frozenset(("SERVICE THAT CHARGES FOR DOWNLOAD",))
