"""keyterms_util.py.

Last Updated: June 28, 2026
Last Tested: June 28

Usage:

from lexos.topwords.keyterms.keyterms_util import (
    _resolve_topn,
    _to_term_sequence,
    is_unicode_punctuation,
    terms_to_strings,
)

"""

import re
import unicodedata
from typing import Callable, Iterable
from spacy.tokens import Doc, Span, Token

def terms_to_strings(
    terms: Iterable[Span | Token | str],
    by: str | Callable[[Span | Token | str], str] | None,
) -> Iterable[str]:
    """Transform a sequence of terms as spaCy Tokens, Spans, or strings into strings.

    Args:
        terms (Iterable[Span | Token | str]): Terms to transform into strings.
        by (str | Callable[[Span | Token | str], str]): Method by which terms are transformed into strings.
            If "orth" or None, terms are represented by their text exactly as written;
            if "lower", by the lowercased form of their text;
            if "lemma", by their base form w/o inflectional suffixes;
            if a callable, must accept a `Token`, `Span`, or `str` and return a string.

    Yields:
        Iterable[str]: Next term in `terms`, as a string.
    """
    terms_: Iterable[str]
    if by in ("orth", None):
        terms_ = (
            term.text if isinstance(term, (Token, Span)) else term for term in terms
        )
    elif by == "lower":
        terms_ = (
            term.lower() if isinstance(term, str) else term.text.lower()
            for term in terms
        )
    elif by == "lemma":
        terms_ = (term.lemma_ if isinstance(term, Token) else term for term in terms)
    elif callable(by):
        terms_ = (by(term) for term in terms)
    else:
        raise ValueError(
            f"by={by} is invalid; must be one of {{'orth', 'lower', 'lemma', Callable}}"
        )
    for term in terms_:
        yield term

def is_unicode_punctuation(ch: str) -> bool:
    """Return True if `ch` is a Unicode punctuation character.

    Args:
        ch (str): Single character to check.

    Returns:
        bool: True if `ch` is a Unicode punctuation character, False otherwise.
    """
    # 'P' stands for Punctuation categories (Pc, Pd, Pe, Pf, Pi, Po, Ps)
    return bool(ch) and all(unicodedata.category(char).startswith("P") for char in ch)

def _to_term_sequence(doc: Doc | str) -> list[Token | str]:
    """Convert input into an ordered term sequence used by all processing stages.

    Args:
        doc (Doc | str): Document to convert into a term sequence.

    Returns:
        list[Token | str]: Ordered term sequence.
    """
    if isinstance(doc, str):
        # Keep punctuation as separate tokens so candidate extraction can filter it.
        return re.findall(r"\w+|[^\w\s]", doc, flags=re.UNICODE)
    return list(doc)

def _resolve_topn(topn: int | float, candidate_count: int) -> int:
    """Resolve a float ratio topn into an absolute integer count."""
    if isinstance(topn, float):
        return int(round(candidate_count * topn))
    return topn
