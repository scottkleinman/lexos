"""yake.py.

Last Updated: May 24, 2026
Last Tested: May 24, 2026

Usage:

from lexos.topwords.keyterms.yake import Yake
keyterms = Yake(doc=doc).keyterms

or

from lexos.topwords.keyterms.yake import yake
keyterms = yake(doc=doc)
"""

import keyterms_util
import collections
import math
import re
import statistics
import unicodedata
from functools import reduce
from operator import mul
from typing import Any, Collection, Iterable, Literal, Optional, Sequence

import pandas as pd
from pydantic import ConfigDict, Field
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Token
from textacy.extract.utils import get_filtered_topn_terms
from textacy.utils import to_set

from lexos.topwords import TopWords
from lexos.topwords.keyterms.keyterms_util import (
    is_unicode_punctuation,
)

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
)

TermLike = str | Token | Any
DocLike = Doc | str | Sequence[TermLike]


class Yake(TopWords):
    """Extracts keyterms using the YAKE algorithm."""

    doc: DocLike = Field(
        ...,
        description="Input as a spaCy doc, raw string, or sequence of terms.",
    )
    normalize: Optional[Literal["orth", "lower", "lemma", "norm"] | None] = Field(
        default="lemma", description="How to normalize terms for scoring."
    )
    include_pos: Optional[str | Collection[str]] = Field(
        default=("NOUN", "PROPN", "ADJ"),
        description="POS tags to include for candidates; ignored when unavailable.",
    )
    stopwords: Optional[str | Collection[str]] = Field(
        default=None,
        description="Custom stopwords to exclude from candidates and scoring.",
    )
    ngrams: Optional[int | Iterable[int] | None] = Field(
        default=(1, 2, 3),
        description="N-gram sizes to consider for keyterm candidates.",
    )
    window_size: Optional[int] = Field(
        2,
        gt=0,
        description="Context window size on each side of each term.",
    )
    topn: Optional[int | float] = Field(
        10,
        gt=0,
        description="Number of top keyterms (or ratio if float in (0, 1]).",
    )

    keyterms: list[tuple[str, float]] | None = Field(
        default=None,
        description="Extracted keyterms as (term, score) tuples.",
    )

    model_config = validation_config

    def __init__(self, **kwargs):
        """Initialize the Yake object and extract keyterms."""
        super().__init__(**kwargs)

        self.keyterms = yake(
            doc=self.doc,
            normalize=self.normalize,
            include_pos=self.include_pos,
            stopwords=self.stopwords,
            ngrams=self.ngrams,
            window_size=self.window_size,
            topn=self.topn,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the extracted keyterms as a dictionary.

        Returns:
            dict[str, Any]: A dictionary containing the extracted keyterms.
        """
        return {
            "keyterms": [
                {"term": term, "score": score} for term, score in (self.keyterms or [])
            ]
        }

    def to_df(self) -> pd.DataFrame:
        """Return the extracted keyterms as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with columns 'term' and 'score' containing the extracted keyterms.
        """
        return pd.DataFrame(getattr(self, "keyterms", []), columns=["term", "score"])


def yake(
    doc: DocLike,
    *,
    normalize: Literal["orth", "lower", "lemma", "norm"] | None = "lemma",
    include_pos: Optional[str | Collection[str]] = ("NOUN", "PROPN", "ADJ"),
    stopwords: Optional[str | Collection[str]] = None,
    ngrams: int | Iterable[int] | None = (1, 2, 3),
    window_size: int = 2,
    topn: int | float = 10,
) -> list[tuple[str, float]]:
    """Extract key terms from a document using the YAKE algorithm.

    This implementation is inspired by Textacy's YAKE extractor but adds
    compatibility for raw strings and sequences of strings/token-like objects.

    Args:
        doc (DocLike): Input document as a spaCy Doc, raw string, or sequence of terms.
        normalize (Literal["orth", "lower", "lemma", "norm"] | None): How to normalize terms for scoring.
        include_pos (Optional[str | Collection[str]]): POS tags to include for candidates; ignored when unavailable.
        stopwords (Optional[str | Collection[str]]): Custom stopwords to exclude from candidates and scoring.
        ngrams (int | Iterable[int] | None): N-gram sizes to consider for keyterm candidates.
        window_size (int): Context window size on each side of each term.
        topn (int | float): Number of top keyterms to return (or ratio if float in (0, 1]).

    Returns:
        list[tuple[str, float]]: Extracted keyterms as (term, score) tuples.
    """
    include_pos_set, stopwords_set, ngram_sizes, topn = _validate_yake_args(
        include_pos=include_pos,
        stopwords=stopwords,
        ngrams=ngrams,
        normalize=normalize,
        topn=topn,
    )
    terms, sent_ids = _to_terms_and_sentence_ids(doc)
    if not terms:
        return []

    stop_words: set[str] = set()
    seen_candidates: set[str] = set()

    word_occ_vals = _get_per_word_occurrence_values(
        terms=terms,
        sent_ids=sent_ids,
        normalize=normalize,
        stop_words=stop_words,
        window_size=window_size,
        custom_stopwords=stopwords_set,
    )
    if not word_occ_vals:
        return []

    word_freqs = {w_id: len(vals["is_uc"]) for w_id, vals in word_occ_vals.items()}
    word_scores = _compute_word_scores(
        sent_ids=sent_ids,
        word_occ_vals=word_occ_vals,
        word_freqs=word_freqs,
        stop_words=stop_words,
    )

    term_scores: dict[str, float] = {}

    if 1 in ngram_sizes:
        candidates = _get_unigram_candidates(
            terms,
            include_pos=include_pos_set,
            custom_stopwords=stopwords_set,
        )
        _score_unigram_candidates(
            candidates=candidates,
            word_freqs=word_freqs,
            word_scores=word_scores,
            term_scores=term_scores,
            stop_words=stop_words,
            seen_candidates=seen_candidates,
            normalize=normalize,
        )

    ngram_candidates = _get_ngram_candidates(
        terms,
        sent_ids,
        n_sizes=tuple(n for n in ngram_sizes if n > 1),
        include_pos=include_pos_set,
        custom_stopwords=stopwords_set,
    )
    ngram_freqs = collections.Counter(
        " ".join(_term_to_str(term, normalize) for term in ngram)
        for ngram in ngram_candidates
    )
    _score_ngram_candidates(
        candidates=ngram_candidates,
        ngram_freqs=ngram_freqs,
        word_scores=word_scores,
        term_scores=term_scores,
        seen_candidates=seen_candidates,
        normalize=normalize,
    )

    if isinstance(topn, float):
        topn = int(round(len(seen_candidates) * topn))

    sorted_term_scores = sorted(term_scores.items(), key=lambda item: item[1])
    return get_filtered_topn_terms(sorted_term_scores, topn, match_threshold=0.8)


def _validate_yake_args(
    *,
    include_pos: Optional[str | Collection[str]],
    stopwords: Optional[str | Collection[str]],
    ngrams: int | Iterable[int] | None,
    normalize: Literal["orth", "lower", "lemma", "norm"] | None,
    topn: int | float,
) -> tuple[Optional[set[str]], Optional[set[str]], tuple[int, ...], int | float]:
    """Validate and normalize YAKE input arguments.

    Args:
        include_pos (Optional[str | Collection[str]]): POS tags to include for candidates; ignored when unavailable.
        stopwords (Optional[str | Collection[str]]): Custom stopwords to exclude from candidates and scoring.
        ngrams (int | Iterable[int] | None): N-gram sizes to consider for keyterm candidates.
        normalize (Literal["orth", "lower", "lemma", "norm"] | None): How to normalize terms for scoring.
        topn (int | float): Number of top keyterms to return (or ratio if float in (0, 1]).

    Returns:
        tuple[Optional[set[str]], Optional[set[str]], tuple[int, ...], int | float]: Normalized YAKE arguments.
    """
    include_pos_set = to_set(include_pos) if include_pos else None
    stopwords_set = {word.lower() for word in to_set(stopwords)} if stopwords else None

    if ngrams is None:
        ngram_sizes = (1,)
    elif isinstance(ngrams, int):
        if ngrams < 1:
            raise ValueError(
                f"ngrams={ngrams} is invalid; all values must be integers >= 1"
            )
        ngram_sizes = (ngrams,)
    else:
        ngram_sizes = tuple(sorted(set(ngrams)))
        if not ngram_sizes or any(
            (not isinstance(n, int) or n < 1) for n in ngram_sizes
        ):
            raise ValueError(
                f"ngrams={ngrams} is invalid; must be an int >= 1 or a non-empty iterable of ints >= 1"
            )

    if normalize not in {"orth", "lower", "lemma", "norm", None}:
        raise ValueError(
            f"normalize={normalize} is invalid; must be one of {{'orth', 'lower', 'lemma', 'norm', None}}"
        )

    if isinstance(topn, float) and not 0.0 < topn <= 1.0:
        raise ValueError(
            f"topn={topn} is invalid; must be an int, or a float between 0.0 and 1.0"
        )

    return include_pos_set, stopwords_set, ngram_sizes, topn


def _to_terms_and_sentence_ids(doc: DocLike) -> tuple[list[TermLike], list[int]]:
    """Convert input into term and sentence-id sequences.

    Args:
        doc (DocLike): Input document as a spaCy Doc, raw string, or sequence of terms.

    Returns:
        tuple[list[TermLike], list[int]]: Term and sentence-id sequences.
    """
    if isinstance(doc, Doc):
        terms = list(doc)
        if not terms:
            return [], []

        if doc.has_annotation("SENT_START"):
            sent_ids: list[int] = []
            for sent_idx, sent in enumerate(doc.sents):
                sent_ids.extend([sent_idx] * len(sent))
            if len(sent_ids) == len(terms):
                return terms, sent_ids

        return terms, [0] * len(terms)

    if isinstance(doc, str):
        terms = re.findall(r"\w+|[^\w\s]", doc, flags=re.UNICODE)
    else:
        terms = list(doc)

    if not terms:
        return [], []

    sent_ids = [0] * len(terms)
    curr_sent = 0
    for i, term in enumerate(terms):
        sent_ids[i] = curr_sent
        text = _term_to_text(term)
        if text in {".", "!", "?"} and i < len(terms) - 1:
            curr_sent += 1

    return terms, sent_ids


def _get_per_word_occurrence_values(
    *,
    terms: Sequence[TermLike],
    sent_ids: Sequence[int],
    normalize: Literal["orth", "lower", "lemma", "norm"] | None,
    stop_words: set[str],
    window_size: int,
    custom_stopwords: Optional[set[str]],
) -> dict[str, dict[str, list]]:
    """Collect per-occurrence stats needed to compute per-word YAKE scores.

    Args:
        terms (Sequence[TermLike]): Sequence of term-like objects in the document.
        sent_ids (Sequence[int]): Corresponding sentence IDs for each term.
        normalize (Literal["orth", "lower", "lemma", "norm"] | None): How to normalize terms for scoring.
        stop_words (set[str]): Set to be populated with normalized stopword IDs.
        window_size (int): Context window size on each side of each term.
        custom_stopwords (Optional[set[str]]): Custom stopwords to exclude from candidates and scoring.

    Returns:
        dict[str, dict[str, list]]: Per-word occurrence stats.
    """
    word_occ_vals: collections.defaultdict = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    by_sent: dict[int, list[TermLike]] = collections.defaultdict(list)
    for term, sent_id in zip(terms, sent_ids, strict=True):
        if _is_space(term) or _is_punct(term):
            continue
        by_sent[sent_id].append(term)

    for sent_id, sent_terms in by_sent.items():
        sent_len = len(sent_terms)
        for i, word in enumerate(sent_terms):
            lwords = sent_terms[max(0, i - window_size) : i]
            rwords = sent_terms[i + 1 : i + 1 + window_size]

            w_id = _term_to_id(word, normalize)
            if _is_stop(word, custom_stopwords):
                stop_words.add(w_id)

            is_sent_start = i == 0
            word_occ_vals[w_id]["is_uc"].append(_is_upper_cased(word, is_sent_start))
            word_occ_vals[w_id]["sent_idx"].append(sent_id)
            word_occ_vals[w_id]["l_context"].extend(
                _term_to_id(w, normalize) for w in lwords if not _is_space(w)
            )
            word_occ_vals[w_id]["r_context"].extend(
                _term_to_id(w, normalize) for w in rwords if not _is_space(w)
            )

    return word_occ_vals


def _compute_word_scores(
    *,
    sent_ids: Sequence[int],
    word_occ_vals: dict[str, dict[str, list]],
    word_freqs: dict[str, int],
    stop_words: set[str],
) -> dict[str, float]:
    """Compute YAKE per-word scores from aggregate occurrence stats.

    Args:
        sent_ids (Sequence[int]): Sentence IDs corresponding to each term occurrence.
        word_occ_vals (dict[str, dict[str, list]]): Per-word occurrence stats collected from the document.
        word_freqs (dict[str, int]): Frequency of each word in the document.
        stop_words (set[str]): Set of normalized stopword IDs to exclude from scoring.

    Returns:
        dict[str, float]: YAKE per-word scores.
    """
    if not word_freqs:
        return {}

    freqs_nsw = [freq for w_id, freq in word_freqs.items() if w_id not in stop_words]
    if not freqs_nsw:
        freqs_nsw = list(word_freqs.values())

    if len(freqs_nsw) > 1:
        freq_baseline = statistics.mean(freqs_nsw) + statistics.stdev(freqs_nsw)
    else:
        freq_baseline = float(freqs_nsw[0]) if freqs_nsw else 1.0
    if freq_baseline <= 0:
        freq_baseline = 1.0

    freq_max = max(word_freqs.values())
    n_sents = max(1, len(set(sent_ids)))

    word_weights: dict[str, dict[str, float]] = {}
    for w_id, vals in word_occ_vals.items():
        freq = word_freqs[w_id]
        n_lctx = len(vals["l_context"])
        n_rctx = len(vals["r_context"])
        n_unique_lc = len(set(vals["l_context"]))
        n_unique_rc = len(set(vals["r_context"]))

        wl = (n_unique_lc / n_lctx) if n_lctx else 0.0
        wr = (n_unique_rc / n_rctx) if n_rctx else 0.0
        pl = n_unique_lc / freq_max
        pr = n_unique_rc / freq_max

        case = sum(vals["is_uc"]) / math.log2(1 + freq)
        pos = math.log2(math.log2(3 + statistics.mean(vals["sent_idx"])))
        freq_w = freq / freq_baseline
        disp = len(set(vals["sent_idx"])) / n_sents
        rel = 1.0 + (wl + wr) * (freq / freq_max) + pl + pr

        word_weights[w_id] = {
            "case": case,
            "pos": pos,
            "freq": freq_w,
            "disp": disp,
            "rel": rel,
        }

    word_scores: dict[str, float] = {}
    for w_id, wts in word_weights.items():
        denom = wts["case"] + (wts["freq"] / wts["rel"]) + (wts["disp"] / wts["rel"])
        if denom <= 0:
            continue
        word_scores[w_id] = (wts["rel"] * wts["pos"]) / denom

    return word_scores


def _get_unigram_candidates(
    terms: Sequence[TermLike],
    *,
    include_pos: Optional[set[str]],
    custom_stopwords: Optional[set[str]],
) -> Iterable[TermLike]:
    """Yield valid unigram candidates.

    Args:
        terms (Sequence[TermLike]): Sequence of term-like objects in the document.
        include_pos (Optional[set[str]]): Set of POS tags to include.
        custom_stopwords (Optional[set[str]]): Custom stopwords to exclude from candidates.

    Yields:
        TermLike: Valid unigram candidates.
    """
    for term in terms:
        if _is_space(term) or _is_punct(term) or _is_stop(term, custom_stopwords):
            continue
        if not _pos_allowed(term, include_pos):
            continue
        yield term


def _score_unigram_candidates(
    *,
    candidates: Iterable[TermLike],
    word_freqs: dict[str, int],
    word_scores: dict[str, float],
    term_scores: dict[str, float],
    stop_words: set[str],
    seen_candidates: set[str],
    normalize: Literal["orth", "lower", "lemma", "norm"] | None,
) -> None:
    """Score unigram candidates and add them to term scores.

    Args:
        candidates (Iterable[TermLike]): Iterable of unigram candidate terms.
        word_freqs (dict[str, int]): Frequency of each word in the document.
        word_scores (dict[str, float]): YAKE per-word scores.
        term_scores (dict[str, float]): Dictionary to populate with candidate term scores.
        stop_words (set[str]): Set of normalized stopword IDs to exclude from scoring.
        seen_candidates (set[str]): Set of normalized candidate strings already scored.
        normalize (Literal["orth", "lower", "lemma", "norm"] | None): How to normalize terms for scoring.

    Returns:
        None: Scores are added to the term_scores dictionary in-place.
    """
    for word in candidates:
        w_id = _term_to_id(word, normalize)
        if w_id in stop_words or w_id in seen_candidates:
            continue
        seen_candidates.add(w_id)

        w_score = word_scores.get(w_id)
        w_freq = word_freqs.get(w_id)
        if w_score is None or w_freq is None:
            continue

        term_scores[_term_to_str(word, normalize)] = w_score / (
            math.log2(1 + w_freq) * (1 + w_score)
        )


def _get_ngram_candidates(
    terms: Sequence[TermLike],
    sent_ids: Sequence[int],
    *,
    n_sizes: tuple[int, ...],
    include_pos: Optional[set[str]],
    custom_stopwords: Optional[set[str]],
) -> list[tuple[TermLike, ...]]:
    """Generate n-gram candidates within sentence boundaries.

    Args:
        terms (Sequence[TermLike]): Sequence of term-like objects in the document.
        sent_ids (Sequence[int]): Sequence of sentence IDs corresponding to each term.
        n_sizes (tuple[int, ...]): Tuple of n-gram sizes to generate.
        include_pos (Optional[set[str]]): Set of POS tags to include.
        custom_stopwords (Optional[set[str]]): Custom stopwords to exclude from candidates.

    Returns:
        list[tuple[TermLike, ...]]: List of n-gram candidate tuples.
    """
    if not n_sizes:
        return []

    by_sent: dict[int, list[TermLike]] = collections.defaultdict(list)
    for term, sent_id in zip(terms, sent_ids, strict=True):
        if _is_space(term) or _is_punct(term) or _is_stop(term, custom_stopwords):
            by_sent[sent_id].append(None)
        elif not _pos_allowed(term, include_pos):
            by_sent[sent_id].append(None)
        else:
            by_sent[sent_id].append(term)

    candidates: list[tuple[TermLike, ...]] = []
    for sent_terms in by_sent.values():
        run: list[TermLike] = []
        for term in sent_terms + [None]:
            if term is None:
                run_len = len(run)
                if run_len:
                    for n in n_sizes:
                        if n > run_len:
                            continue
                        for i in range(run_len - n + 1):
                            candidates.append(tuple(run[i : i + n]))
                run = []
            else:
                run.append(term)

    return candidates


def _score_ngram_candidates(
    *,
    candidates: list[tuple[TermLike, ...]],
    ngram_freqs: dict[str, int],
    word_scores: dict[str, float],
    term_scores: dict[str, float],
    seen_candidates: set[str],
    normalize: Literal["orth", "lower", "lemma", "norm"] | None,
) -> None:
    """Score n-gram candidates and add them to term scores.

    Args:
        candidates (list[tuple[TermLike, ...]]): List of n-gram candidate tuples.
        ngram_freqs (dict[str, int]): Frequency of each n-gram candidate in the document.
        word_scores (dict[str, float]): YAKE per-word scores.
        term_scores (dict[str, float]): Dictionary to populate with candidate term scores.
        seen_candidates (set[str]): Set of normalized candidate strings already scored.
        normalize (Literal["orth", "lower", "lemma", "norm"] | None): How to normalize terms for scoring.

    Returns:
    None: Scores are added to the term_scores dictionary in-place.
    """
    for ngram in candidates:
        ngtxt = " ".join(_term_to_str(word, normalize) for word in ngram)
        if ngtxt in seen_candidates:
            continue
        seen_candidates.add(ngtxt)

        ngram_word_scores = [
            word_scores.get(_term_to_id(word, normalize), 0.0) for word in ngram
        ]
        if any(score <= 0.0 for score in ngram_word_scores):
            continue

        numerator = reduce(mul, ngram_word_scores, 1.0)
        denominator = math.log2(1 + ngram_freqs[ngtxt]) * (1.0 + sum(ngram_word_scores))
        if denominator <= 0:
            continue

        term_scores[ngtxt] = numerator / denominator


def _term_to_text(term: TermLike) -> str:
    """Get term text from a string or token-like object.

    Args:
        term (TermLike): A term-like object, which may be a string or have a 'text' attribute.

    Returns:
        str: The text representation of the term.
    """
    if isinstance(term, str):
        return term
    return str(getattr(term, "text", term))


def _term_to_id(
    term: TermLike,
    normalize: Literal["orth", "lower", "lemma", "norm"] | None,
) -> str:
    """Get normalized identifier for a term.

    Args:
        term (TermLike): A term-like object, which may be a string or have attributes for normalization.
        normalize (Literal["orth", "lower", "lemma", "norm"] | None): How to normalize the term for scoring.

    Returns:
        str: The normalized identifier for the term.
    """
    text = _term_to_text(term)
    if normalize in (None, "orth"):
        return text
    if normalize == "lower":
        if isinstance(term, str):
            return term.lower()
        return str(getattr(term, "lower_", text.lower()))
    if normalize == "lemma":
        if isinstance(term, str):
            return term
        return str(getattr(term, "lemma_", text))
    if normalize == "norm":
        if isinstance(term, str):
            return term
        return str(getattr(term, "norm_", text))
    return text


def _term_to_str(
    term: TermLike,
    normalize: Literal["orth", "lower", "lemma", "norm"] | None,
) -> str:
    """Get normalized string form for term output and candidate keys.

    Args:
        term (TermLike): A term-like object, which may be a string or have attributes for normalization.
        normalize (Literal["orth", "lower", "lemma", "norm"] | None): How to normalize the term for scoring.

    Returns:
        str: The normalized string form of the term.
    """
    return _term_to_id(term, normalize)


def _is_stop(term: TermLike, custom_stopwords: Optional[set[str]]) -> bool:
    """Return True if term should be treated as a stopword.

    Args:
        term (TermLike): A term-like object, which may be a string or have a 'text' attribute.
        custom_stopwords (Optional[set[str]]): An optional set of custom stopwords.

    Returns:
        bool: True if the term is a stopword, False otherwise.
    """
    if custom_stopwords is not None:
        return _term_to_text(term).lower() in custom_stopwords
    return bool(getattr(term, "is_stop", False))


def _is_space(term: TermLike) -> bool:
    """Return True if term is a space token.

    Args:
        term (TermLike): A term-like object, which may be a string or have a 'text' attribute.

    Returns:
        bool: True if the term is a space token, False otherwise.
    """
    if isinstance(term, str):
        return term.isspace()
    return bool(getattr(term, "is_space", False))


def _is_punct(term: TermLike) -> bool:
    """Return True if term is punctuation.

    Args:
        term (TermLike): A term-like object, which may be a string or have a 'text' attribute.

    Returns:
        bool: True if the term is punctuation, False otherwise.
    """
    if isinstance(term, str):
        return is_unicode_punctuation(term)

    is_punct_attr = getattr(term, "is_punct", None)
    if is_punct_attr is not None:
        return bool(is_punct_attr)

    return is_unicode_punctuation(_term_to_text(term))


def _is_upper_cased(term: TermLike, is_sent_start: bool) -> bool:
    """YAKE case feature for token-like terms.

    Args:
        term (TermLike): A term-like object, which may be a string or have a 'text' attribute.
        is_sent_start (bool): A boolean indicating if the term is at the start of a sentence.

    Returns:
        bool: True if the term is upper-cased, False otherwise.
    """
    if isinstance(term, str):
        return term.isupper() or (term.istitle() and not is_sent_start)

    if bool(getattr(term, "is_upper", False)):
        return True
    if bool(getattr(term, "is_title", False)) and not is_sent_start:
        return True
    return False


def _pos_allowed(term: TermLike, include_pos: Optional[set[str]]) -> bool:
    """Return True if term passes POS filtering or has no POS annotation.

    Args:
        term (TermLike): A term-like object, which may be a string or have a 'pos_' attribute.
        include_pos (Optional[set[str]]): An optional set of POS tags to include.

    Returns:
        bool: True if the term passes POS filtering or has no POS annotation, False otherwise.
    """
    if include_pos is None:
        return True

    pos_val = getattr(term, "pos_", None)
    if not pos_val:
        # Compatibility fix: if POS is unavailable, skip POS filtering.
        return True
    return pos_val in include_pos


