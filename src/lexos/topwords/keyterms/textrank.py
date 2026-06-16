"""textrank.py.

Last Updated: June 16, 2026
Last Tested: May 24, 2026

Usage:

from lexos.topwords.keyterms.textrank import TextRank
keyterms = TextRank(doc=doc).keyterms

or

from lexos.topwords.keyterms.textrank import textrank
keyterms = textrank(doc=doc)
"""

import collections
import heapq
import itertools
from operator import itemgetter
from typing import Callable, Collection, Iterable, Literal, Optional, Sequence

import pandas as pd
from pydantic import ConfigDict, Field
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token
from textacy.extract.utils import get_filtered_topn_terms
from textacy.representations import network
from textacy.utils import to_set

from lexos.topwords import TopWords
from lexos.topwords.keyterms.keyterms_util import (
    _resolve_topn,
    _to_term_sequence,
    is_unicode_punctuation,
    terms_to_strings,
)

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
)

class TextRank(TopWords):
    """Extracts keyterms using the TextRank algorithm."""

    doc: str | Doc = Field(..., description="The raw text or spaCy doc to analyze.")
    normalize: Optional[Literal["orth", "lower", "lemma"]] = Field(
        default=None, description="How to normalize tokens for candidate selection."
    )
    include_pos: Optional[str | Collection[str]] = Field(
        default=("NOUN", "PROPN", "ADJ"),
        description="POS tags to include for candidate selection.",
    )
    stopwords: Optional[str | Collection[str]] = Field(
        default=None, description="Custom stopwords to exclude from candidates."
    )
    ngrams: Optional[int | Iterable[int]] = Field(
        default=1,
        description="The ngram range for candidate selection, e.g., 1 for unigrams, (1, 2) for unigrams and bigrams.",
    )
    window_size: Optional[int] = Field(
        2, gt=0, description="The size of the sliding window for co-occurrence."
    )
    edge_weighting: Optional[str] = Field(
        "binary", description='How to weight edges in the graph: "binary" or "count".'
    )
    position_bias: Optional[bool] = Field(
        False,
        description="Whether to bias towards candidates appearing earlier in the text.",
    )
    candidate_weighting: Optional[Literal["unique", "frequency"]] = Field(
        "unique",
        description="How to weight candidates based on their frequency or uniqueness.",
    )
    topn: Optional[int | float] = Field(
        10,
        gt=0,
        description="The number of top keyterms to return (int or float ratio of candidates).",
    )

    keyterms: list[tuple[str, float]] | None = Field(
        default=None,
        description="Extracted keyterms.",
    )

    model_config = validation_config

    def __init__(self, **kwargs) -> None:
        """Initialize the TextRank object and extract keyterms."""
        super().__init__(**kwargs)
        # Extract keyterms and store them in the `keyterms` field
        self.keyterms = textrank(
            doc=self.doc,
            normalize=self.normalize,
            include_pos=self.include_pos,
            stopwords=self.stopwords,
            ngrams=self.ngrams,
            window_size=self.window_size,
            edge_weighting=self.edge_weighting,
            position_bias=self.position_bias,
            candidate_weighting=self.candidate_weighting,
            topn=self.topn,
        )

    def to_dict(self):
        """Return the extracted keyterms as a dictionary."""
        return {
            "keyterms": [
                {"term": term, "score": score} for term, score in (self.keyterms or [])
            ]
        }

    def to_df(self):
        """Return the extracted keyterms as a pandas DataFrame."""
        return pd.DataFrame(getattr(self, "keyterms", []), columns=["term", "score"])


def textrank(
    doc: Doc | str,
    *,
    normalize: Literal["orth", "lower", "lemma"] | None = None,
    include_pos: Optional[str | Collection[str]] = ("NOUN", "PROPN", "ADJ"),
    stopwords: Optional[str | Collection[str]] = None,
    ngrams: int | Iterable[int] | None = 1,
    window_size: int = 2,
    edge_weighting: str = "binary",
    position_bias: bool = False,
    candidate_weighting: Literal["unique", "frequency"] = "unique",
    topn: int | float = 10,
) -> list[tuple[str, float]]:
    """Extract key terms from a document using the TextRank algorithm, or a variation thereof.

    Args:
        doc (Doc | str): spaCy `Doc` or plain string from which to extract keyterms.
        normalize (Literal["orth", "lower", "lemma"] | None): If "lemma", lemmatize
            terms; if "lower", lowercase terms; if None, use the orthographic forms
            that appear in `doc`.
        include_pos (str | Collection[str] | None): One or more POS tags with which
            to filter for good candidate keyterms. If `None`, include tokens of all POS
            tags (which also allows keyterm extraction from docs without POS-tagging.)
        stopwords (str | Collection[str] | None): One or more stopwords to filter out.
            When provided for spaCy `Doc` inputs, this overrides the model's built-in
            stopword flags and only filters tokens in this list.
        ngrams (int | Iterable[int] | None): Candidate n-gram lengths to
            extract (for example, `1` or `(1, 2, 3)`). If `None`, defaults to
            unigram extraction.
        window_size (int): Size of sliding window in which term co-occurrences are
            determined.
        edge_weighting (str: {"count", "binary"}): : If "count", the nodes for
            all co-occurring terms are connected by edges with weight equal to
            the number of times they co-occurred within a sliding window;
            if "binary", all such edges have weight = 1.
        position_bias (bool): If True, bias the PageRank algorithm for weighting
            nodes in the word graph, such that words appearing earlier and more
            frequently in `doc` tend to get larger weights.
        candidate_weighting (Literal["unique", "frequency"]): Weighting mode for
            candidate phrase scoring. If "unique", score each unique candidate phrase
            once (Textacy behaviour). If "frequency", multiply each candidate phrase's
            score by its observed frequency in the document.
        topn (int | float): Number of top-ranked terms to return as key terms.
            If an integer, represents the absolute number; if a float, value
            must be in the interval (0.0, 1.0], which is converted to an int by
            `int(round(len(candidates) * topn))`.

    Returns:
        list[tuple[str, float]]: Sorted list of top `topn` key terms and their
            corresponding TextRank ranking scores.

    Notes:
        Example parameter settings for different TextRank variations:

        - TextRank: `window_size=2, edge_weighting="binary", position_bias=False`
        - SingleRank: `window_size=10, edge_weighting="count", position_bias=False`
        - PositionRank: `window_size=10, edge_weighting="count", position_bias=True`
    """
    include_pos, stopwords, ngrams, topn = _validate_textrank_args(
        include_pos,
        stopwords,
        ngrams,
        candidate_weighting,
        topn,
    )

    # Build aligned term sequences once and reuse across all processing stages.
    terms = _to_term_sequence(doc)
    if not terms:
        return []
    normalized_terms = list(terms_to_strings(terms, normalize))

    word_pos = _build_position_bias(normalized_terms) if position_bias else None

    # Build a graph from all words in doc, then score them
    graph = network.build_cooccurrence_network(
        normalized_terms,
        window_size=window_size,
        edge_weighting=edge_weighting,
    )
    word_scores = network.rank_nodes_by_pagerank(
        graph, weight="weight", personalization=word_pos
    )

    # Generate candidate terms with frequencies in a single streaming pass.
    # Algorithm optimization: frequency-aware candidate extraction.
    candidate_counts = _get_candidate_counts(
        terms, normalized_terms, include_pos, stopwords, ngrams
    )
    topn = _resolve_topn(topn, len(candidate_counts))

    candidate_scores = _score_candidates(
        candidate_counts, word_scores, candidate_weighting
    )
    sorted_candidate_scores = _rank_candidate_scores(candidate_scores, topn)

    return get_filtered_topn_terms(sorted_candidate_scores, topn, match_threshold=0.8)


def _validate_textrank_args(
    include_pos: Optional[str | Collection[str]],
    stopwords: Optional[str | Collection[str]],
    ngrams: int | Iterable[int] | None,
    candidate_weighting: Literal["unique", "frequency"],
    topn: int | float,
) -> tuple[Optional[set[str]], Optional[set[str]], tuple[int, ...], int | float]:
    """Validate input args and return transformed values used by textrank()."""
    include_pos_set = to_set(include_pos) if include_pos else None
    stopwords_set = {word.lower() for word in to_set(stopwords)} if stopwords else None
    if ngrams is None:
        ngram_sizes: tuple[int, ...] = (1,)
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
    if candidate_weighting not in {"unique", "frequency"}:
        raise ValueError(
            f"candidate_weighting={candidate_weighting} is invalid; must be one of {{'unique', 'frequency'}}"
        )
    if isinstance(topn, float) and not 0.0 < topn <= 1.0:
        raise ValueError(
            f"topn={topn} is invalid; must be an int, or a float between 0.0 and 1.0"
        )
    return include_pos_set, stopwords_set, ngram_sizes, topn


def _build_position_bias(normalized_terms: Sequence[str]) -> dict[str, float]:
    """Compute an in-place normalized position-bias mapping for PageRank personalization."""
    word_pos: dict[str, float] = {}
    for i, norm_word in enumerate(normalized_terms):
        word_pos[norm_word] = word_pos.get(norm_word, 0.0) + (1 / (i + 1))
    sum_word_pos = sum(word_pos.values())
    if sum_word_pos > 0.0:
        inv_sum_word_pos = 1.0 / sum_word_pos
        for word in word_pos:
            word_pos[word] *= inv_sum_word_pos
    return word_pos

def _score_candidates(
    candidate_counts: collections.Counter[tuple[str, ...]],
    word_scores: dict[str, float],
    candidate_weighting: Literal["unique", "frequency"],
) -> dict[str, float]:
    """Compute phrase scores from word scores with optional frequency weighting."""
    score_get = word_scores.get
    use_frequency = candidate_weighting == "frequency"
    candidate_scores: dict[str, float] = {}
    for candidate, count in candidate_counts.items():
        multiplier = count if use_frequency else 1
        candidate_scores[" ".join(candidate)] = multiplier * sum(
            score_get(word, 0.0) for word in candidate
        )
    return candidate_scores

def _rank_candidate_scores(
    candidate_scores: dict[str, float], topn: int
) -> list[tuple[str, float]]:
    """Rank candidate phrase scores, using heap-based preselection for small topn."""
    if len(candidate_scores) > topn and topn > 0:
        overfetch_n = min(len(candidate_scores), max(topn * 5, topn + 25))
        return heapq.nlargest(
            overfetch_n,
            candidate_scores.items(),
            key=lambda item: (item[1], item[0]),
        )
    return sorted(candidate_scores.items(), key=itemgetter(1, 0), reverse=True)

def _get_candidate_counts(
    terms: Sequence[Token | str],
    normalized_terms: Sequence[str],
    include_pos: Optional[set[str]],
    stopwords: Optional[set[str]],
    ngrams: Sequence[int],
) -> collections.Counter[tuple[str, ...]]:
    """Get frequency counts of candidate terms by joining the longest subsequences of valid words.

    A valid word is non-stopword and non-punct, filtered to nouns, proper nouns, and adjectives
    if `doc` is POS-tagged -- then normalized into strings.

    Args:
        terms (Sequence[Token | str]): Original term sequence from the document.
        normalized_terms (Sequence[str]): Normalized term sequence aligned with `terms`.
        include_pos (set[str] | None): Set of POS tags to include for candidate selection, or None to include all.
        stopwords (set[str] | None): Set of stopwords to exclude from candidates, or None to use spaCy's built-in stopword flags.
        ngrams (Sequence[int]): Sequence of n-gram sizes to extract as candidates.

    Returns:
        collections.Counter[tuple[str, ...]]: A counter mapping candidate term tuples to their frequency counts in the document.
    """

    def _is_valid_tok(tok: Token | str) -> bool:
        """Return True if `tok` is a valid token for candidate keyterms, False otherwise.

        Args:
            tok (Token | str): Token or string to check.

        Returns:
            bool: True if `tok` is a valid token for candidate keyterms, False otherwise.
        """
        if isinstance(tok, str):
            is_custom_stopword = stopwords is not None and tok.lower() in stopwords
            return not (
                is_unicode_punctuation(tok) or tok.isspace() or is_custom_stopword
            )
        else:
            is_stopword = (
                tok.lower_ in stopwords if stopwords is not None else tok.is_stop
            )
            # If the doc was produced by a model without a tagger, tok.pos_ is "".
            # In that case skip the POS filter rather than excluding everything.
            pos_ok = include_pos is None or not tok.pos_ or tok.pos_ in include_pos
            return not (is_stopword or tok.is_punct or tok.is_space) and pos_ok

    candidate_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    ngram_sizes = tuple(sorted(set(ngrams)))

    valid_run: list[str] = []

    def _consume_run(run: list[str]) -> None:
        """Consume a run of valid normalized tokens by extracting n-grams and updating `candidate_counts`.

        Args:
            run (list[str]): List of valid normalized tokens.

        Returns:
            None
        """
        run_len = len(run)
        if run_len == 0:
            return
        for n in ngram_sizes:
            if n > run_len:
                continue
            for i in range(run_len - n + 1):
                candidate_counts[tuple(run[i : i + n])] += 1

    for tok, norm_tok in zip(terms, normalized_terms, strict=True):
        if _is_valid_tok(tok):
            valid_run.append(norm_tok)
        else:
            _consume_run(valid_run)
            valid_run = []

    _consume_run(valid_run)

    return candidate_counts

def get_longest_subsequence_candidates(
    doc: Doc | str,
    match_func: Callable[[Token | str], bool],
) -> Iterable[tuple[Token | str, ...]]:
    """Get candidate keyterms from `doc`.

    Candidates are longest consecutive subsequences of tokens for which all `match_func(token)` is True.

    Args:
        doc (Doc | str): Document from which to extract candidate keyterms.
        match_func (Callable[[Token | str], bool]): Function applied sequentially to each `Token` or string in `doc`
            that returns True for matching ("good") tokens, False otherwise.

    Yields:
        Iterable[tuple[Token | str, ...]]: Next longest consecutive subsequence candidate, as a tuple
            of constituent terms.
    """
    if isinstance(doc, str):
        # Note: This must eventually allow a custom tokenizer to be passed in, but for now we just split on whitespace.
        doc = doc.split()
    for key, words_grp in itertools.groupby(doc, key=match_func):
        if key is True:
            yield tuple(words_grp)
