"""sgrank.py.

Last Updated: 
Last Tested: 

Usage:

from lexos.topwords.keyterms.sgrank import SGRank
keyterms = SGRank(doc=doc).keyterms

or

from lexos.topwords.keyterms.sgrank import sgrank
keyterms = sgrank(doc=doc)
"""

import collections
from operator import itemgetter
from typing import Callable, Collection, NamedTuple, Optional

import networkx as nx
import pandas as pd
from pydantic import ConfigDict, Field
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token
from textacy.extract.utils import get_filtered_topn_terms
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


class Candidate(NamedTuple):
    """A single candidate keyterm in the document.
    """
    text: str
    idx: int
    length: int
    count: int


class SGRank(TopWords):
    """Extracts keyterms using the SGRank algorithm."""

    doc: str | Doc = Field(..., description="The raw text or spaCy doc to analyze.")
    normalize: Optional[str | Callable[[Span], str]] = Field(
        default="lemma", description="How to normalize candidates for scoring."
    )
    ngrams: int | Collection[int] = Field(
        default=(1, 2, 3, 4, 5, 6),
        description="N-gram sizes to consider for keyterm candidates.",
    )
    include_pos: Optional[str | Collection[str]] = Field(
        default=("NOUN", "PROPN", "ADJ"),
        description="POS tags to include for candidate selection.",
    )
    window_size: Optional[int] = Field(
        1500,
        gt=1,
        description="Size of the sliding window for co-occurrence, in tokens.",
    )
    topn: Optional[int | float] = Field(
        10,
        gt=0,
        description="The number of top keyterms to return (int or float ratio of candidates).",
    )
    idf: Optional[dict[str, float]] = Field(
        default=None,
        description="Mapping of normalized term to inverse document frequency.",
    )

    keyterms: list[tuple[str, float]] | None = Field(
        default=None,
        description="Extracted keyterms.",
    )

    model_config = validation_config

    def __init__(self, **kwargs) -> None:
        """Initialize the SGRank object and extract keyterms."""
        super().__init__(**kwargs)
        self.keyterms = sgrank(
            doc=self.doc,
            normalize=self.normalize,
            ngrams=self.ngrams,
            include_pos=self.include_pos,
            window_size=self.window_size,
            topn=self.topn,
            idf=self.idf,
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


def sgrank(
    doc: Doc | str,
    *,
    normalize: Optional[str | Callable[[Span], str]] = "lemma",
    ngrams: int | Collection[int] = (1, 2, 3, 4, 5, 6),
    include_pos: Optional[str | Collection[str]] = ("NOUN", "PROPN", "ADJ"),
    window_size: int = 1500,
    topn: int | float = 10,
    idf: Optional[dict[str, float]] = None,
) -> list[tuple[str, float]]:
    """
    Extract key terms from a document using the SGRank algorithm. 

    This comment is just taken straight from textacy, will adjust later

    Args:
        doc: spaCy ``Doc`` from which to extract keyterms.
        normalize: If "lemma", lemmatize terms; if "lower", lowercase terms; if None,
            use the form of terms as they appeared in ``doc``; if a callable,
            must accept a ``Span`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`
        ngrams: n of which n-grams to include. For example, ``(1, 2, 3, 4, 5, 6)`` (default)
            includes all ngrams from 1 to 6; `2` if only bigrams are wanted
        include_pos: One or more POS tags with which to filter for good candidate keyterms.
            If None, include tokens of all POS tags
            (which also allows keyterm extraction from docs without POS-tagging.)
        window_size: Size of sliding window in which term co-occurrences are determined
            to occur. Note: Larger values may dramatically increase runtime, owing to
            the larger number of co-occurrence combinations that must be counted.
        topn: Number of top-ranked terms to return as keyterms.
            If int, represents the absolute number; if float, must be in the open interval
            (0.0, 1.0), and is converted to an integer by ``int(round(len(candidates) * topn))``
        idf: Mapping of ``normalize(term)`` to inverse document frequency
            for re-weighting of unigrams (n-grams with n > 1 have df assumed = 1).
            Results are typically better with idf information.

    Returns:
        Sorted list of top ``topn`` key terms and their corresponding SGRank scores

    Raises:
        ValueError: if ``topn`` is a float but not in (0.0, 1.0] or ``window_size`` < 2

    References:
        Danesh, Sumner, and Martin. "SGRank: Combining Statistical and Graphical
        Methods to Improve the State of the Art in Unsupervised Keyphrase Extraction."
        Lexical and Computational Semantics (* SEM 2015) (2015): 117.
    """
    include_pos_set, ngram_sizes, topn = _validate_sgrank_args(
        include_pos, ngrams, window_size, topn
    )

    terms = _to_term_sequence(doc)
    if not terms:
        return []
    normalized_terms = list(terms_to_strings(terms, normalize))

    # Gather every valid n-gram candidate and its position in the doc.
    candidates = _get_candidates(
        terms, normalized_terms, include_pos_set, ngram_sizes
    )
    if not candidates:
        return []

    # Score each candidate a modified measure.
    term_weights = _compute_term_weights(candidates, idf)

    # Build a position-weighted co-occurrence graph and rank candidates within it.
    graph = _build_weighted_graph(candidates, term_weights, window_size)
    if graph.number_of_nodes() == 0:
        return []
    word_scores = nx.pagerank(graph, alpha=0.85, weight="weight")

    #Score thefull candidate phrases and limit overlapping candidates.
    candidate_scores = _score_candidate_phrases(candidates, word_scores)
    topn = _resolve_topn(topn, len(candidate_scores))
    sorted_candidate_scores = sorted(
        candidate_scores.items(), key=itemgetter(1, 0), reverse=True
    )

    return get_filtered_topn_terms(sorted_candidate_scores, topn, match_threshold=0.8)


def _validate_sgrank_args(
    include_pos: Optional[str | Collection[str]],
    ngrams: int | Collection[int],
    window_size: int,
    topn: int | float,
) -> tuple[Optional[set[str]], tuple[int, ...], int | float]:
    """Validate SGRank arguments"""
    include_pos_set = to_set(include_pos) if include_pos else None

    if isinstance(ngrams, int):
        if ngrams < 1:
            raise ValueError(
                f"ngrams={ngrams} is invalid; all values must be integers >= 1"
            )
        ngram_sizes: tuple[int, ...] = (ngrams,)
    else:
        ngram_sizes = tuple(sorted(set(ngrams)))
        if not ngram_sizes or any(
            (not isinstance(n, int) or n < 1) for n in ngram_sizes
        ):
            raise ValueError(
                f"ngrams={ngrams} is invalid; must be an int >= 1 or a non-empty iterable of ints >= 1"
            )

    if window_size < 2:
        raise ValueError(f"window_size={window_size} is invalid; must be >= 2")

    if isinstance(topn, float) and not 0.0 < topn <= 1.0:
        raise ValueError(
            f"topn={topn} is invalid; must be an int, or a float between 0.0 and 1.0"
        )

    return include_pos_set, ngram_sizes, topn


def _get_candidates(
    terms: list[Token | str],
    normalized_terms: list[str],
    include_pos: Optional[set[str]],
    ngram_sizes: tuple[int, ...],
) -> list[Candidate]:
    """Extract n-gram candidates as runs, recording the position and length.

    Args:
        terms: Token/string sequence produced by `_to_term_sequence`.
        normalized_terms: Normalized strings aligned with `terms`.
        include_pos: Allowed POS tags, or None to allow all.
        ngram_sizes: Sizes of n-grams to extract as candidates.

    Returns:
        list[Candidate]: All candidate occurrences found in the document.
    """

    def _is_valid(tok: Token | str) -> bool:
        if isinstance(tok, str):
            return not (is_unicode_punctuation(tok) or tok.isspace())
        if tok.is_stop or tok.is_punct or tok.is_space:
            return False
        return include_pos is None or not tok.pos_ or tok.pos_ in include_pos

    candidates: list[Candidate] = []
    run_start = 0
    run: list[str] = []

    def _consume_run(start: int, run_terms: list[str]) -> None:
        """Extract all n-gram candidates from one run of consecutive valid tokens."""
        run_len = len(run_terms)
        for n in ngram_sizes:
            if n > run_len:
                continue
            for i in range(run_len - n + 1):
                text = " ".join(run_terms[i : i + n])
                candidates.append(Candidate(text=text, idx=start + i, length=n, count=0))

    for i, (tok, norm) in enumerate(zip(terms, normalized_terms)):
        if _is_valid(tok):
            if not run:
                run_start = i
            run.append(norm)
        else:
            if run:
                _consume_run(run_start, run)
                run = []
    if run:
        _consume_run(run_start, run)

    return candidates

#come back to this
def _compute_term_weights(
    candidates: list[Candidate],
    idf: Optional[dict[str, float]],
    ) -> dict[str, float]:

    #to be done
    return term_weights


def _build_weighted_graph(
    candidates: list[Candidate],
    term_weights: dict[str, float],
    window_size: int,
) -> nx.Graph:
    """Build a co-occurrence graph weighted by inverse positional distance and term weight.

    Args:
        candidates: All candidate occurrences from `_get_candidates`.
        term_weights: Mapping of candidate text to its tf-idf-style weight.
        window_size: Maximum token distance for two candidates to co-occur.

    Returns:
        nx.Graph: Weighted graph of candidate texts.
    """
    edge_weights: collections.defaultdict[tuple[str, str], float] = (
        collections.defaultdict(float)
    )

    # Sliding window over candidates (by token index).
    n = len(candidates)
    for i in range(n):
        c1 = candidates[i]
        for j in range(i + 1, n):
            c2 = candidates[j]
            distance = c2.idx - c1.idx
            if distance > window_size:
                break
            if distance <= 0 or c1.text == c2.text:
                continue
            weight = (term_weights[c1.text] * term_weights[c2.text]) / distance
            edge_weights[(c1.text, c2.text)] += weight

    graph = nx.Graph()
    graph.add_weighted_edges_from(
        (t1, t2, w) for (t1, t2), w in edge_weights.items()
    )
    return graph

#come back to this
def _score_candidate_phrases(
    candidates: list[Candidate],
    word_scores: dict[str, float],
) -> dict[str, float]:
    """Score each unique candidate phrase."""

    #to be done
    return candidate_scores