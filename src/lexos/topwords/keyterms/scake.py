"""scake.py.

Last Updated: June 16, 2026
Last Tested: June 2, 2026

Usage:

from lexos.topwords.keyterms.scake import SCake
keyterms = SCake(doc=doc).keyterms

or

from lexos.topwords.keyterms.scake import scake
keyterms = scake(doc=doc)
"""

import collections
import itertools
from operator import itemgetter
from typing import Callable, Collection, DefaultDict, Iterable, Literal, Optional

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

class SCake(TopWords):
    """Extracts keyterms using the sCAKE (Semantic Connectivity Aware Keyword Extraction) algorithm."""

    doc: str | Doc = Field(..., description="The raw text or spaCy doc to analyze.")
    normalize: Optional[Literal["orth", "lower", "lemma"]] = Field(
        default="lemma", description="How to normalize tokens for candidate selection.",
    )

    include_pos: Optional[str | Collection[str]] = Field(
        default=("NOUN", "PROPN", "ADJ"), description="POS tags to include for candidate selection.",
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
        """Initialize the SCake object and extract keyterms."""

        super().__init__(**kwargs)
        # Extract keyterms and put them in field
        self.keyterms = scake(
            doc=self.doc,
            normalize=self.normalize,
            include_pos=self.include_pos,
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


def scake(
    doc: Doc | str,
    *,
    normalize: Literal["orth", "lower", "lemma"] | None = "lemma",
    include_pos: Optional[str | Collection[str]] = ("NOUN", "PROPN", "ADJ"),
    topn: int | float = 10,
) -> list[tuple[str, float]]:
    """Extract key terms from a document using the sCAKE algorithm.

    Args:
        doc (Doc | str): spaCy `Doc` or plain string from which to extract keyterms.
            If a `Doc`, it should be sentence-segmented for best results.
        normalize (Literal["orth", "lower", "lemma"] | None): If "lemma", lemmatize
            terms; if "lower", lowercase terms; if None, use the orthographic forms
            that appear in `doc`; if a callable, must accept a `Token` and return a str.
        include_pos (str | Collection[str] | None): One or more POS tags with which
            to filter for good candidate keyterms. If `None`, include tokens of all POS
            tags (which also allows keyterm extraction from docs without POS-tagging.)
        topn (int | float): Number of top-ranked terms to return as key terms.
            If an integer, represents the absolute number; if a float, value
            must be in the interval (0.0, 1.0], which is converted to an int by
            `int(round(len(candidates) * topn))`.

    Returns:
        list[tuple[str, float]]: Sorted list of top `topn` key terms and their
            corresponding sCAKE scores.
    Notes:
        1. normalize the arguemnts
        2. then make the co-occurence matrix
        3. makes a graph based on the matrix
        4. compute the scores based on the graph
        5. get the key phrases and return topn
    """
    include_pos_set, topn = _validate_scake_args(include_pos, topn)

    # Convert str input to term sequence 
    terms = _to_term_sequence(doc)
    if not terms:
        return []

    # Build normalized str sequence aligned with terms
    normalized_terms = list(terms_to_strings(terms, normalize))

    # Build co-occurrence matrix over sentence segments(or the whole doc if plain-string input, which has no sentence boundaries)
    cooc_mat = _build_cooc_matrix(doc, terms, normalized_terms, include_pos_set)
    if not cooc_mat:
        return []

    # Build the word graph from the co-occurrence matrix
    graph = nx.Graph()
    graph.add_edges_from(
        (w1, w2, {"weight": weight}) for (w1, w2), weight in cooc_mat.items()
    )

    # Compute the scores
    word_scores = _compute_word_scores(terms, normalized_terms, graph, cooc_mat)
    if not word_scores:
        return []

    # Get candidate phrases and resolve topn
    candidates = _get_candidates(terms, normalized_terms, include_pos_set)
    topn = _resolve_topn(topn, len(candidates))

    candidate_scores = {}
    for candidate in candidates:
        phrase = " ".join(candidate)
        score = sum(word_scores.get(word, 0.0) for word in candidate)
        candidate_scores[phrase] = score
        sorted_candidate_scores = sorted(candidate_scores.items(), key=itemgetter(1, 0), reverse=True)

    return get_filtered_topn_terms(sorted_candidate_scores, topn, match_threshold=0.8)


def _validate_scake_args(
    include_pos: Optional[str | Collection[str]],
    topn: int | float,
) -> tuple[Optional[set[str]], int | float]:
    """validate and transform sCAKE arguments.

    Args:
        include_pos: POS tag filter passed by the user.
        topn: Top-N count or ratio passed by the user.

    Returns:
        tuple: validated (include_pos_set, topn).
    """
    include_pos_set = to_set(include_pos) if include_pos else None

    if isinstance(topn, float) and not 0.0 < topn <= 1.0:
        raise ValueError(
            f"topn={topn} is invalid; must be an int, or a float between 0.0 and 1.0"
        )
    
    return include_pos_set, topn


def _build_cooc_matrix(
    doc: Doc | str,
    terms: list[Token | str],
    normalized_terms: list[str],
    include_pos: Optional[set[str]],
) -> collections.Counter[tuple[str, str]]:
    """Builds a co-occurrence counter over sentence windows.

    For spaCy docs, inputs the algorithm over adjacent sentence pairs
    For plain-string inputs/ normal docs(which have no sentence boundaries) the entire 
    token sequence is treated as a single window so that the algorithm still 
    produces meaningful co-occurrences

    Args:
        doc: Original document (used only to retrieve sentence spans when it is
            a Doc).
        terms: Token/string sequence produced by `_to_term_sequence`.
        normalized_terms: Normalized strings aligned with `terms`.
        include_pos: Allowed POS tags, or None to allow all.

    Returns:
        collections.Counter mapping sorted word-pair tuples to co-occurrence counts
    """
    cooc_mat: collections.Counter[tuple[str, str]] = collections.Counter()

    if isinstance(doc, Doc):
        # Builds a mapping from token index to normalized string so we can look
        # Up the correct form for each sentence span
        idx_to_norm: dict[int, str] = {
            tok.i: norm for tok, norm in zip(terms, normalized_terms)
        }

        sentences = list(doc.sents)
        n_sents = len(sentences)

        for i in range(n_sents):
            # Window covers the current sentence and the next one if it exists
            window = list(sentences[i])
            if i + 1 < n_sents:
                window += list(sentences[i + 1])

            window_words = [
                idx_to_norm[tok.i]
                for tok in window
                if _is_valid_tok_doc(tok, include_pos)
            ]
            cooc_mat.update(
                w1_w2
                for w1_w2 in itertools.combinations(sorted(window_words), 2)
                if w1_w2[0] != w1_w2[1]
            )
    else:
        # Plain-string/normal doc path: one global window over all valid tokens
        window_words = [
            norm
            for tok, norm in zip(terms, normalized_terms)
            if _is_valid_tok_str(tok)
        ]
        cooc_mat.update(
            w1_w2
            for w1_w2 in itertools.combinations(sorted(window_words), 2)
            if w1_w2[0] != w1_w2[1]
        )

    return cooc_mat


def _is_valid_tok_doc(tok: Token, include_pos: Optional[set[str]]) -> bool:
    """Return True if a spaCy Token should be included in co-occurrence windows.

    POS filtering is skipped when the pipeline has no tagger (tok.pos_ == ""),
    so this fixes that bug

    Args:
        tok: spaCy Token to test.
        include_pos: Allowed POS tags, or None to allow all.

    Returns:
        bool
    """
    if tok.is_stop or tok.is_punct or tok.is_space:
        return False
    pos_ok = include_pos is None or not tok.pos_ or tok.pos_ in include_pos

    return pos_ok


def _is_valid_tok_str(tok: str) -> bool:
    """Return True if a plain-string token should be included in co-occurrence windows.

    Args:
        tok: String token to test.

    Returns:
        bool
    """
    return not (is_unicode_punctuation(tok) or tok.isspace())


def _compute_word_scores( 
    terms: list[Token | str],
    normalized_terms: list[str],
    graph: nx.Graph,
    cooc_mat: collections.Counter[tuple[str, str]],
) -> dict[str, float]:
    """Computes per-word sCAKE scores based on
        - truss level
        - semantic strength
        - semantic connectivity
        - positional weight

    Args:
        terms: Token/string sequence produced by `_to_term_sequence`.
        normalized_terms: Normalized strings aligned with `terms`.
        graph: Word co-occurrence graph.
        cooc_mat: Co-occurrence counter used to build `graph`.

    Returns:
        Dict mapping normalized word strings to their sCAKE scores
    """
    word_strs: list[str] = list(graph.nodes())

    # K-truss decomposition - - - - - - - - - - - - - - - - - - - -
    max_truss_levels = _compute_node_truss_levels(graph)
    max_truss_level = max(max_truss_levels.values(), default=0)
    # If all words are isolated nodes, the scores would be zero or undefined
    if not max_truss_level:
        return {}

    # Semantic strength component - - - - - - - - - - - - - - - - -
    sem_strengths: dict[str, int] = {}
    for w in word_strs:
        total = 0
        for nbr in graph.neighbors(w):
            pair = tuple(sorted([w, nbr]))
            total += cooc_mat[pair] * max_truss_levels[nbr]
        sem_strengths[w] = total

    # Semantic connectivity component - - - - - - - - - - - - - -
    sem_connectivities: dict[str, float] = {}
    for w in word_strs:
        neighbour_levels = set()
        for nbr in graph.neighbors(w):
            neighbour_levels.add(max_truss_levels[nbr])
        sem_connectivities[w] = len(neighbour_levels) / max_truss_level

    # Positional weight component - - - - - - - - - - - - - - - - -
    # For spaCy Token sequences we use the index (tok.i)
    # Plain strings just use sequential position.
    word_pos: DefaultDict[str, float] = collections.defaultdict(float)
    for idx, (tok, norm) in enumerate(zip(terms, normalized_terms)):
        pos = tok.i if isinstance(tok, Token) else idx
        word_pos[norm] += 1.0 / (pos + 1)

    # Score all the components together - - - - - - - - - - - - - - -
    word_scores: dict[str, float] = {}
    for w in word_strs:
        word_scores[w] = (
            word_pos[w]
            * max_truss_levels[w]
            * sem_strengths[w]
            * sem_connectivities[w]
        )
    
    return word_scores

def _get_candidates(
    terms: list[Token | str],
    normalized_terms: list[str],
    include_pos: Optional[set[str]],
) -> set[tuple[str, ...]]:
    """Extract candidate keyphrases as the longest valid subsequences of tokens.

    A valid token is non-stopword, non-punctuation, and non-space. When
    `include_pos` is supplied and the token carries POS information, only then
    do tokens whose POS tag is in `include_pos` are they considered valid.

    Args:
        terms: Token/string sequence produced by `_to_term_sequence`.
        normalized_terms: Normalized strings aligned with `terms`.
        include_pos: Allowed POS tags, or None to allow all.

    Returns:
        Set of candidate term tuples (each tuple is a sequence of normalized strings)
    """

    def _is_valid(tok: Token | str) -> bool:
        if isinstance(tok, str):
            return not (is_unicode_punctuation(tok) or tok.isspace())
        is_stop = tok.is_stop
        if tok.is_punct or tok.is_space or is_stop:
            return False
        pos_ok = include_pos is None or not tok.pos_ or tok.pos_ in include_pos
        return pos_ok

    candidates: set[tuple[str, ...]] = set()
    run: list[str] = []
    for tok, norm in zip(terms, normalized_terms):
        if _is_valid(tok):
            run.append(norm)
        else:
            if run:
                candidates.add(tuple(run))
                run = []
    if run:
        candidates.add(tuple(run))
    return candidates

def _compute_node_truss_levels(graph: nx.Graph) -> dict[str, int]:
    """Computes the maximum k-truss level for each node in the graph

    The k-truss level of a node is the maximum k such that the node belongs
    to a k-truss subgraph (every edge is in at least k-1 triangles(its part of a set))

    Args:
        graph: Undirected word co-occurrence graph.

    Returns:
        Dict mapping each node name to its maximum k-truss level
    """
    max_edge_ks: dict[tuple, int] = {}
    is_removed: collections.defaultdict[tuple, int] = collections.defaultdict(int)

    triangle_counts: dict[tuple, int] = {}
    for edge in graph.edges():
        shared_neighbours = set(graph.neighbors(edge[0])) & set(graph.neighbors(edge[1]))
        triangle_counts[edge] = len(shared_neighbours)

    k = 1
    while True:
        to_remove = collections.deque(
            edge
            for edge, tcount in triangle_counts.items()
            if tcount < k and not is_removed[edge]
        )

        while to_remove:
            edge = to_remove.popleft()
            is_removed[edge] = 1
            for nbr in set(graph.neighbors(edge[0])) & set(graph.neighbors(edge[1])):
                for node in edge:
                    nbr_edge = (node, nbr)
                    try:
                        triangle_counts[nbr_edge] -= 1
                    except KeyError:
                        nbr_edge = (nbr, node)
                        triangle_counts[nbr_edge] -= 1
                    if triangle_counts[nbr_edge] == k - 1:
                        to_remove.append(nbr_edge)
                        is_removed[nbr_edge] = 1
            max_edge_ks[edge] = k - 1

        if len(is_removed) == len(triangle_counts):
            break
        k += 1

    # A nodes truss level is the maximum over the k values of its edges
    max_node_ks: dict[str, int] = {}
    for node in graph.nodes():
        node_edge_ks = []
        for edge, k in max_edge_ks.items():
            if node in edge:
                node_edge_ks.append(k)
        max_node_ks[node] = max(node_edge_ks, default=0)

    return max_node_ks
