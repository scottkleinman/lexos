"""structural_stylometry.py.

Scripts for performing stylometric analysis on structural features like punctuation and whitespace.

Last Update: July 22, 2026
Last Tested: July 22, 2026
"""

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from pydantic import BaseModel, ConfigDict, Field
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from spacy.tokens import Doc

from lexos.corpus import Corpus
from lexos.io.loader import Loader


class StructuralAnalyzer(BaseModel):
    """Builds a stylometric matrix from a corpus of documents, with options for punctuation and whitespace tracking."""

    corpus: dict[str, str | Doc] | Corpus | Loader = Field(
        ...,
        description="Dictionary of {doc_id: str | Doc} representing the corpus or a Lexos `Corpus` or `Loader` object.",
    )
    model: Optional[str] = Field(
        default="xx_sent_ud_sm",
        description="spaCy language model to use for processing text.",
    )
    max_features: Optional[int] = Field(
        100, description="Maximum number of features to track in the vocabulary"
    )
    include_whitespace: Optional[bool] = Field(
        True, description="Whether to include whitespace markers as features"
    )
    feature_mode: Optional[str] = Field(
        "all", description='"all", "punctuation_only", or "structural_only"'
    )
    min_punctuation_threshold: Optional[int] = Field(
        20, description="Minimum total punctuation marks required per doc"
    )
    action_on_low_count: Optional[str] = Field(
        "warn",
        description='"warn" to log console warnings, "drop" to filter out doc completely',
    )

    # Assigned after initialisation
    include_whitespace: bool = Field(
        True, description="Whether to include whitespace markers as features"
    )
    feature_mode: str = Field(
        "all", description='"all", "punctuation_only", or "structural_only"'
    )
    min_punctuation_threshold: int = Field(
        20, description="Minimum total punctuation marks required per doc"
    )
    tokenized_corpus: dict[str, list[str]] = Field(
        default_factory=dict, description="Dictionary of {doc_id: list_of_tokens}"
    )
    doc_ids: list[str] = Field(
        default_factory=list, description="List of document IDs that passed filtering"
    )
    vocabulary: list[str] = Field(
        default_factory=list,
        description="List of tracked features (punctuation/whitespace tokens)",
    )
    vocab_idx: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of token to column index in the matrix",
    )
    raw_matrix: np.ndarray = Field(
        default_factory=lambda: np.array([]),
        description="Raw frequency matrix of shape (num_docs, num_features)",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Constructor for stylometric matrix builder."""
        super().__init__(**data)

        # Convert Lexos Corpus and Loader objects
        if isinstance(self.corpus, Loader):
            self.corpus = {
                name: text for name, text in zip(self.corpus.names, self.corpus.texts)
            }
        if isinstance(self.corpus, Corpus):
            self.corpus = {
                record.name: record.content for record in self.corpus.records.values()
            }

        # Process texts, count total punctuation, and apply constraint flags
        for d_id, text in self.corpus.items():
            # Check punctuation density before final filtering
            total_punct = self._count_total_punctuation(text)

            if total_punct < self.min_punctuation_threshold:
                warning_msg = (
                    f"⚠️ DOCUMENT CONSTRAINT ALERT: '{d_id}' only contains {total_punct} "
                    f"punctuation tokens (Minimum threshold: {self.min_punctuation_threshold}). "
                    f"Statistical reliability will be degraded."
                )

                if self.action_on_low_count == "drop":
                    warnings.warn(f"{warning_msg} -> DROPPING document from corpus.")
                    continue  # Skip adding this document to the active indices
                else:
                    warnings.warn(f"{warning_msg} -> RETAINING document anyway.")

            # Store validated/retained tokens
            self.doc_ids.append(d_id)
            self.tokenized_corpus[d_id] = self._tokenize_structural(text)

        if not self.doc_ids:
            raise ValueError(
                "All documents were dropped! Lower your min_punctuation_threshold or check inputs."
            )

        # Build vocabulary mapping from retained documents
        global_counts = Counter()
        for tokens in self.tokenized_corpus.values():
            global_counts.update(tokens)

        self.vocabulary = [
            token for token, _ in global_counts.most_common(self.max_features)
        ]
        self.vocab_idx = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.raw_matrix = self._build_raw_matrix()

    def _count_total_punctuation(self, text: str) -> int:
        """Quick internal count of total punctuation characters using spaCy token rules.

        Args:
            text (str): Raw text to analyze

        Returns:
            int: Total number of punctuation characters in the text
        """
        if isinstance(text, Doc):
            doc = text
        else:
            nlp = spacy.load(self.model)
            doc = nlp(text)
        return sum(1 for token in doc if token.pos_ == "PUNCT" or token.is_punct)

    def _tokenize_structural(self, text: str | Doc, lowercase: bool = True) -> list:
        """Extracts and filters tokens based on configuration rules.

        Args:
            text (str): Raw text to tokenize
            lowercase (bool): Whether to convert tokens to lowercase

        Returns:
            list: List of tokens, including punctuation and optional whitespace markers
        """
        tokens = []
        if isinstance(text, Doc):
            doc = text
        else:
            if self.include_whitespace:
                tokens.extend(["[WS_DOUBLE_SPACE]"] * len(re.findall(r" {2,}", text)))
                tokens.extend(
                    ["[WS_MULTIPLE_NEWLINE]"] * len(re.findall(r"\n{2,}", text))
                )
                tokens.extend(["[WS_TRAILING_SPACE]"] * len(re.findall(r" \n", text)))
            nlp = spacy.load(self.model)
            doc = nlp(text)
        for token in doc:
            if token.is_space:
                continue
            if token.pos_ == "PUNCT" or token.is_punct:
                tokens.append(token.text)
            else:
                tokens.append(token.text.lower() if lowercase else token.text)

        if self.feature_mode == "punctuation_only":
            return [t for t in tokens if not t.startswith("[WS_") and not t.isalnum()]
        elif self.feature_mode == "structural_only":
            return [t for t in tokens if t.startswith("[WS_") or not t.isalnum()]
        return tokens

    def _build_raw_matrix(self):
        """Constructs the raw frequency matrix for the corpus."""
        matrix = np.zeros((len(self.doc_ids), len(self.vocabulary)))
        for r_idx, d_id in enumerate(self.doc_ids):
            tokens = self.tokenized_corpus[d_id]
            doc_len = len(tokens) if len(tokens) > 0 else 1
            counts = Counter(tokens)
            for token, count in counts.items():
                if token in self.vocab_idx:
                    matrix[r_idx, self.vocab_idx[token]] = count / doc_len
        return matrix

    def get_distance_matrix(
        self, method: str = "classic", as_df: bool = False
    ) -> np.ndarray | pd.DataFrame:
        """Computes the Burrows' Delta distance matrix.

        Use classic pairwise Burrows' Delta distance if you are benchmarking your results against older stylometry frameworks, or if your dataset relies on a mixture of words and punctuation where you want every feature to contribute linearly.

        Use the Argamon quadratic variant (applies a Euclidean metric to the Z-scored feature values) if you are utilizing the `structural_only` or `punctuation_only` feature modes. Because punctuation marks often show dramatic outliers (e.g., one text using ten times as many semicolons as the corpus average), squaring the differences prevents these massive single-feature spikes from distorting your overall unsupervised clusters.

        Args:
            method (str): "classic" or "quadratic" for the distance calculation method.
            as_df (bool): Whether to return the result as a pandas DataFrame with labeled rows and columns.

        Returns:
            np.ndarray | pd.DataFrame: A square NumPy array or pandas DataFrame where matrix[i, j] is the stylistic distance.
        """
        # Grab the computed Burrows' Z-scores from our existing method
        z_scores = self.get_feature_matrix(method="burrows_z")
        num_docs = len(self.doc_ids)

        # Compute the classic Burrows' Delta (Manhattan distance on Z-scores)
        if method == "classic":
            # Initialize a square distance matrix
            delta_matrix = np.zeros((num_docs, num_docs))

            # Calculate pairwise average absolute difference (Manhattan Distance)
            for i in range(num_docs):
                for j in range(num_docs):
                    delta_matrix[i, j] = np.mean(np.abs(z_scores[i] - z_scores[j]))
        elif method == "quadratic":
            # Calculate pairwise Argamon Quadratic distance (Euclidean on Z-scores)
            delta_matrix = np.zeros((num_docs, num_docs))
            for i in range(num_docs):
                for j in range(num_docs):
                    # Square root of the sum of squared Z-score differences
                    squared_diff = (z_scores[i] - z_scores[j]) ** 2
                    delta_matrix[i, j] = np.sqrt(np.sum(squared_diff))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'classic' or 'quadratic'.")
        # Convert the result to a DataFrame if requested
        if as_df:
            import pandas as pd

            return pd.DataFrame(delta_matrix, index=self.doc_ids, columns=self.doc_ids)
        return delta_matrix

    def get_feature_matrix(self, method: str = "tfidf"):
        """Returns the stylometric matrix in the specified representation.

        Args:
            method (str): "raw", "tfidf", or "burrows_z"
        """
        if method == "raw":
            return self.raw_matrix
        elif method == "tfidf":
            N = len(self.doc_ids)
            df = np.sum(self.raw_matrix > 0, axis=0)
            idf = np.log((1 + N) / (1 + df)) + 1
            return self.raw_matrix * idf
        elif method == "burrows_z":
            means = np.mean(self.raw_matrix, axis=0)
            stds = np.std(self.raw_matrix, axis=0)
            stds[stds == 0] = 1e-6
            return (self.raw_matrix - means) / stds
        else:
            raise ValueError("Unknown transformation method.")

    def to_csv(
        self,
        filepath: str,
        method: str,
    ):
        """Exports a stylometric representation matrix to CSV.

        Args:
            filepath (str): Local disk save path destination string.
            method (str): "raw", "tfidf", or "burrows_z"
        """
        df = self.to_df(method=method)
        filepath = Path(filepath)
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False, encoding="utf-8")

    def to_df(
        self,
        method: str,
    ) -> pd.DataFrame:
        """Converts a stylometric representation matrix into a labeled DataFrame for export.

        Args:
        - method (str): "raw", "tfidf", or "burrows_z"

        Returns:
        - pd.DataFrame: A DataFrame with document IDs as the first column and vocabulary features as subsequent columns.
        """
        # Get the representation matrix based on the specified method
        matrix = self.get_feature_matrix(method=method)

        # Build explicit data frame with structural string identifiers
        df = pd.DataFrame(data=matrix, index=self.doc_ids, columns=self.vocabulary)

        # Reset index to make Document IDs an explicit named data column in row zero
        df.index.name = "Document_ID"
        df = df.reset_index()

        return df

    def visualize(
        self,
        method: str = "tfidf",
        top_n: int = 5,
        show_plots: bool = True,
        show_loadings: bool = True,
    ):
        """Generates Dendrogram and PCA plots.

        Also prints the specific structural features driving the variance on PC1 and PC2.

        Args:
            method (str): "raw", "tfidf", or "burrows_z" for the stylometric representation.
            top_n (int): Number of top features to display for each principal component.
            show_plots (bool): Whether to display the plots. If False, only prints the loadings.
            show_loadings (bool): Whether to print the top feature loadings for PC1 and PC2.
        """
        matrix = self.get_feature_matrix(method=method)
        labels = self.doc_ids
        method_name = method.replace("_", " ").upper()

        # Calculate PCA projections
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(matrix)

        if show_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # ---------------- DENDROGRAM ----------------
            metric_choice = "cityblock" if "BURROWS" in method_name else "euclidean"
            if metric_choice == "euclidean":
                linked = linkage(matrix, method="ward", metric="euclidean")
            else:
                linked = linkage(matrix, method="average", metric="cityblock")

            dendrogram(
                linked,
                orientation="top",
                labels=labels,
                ax=ax1,
                distance_sort="descending",
            )
            ax1.set_title(f"Hierarchical Dendrogram ({method_name})")
            ax1.set_ylabel("Stylistic Distance Index")
            ax1.tick_params(axis="x", rotation=45)

            # ---------------- PCA PLOT ----------------
            ax2.scatter(
                pca_results[:, 0],
                pca_results[:, 1],
                color="darkblue",
                s=100,
                edgecolors="black",
                alpha=0.7,
            )

            for i, label in enumerate(labels):
                ax2.annotate(
                    label,
                    (pca_results[i, 0], pca_results[i, 1]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )

            ax2.set_title(
                f"PCA Projections (Variance: {np.sum(pca.explained_variance_ratio_) * 100:.1f}%)"
            )
            ax2.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
            ax2.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
            ax2.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.show()

        # ---------------- DISPLAY PCA LOADINGS ----------------
        if show_loadings:
            # Create a dict of sorted features by loading value for each component
            loadings = self.get_loadings(method=method, pca=pca)

            print("\n" + "=" * 50)
            print(f" TOP FEATURE LOADINGS ({method_name}) ")
            print("=" * 50)

            for component_name, component in loadings.items():
                print(f"\nDriving features for {component_name}")
                print("-" * 50)

                print(f"\n**Top Positive Features (Pushes RIGHT/TOP)**")
                print(
                    component.head(top_n).to_markdown(
                        floatfmt=".4f", headers=["Feature", "Weight"], tablefmt="pretty"
                    )
                )

                print(f"\n**Top Negative Features (Pushes LEFT/BOTTOM)**")
                print(
                    component.tail(top_n).to_markdown(
                        floatfmt=".4f", headers=["Feature", "Weight"], tablefmt="pretty"
                    )
                )

    def get_loadings(
        self, method: str = "tfidf", pca: PCA | None = None
    ) -> dict[str, pd.DataFrame]:
        """Returns a dictionary of PCA loadings for each principal component.

        Args:
            method (str): "raw", "tfidf", or "burrows_z" for the stylometric representation.
            pca (PCA | None): An optional pre-fitted PCA object. If None, a new PCA will be fitted.

        Returns:
            dict: A dictionary where keys are component names (e.g., 'PC1', 'PC2') and values are DataFrames of features and their corresponding loadings.
        """
        matrix = self.get_feature_matrix(method=method)

        # Calculate PCA projections
        if pca is None:
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(matrix)

        # Create a clean DataFrame of the loadings mapping feature names to component weights
        loadings_df = pd.DataFrame(
            pca.components_.T, columns=["PC1", "PC2"], index=self.vocabulary
        )

        # Create a dict of sorted features by loading value for each component
        loadings = {
            "PC1": loadings_df["PC1"].sort_values(ascending=False).to_frame(),
            "PC2": loadings_df["PC2"].sort_values(ascending=False).to_frame(),
        }

        return loadings
