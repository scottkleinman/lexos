"""seetrees.py.

This module is adapted from the R code from the 'see' package by Artjoms Šeļa (https://github.com/perechen/seetrees).

Last Updated: August 16, 2026
Last Tested: August 16, 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, ConfigDict, Field
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from lexos.dtm import DTM


class SeeTrees(BaseModel):
    """SeeTrees class for stylometric analysis and visualization."""

    distance_table: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Optional distance matrix."
    )
    dtm: DTM | None = Field(
        default=None, description="Optional Lexos DTM to initialize frequencies from."
    )
    features: list[str] = Field(
        default_factory=list, description="Optional list of feature names."
    )
    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Optional frequency table."
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Document labels derived from the distance matrix or frequency table.",
    )
    stylo_res: dict | None = Field(
        default=None,
        description="Optional dictionary containing stylo output keys such as `frequencies`, `distance_table`, and `features`.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data):
        """Initialize a SeeTrees instance."""
        super().__init__(**data)
        if self.dtm is not None:
            self._init_from_dtm()
        elif self.stylo_res is not None:
            self._init_from_stylo_res()
        else:
            self._init_from_raw()

        self._init_labels()

    def _init_from_dtm(self):
        """Initialize frequencies from a Lexos DTM."""
        self.frequencies = self.dtm.to_df(transpose=True)
        self.distance_table = (
            pd.DataFrame(self.distance_table)
            if self.distance_table is not None
            else pd.DataFrame()
        )
        self.features = list(self.features) if self.features is not None else []

    def _init_from_stylo_res(self):
        """Initialize frequencies, distance table, and features from a stylo result dictionary."""
        self.frequencies = pd.DataFrame(self.stylo_res.get("frequencies", {}))
        self.distance_table = pd.DataFrame(self.stylo_res.get("distance_table", {}))
        self.features = list(self.stylo_res.get("features", []))

    def _init_from_raw(self):
        self.frequencies = (
            pd.DataFrame(self.frequencies)
            if self.frequencies is not None
            else pd.DataFrame()
        )
        self.distance_table = (
            pd.DataFrame(self.distance_table)
            if self.distance_table is not None
            else pd.DataFrame()
        )
        self.features = list(self.features) if self.features is not None else []

    def _init_labels(self):
        if not self.distance_table.empty:
            self.labels = list(self.distance_table.index)
        else:
            self.labels = (
                list(self.frequencies.index) if not self.frequencies.empty else []
            )

    def compare_scores(
        self,
        source_text: str,
        target_text: str,
        top_diff: int = 10,
        view_type: str = "profile",
    ):
        """Compares two documents based on feature usage profiles or direct differences.

        Args:
            source_text (str): The reference text to compare against.
            target_text (str): The text to compare with the source.
            top_diff (int): Number of top differing features to display.
            view_type (str): Type of comparison view. Options are:
                - "profile": Overlay of z-score profiles for both texts.
                - "difference": Bar chart of z-score differences (target minus source).

        Raises:
            ValueError: If the frequency table is empty or if either text is not found in the corpus.
        """
        if self.frequencies.empty:
            raise ValueError("Frequency data is required for compare_scores.")
        if source_text not in self.frequencies.index:
            raise ValueError(f"Source text '{source_text}' not found in corpus.")
        if target_text not in self.frequencies.index:
            raise ValueError(f"Target text '{target_text}' not found in corpus.")

        z_scores = (self.frequencies - self.frequencies.mean()) / self.frequencies.std()
        z_scores = z_scores.fillna(0)

        src_profile = z_scores.loc[source_text]
        tgt_profile = z_scores.loc[target_text]
        diff_profile = tgt_profile - src_profile

        # Isolate largest absolute differences
        top_features = (
            diff_profile.abs().sort_values(ascending=False).head(top_diff).index
        )

        df_comp = (
            pd.DataFrame(
                {
                    "Source": src_profile.loc[top_features],
                    "Target": tgt_profile.loc[top_features],
                    "Difference": diff_profile.loc[top_features],
                }
            )
            .reset_index()
            .rename(columns={"index": "Feature"})
        )

        plt.figure(figsize=(10, 6))
        if view_type == "profile":
            # Profile overlay view
            x = np.arange(len(df_comp))
            width = 0.35
            plt.barh(
                x - width / 2,
                df_comp["Source"],
                width,
                label=source_text,
                color="#bae1ff",
                edgecolor="grey",
            )
            plt.barh(
                x + width / 2,
                df_comp["Target"],
                width,
                label=target_text,
                color="#ffb3ba",
                edgecolor="grey",
            )
            plt.yticks(x, df_comp["Feature"])
            plt.title(f"Z-Score Profile Contrast: {source_text} vs {target_text}")
        else:
            # Profile of differences view
            colors = np.where(df_comp["Difference"] > 0, "#ffb3ba", "#bae1ff")
            plt.barh(
                df_comp["Feature"],
                df_comp["Difference"],
                color=colors,
                edgecolor="grey",
            )
            plt.title(f"Z-Score Differences ({target_text} minus {source_text})")

        plt.axvline(0, color="black", linestyle="-", alpha=0.7)
        plt.gca().invert_yaxis()
        plt.xlabel("Z-Score Scale")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compute_distances(self, metric: str = "delta") -> pd.DataFrame:
        """Compute a stylometric distance matrix from frequency data.

        Supports multiple stylometric metrics including Burrows' Delta,
        Eder's Delta, and cosine variants.

        Args:
            metric (str): Distance metric to compute. Valid values are:
                `'delta'`, `'eder_delta'`, `'cosine_delta'`, `'manhattan'`,
                and `'cosine'`.

        Returns:
            pd.DataFrame: Pairwise distance matrix indexed by the original labels.

        Raises:
            ValueError: If the frequency table is empty or the metric is unknown.
        """
        if self.frequencies.empty:
            raise ValueError("Frequency table is required to compute distance metrics.")

        # Features must be ordered from most frequent to least frequent for Eder's Delta
        # Calculate mean frequency across corpus to determine exact rank
        mean_freqs = self.frequencies.mean().sort_values(ascending=False)
        ordered_freqs = self.frequencies[mean_freqs.index]

        # Calculate standard Z-scores
        z_scores = (ordered_freqs - ordered_freqs.mean()) / ordered_freqs.std()
        z_scores = z_scores.fillna(0)
        n_features = z_scores.shape[1]

        if metric.lower() == "delta":
            distances = pdist(z_scores.to_numpy(), metric="cityblock") / n_features

        elif metric.lower() == "eder_delta":
            # 1-based ranks for the features
            ranks = np.arange(1, n_features + 1)
            # Apply Eder's descending linear weight formula
            eder_weights = -(ranks / n_features) + 1 + (1 / n_features)

            # Multiply scaled Z-scores by the weights
            weighted_z = z_scores.to_numpy() * eder_weights
            distances = pdist(weighted_z, metric="cityblock")

        elif metric.lower() == "cosine_delta":
            distances = pdist(z_scores.to_numpy(), metric="cosine")
        elif metric.lower() == "manhattan":
            distances = pdist(ordered_freqs.to_numpy(), metric="cityblock")
        elif metric.lower() == "cosine":
            distances = pdist(ordered_freqs.to_numpy(), metric="cosine")
        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from: 'delta', 'eder_delta', 'cosine_delta', 'manhattan', 'cosine'."
            )

        # Update the module's distance table matrix
        self.distance_table = pd.DataFrame(
            squareform(distances),
            index=ordered_freqs.index,
            columns=ordered_freqs.index,
        )
        self.labels = list(self.distance_table.index)
        return self.distance_table

    def view_distances(
        self, method: str = "MDS", metric: str | None = None, random_state: int = 42
    ):
        """Visualize document relationships in 2D space.

        Args:
            method (str): Projection method to use. Valid values are `'MDS'` or
                `'PCA'`.
            metric (str | None): Optional metric name to compute a distance
                matrix before projecting. If provided, `compute_distances` is
                called with this metric.
            random_state (int): Random seed for reproducible projection results.

        Raises:
            ValueError: If required data is missing for the selected method.
        """
        if metric is not None:
            self.compute_distances(metric=metric)

        plt.figure(figsize=(10, 8))

        if method.upper() == "MDS":
            if self.distance_table.empty:
                raise ValueError(
                    "Distance table is missing. Provide a matrix or pass a 'metric' parameter."
                )

            mds = MDS(
                n_components=2, dissimilarity="precomputed", random_state=random_state
            )
            coords = mds.fit_transform(self.distance_table.to_numpy())
            title_suffix = f"MDS (Distance Matrix via '{metric or 'precomputed'}')"

        elif method.upper() == "PCA":
            if self.frequencies.empty:
                raise ValueError("Frequency metrics are required to calculate PCA.")
            z_scores = (
                self.frequencies - self.frequencies.mean()
            ) / self.frequencies.std()
            pca = PCA(n_components=2, random_state=random_state)
            coords = pca.fit_transform(z_scores.fillna(0).to_numpy())
            title_suffix = "PCA (Z-scored Profiles)"
        else:
            raise ValueError("Method must be either 'MDS' or 'PCA'.")

        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            color="#e7a",
            edgecolors="black",
            s=100,
            alpha=0.8,
        )

        for i, label in enumerate(self.labels):
            plt.text(
                coords[i, 0] + (coords[:, 0].max() * 0.015),
                coords[i, 1] + (coords[:, 1].max() * 0.015),
                label,
                fontsize=9,
                alpha=0.8,
            )

        plt.title(f"Stylometric Distribution via {title_suffix}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def view_scores(self, target_text: str, top: int = 20):
        """Visualizes preferred and avoided features for a target text using z-scores.

        Args:
            target_text (str): The text to analyze for distinctive features.
            top (int): Number of top features to display based on absolute z-score.

        Raises:
            ValueError: If the frequency table is empty or if the target text is not found in the corpus.
        """
        if self.frequencies.empty:
            raise ValueError("Frequency data is required for view_scores.")
        if target_text not in self.frequencies.index:
            raise ValueError(f"Target text '{target_text}' not found in corpus.")

        # Z-score scale the frequencies across the corpus matching latest alignment
        z_scores = (self.frequencies - self.frequencies.mean()) / self.frequencies.std()
        z_scores = z_scores.fillna(0)

        text_profile = z_scores.loc[target_text]

        # Find features deviating the most in both directions
        sorted_profile = text_profile.reindex(
            text_profile.abs().sort_values(ascending=False).index
        )
        top_deviations = sorted_profile.head(top)

        # Format for visualization
        df_plot = (
            pd.DataFrame({"Z-Score": top_deviations})
            .reset_index()
            .rename(columns={"index": "Feature"})
        )
        colors = np.where(
            df_plot["Z-Score"] > 0, "#ffb3ba", "#bae1ff"
        )  # Pink for preferred, Blue for avoided

        plt.figure(figsize=(10, 6))
        plt.barh(df_plot["Feature"], df_plot["Z-Score"], color=colors, edgecolor="grey")

        # Reference lines for standard deviations
        plt.axvline(0, color="black", linestyle="-")
        for sd in [-2, -1, 1, 2]:
            plt.axvline(sd, color="gray", linestyle="--", alpha=0.6)

        plt.gca().invert_yaxis()
        plt.title(f"Most Distinctive Features in: {target_text}")
        plt.xlabel("Corpus-wide Deviation (Z-Score)")
        plt.tight_layout()
        plt.show()

    def view_tree(self, k: int = 2, right_margin: int = 12):
        """Render a dendrogram and compute top eta-squared features.

        Args:
            k (int): Number of clusters to cut the dendrogram into.
            right_margin (int): Margin for the right side of the dendrogram.

        Raises:
            ValueError: If the distance table has not been computed.
        """
        if self.distance_table.empty:
            raise ValueError(
                "Distance table is required. Run compute_distances() first."
            )

        z = linkage(self.distance_table, method="ward")
        clusters = fcluster(z, k, criterion="maxclust")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        dendrogram(z, labels=self.labels, orientation="right", ax=ax1)
        ax1.set_title(f"Dendrogram Cut into k={k} Groups")

        eta_scores = {}
        for feature in self.frequencies.columns:
            f_vec = self.frequencies[feature]
            grand_mean = f_vec.mean()
            ss_total = np.sum((f_vec - grand_mean) ** 2)
            if ss_total == 0:
                continue

            ss_between = 0
            for cluster_id in np.unique(clusters):
                cluster_vec = f_vec[clusters == cluster_id]
                ss_between += len(cluster_vec) * (
                    (cluster_vec.mean() - grand_mean) ** 2
                )

            eta_scores[feature] = ss_between / ss_total

        top_features = sorted(eta_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        words, scores = zip(*top_features) if top_features else ([], [])

        sns.barplot(
            x=list(scores),
            y=list(words),
            hue=list(words),
            ax=ax2,
            palette="flare",
            legend=False,
        )
        ax2.set_title(rf"Top Features Distinctive of Clusters ($η^{{2}}$)")
        ax2.set_xlabel(rf"Correlation Ratio ($η^{{2}}$)")
        plt.tight_layout()
        plt.show()
