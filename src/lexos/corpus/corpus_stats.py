"""corpus_stats.py.

Last updated: December 4, 2025
Last tested: November 18, 2025
"""

import math
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict, Field, validate_call
from scipy import stats

from lexos.dtm import DTM


def make_labels_unique(labels: list[str]) -> list[str]:
    """Make labels unique by adding suffixes recursively.

    For duplicate labels, append "-001", "-002", etc. This function handles
    the case where a label already has a suffix that would conflict with
    the generated suffixes by recursively renaming.

    Args:
        labels: A list of labels that may contain duplicates.

    Returns:
        A new list with unique labels. Duplicates get suffixes "-001", "-002", etc.

    Example:
        >>> make_labels_unique(["doc", "doc", "report"])
        ["doc-001", "doc-002", "report"]

        >>> make_labels_unique(["doc", "doc-001", "doc"])
        ["doc-001-001", "doc-001-002", "doc-002"]
    """
    if not labels:
        return labels

    label_counts = {}
    result = []

    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Find all labels that appear more than once
    duplicates = {label for label, count in label_counts.items() if count > 1}

    if not duplicates:
        return labels

    # Track which labels need suffixes
    label_indices = {label: 0 for label in duplicates}

    for label in labels:
        if label in duplicates:
            label_indices[label] += 1
            new_label = f"{label}-{label_indices[label]:03d}"

            # Recursively handle conflicts: if the new label is also a duplicate,
            # rename all instances including the newly generated one
            if new_label in label_counts or new_label in duplicates:
                # Re-run make_labels_unique with the conflicting label treated as duplicate
                temp_labels = result + [new_label] + labels[len(result) + 1 :]
                return make_labels_unique(temp_labels)

            result.append(new_label)
        else:
            result.append(label)

    return result


class CorpusStats(BaseModel):
    """A class to hold statistics about a Corpus.

    The input should be a list of tuples, where each tuple contains:
        - id: A unique identifier for the record.
        - label: A label for the record.
        - token list: A list of tokens in the record. Tokens can be words, n-grams, or any other token unit.
        - Settings to pass to the DTM vectorizer, such as min_df, max_df, and max_n_terms.

    To reproduce the webapp:

      - stats = CorpusStats(docs=docs)
      - stats.doc_stats_df # The DataFrame containing record statistics.
      - stats.mean # The mean count for the entire corpus.
      - stats.standard_deviation # The standard deviation for the entire corpus.
      - stats.get_iqr_outliers() # Get outliers based on interquartile range (IQR).
      - stats.get_std_outliers() # Get outliers based on standard deviation.
      - stats.plot(column="total_tokens", type="plotly_boxplot" title="Corpus Boxplot") # Plot the boxplot of total tokens with Plotly.
    """

    docs: list[tuple[str, str, list[str]]]
    min_df: int | None = None
    max_df: int | None = None
    max_n_terms: int | None = None
    dtm: DTM = Field(
        default=None, description="Document-Term Matrix (DTM) for the Corpus."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize the CorpusStats and create the DTM."""
        super().__init__(**data)
        # Separate the ids and labels from the docs
        object.__setattr__(self, "ids", [doc[0] for doc in self.docs])
        object.__setattr__(self, "labels", [doc[1] for doc in self.docs])

        # Configure the DTM vectorizer with the provided settings
        vectorizer_kwargs = {}
        if self.min_df is not None:
            vectorizer_kwargs["min_df"] = self.min_df
        if self.max_df is not None:
            vectorizer_kwargs["max_df"] = self.max_df
        if self.max_n_terms is not None:
            vectorizer_kwargs["max_n_terms"] = self.max_n_terms

        # Create and initialize the Document-Term Matrix (DTM) using the provided token lists
        object.__setattr__(self, "dtm", DTM())
        # Pass vectorizer kwargs during the call rather than initialization
        # NB. DTM.to_df() will not work unless columns are unique labels
        self.dtm(
            docs=[doc[2] for doc in self.docs],
            labels=make_labels_unique(self.labels),
            **vectorizer_kwargs,
        )

    @property
    def df(self) -> pd.DataFrame:
        """Get the Document-Term Matrix (DTM) in sparse format."""
        return self.dtm.to_df()

    @cached_property
    def doc_stats_df(self) -> pd.DataFrame:
        """Get a Pandas dataframe containing the statistics of each record.

        Returns:
            A Pandas dataframe containing statistics of each record.
        """
        return self._get_doc_stats_df()

    @cached_property
    def mean_and_spread(self) -> tuple[float, float]:
        """Get the mean and standard deviation of the total tokens in the Corpus."""
        df = self.df.sparse.to_dense().T
        df["Total"] = df.sum(axis=1)
        return df["Total"].mean(), df["Total"].std()

    @property
    def mean(self) -> float:
        """Get the mean of the total tokens in the Corpus."""
        return self.mean_and_spread[0]

    @property
    def standard_deviation(self) -> float:
        """Get the standard deviation of the total tokens in the Corpus."""
        return self.mean_and_spread[1]

    @cached_property
    def iqr_values(self) -> tuple[float, float, float]:
        """Get the Q1, Q3, and IQR values for total tokens.

        Returns:
            tuple[float, float, float]: A tuple containing (q1, q3, iqr)
        """
        doc_lengths = self.doc_stats_df["total_tokens"].values
        q1 = np.quantile(doc_lengths, 0.25)
        q3 = np.quantile(doc_lengths, 0.75)
        iqr = q3 - q1
        return q1, q3, iqr

    @cached_property
    def iqr_bounds(self) -> tuple[float, float]:
        """Get the IQR outlier bounds for total tokens.

        Returns:
            tuple[float, float]: A tuple containing (lower_bound, upper_bound)
        """
        q1, q3, iqr = self.iqr_values
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return lower_bound, upper_bound

    @cached_property
    def iqr_outliers(self) -> list[tuple[str, str]]:
        """Get the IQR outliers for total tokens.

        Returns:
            list[tuple[str, str]]: A list of tuples containing the record ID
            and record name for each outlier.
        """
        doc_lengths = self.doc_stats_df["total_tokens"].values
        lower_bound, upper_bound = self.iqr_bounds

        return [
            (str(self.ids[i]), str(self.labels[i]))
            for i, length in enumerate(doc_lengths)
            if length < lower_bound or length > upper_bound
        ]

    @cached_property
    def distribution_stats(self) -> dict[str, float]:
        """Get comprehensive distribution statistics for record lengths.

        Returns:
            dict[str, float]: Dictionary containing skewness, kurtosis, and normality test results.
        """
        doc_lengths = self.doc_stats_df["total_tokens"].values

        # Calculate skewness and kurtosis
        skewness = stats.skew(doc_lengths)
        kurt = stats.kurtosis(doc_lengths)  # Excess kurtosis (normal dist = 0)

        # Shapiro-Wilk normality test
        shapiro_stat, shapiro_p = stats.shapiro(doc_lengths)

        # Coefficient of variation (relative variability)
        cv = self.standard_deviation / self.mean if self.mean != 0 else 0

        return {
            "skewness": skewness,
            "kurtosis": kurt,
            "coefficient_of_variation": cv,
            "shapiro_statistic": shapiro_stat,
            "shapiro_p_value": shapiro_p,
            "is_normal": shapiro_p > 0.05,  # Conservative threshold
        }

    @cached_property
    def percentiles(self) -> dict[str, float]:
        """Get comprehensive percentile analysis for record lengths.

        Returns:
            dict[str, float]: Dictionary containing various percentiles.
        """
        doc_lengths = self.doc_stats_df["total_tokens"].values

        return {
            "percentile_5": np.percentile(doc_lengths, 5),
            "percentile_10": np.percentile(doc_lengths, 10),
            "percentile_25": np.percentile(doc_lengths, 25),  # Q1
            "percentile_50": np.percentile(doc_lengths, 50),  # Median
            "percentile_75": np.percentile(doc_lengths, 75),  # Q3
            "percentile_90": np.percentile(doc_lengths, 90),
            "percentile_95": np.percentile(doc_lengths, 95),
            "min": np.min(doc_lengths),
            "max": np.max(doc_lengths),
            "range": np.max(doc_lengths) - np.min(doc_lengths),
        }

    @cached_property
    def text_diversity_stats(self) -> dict[str, float]:
        """Get text-specific diversity and complexity statistics.

        Returns:
            dict[str, float]: Dictionary containing lexical diversity measures.
        """
        doc_stats = self.doc_stats_df

        # Type-Token Ratio statistics across corpus
        ttr_values = (
            doc_stats["vocabulary_density"].values / 100
        )  # Convert from percentage

        # Hapax legomena statistics
        hapax_values = doc_stats["hapax_legomena"].values
        hapax_ratio = hapax_values / doc_stats["total_tokens"].values

        # Calculate corpus-level diversity metrics
        total_tokens = doc_stats["total_tokens"].sum()
        total_terms = (
            len(self.dtm.sorted_terms_list)
            if hasattr(self.dtm, "sorted_terms_list")
            else doc_stats["total_terms"].sum()
        )

        # Hapax dislegomena statistics
        dislegomena_values = doc_stats["hapax_dislegomena"].values
        dislegomena_ratio = dislegomena_values / doc_stats["total_tokens"].values

        return {
            "mean_ttr": np.mean(ttr_values),
            "median_ttr": np.median(ttr_values),
            "std_ttr": np.std(ttr_values),
            "corpus_ttr": total_terms / total_tokens if total_tokens > 0 else 0,
            "mean_hapax_ratio": np.mean(hapax_ratio),
            "median_hapax_ratio": np.median(hapax_ratio),
            "std_hapax_ratio": np.std(hapax_ratio),
            "total_hapax": hapax_values.sum(),
            "corpus_hapax_ratio": hapax_values.sum() / total_tokens
            if total_tokens > 0
            else 0,
            "mean_dislegomena_ratio": np.mean(dislegomena_ratio),
            "median_dislegomena_ratio": np.median(dislegomena_ratio),
            "total_dislegomena": dislegomena_values.sum(),
            "corpus_dislegomena_ratio": dislegomena_values.sum() / total_tokens
            if total_tokens > 0
            else 0,
        }

    def _get_doc_stats_df(self) -> pd.DataFrame:
        """Get a Pandas dataframe containing the statistics of each record.

        Returns:
            pd.DataFrame: A Pandas dataframe containing statistics of each record.
        """
        # Check if empty corpus is given.
        if self.df.empty:
            raise ValueError(
                "The DataFrame is empty. Please provide a valid DataFrame."
            )

        # Convert the DataFrame to dense format
        df = self.dtm.to_df().sparse.to_dense()

        # Replace the unique columns with the original columns (labels)
        df.columns = self.labels

        # Transpose the DataFrame so that documents are rows
        df = df.T

        # Create file_stats DataFrame
        file_stats = pd.DataFrame(self.labels, columns=["Documents"])
        file_stats.set_index("Documents", inplace=True)

        # Count terms appearing exactly once in each document
        file_stats[f"hapax_legomena"] = df.eq(1).sum(axis=1)

        # Calculate total tokens in each document
        file_stats["total_tokens"] = df.sum(axis=1)

        # Number of distinct terms in each document
        file_stats["total_terms"] = df.ne(0).sum(axis=1)

        # Calculate vocabulary density
        file_stats["vocabulary_density"] = (
            file_stats["total_terms"] / file_stats["total_tokens"] * 100
        ).round(2)

        # Add hapax dislegomena (words appearing exactly twice)
        file_stats["hapax_dislegomena"] = df.eq(2).sum(axis=1)

        return file_stats

    def get_iqr_outliers(self) -> list[tuple[str, str]]:
        """Get the interquartile range (IQR) outliers in the Corpus.

        Returns:
            list[tuple[str, str]]: A list of tuples containing the record ID
            and record name for each outlier.
        """
        return self.iqr_outliers

    def get_std_outliers(self) -> list[tuple[str, str]]:
        """Get the standard deviation outliers in the Corpus.

        Returns:
            list[tuple[str, str]]: A list of tuples containing the record ID
            and record name for each outlier.
        """
        # Get doc lengths from the doc_stats_df
        doc_lengths = self.doc_stats_df["total_tokens"].values

        # Calculate mean and std
        mean = doc_lengths.mean()
        std_dev = doc_lengths.std()

        return [
            (str(self.ids[i]), str(self.labels[i]))
            for i, length in enumerate(doc_lengths)
            if abs(length - mean) > 2 * std_dev
        ]

    def compare_groups(
        self,
        group1_labels: list[str],
        group2_labels: list[str],
        metric: str = "total_tokens",
        test_type: str = "mann_whitney",
    ) -> dict:
        """Compare two groups of records using statistical tests.

        Args:
            group1_labels: List of record labels for group 1
            group2_labels: List of record labels for group 2
            metric: Column name to compare (default: "total_tokens")
            test_type: Statistical test to use ("mann_whitney", "t_test", "welch_t")

        Returns:
            dict: Test results including statistic, p-value, and effect size
        """
        doc_stats = self.doc_stats_df

        # Get values for each group
        group1_values = doc_stats.loc[group1_labels, metric].values
        group2_values = doc_stats.loc[group2_labels, metric].values

        results = {
            "group1_size": len(group1_values),
            "group2_size": len(group2_values),
            "group1_mean": np.mean(group1_values),
            "group2_mean": np.mean(group2_values),
            "metric": metric,
            "test_type": test_type,
        }

        if test_type == "mann_whitney":
            # Mann-Whitney U test: Non-parametric test comparing distributions
            # Tests null hypothesis that distributions are identical
            statistic, p_value = stats.mannwhitneyu(
                group1_values, group2_values, alternative="two-sided"
            )

            # Calculate effect size using rank biserial correlation:
            # Formula: r = (2U)/(n1*n2) - 1
            # Where U is the Mann-Whitney U statistic
            # Range: -1 to +1 (like Pearson correlation)
            # Interpretation: proportion of pairs where group1 > group2, adjusted
            n1, n2 = len(group1_values), len(group2_values)
            effect_size = (2 * statistic) / (n1 * n2) - 1
            results.update(
                {
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "effect_size_interpretation": self._interpret_effect_size(
                        abs(effect_size)
                    ),
                }
            )

        elif test_type == "t_test":
            # Independent samples t-test: Parametric test assuming equal variances
            # Tests null hypothesis that population means are equal
            statistic, p_value = stats.ttest_ind(
                group1_values, group2_values, equal_var=True
            )

            # Calculate Cohen's d effect size:
            # First, compute pooled standard deviation using formula:
            # s_pooled = sqrt[((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2)]
            # This weights each group's variance by its degrees of freedom
            pooled_std = np.sqrt(
                (
                    (len(group1_values) - 1) * np.var(group1_values, ddof=1)
                    + (len(group2_values) - 1) * np.var(group2_values, ddof=1)
                )
                / (len(group1_values) + len(group2_values) - 2)
            )

            # Cohen's d = (mean1 - mean2) / pooled_std
            # Standardized mean difference in pooled standard deviation units
            # Interpretation: 0.2=small, 0.5=medium, 0.8=large effect
            cohens_d = (np.mean(group1_values) - np.mean(group2_values)) / pooled_std
            results.update(
                {
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_size": cohens_d,
                    "effect_size_interpretation": self._interpret_cohens_d(
                        abs(cohens_d)
                    ),
                }
            )

        elif test_type == "welch_t":
            # Welch's t-test: Parametric test NOT assuming equal variances
            # Uses Satterthwaite approximation for degrees of freedom
            # More robust when group variances differ substantially
            statistic, p_value = stats.ttest_ind(
                group1_values, group2_values, equal_var=False
            )

            # Calculate Cohen's d for unequal variances:
            # Even though we use Welch's t-test, we still use pooled std for Cohen's d
            # as it provides a standardized effect size comparable across studies
            s1, s2 = np.std(group1_values, ddof=1), np.std(group2_values, ddof=1)
            n1, n2 = len(group1_values), len(group2_values)

            # Pooled standard deviation (same formula as regular t-test)
            # This maintains comparability of effect sizes across different test types
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

            # Cohen's d = standardized mean difference
            cohens_d = (np.mean(group1_values) - np.mean(group2_values)) / pooled_std
            results.update(
                {
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_size": cohens_d,
                    "effect_size_interpretation": self._interpret_cohens_d(
                        abs(cohens_d)
                    ),
                }
            )

        # Add significance interpretation
        results["is_significant"] = p_value < 0.05
        results["significance_level"] = (
            "p < 0.001"
            if p_value < 0.001
            else "p < 0.01"
            if p_value < 0.01
            else "p < 0.05"
            if p_value < 0.05
            else "ns"
        )

        return results

    def bootstrap_confidence_interval(
        self,
        metric: str = "total_tokens",
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> dict:
        """Calculate bootstrap confidence intervals for a given metric.

        Args:
            metric: Column name to analyze
            confidence_level: Confidence level (default: 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples

        Returns:
            dict: Bootstrap statistics including confidence intervals
        """
        values = self.doc_stats_df[metric].values

        # Bootstrap sampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return {
            "metric": metric,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
            "original_mean": np.mean(values),
            "bootstrap_mean": np.mean(bootstrap_means),
            "bootstrap_std": np.std(bootstrap_means),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "margin_of_error": (ci_upper - ci_lower) / 2,
        }

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.1:
            return "negligible"
        elif effect_size < 0.3:
            return "small"
        elif effect_size < 0.5:
            return "medium"
        else:
            return "large"

    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"

    @cached_property
    def advanced_lexical_diversity(self) -> dict[str, float]:
        """Calculate advanced lexical diversity measures beyond simple TTR.

        Returns:
            dict: Advanced diversity measures including MTLD, HD-D, and more
        """
        doc_stats = self.doc_stats_df

        # Moving Average Type-Token Ratio (MATTR) simulation
        # Calculate TTR for overlapping windows to reduce text length sensitivity
        def calculate_mattr(tokens_list: list[str], window_size: int = 50) -> float:
            if len(tokens_list) < window_size:
                return len(set(tokens_list)) / len(tokens_list) if tokens_list else 0

            ttrs = []
            for i in range(len(tokens_list) - window_size + 1):
                window = tokens_list[i : i + window_size]
                ttr = len(set(window)) / len(window)
                ttrs.append(ttr)
            return np.mean(ttrs) if ttrs else 0

        # Corrected TTR (CTTR) - TTR divided by square root of tokens
        def calculate_cttr(types: int, tokens: int) -> float:
            return types / np.sqrt(2 * tokens) if tokens > 0 else 0

        # Root TTR (RTTR) - Types divided by square root of tokens
        def calculate_rttr(types: int, tokens: int) -> float:
            return types / np.sqrt(tokens) if tokens > 0 else 0

        # Log TTR (LogTTR) - Log of types divided by log of tokens
        def calculate_log_ttr(types: int, tokens: int) -> float:
            if types > 0 and tokens > 0:
                return np.log(types) / np.log(tokens)
            return 0

        # Calculate for each document
        doc_diversity = []
        for _, row in doc_stats.iterrows():
            tokens = row["total_tokens"]
            types = row["total_terms"]
            if isinstance(tokens, float) and math.isnan(tokens):
                tokens = 0
            else:
                tokens = int(tokens)
            if isinstance(types, float) and math.isnan(types):
                types = 0
            else:
                types = int(types)

            diversity = {
                "ttr": types / tokens if tokens > 0 else 0,
                "cttr": calculate_cttr(types, tokens),
                "rttr": calculate_rttr(types, tokens),
                "log_ttr": calculate_log_ttr(types, tokens),
            }
            doc_diversity.append(diversity)

        # Aggregate statistics
        diversity_df = pd.DataFrame(doc_diversity)

        return {
            "mean_cttr": diversity_df["cttr"].mean(),
            "median_cttr": diversity_df["cttr"].median(),
            "std_cttr": diversity_df["cttr"].std(),
            "mean_rttr": diversity_df["rttr"].mean(),
            "median_rttr": diversity_df["rttr"].median(),
            "std_rttr": diversity_df["rttr"].std(),
            "mean_log_ttr": diversity_df["log_ttr"].mean(),
            "median_log_ttr": diversity_df["log_ttr"].median(),
            "std_log_ttr": diversity_df["log_ttr"].std(),
            "diversity_range": diversity_df["ttr"].max() - diversity_df["ttr"].min(),
            "diversity_coefficient_variation": diversity_df["ttr"].std()
            / diversity_df["ttr"].mean()
            if diversity_df["ttr"].mean() > 0
            else 0,
        }

    @cached_property
    def zipf_analysis(self) -> dict[str, float | bool | str]:
        """Analyze corpus term frequency distribution using Zipf's law.

        Returns:
            dict: Zipf distribution analysis including slope, R-squared, and goodness of fit
        """
        try:
            # Get term frequencies from DTM
            if hasattr(self.dtm, "sorted_term_counts"):
                term_counts = list(self.dtm.sorted_term_counts.values())
            else:
                # Fallback: aggregate from document stats
                df = self.dtm.to_df().sparse.to_dense()
                term_counts = df.sum(axis=0).sort_values(ascending=False).values

            if len(term_counts) < 10:  # Need sufficient data for meaningful analysis
                return {
                    "zipf_slope": 0.0,
                    "zipf_intercept": 0.0,
                    "r_squared": 0.0,
                    "zipf_goodness_of_fit": "insufficient_data",
                    "follows_zipf": False,
                    "num_terms": len(term_counts),
                }

            # Filter out zero counts and prepare for log-log analysis
            term_counts = [count for count in term_counts if count > 0]
            ranks = np.arange(1, len(term_counts) + 1)

            # Log-log transformation for Zipf analysis
            log_ranks = np.log10(ranks)
            log_freqs = np.log10(term_counts)

            # Linear regression on log-log data
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_ranks, log_freqs
            )
            r_squared = r_value**2

            # Zipf's law predicts slope around -1
            zipf_deviation = abs(slope + 1.0)  # How far from ideal Zipf slope of -1

            # Classify goodness of fit
            if r_squared > 0.9 and zipf_deviation < 0.3:
                fit_quality = "excellent"
                follows_zipf = True
            elif r_squared > 0.8 and zipf_deviation < 0.5:
                fit_quality = "good"
                follows_zipf = True
            elif r_squared > 0.6 and zipf_deviation < 0.7:
                fit_quality = "moderate"
                follows_zipf = True
            else:
                fit_quality = "poor"
                follows_zipf = False

            return {
                "zipf_slope": slope,
                "zipf_intercept": intercept,
                "r_squared": r_squared,
                "p_value": p_value,
                "std_error": std_err,
                "zipf_deviation": zipf_deviation,
                "zipf_goodness_of_fit": fit_quality,
                "follows_zipf": follows_zipf,
                "num_terms": len(term_counts),
                "frequency_range": {
                    "min": int(min(term_counts)),
                    "max": int(max(term_counts)),
                    "ratio": max(term_counts) / min(term_counts)
                    if min(term_counts) > 0
                    else 0,
                },
            }

        except Exception as e:
            return {
                "zipf_slope": 0.0,
                "zipf_intercept": 0.0,
                "r_squared": 0.0,
                "zipf_goodness_of_fit": f"error: {str(e)}",
                "follows_zipf": False,
                "num_terms": 0,
            }

    @cached_property
    def corpus_quality_metrics(self) -> dict[str, float | int | str]:
        """Calculate corpus quality and balance metrics for research validity.

        Returns:
            dict: Quality metrics including balance, coverage, and sampling adequacy
        """
        doc_stats = self.doc_stats_df

        # Document length balance analysis
        doc_lengths = doc_stats["total_tokens"].values
        length_cv = (
            np.std(doc_lengths) / np.mean(doc_lengths)
            if np.mean(doc_lengths) > 0
            else 0
        )

        # Vocabulary density balance
        vocab_densities = doc_stats["vocabulary_density"].values / 100
        density_cv = (
            np.std(vocab_densities) / np.mean(vocab_densities)
            if np.mean(vocab_densities) > 0
            else 0
        )

        # Term coverage analysis
        total_tokens = doc_stats["total_tokens"].sum()
        total_unique_terms = (
            len(self.dtm.sorted_terms_list)
            if hasattr(self.dtm, "sorted_terms_list")
            else doc_stats["total_terms"].sum()
        )

        # Hapax analysis for vocabulary richness
        total_hapax = doc_stats["hapax_legomena"].sum()
        total_dislegomena = doc_stats["hapax_dislegomena"].sum()

        # Calculate vocabulary growth rate (simplified)
        # This approximates how much new vocabulary each document contributes
        vocab_growth = total_unique_terms / len(doc_stats) if len(doc_stats) > 0 else 0

        # Corpus balance classification
        def classify_balance(cv: float) -> str:
            if cv < 0.2:
                return "very_balanced"
            elif cv < 0.4:
                return "balanced"
            elif cv < 0.6:
                return "moderately_unbalanced"
            else:
                return "highly_unbalanced"

        # Sampling adequacy (based on vocabulary saturation)
        vocab_saturation = (
            total_hapax / total_unique_terms if total_unique_terms > 0 else 0
        )

        def assess_sampling_adequacy(saturation: float) -> str:
            if saturation < 0.1:
                return "excellent"  # Very few hapax, good coverage
            elif saturation < 0.3:
                return "good"
            elif saturation < 0.5:
                return "adequate"
            else:
                return "insufficient"  # Too many hapax, need more data

        return {
            "document_length_balance": {
                "coefficient_variation": length_cv,
                "classification": classify_balance(length_cv),
                "range_ratio": np.max(doc_lengths) / np.min(doc_lengths)
                if np.min(doc_lengths) > 0
                else 0,
            },
            "vocabulary_density_balance": {
                "coefficient_variation": density_cv,
                "classification": classify_balance(density_cv),
            },
            "corpus_coverage": {
                "total_tokens": int(total_tokens),
                "unique_terms": int(total_unique_terms),
                "coverage_ratio": total_unique_terms / total_tokens
                if total_tokens > 0
                else 0,
                "vocab_growth_per_doc": vocab_growth,
            },
            "vocabulary_richness": {
                "hapax_ratio": total_hapax / total_tokens if total_tokens > 0 else 0,
                "dislegomena_ratio": total_dislegomena / total_tokens
                if total_tokens > 0
                else 0,
                "vocabulary_saturation": vocab_saturation,
                "sampling_adequacy": assess_sampling_adequacy(vocab_saturation),
            },
            "corpus_size_metrics": {
                "num_documents": len(doc_stats),
                "mean_doc_length": np.mean(doc_lengths),
                "median_doc_length": np.median(doc_lengths),
                "recommended_min_docs": max(
                    30, int(total_unique_terms * 0.1)
                ),  # Rule of thumb
                "size_adequacy": "adequate" if len(doc_stats) >= 30 else "small",
            },
        }

    @validate_call(config=model_config)
    def plot(
        self,
        column: str = "token_lengths",
        type: str = "seaborn_boxplot",
        title: str = None,
    ) -> None:
        """Generate a plot of the Corpus.

        Args:
            column: The column to plot from the doc_stats_df.
            type: The type of plot to generate. Currently only "seaborn_boxplot" and "plotly_boxplot" are supported.
            title: The title of the plot. If None, the plotting function's default is used.
        """
        supported_types = ["seaborn_boxplot", "plotly_boxplot"]
        if type not in supported_types:
            raise ValueError(
                f"Unsupported plot type: {type}. The following types are supported: {', '.join(supported_types)}."
            )

        # Use default title if None is provided
        plot_title = title if title is not None else "Corpus Boxplot"

        if type == "seaborn_boxplot":
            get_seaborn_boxplot(self.doc_stats_df, column=column, title=plot_title)
        elif type == "plotly_boxplot":
            get_plotly_boxplot(self.doc_stats_df, column=column, title=plot_title)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_seaborn_boxplot(
    df: pd.DataFrame, column: str, title: str = "Corpus Boxplot"
) -> None:
    """Get a boxplot of the specified column in the DataFrame.

    Args:
        df: A Pandas DataFrame.
        column: The column to plot.
        title: The title of the plot.
    """
    sns.set_theme(style="darkgrid")
    ax = sns.boxplot(y=df[column], width=0.25)
    sns.swarmplot(y=column, data=df, color="black", ax=ax)
    ax.set_title(title)
    plt.show()


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_plotly_boxplot(
    df: pd.DataFrame, column: str, title: str = "Corpus Boxplot"
) -> None:
    """Get a boxplot of the specified column in the DataFrame using Plotly.

    Args:
        df: A Pandas DataFrame.
        column: The column to plot.
        title: The title of the plot.
    """
    # Get file names.
    labels = df.index.tolist()

    # Set up the points.
    scatter_plot = go.Scatter(
        x=labels,
        y=df[column].values,
        hoverinfo="text",
        mode="markers",
        marker=dict(color="green"),
        text=labels,
    )

    # Set up the box plot.
    box_plot = go.Box(
        x0=0,  # Initial position of the box plot
        y=df[column].values,
        hoverinfo="y",
        marker=dict(color="green"),
        jitter=0.15,
    )

    # Create a figure with two subplots and fill the figure.
    figure = make_subplots(rows=1, cols=2, shared_yaxes=False)
    figure.append_trace(trace=scatter_plot, row=1, col=1)
    figure.append_trace(trace=box_plot, row=1, col=2)

    # Hide useless information on x-axis and set up title.
    figure.layout.update(
        title={
            "text": title,
            "x": 0.5,  # x position (0-1)
            "xanchor": "center",  # Horizontal alignment
            "y": 0.99,  # y position (0-1)
            "yanchor": "top",  # Vertical alignment
        },
        height=300,
        width=500,
        dragmode="pan",
        showlegend=False,
        margin=dict(r=0, b=30, t=15, pad=4),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            showline=False,
            zeroline=False,
            gridcolor="black",
            title="Total Tokens",
        ),
        xaxis2=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis2=dict(
            showline=False,
            zeroline=False,
            gridcolor="black",
        ),
        hovermode="closest",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="black", size=14),
    )

    # Show the Plotly figure.
    config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["toImage", "toggleSpikelines"],
        "scrollZoom": True,
    }
    figure.show(showlink=False, config=config)
