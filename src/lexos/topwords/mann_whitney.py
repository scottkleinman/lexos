"""mann_whitney.py.

Implements the Mann-Whitney U (AKA Wilcoxon Rank-Sum) Test.

Last Updated: November 10, 2025
Last Tested: November 14, 2025
"""

from typing import Optional

import pandas as pd
from pydantic import ConfigDict, Field
from scipy.stats import mannwhitneyu
from wasabi import msg

from lexos.exceptions import LexosException
from lexos.topwords import TopWords


class MannWhitney(TopWords):
    """Mann-Whitney U test model."""

    target: pd.DataFrame = Field(
        ..., description="DataFrame containing frequencies for control documents."
    )
    comparison: pd.DataFrame = Field(
        ..., description="DataFrame containing frequencies for compare documents."
    )
    add_freq: Optional[bool] = Field(
        True, description="If True, adds average frequency and increase in frequency."
    )
    result: Optional[dict] = Field(
        default=None, description="Result of the Mann-Whitney U test."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initializes the MannWhitney model with data."""
        super().__init__(**data)
        if not isinstance(self.target, pd.DataFrame) or not isinstance(
            self.comparison, pd.DataFrame
        ):
            raise LexosException("Error: Inputs must be Pandas DataFrames.")
        if self.target.empty or self.comparison.empty:
            msg.warn("Warning: One or both input DataFrames are empty.")

        results = []

        # Iterate through columns of the first DataFrame (assumes y has same columns)
        for col in self.target.columns:
            if col not in self.comparison.columns:
                msg.warn(f"Warning: Column '{col}' not found in y. Skipping.")
                continue

            # Extract data for the current column
            x_data = self.target[
                col
            ].dropna()  # Drop NaNs as mannwhitneyu can't handle them
            y_data = self.comparison[col].dropna()

            # Ensure data is numeric and there's enough data to perform the test
            if not pd.api.types.is_numeric_dtype(
                x_data
            ) or not pd.api.types.is_numeric_dtype(y_data):
                msg.warn(
                    f"Warning: Column '{col}' is not numeric in one or both DataFrames. Skipping."
                )
                continue

            if len(x_data) == 0 or len(y_data) == 0:
                msg.warn(
                    f"Warning: Column '{col}' has no non-NaN data in one or both groups after dropping NaNs. Skipping."
                )
                continue

            try:
                # Perform the Mann-Whitney U test
                # Use alternative="two-sided" for a standard two-tailed test
                statistic, p_value = mannwhitneyu(
                    x_data, y_data, alternative="two-sided"
                )
                results.append(
                    {"term": col, "statistic": statistic, "p_value": p_value}
                )
            except ValueError as e:
                # This can happen if all values are identical in one or both samples, or other edge cases.
                msg.warn(
                    f"Warning: Could not calculate Mann-Whitney U for column '{col}': {e}"
                )
                results.append(
                    {"term": col, "statistic": float("nan"), "p_value": float("nan")}
                )

        if not results:
            msg.warn(
                "Warning: No suitable columns found to perform the Mann-Whitney U test."
            )
            # Set result to empty list for consistency
            self.result = []
            return

        result = pd.DataFrame(results)

        if self.add_freq:
            # Make sure that the Mann-Whitney U results are sorted by term
            result = result.sort_values(by="term", ascending=True)

            # Get the control mean and difference only for valid terms
            valid_terms = result["term"].tolist()
            freq_stats = self._get_freq_stats(valid_terms)

            # Add the average frequency and difference to the result DataFrame
            result["ave_freq"] = freq_stats["ave_freq"].tolist()
            result["difference"] = freq_stats["difference"].tolist()

        # Sort by the statistic
        self.result = result.sort_values(by="statistic", ascending=False).to_dict(
            orient="records"
        )

    def _get_freq_stats(self, valid_terms: list = None) -> pd.DataFrame:
        """Calculates the mean frequencies for control and comparison DataFrames.

        Also computes the difference in frequency from control to comparison.

        Args:
            valid_terms: List of terms to calculate statistics for. If None,
                        calculates for all columns.

        Returns:
        A DataFrame with means and differences in frequency.
        """
        # If valid_terms is provided, only calculate stats for those terms
        if valid_terms is not None:
            x_subset = self.target[valid_terms]
            y_subset = self.comparison[valid_terms]
        else:
            x_subset = self.target
            y_subset = self.comparison

        x_sorted = x_subset.T.sort_index(ascending=True)
        # Use numeric_only=True to handle non-numeric columns gracefully
        x_sorted["Mean"] = x_sorted.mean(axis=1, numeric_only=True)
        x_mean = x_sorted["Mean"].tolist()

        y_sorted = y_subset.T.sort_index(ascending=True)
        y_sorted["Mean"] = y_sorted.mean(axis=1, numeric_only=True)
        y_mean = y_sorted["Mean"].tolist()

        difference = [v1 - v2 for v1, v2 in zip(x_mean, y_mean)]
        difference = [f"{d * 100:.2f}%" for d in difference]
        df = pd.DataFrame({"ave_freq": x_mean, "difference": difference})
        df.index = x_sorted.index
        return df

    def to_df(self) -> pd.DataFrame:
        """Returns the result of the Mann-Whitney U test as a DataFrame."""
        if self.result is None:
            raise LexosException(
                "No results available. Ensure the model has been initialized correctly."
            )
        return pd.DataFrame.from_records(self.result)
