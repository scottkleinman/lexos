"""ratios.py.

Last update: August 6, 2025
Last Tested: February 16, 2025
"""

from typing import ClassVar, Optional

import pandas as pd
from pydantic import ConfigDict, validate_call
from spacy.schemas import DocJSONSchema

from lexos.exceptions import LexosException
from lexos.rolling_windows import Windows
from lexos.rolling_windows.calculators.counts import Counts, spacy_rule_to_lower

validation_config = ConfigDict(
    arbitrary_types_allowed=True,
    json_schema_extra=DocJSONSchema.schema(),
    validate_assignment=True,
)


class Ratios(Counts):
    """A calculator for calculating ratios of counts in rolling windows."""

    _id: ClassVar[str] = "ratios"

    def __call__(
        self,
        patterns: Optional[list | str] = None,
        windows: Optional[Windows] = None,
        mode: Optional[bool] = None,
        case_sensitive: Optional[bool] = None,
        alignment_mode: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Call the calculator."""
        attrs = {
            "patterns": patterns,
            "windows": windows,
            "mode": mode,
            "case_sensitive": case_sensitive,
            "alignment_mode": alignment_mode,
            "model": model,
        }
        self._set_attrs(attrs)
        if not isinstance(self.patterns, list):
            raise LexosException(
                "You must supply a list of two patterns to calculate ratios."
            )
        if len(self.patterns) != 2:
            raise LexosException("You can only calculate ratios for two patterns.")
        if self.windows is not None:
            self.data = [
                self._get_ratio(
                    [
                        self._get_window_count(window, pattern)
                        for pattern in self.patterns
                    ]
                )
                for window in self.windows
            ]
            return self.data
        else:
            raise LexosException("Calculator `windows` attribute is empty.")

    def _get_ratio(self, counts: list[int]) -> float:
        """Calculate the ratio between two counts.

        Args:
            counts (List[int]): A list of two counts.

        Returns:
            The calculated ratio.
        """
        numerator = counts[0]
        denominator = counts[1]
        # Handle division by 0
        if denominator + numerator == 0:
            return 0.0
        else:
            return numerator / (denominator + numerator)

    @validate_call(config=validation_config)
    def to_df(self, show_spacy_rules: Optional[bool] = False) -> pd.DataFrame:
        """Convert the data to a pandas dataframe.

        Args:
            show_spacy_rules (Optional[bool]): If True, use full spaCy rules for labels; otherwise use only the string pattern.

        Returns:
                pd.DataFrame: A pandas DataFrame.
        """
        if show_spacy_rules:
            patterns = self.patterns
        else:
            patterns = []
            # Extract strings from spaCy rules
            for pattern in self.patterns:
                if isinstance(pattern, list):
                    patterns.append(self._extract_string_pattern(pattern))
                else:
                    patterns.append(pattern)
        # Assign column labels
        cols = []
        for pattern in patterns:
            if not self.case_sensitive and isinstance(pattern, str):
                pattern = pattern.lower()
            elif not self.case_sensitive and isinstance(pattern, list):
                pattern = str(spacy_rule_to_lower(pattern))
            cols.append(str(pattern))
        # Merge columns for ratios
        cols = [":".join(cols)]
        # Generate dataframe
        return pd.DataFrame(self.data, columns=cols)
