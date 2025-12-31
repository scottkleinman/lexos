"""counts.py.

Last update: February 16, 2025
Last tested: February 16, 2025
"""

from typing import ClassVar, Optional

import pandas as pd
from pydantic import ConfigDict, validate_call
from spacy.schemas import DocJSONSchema

from lexos.exceptions import LexosException
from lexos.rolling_windows import Windows
from lexos.rolling_windows.calculators.base_calculator import (
    BaseCalculator,
    spacy_rule_to_lower,
)

validation_config = ConfigDict(
    arbitrary_types_allowed=True,
    json_schema_extra=DocJSONSchema.schema(),
    validate_assignment=True,
)


class Counts(BaseCalculator):
    """A calculator for counting patterns in rolling windows."""

    _id: ClassVar[str] = "counts"

    @validate_call(config=validation_config)
    def __call__(
        self,
        patterns: Optional[list | str] = None,
        windows: Optional[Windows] = None,
        mode: Optional[bool | str] = None,
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
        if self.windows is not None:
            self.data = [
                [self._get_window_count(window, pattern) for pattern in self.patterns]
                for window in self.windows
            ]
            return self.data
        else:
            raise LexosException("Calculator `windows` attribute is empty.")

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
        # Generate dataframe
        return pd.DataFrame(self.data, columns=cols)
