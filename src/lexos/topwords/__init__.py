"""Public API for the `lexos.topwords` package.

Topwords classes are used to identify keywords and significant terms in a corpus.

Phase 1 export surface:
- TopWords
- KeyTerms
- MannWhitney
- ZTest
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class TopWords(BaseModel):
    """Base class for topwords plugins, providing a common API."""

    def to_df(self):
        """Return a pandas DataFrame representation of the model."""
        return pd.DataFrame()


from lexos.topwords.keyterms import KeyTerms
from lexos.topwords.mann_whitney import MannWhitney
from lexos.topwords.ztest import ZTest

__all__ = ["TopWords", "KeyTerms", "MannWhitney", "ZTest"]
