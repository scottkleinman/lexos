"""__init__.py.

Last Updated: June 25, 2025
Last Tested: June 25, 2025
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TopWords(BaseModel):
    """Base class for topwords plugins, providing a common API."""

    def to_df(self):
        """Return a pandas DataFrame representation of the model."""
        raise NotImplementedError("Subclasses must implement the to_df method")
