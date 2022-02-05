"""lexosdoc.py."""
from typing import List
import pandas as pd
from . import extensions

class LexosDoc():
    """A wrapper class for a spaCy doc which allows for extra methods.

    A convenience that allows you to use Doc extensions without the
    underscore prefix.

    Note: There is probably no need for this class. We can just keep a
    library of functions in a file called `tokenizer.py` and import them.
    If certain functions get used commonly, they can be turned into Doc
    extensions.
    """
    def __init__(self, doc: object):
        self.doc = doc

    def get_tokens(self):
        """Return a list of tokens in the doc."""
        return self.doc._.get_tokens()

    def get_token_attrs(self):
        """Get a list of attributes for each token in the doc.

        Returns a dict with "spacy_attributes" and "extensions".

        Note: This function relies on sampling the first token in a doc
        to compile the list of attributes. It does not check for consistency.
        Currently, it is up to the user to reconcile inconsistencies between
        docs.
        """
        sample = self.doc[0]
        attrs = sorted([x for x in dir(sample) if not x.startswith("__") and x != "_"])
        exts = sorted([f"_{x}" for x in dir(sample._) if x not in ["get", "has", "set"]])
        return {"spacy_attributes": attrs, "extensions": exts}

    def to_dataframe(self,
                     cols: List[str] = ["text"],
                     show_ranges: bool = True) -> pd.DataFrame:
        """Get a pandas dataframe of the doc attributes.

        Args:
            cols: A list of columns to include in the dataframe.
            show_ranges: Whether to include the token start and end positions in the dataframe.

        Returns a pandas dataframe of the doc attributes.

        Note: It is a good idea to call `LexosDoc.get_token_attrs()` first
        to check which attributes are available for the doc.
        """
        rows = []
        for i, token in enumerate(self.doc):
            t = []
            for col in cols:
                t.append(getattr(token, col))
            if show_ranges:
                ranges = self.doc.to_json()["tokens"][i]
                t.append(ranges["start"])
                t.append(ranges["end"])
            rows.append(t)
        if show_ranges:
            cols = cols + ["start", "end"]
        return pd.DataFrame(rows, columns=cols)
