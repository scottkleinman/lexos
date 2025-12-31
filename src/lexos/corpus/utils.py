"""utils.py.

Utility functions and classes for the Lexos corpus module.

Last updated: December 7, 2025
Last tested: December 7, 2025
"""

from collections.abc import Mapping
from typing import Any

from spacy.language import Language

from lexos.util import load_spacy_model


class LexosModelCache:
    """A simple cache for spaCy models."""

    def __init__(self):
        """Initialize the cache."""
        self._cache = {}

    def get_model(self, model_name: str) -> Language:
        """Get a model from the cache or load it if not cached.

        Args:
            model_name (str): The spaCy model name to load (e.g., 'en_core_web_sm').

        Returns:
            Language: The loaded spaCy language model.
        """
        if model_name not in self._cache:
            self._cache[model_name] = load_spacy_model(model_name)
        return self._cache[model_name]


class RecordsDict(dict):
    """A dictionary-like class for storing Record objects.

    This class ensures that no ids can be overwritten, and it raises an error if an attempt is made to do so.
    """

    def __setitem__(self, key, value):
        """Set an item in the Records dictionary.

        Args:
            key (str): The ID of the Record.
            value (Record): The Record object to set.

        Raises:
            KeyError: If the key already exists in the Records dictionary.
        """
        if not key in self:
            super(RecordsDict, self).__setitem__(key, value)
        else:
            raise Exception(f"ID '{key}' already exists. Cannot overwrite.")

    def update(self, other=None, **kwargs: Any) -> None:
        """Update the Records dictionary with a non-prexisting mapping or keyword arguments.

        Args:
            other (Mapping or iterable): An optional mapping or iterable of key-value pairs to update the Records dictionary.
            **kwargs (Any): Additional keyword arguments to update the Records dictionary.
        """
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v
