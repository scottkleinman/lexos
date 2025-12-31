"""__init__.py.

Filter module for Lexos.

This module provides filters for identifying and working with specific types of tokens
in spaCy Doc objects.

Available filters:

- BaseFilter: Base class for all filters
- IsWordFilter: Identifies word tokens
- IsRomanFilter: Identifies Roman numeral tokens
- IsStopwordFilter: Manages stop words in a model
"""

from .filters import BaseFilter, IsRomanFilter, IsStopwordFilter, IsWordFilter

__all__ = ["BaseFilter", "IsWordFilter", "IsRomanFilter", "IsStopwordFilter"]
