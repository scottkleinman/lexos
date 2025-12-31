"""base_calculator.py.

Last update: September 11, 2025
Last tested: February 16, 2025
"""

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, Optional

import pandas as pd
import spacy
from pydantic import BaseModel, ConfigDict, Field
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token

from lexos.exceptions import LexosException
from lexos.rolling_windows import Windows

validation_config = ConfigDict(
    arbitrary_types_allowed=True,
    json_schema_extra=DocJSONSchema.schema(),
    validate_assignment=True,
)


def flatten(input: dict | list | str) -> Iterable:
    """Yield items from any nested iterable.

    Args:
        input (dict | list | str): A list of lists or dicts.

    Yields:
        Iterable: Items from the nested iterable.

    Notes:
        See https://stackoverflow.com/a/40857703.
    """
    for x in input:
        if isinstance(x, Iterable) and not isinstance(x, str):
            if isinstance(x, list):
                for sub_x in flatten(x):
                    yield sub_x
            elif isinstance(x, dict):
                yield list(x.values())[0]
        else:
            yield x


def regex_escape(s: str) -> str:
    """Escape only regex special characters.

    Args:
        s (str): A string.

    Returns:
        An escaped string.

    Note:
        See https://stackoverflow.com/a/78136529/22853742.
    """
    if isinstance(s, bytes):
        return re.sub(rb"[][(){}?*+.^$]", lambda m: b"\\" + m.group(), s)
    return re.sub(r"[][(){}?*+.^$]", lambda m: "\\" + m.group(), s)


def spacy_rule_to_lower(
    patterns: dict | list[dict],
    old_key: Optional[list[str] | str] = ["TEXT", "ORTH"],
    new_key: Optional[str] = "LOWER",
) -> list:
    """Convert spacy Rule Matcher patterns to lowercase.

    Args:
        patterns (dict | list[dict]): A list of spacy Rule Matcher patterns.
        old_key (list[str] | str): A dictionary key or list of keys to rename.
        new_key (Optional[str]): The new key name.

    Returns:
        A list of spacy Rule Matcher patterns
    """

    def convert(key):
        """Converts the key to lowercase."""
        if key in old_key:
            return new_key
        else:
            return key

    if isinstance(patterns, dict):
        new_dict = {}
        for key, value in patterns.items():
            key = convert(key)
            new_dict[key] = value
        return new_dict

    if isinstance(patterns, list):
        new_list = []
        for value in patterns:
            new_list.append(spacy_rule_to_lower(value))
        return new_list


class BaseCalculator(ABC, BaseModel):
    """An abstract base class for calculators."""

    id: ClassVar[str] = "base_calculator"

    patterns: Optional[list | str] = Field(
        default=None, description="A pattern or list of patterns to search in windows."
    )
    windows: Optional[Windows] = Field(
        default=None, description="A Windows object containing the windows to search."
    )
    mode: Optional[bool | str] = Field(
        default="exact",
        description="The search method to use ('regex', 'spacy_rule', 'multi_token', 'multi_token_exact').",
    )
    case_sensitive: Optional[bool] = Field(
        default=False, description="Whether to make searches case-sensitive."
    )
    alignment_mode: Optional[str] = Field(
        default="strict",
        description="Whether to snap searches to token boundaries. Values are 'strict', 'contract', and 'expand'.",
    )
    model: Optional[str] = Field(
        default="xx_sent_ud_sm",
        description="The language model to be used for searching spaCy tokens/spans.",
    )
    nlp: Optional[Language] = Field(default=None, description="The spaCy nlp object.")
    data: Optional[list] = Field(
        default=[], description="A container for the calculated data."
    )

    model_config = validation_config

    @property
    def metadata(self) -> dict:
        """Return metadata for the calculator."""
        # Note: model_dump() may evaluate computed fields on this model.
        # If any computed properties rely on external state or are expensive
        # to compute, accessing them via model_dump() could raise or cause
        # performance issues. Subclasses should ensure their computed fields
        # are safe to evaluate here or override this property to exclude
        # such fields explicitly (e.g., model_dump(exclude=[...])).
        #
        # Historically, this property intentionally did not return the
        # result of `model_dump()`; it was used to ensure that validators or
        # other side effects ran without exposing the entire dict to the
        # caller. Maintain that behavior by calling `model_dump()` but not
        # returning the value. If you need the metadata dict, override this
        # property in subclasses to return it explicitly.
        self.model_dump()

    @property
    def n(self):
        """Get the number of units per window."""
        if self.windows.n is not None:
            return self.windows.n
        return None

    @property
    def regex_flags(self):
        """Return regex flags based on case_sensitive setting."""
        if not self.case_sensitive:
            return re.IGNORECASE | re.UNICODE
        else:
            return re.UNICODE

    @property
    def window_type(self):
        """Get the type of units in the windows."""
        if self.windows.window_type is not None:
            return self.windows.window_type
        return None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Call the instance."""
        ...

    def _count_character_patterns_in_character_windows(
        self, window: str, pattern: str
    ) -> int:
        """Use Python count() to count exact character matches in a character window.

        Args:
            window (str): A string window.
            pattern (str): A string pattern to search for.

        Returns:
            The number of occurrences of the pattern in the window.
        """
        if self.mode == "regex":
            return len(re.findall(pattern, window, self.regex_flags))
        else:
            if not self.case_sensitive:
                window = window.lower()
                pattern = pattern.lower()
            return window.count(pattern)

    def _count_in_character_window(self, window: str, pattern: str) -> int:
        """Choose function for counting matches in character windows.

        Args:
            window (str): A string window.
            pattern (str): A string pattern to search for.

        Returns:
            The number of occurrences of the pattern in the window.
        """
        if self.mode in ["exact", "regex"]:
            return self._count_character_patterns_in_character_windows(window, pattern)
        else:
            raise LexosException("Invalid mode for character windows.")

    def _count_token_patterns_in_token_lists(
        self, window: list[str], pattern: str
    ) -> int:
        """Count patterns in lists of token strings.

        Args:
            window (list[str]): A window consisting of a list of strings.
            pattern (str): A string pattern to search for.

        Returns:
            The number of occurrences of the pattern in the window.
        """
        if self.mode == "regex":
            return sum(
                [len(re.findall(pattern, token, self.regex_flags)) for token in window]
            )
        else:
            if not self.case_sensitive:
                window = [token.lower() for token in window]
                pattern = pattern.lower()
            return window.count(pattern)

    def _count_token_patterns_in_span(self, window: Span, pattern: list | str) -> int:
        """Count patterns in spans or docs.

        Args:
            window (Span): A window consisting of a list of spaCy spans or a spaCy doc.
            pattern (list | str): A string pattern or spaCy rule to search for.

        Returns:
            The number of occurrences of the pattern in the window.
        """
        if self.mode == "exact":
            if not self.case_sensitive:
                window = [token.lower_ for token in window]
                pattern = pattern.lower()
            else:
                window = [token.text for token in window]
            return window.count(pattern)
        elif self.mode == "regex":
            return sum(
                [
                    len(re.findall(pattern, token.text, self.regex_flags))
                    for token in window
                ]
            )
        elif self.mode == "spacy_rule":
            if not self.case_sensitive:
                pattern = spacy_rule_to_lower(pattern)
            matcher = Matcher(self.nlp.vocab)
            matcher.add("Pattern", [pattern])
            return len(matcher(window))

    def _count_token_patterns_in_span_text(self, window: Span, pattern: str) -> int:
        """Count patterns in span or doc text with token alignment.

        Args:
            window (Span): A Span window.
            pattern (str): A string pattern to search for.

        Returns:
            The number of occurrences of the pattern in the window.
        """
        count = 0
        if self.mode == "multi_token_exact":
            pattern = regex_escape(pattern)
        for match in re.finditer(pattern, window.text, self.regex_flags):
            start, end = match.span()
            span = window.char_span(start, end, self.alignment_mode)
            if span is not None:
                count += 1
        return count

    def _count_in_token_window(
        self, window: list[str] | list[Token] | Doc | Span, pattern: list | str
    ) -> int:
        """Choose function for counting matches in token windows.

        Args:
            window (list[str] | Span): A window consisting of a list of token strings, a list of spaCy spans, or a spaCy doc.
            pattern (list | str): A string pattern or spaCy rule to search for.

        Returns:
            The number of occurrences of the pattern in the window.
        """
        # Validate window type for multi_token and spacy_rule modes
        if self.mode in ["multi_token", "spacy_rule"]:
            if not isinstance(window, (Doc, Span)):
                raise LexosException(
                    "You cannot use spaCy rules or perform multi-token searches with a string or list of token strings."
                )
        if isinstance(window, (list)) and self.mode in ["multi_token", "spacy_rule"]:
            raise LexosException(
                "You cannot use spaCy rules or perform multi-token searches with a string or list of token strings."
            )
        elif isinstance(window, list) and all(isinstance(i, str) for i in window):
            # Match in single tokens
            return self._count_token_patterns_in_token_lists(window, pattern)
        elif isinstance(window, Doc | Span):
            # Iterate over the full text with token boundary alignment
            if self.mode.startswith("multi_token"):
                return self._count_token_patterns_in_span_text(window, pattern)
            # Match in single tokens
            else:
                return self._count_token_patterns_in_span(window, pattern)

    def _extract_string_pattern(self, pattern: list[list[dict[str, Any]]]) -> str:
        """Extract a string pattern from a spaCy rule.

        Args:
            pattern (list[list[dict[str, Any]]]): A list of spaCy rule patterns to search.

        Returns:
            str: A string pattern.
        """
        return "|".join(
            [
                item if isinstance(item, str) else list(item.values())[0]
                for item in list(flatten(pattern))
            ]
        )

    def _get_window_count(
        self, window: list[str] | Span | str, pattern: list | str
    ) -> int:
        """Call character or token window methods, as appropriate.

        Args:
            window (list[str] | Span | str]): A window consisting of a list of token strings, a list of spaCy spans, a spaCy doc, or a string.
            pattern (list | str): A string pattern or spaCy rule to search for.

        Returns:
            The number of occurrences of the pattern in the window.
        """
        if self.window_type == "characters":
            return self._count_in_character_window(window, pattern)
        else:
            return self._count_in_token_window(window, pattern)

    def _set_attrs(self, attrs: dict) -> None:
        """Set instance attributes when public method is called.

        Args:
            attrs (dict): A dict of keyword arguments and their values.
        """
        for key, value in attrs.items():
            if value is not None:
                setattr(self, key, value)
            if key == "model" and value is not None:
                self.nlp = spacy.load(self.model)

    @abstractmethod
    def to_df(self, *args, **kwargs) -> pd.DataFrame:
        """Output the calcualtions as a pandas DataFrame."""
        ...
