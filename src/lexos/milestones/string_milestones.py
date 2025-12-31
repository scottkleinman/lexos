"""string_milestones.py.

The StringMilestones class is an efficient model for extracting and storing
milestones from character strings. It will accept spaCy Doc objects but use
the Doc.text attribute to search for milestones. The class uses regex to
match milestone patterns and stores them along with the milestone text in
StringSpan objects. The class is iterable and will return a list of StringSpans
if the spans property is accessed. The set method will update the object's
pattern and case sensitivity settings.

Example:
doc = "The quick brown fox jumps over the lazy dog."
milestones = StringMilestones(doc=doc, patterns="quick")
for milestone in milestones:
    print(milestone.start, milestone.end, milestone.text)
milestones.set("The", case_sensitive=False)
spans = milestones.spans

Last Update: Jan 14 2025
Last Tested: Dec 21 2024
"""

import re
from enum import Enum
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span

from .util import ensure_list

type Doclike = str | Doc | Span
case_insensitive_flags: Enum = re.DOTALL | re.IGNORECASE | re.MULTILINE | re.UNICODE
case_sensitive_flags: Enum = re.DOTALL | re.MULTILINE | re.UNICODE


class StringSpan(BaseModel):
    """StringSpan class.

    A Pydantic model containing the milestone text, and the start
    and character indices of the milestone in the original text.
    """

    text: str
    start: int
    end: int


class StringMilestones(BaseModel):
    """String Milestones class.

    Milestones object for text strings or spaCy Doc objects to
    be treated as strings.
    """

    doc: Doclike = Field(
        json_schema_extra={"description": "A string or spaCy Doc object."}
    )

    patterns: str | list[str] = Field(
        default=None,
        json_schema_extra={"description": "The pattern(s) used to match milestones."},
    )
    case_sensitive: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Whether to perform case-sensitive searches."
        },
    )
    flags: Enum = Field(
        default=case_sensitive_flags,
        json_schema_extra={"description": "The regex flags to use."},
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data) -> None:
        """Set regex flags and milestone IOB extensions after initialization."""
        super().__init__(**data)
        self._spans: list = []
        if not self.case_sensitive:
            self.flags = case_insensitive_flags
        self.patterns = ensure_list(self.patterns)
        if self.patterns != [None]:
            self.set()

    @property
    def spans(self) -> list[StringSpan]:
        """Return the Spans.

        Returns:
            list[StringSpan]: A list of StringSpans.
        """
        return self._spans or []

    def __iter__(self) -> Iterator:
        """Make the class iterable.

        Returns:
            Iterator: A generator containing the object's spans.
        """
        return (span for span in self.spans)

    def _set_case_sensitivity(self, case_sensitive: Optional[bool] = None) -> None:
        """Set the object's case sensitivity.

        Args:
            case_sensitive (optional, bool): Whether or not to use case-sensitive searching.
        """
        if case_sensitive is not None:
            self.case_sensitive = case_sensitive
        if self.case_sensitive is True:
            self.flags = case_sensitive_flags
        else:
            self.flags = case_insensitive_flags

    @validate_call()
    def set(
        self,
        patterns: Optional[str | list[str]] = None,
        case_sensitive: Optional[bool] = None,
    ) -> None:
        """Return the milestones.

        Args:
            patterns (Optional[str | list[str]]): The pattern(s) used to match milestones.
            case_sensitive (bool, optional): Whether to perform case-sensitive searches. Defaults to True.

        Note:
            If no parameters are set, the method will use the object's current patterns and case sensitivity.
        """
        if patterns:
            self.patterns = ensure_list(patterns)
        self._set_case_sensitivity(case_sensitive)
        text = self.doc if isinstance(self.doc, str) else self.doc.text
        all_matches = []
        for pattern in self.patterns:
            matches = re.finditer(pattern, text, self.flags)
            all_matches.extend(
                StringSpan(text=match.group(), start=match.start(), end=match.end())
                for match in matches
            )
        all_matches.sort(key=lambda match: match.start)
        self._spans = all_matches
