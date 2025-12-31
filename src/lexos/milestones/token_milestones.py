"""milestones.py.

Last Update: December 23, 2025
Last Tested: December 23, 2025
"""

import re
from enum import Enum
from string import punctuation
from typing import Any, Iterator, Match, Optional

import spacy
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.matcher import Matcher, PhraseMatcher
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token

from .util import (
    chars_to_tokens,
    ensure_list,
    filter_doc,
    lowercase_spacy_rules,
    move_milestone,
)

type Doclike = Doc | Span | str
case_insensitive_flags: Enum = re.DOTALL | re.IGNORECASE | re.MULTILINE | re.UNICODE
case_sensitive_flags: Enum = re.DOTALL | re.MULTILINE | re.UNICODE

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
)


class TokenMilestones(BaseModel):
    """Milestones class.

    - Referencing the Milestones instance yields an iterator of the spans in the Doc.
    - Referencing Milestones.spans returns an indexed list of spans in the Doc.
    - Referencing milestones.doc.spans["milestones"] returns a SpanGroup.
    """

    doc: Doclike = Field(
        json_schema_extra={"description": "A string or spaCy Doc object."}
    )

    patterns: Optional[Any] = Field(
        default=None,
        json_schema_extra={"description": "The pattern(s) used to match milestones."},
    )
    case_sensitive: Optional[bool] = Field(
        default=True,
        json_schema_extra={
            "description": "Whether to perform case-sensitive searches."
        },
    )
    character_map: Optional[dict] = Field(
        default=None,
        json_schema_extra={"description": "A map of characters to token indexes."},
    )
    attr: Optional[str] = Field(
        default="ORTH",
        json_schema_extra={
            "description": "The spaCy token attribute to search ('ORTH' or 'LOWER')."
        },
    )
    flags: Optional[Enum] = Field(
        default=case_sensitive_flags,
        json_schema_extra={"description": "The regex flags to use."},
    )
    mode: Optional[str] = Field(
        default="string",
        json_schema_extra={"description": "The mode used to match patterns."},
    )
    nlp: Optional[str] = Field(
        default="xx_sent_ud_sm",
        json_schema_extra={"description": "The language model to use."},
    )
    type: Optional[str] = Field(
        default=None, json_schema_extra={"description": "The type of milestone."}
    )

    model_config = validation_config

    def __init__(self, **data) -> None:
        """Set regex flags and milestone IOB extensions after initialization."""
        super().__init__(**data)
        if not self.case_sensitive:
            self.flags = case_insensitive_flags
            self.attr = "LOWER"
        if not Token.has_extension("milestone_iob"):
            Token.set_extension("milestone_iob", default="O", force=True)
        if not Token.has_extension("milestone_label"):
            Token.set_extension("milestone_label", default="", force=True)

    @property
    def spans(self) -> list[Span]:
        """Return the Spans.

        Returns:
            list[Span]: A list of spaCy Spans.
        """
        if "milestones" in self.doc.spans:
            return list(self.doc.spans["milestones"])
        else:
            return []

    def __iter__(self) -> Iterator:
        """Make the class iterable.

        Returns:
            Iterator: A generator containing the object's spans.
        """
        return (span for span in self.spans)

    def _assign_token_attributes(
        self, spans: list[Span], max_label_length: int = 20
    ) -> None:
        """Assign token attributes in the doc based on spans.

        Args:
            spans (list[Span]): A list of spaCy Spans.
            max_label_length (int): The maximum number of characters to include in the label.
        """
        # Early return if no spans
        if not spans:
            for token in self.doc:
                self.doc[token.i]._.milestone_iob = "O"
                self.doc[token.i]._.milestone_label = ""
            return

        # Pre-compute token positions and labels
        milestone_starts = {span.start: span for span in spans}
        milestone_ranges = {token.i for span in spans for token in span[1:]}

        # Assign attributes in single pass
        for token in self.doc:
            if span := milestone_starts.get(token.i):
                self.doc[token.i]._.milestone_iob = "B"
                self.doc[
                    token.i
                ]._.milestone_label = f"{span.text:.{max_label_length}}{'...' if len(span.text) > max_label_length else ''}"
            elif token.i in milestone_ranges:
                self.doc[token.i]._.milestone_iob = "I"
                self.doc[token.i]._.milestone_label = ""
            else:
                self.doc[token.i]._.milestone_iob = "O"
                self.doc[token.i]._.milestone_label = ""

    def _autodetect_mode(self, patterns: str | list) -> str:
        """Autodetect mode for matching milestones if not supplied (experimental).

        Args:
            patterns (str | list): A pattern to match.

        Returns:
            str: A string to supply to the get_matches() mode argument.
        """
        for pattern in patterns:
            if not isinstance(pattern, (str, list)):
                raise ValueError(
                    f"Pattern {pattern} must be a string or a spaCy Matcher rule."
                )
            if isinstance(pattern, str):
                if re.search(r"\s", pattern):
                    self.mode = "phrase"
                else:
                    self.mode = "string"
            else:
                try:
                    matcher = Matcher(self.doc.vocab, validate=True)
                    matcher.add("Pattern", [pattern])
                    self.mode = "rule"
                # Raise an error if the pattern is not a valid Matcher pattern
                except BaseException:
                    raise BaseException(
                        f"The pattern `{pattern}` could not be matched automatically. Check that the pattern is correct and try setting the `mode` argument in `get_matches()`."
                    )
        return self.mode

    def _get_string_matches(self, patterns: Any, flags: Enum) -> list[Span]:
        """Get matches to milestone patterns.

        Args:
            patterns (Any): A pattern to match.
            flags (Enum): An enum of regex flags.

        Returns:
            list[Span]: A list of Spans matching the pattern.
        """
        if patterns is None or patterns == []:
            raise ValueError("Patterns cannot be empty")
        patterns = ensure_list(patterns)
        if self.character_map is None:
            self.character_map = chars_to_tokens(self.doc)
        pattern_matches = []
        for pattern in patterns:
            matches = re.finditer(pattern, self.doc.text, flags=flags)
            for match in matches:
                pattern_matches.append(match)
        return [self._to_spacy_span(match) for match in pattern_matches]

    def _get_phrase_matches(self, patterns: Any, attr: str = "ORTH") -> list[Span]:
        """Get matches to milestone patterns in phrases.

        Args:
            patterns (Any): A pattern to match.
            attr (str): A spaCy Token attribute to search.

        Returns:
            list[Span]: A list of Spans matching the pattern.
        """
        nlp = spacy.load(self.nlp)
        matcher = PhraseMatcher(self.doc.vocab, attr=attr)
        patterns = [nlp.make_doc(text) for text in patterns]
        matcher.add("PatternList", patterns)
        matches = matcher(self.doc)
        return [self.doc[start:end] for _, start, end in matches]

    def _get_rule_matches(self, patterns: Any) -> list[Span]:
        """Get matches to milestone patterns with spaCy rules.

        Args:
            patterns (Any): A pattern to match.

        Returns:
            list[Span]: A list of Spans matching the pattern.
        """
        nlp = spacy.load(self.nlp)
        spans = []
        if not self.case_sensitive:
            patterns = lowercase_spacy_rules(patterns)
        for pattern in patterns:
            matcher = Matcher(nlp.vocab, validate=True)
            matcher.add("Pattern", [pattern])
            matches = matcher(self.doc)
            spans.extend([self.doc[start:end] for _, start, end in matches])
        return spans

    def _remove_duplicate_spans(self, spans: list[Span]) -> list[Span]:
        """Remove duplicate spans, generally created when a pattern is added.

        Args:
            spans (list[Span]): A list of Spans.

        Returns:
            list[Span]: A list of de-duplicated Spans.
        """
        result = []
        seen = set()
        for span in spans:
            key = (span.start, span.end)
            if key not in seen:
                result.append(span)
                seen.add(key)
        return result

    def _set_case_sensitivity(self, case_sensitive: Optional[bool] = None) -> None:
        """Set the object's case sensitivity.

        Args:
            case_sensitive (optional, bool): Whether or not to use case-sensitive searching.
        """
        if case_sensitive is not None:
            self.case_sensitive = case_sensitive
        if self.case_sensitive is True:
            self.flags: Enum = re.DOTALL | re.MULTILINE | re.UNICODE
            self.attr = "ORTH"
        else:
            self.flags: Enum = re.DOTALL | re.IGNORECASE | re.MULTILINE | re.UNICODE
            self.attr = "LOWER"

    def _to_spacy_span(self, match: Match) -> Span:
        """Convert a re.match object to a Span.

        Args:
            match (Match): A re.match object.

        Returns:
            Span: A spaCy Span.

        Raises:
            ValueError: If match is None or span cannot be created.
        """
        if not match:
            raise ValueError("Match object is None.")

        # Lazy load character map
        if not self.character_map:
            self.character_map = chars_to_tokens(self.doc)

        # Get character positions
        start_char, end_char = match.span()

        # Try direct char_span first
        if span := self.doc.char_span(start_char, end_char):
            return span

        # Fallback to character map
        start_token = self.character_map.get(start_char)
        end_token = self.character_map.get(end_char)

        if start_token is not None and end_token is not None:
            if span := self.doc[start_token : end_token + 1]:
                return span

        raise ValueError(
            f"Could not create span for match at positions {start_char}:{end_char}"
        )

    @validate_call(config=validation_config)
    def get_matches(
        self,
        patterns: Optional[Any] = None,
        mode: Optional[str] = None,
        case_sensitive: Optional[bool] = None,
    ) -> list[Span]:
        """Get matches to milestone patterns.

        Args:
            patterns (Optional[Any]): The pattern(s) to match.
            mode (Optional[str]): The mode to use for matching ('string', 'phrase', 'rule').
            case_sensitive (Optional[bool]): Whether to use case sensitive matching. Defaults to True.

        Returns:
            list[Span]: A list of spaCy Spans matching the pattern.

        Raises:
            ValueError: If patterns is None or empty.
        """
        self._set_case_sensitivity(case_sensitive)

        # Update patterns list
        if patterns:
            self.patterns = ensure_list(patterns)

        # Define mode handlers
        mode_handlers = {
            "string": lambda: self._get_string_matches(patterns, self.flags),
            "phrase": lambda: self._get_phrase_matches(patterns, self.attr),
            "rule": lambda: self._get_rule_matches(patterns),
        }

        # If mode not provided or invalid, autodetect
        if not mode or mode not in mode_handlers:
            spans = self.get_matches(patterns, mode=self._autodetect_mode(patterns))
        # Get spans using appropriate handler
        else:
            spans = mode_handlers[mode]()
        return self._remove_duplicate_spans(spans)

    @validate_call(config=validation_config)
    def remove(self, patterns: Any, *, mode: Optional[str] = "string") -> None:
        """Remove patterns.

        Args:
            patterns (Any): The pattern(s) to match.
            mode (Optional[str]): The mode to use for matching.
        """
        patterns = ensure_list(patterns)
        spans = self.get_matches(patterns, mode=mode)

        # Create a set of spans to remove for faster lookup
        remove_spans = {f"{span.start},{span.end}" for span in spans}

        # Filter out the spans to be removed
        new_spans = [
            span
            for span in self.doc.spans["milestones"]
            if f"{span.start},{span.end}" not in remove_spans
        ]

        # Reset the token attributes for the spans to be removed
        for span in spans:
            for token in self.doc[span.start : span.end]:
                token._.milestone_iob = "O"
                token._.milestone_label = ""

        # Re-set the milestones with the remaining spans
        self.set_milestones(new_spans)

        # Remove the patterns from the object's patterns list
        self.patterns = [p for p in self.patterns if p not in patterns]

    def reset(self):
        """Reset all `milestone` values to defaults.

        Note: Does not modify patterns or any other settings.
        """
        self.doc.spans["milestones"] = []
        for i, _ in enumerate(self.doc):
            self.doc[i]._.milestone_iob = "O"
            self.doc[i]._.milestone_label = ""

    @validate_call(config=validation_config)
    def set_milestones(
        self,
        spans: list[Span],
        *,
        start: Optional[str | None] = None,
        remove: Optional[bool] = False,
        max_label_length: Optional[int] = 20,
    ) -> None:
        """Commit milestones to the object instance.

        Args:
            spans (list[Span]): The span(s) to use for identifying token attributes.
            start (Optional[str | None]): Set milestone start to the token before or after the milestone span. May be "before" or "after".
            remove (Optional[bool]): Set milestone start to the token following the milestone span and
                remove the milestone span tokens from the Doc.
            max_label_length (Optional[int]): The maximum number of characters to include in the label.
        """
        if start not in [None, "before", "after"]:
            raise ValueError("Start must be None, 'before', or 'after'.")
        if remove:
            self.doc = filter_doc(self.doc, spans)
        elif start is not None:
            # Update the doc's milestones
            self.doc.spans["milestones"] = move_milestone(self.doc, spans, start)
        else:
            self.doc.spans["milestones"] = spans
            self._assign_token_attributes(
                self.doc.spans["milestones"], max_label_length
            )
        self.type = "tokens"

    @validate_call(config=validation_config)
    def to_list(self, *, strip_punct: Optional[bool] = True) -> list[dict]:
        """Get a list of milestone dicts.

        Args:
            strip_punct (Optional[bool]): Strip single punctation mark at the end of the character string.

        Returns:
            list[dict]: A list of milestone dicts.

        Note:
            Some language models include a final punctuation mark in the token string,
            particularly at the end of a sentence. The strip_punct argument is a
            somewhat hacky convenience method to remove it. However, the user may wish
            instead to do some post-processing in order to use the output for their
            own purposes.
        """
        milestone_dicts = []
        for span in self.doc.spans["milestones"]:
            start_char = self.doc[span.start].idx
            end_char = start_char + len(span.text)
            chars = self.doc.text[start_char:end_char]
            if strip_punct:
                chars = chars.rstrip(punctuation)
                end_char -= 1
            milestone_dicts.append(
                {
                    "text": span.text,
                    "characters": chars,
                    "start_token": span.start,
                    "end_token": span.end,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

        return milestone_dicts
