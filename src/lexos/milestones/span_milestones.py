"""milestones.py.

Last Update: Jan 14 2025

Span milestones are used to group spans together for analysis or visualization.
Span milestones differ from normal milestones in that milestones are "invisible"
structural boundaries between spans or groups of spans (e.g. sentence or line breaks).
Thus, instead of storing a list of patterns representing milestones, span milestones
store the groups of spans themselves.

Last Update: 12/27/2024
Last Tested: 12/27/2024
"""

from itertools import zip_longest
from string import punctuation
from typing import ClassVar, Iterator, Optional

import spacy
from pydantic import BaseModel, ConfigDict, Field, ValidationError, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, SpanGroup, Token

# from . import helpers, util
# from .util import LexosBaseModel

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
)


class SpanMilestones(BaseModel):
    """Span Milestones class.

    - Referencing the Milestones instance yields an iterator of the spans in the Doc.
    - Referencing Milestones.spans returns an indexed list of spans in the Doc.
    - Referencing milestones.doc.spans["milestones"] returns a SpanGroup.
    """

    doc: Doc | Span = Field(json_schema_extra={"description": "A spaCy Doc object."})
    nlp: str = Field(
        default="xx_sent_ud_sm",
        json_schema_extra={"description": "The language model to use."},
    )

    model_config = validation_config

    def __init__(self, **data) -> None:
        """Set regex flags and milestone IOB extensions after initialization."""
        super().__init__(**data)
        self.doc.spans["milestones"] = []
        if not Token.has_extension("milestone_iob"):
            Token.set_extension("milestone_iob", default="O", force=True)
        if not Token.has_extension("milestone_label"):
            Token.set_extension("milestone_label", default="", force=True)

    @property
    def spans(self) -> list[Span]:
        """Return the milestone Spans.

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

    def _get_list(self, *, strip_punct: Optional[bool] = True) -> list[dict]:
        """Get a list of milestone dicts.

        Args:
            strip_punct (Optional[bool]): Strip single punctation mark at the end of the character string.
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

    def _reset(self) -> None:
        """Reset token attributes."""
        self.doc.spans["milestones"] = []
        for i, _ in enumerate(self.doc):
            self.doc[i]._.milestone_iob = "O"
            self.doc[i]._.milestone_label = ""


class SentenceMilestones(SpanMilestones):
    """Sentence Milestones class.

    - Referencing the Milestones instance yields an iterator of the spans in the Doc.
    - Referencing Milestones.spans returns an indexed list of spans in the Doc.
    - Referencing milestones.doc.spans["milestones"] returns a SpanGroup.
    """

    type: ClassVar[str] = "sentences"

    doc: Doc | Span = Field(json_schema_extra={"description": "A spaCy Doc object."})

    def __init__(self, **data) -> None:
        """Set regex flags and milestone IOB extensions after initialization."""
        super().__init__(**data)
        if not self.doc.has_annotation("SENT_START"):
            raise ValueError(
                "Either the document's model does not parse sentence boundaries or the sentence boundary component has been disabled in the pipeline."
            )
        if not Token.has_extension("milestone_iob"):
            Token.set_extension("milestone_iob", default="O", force=True)
        if not Token.has_extension("milestone_label"):
            Token.set_extension("milestone_label", default="", force=True)

    def reset(self):
        """Reset all `milestone` values to defaults."""
        self._reset()

    @validate_call(config=validation_config)
    def set(
        self, *, step: Optional[int] = 1, max_label_length: Optional[int] = 20
    ) -> None:
        """Generate spans with n sentences per span.

        Args:
            step (Optional[int]): The number of sentences to group under a single milestone
            max_label_length (Optional[int]): The maximum number of characters to include in the label.
        """
        self.reset()
        # Apply the step and set new milestone spans
        sents = list(self.doc.sents)
        segments = [sents[x : x + step] for x in range(0, len(sents), step)]
        self.doc.spans["milestones"] = [
            self.doc[span[0].start : span[-1].end] for span in segments
        ]
        # Set the token attributes
        for span in self.doc.spans["milestones"]:
            self.doc[span.start]._.milestone_iob = "B"
            self.doc[
                span.start
            ]._.milestone_label = f"{span.text[:max_label_length]}{'...' if len(span.text) > max_label_length else ''}"

    @validate_call(config=validation_config)
    def to_list(self, *, strip_punct: Optional[bool] = True) -> list[dict]:
        """Get a list of milestone dicts.

        Args:
            strip_punct (Optional[bool]): Strip single punctation mark at the end of the character string.

        Returns:
            list[dict]: A list of milestone dicts.
        """
        return self._get_list(strip_punct=strip_punct)


class LineMilestones(SpanMilestones):
    """Line Milestones class.

    - Referencing the Milestones instance yields an iterator of the spans in the Doc.
    - Referencing Milestones.spans returns an indexed list of spans in the Doc.
    - Referencing milestones.doc.spans["milestones"] returns a SpanGroup.
    """

    type: ClassVar[str] = "lines"

    doc: Doc | Span = Field(json_schema_extra={"description": "A spaCy Doc object."})

    def __init__(self, **data) -> None:
        """Set regex flags and milestone IOB extensions after initialization."""
        super().__init__(**data)
        if not Token.has_extension("milestone_iob"):
            Token.set_extension("milestone_iob", default="O", force=True)
        if not Token.has_extension("milestone_label"):
            Token.set_extension("milestone_label", default="", force=True)

    def reset(self):
        """Reset all `milestone` values to defaults."""
        self._reset()

    @validate_call(config=validation_config)
    def set(
        self,
        pattern: Optional[str] = "\n",
        *,
        step: Optional[int] = 1,
        remove_linebreak: Optional[bool] = True,
        max_label_length: Optional[int] = 20,
    ) -> list[Span]:
        """Generate spans based on line breaks.

        Args:
            pattern (Optional[str]): The string or regex pattern to use to identify the milestone.
            step (Optional[int]): The number of lines to include in the spans. By default, all lines are included.
            remove_linebreak (Optional[bool]): Whether or not to remove the linebreak character.
            max_label_length (Optional[int]): The maximum number of characters to include in the label.

        Returns:
            list[Span]: A list of spaCy spans.
        """
        self.reset()
        spans = []
        start = 0
        for token in self.doc:
            if token.text == pattern:
                if not remove_linebreak:
                    new_span = self.doc[start : token.i + 1]
                    spans.append(new_span)
                    start = token.i + 1
                else:
                    spans.append(self.doc[start : token.i])
                    start = token.i + 1
        # Append any remaining span
        if start < len(self.doc):
            spans.append(self.doc[start:])

        if step:
            steps = zip_longest(*[iter(spans)] * step, fillvalue=None)
            new_spans = [
                self.doc[
                    min(span.start for span in group if span) : max(
                        span.end for span in group if span
                    )
                ]
                for group in steps
            ]
            group = SpanGroup(self.doc, name="milestones", spans=new_spans)
            self.doc.spans["milestones"] = group
        else:
            self.doc.spans["milestones"] = spans

        # Set the token attributes
        for span in self.doc.spans["milestones"]:
            self.doc[span.start]._.milestone_iob = "B"
            self.doc[
                span.start
            ]._.milestone_label = f"{span.text[:max_label_length]}{'...' if len(span.text) > max_label_length else ''}"

    @validate_call(config=validation_config)
    def to_list(self, *, strip_punct: Optional[bool] = True) -> list[dict]:
        """Get a list of milestone dicts.

        Args:
            strip_punct (Optional[bool]): Strip single punctation mark at the end of the character string.

        Returns:
            list[dict]: A list of milestone dicts.
        """
        return self._get_list(strip_punct=strip_punct)


class CustomMilestones(SpanMilestones):
    """Custom Milestones class.

    - Referencing the Milestones instance yields an iterator of the spans in the Doc.
    - Referencing Milestones.spans returns an indexed list of spans in the Doc.
    - Referencing milestones.doc.spans["milestones"] returns a SpanGroup.
    """

    type: ClassVar[str] = "custom"

    doc: Doc | Span = Field(json_schema_extra={"description": "A spaCy Doc object."})

    def __init__(self, **data) -> None:
        """Set regex flags and milestone IOB extensions after initialization."""
        super().__init__(**data)
        if not Token.has_extension("milestone_iob"):
            Token.set_extension("milestone_iob", default="O", force=True)
        if not Token.has_extension("milestone_label"):
            Token.set_extension("milestone_label", default="", force=True)

    def reset(self):
        """Reset all `milestone` values to defaults."""
        self._reset()

    @validate_call(config=validation_config)
    def set(
        self,
        spans: list[Span],
        *,
        step: Optional[int] = 1,
        max_label_length: Optional[int] = 20,
    ) -> list[Span]:
        """Generate spans based on a custom list.

        Args:
            spans (list[Span]): A list of spaCy spans.
            step (Optional[int]): The number of spans to group into each milestone span. By default, all spans are included.
            max_label_length (Optional[int]): The maximum number of characters to include in the label.

        Returns:
            list[Span]: A list of spaCy spans.
        """
        self.reset()

        if step:
            segments = [
                (spans[i].start, spans[min(i + step, len(spans)) - 1].end)
                for i in range(0, len(spans), step)
            ]
            # Use the segment start and end indexes to generate new spans
            self.doc.spans["milestones"] = [
                self.doc[start:end] for start, end in segments
            ]
        else:
            self.doc.spans["milestones"] = spans

        # Set the token attributes
        for span in self.doc.spans["milestones"]:
            self.doc[span.start]._.milestone_iob = "B"
            self.doc[
                span.start
            ]._.milestone_label = f"{span.text[:max_label_length]}{'...' if len(span.text) > max_label_length else ''}"

    @validate_call(config=validation_config)
    def to_list(self, *, strip_punct: Optional[bool] = True) -> list[dict]:
        """Get a list of milestone dicts.

        Args:
            strip_punct (Optional[bool]): Strip single punctation mark at the end of the character string.

        Returns:
            list[dict]: A list of milestone dicts.
        """
        return self._get_list(strip_punct=strip_punct)
