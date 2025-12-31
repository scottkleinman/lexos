"""util.py.

Utilities for the Milestones class.

Last Updated: 12/21/2024
Last Tested: 12/27/2024
"""

from typing import Any

import spacy
from pydantic import BaseModel, ConfigDict, ValidationError, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token


class LexosBaseModel(BaseModel):
    """Base model inherits from Pydantic base model but validates spaCy objects."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )


def chars_to_tokens(doc: Doc) -> dict[int, int]:
    """Generate a characters to tokens mapping for _match_regex().

    Args:
        doc: A spaCy doc.

    Returns:
        A dict mapping character indexes to token indexes.
    """
    chars_to_tokens = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token.text)):
            chars_to_tokens[i] = token.i
    return chars_to_tokens


def ensure_list(item: Any) -> list[Any]:
    """Ensure that the input is a list.

    Args:
        item (Any): The item to ensure is a list.

    Returns:
        list[Any]: The item as a list.
    """
    if not isinstance(item, list):
        return [item]
    return item


def filter_doc(doc: Doc, spans: list[Span]) -> Doc:
    """Filter a doc to remove tokens by index, retaining custom extensions.

    Args:
        doc: A spaCy doc.
        spans (list[Span]): The span(s) to remove from the doc.

    Returns:
        A new doc with the spans removed.
    """
    # Check if the milestone extensions have been set
    if not Token.has_extension("milestone_iob") or not Token.has_extension(
        "milestone_label"
    ):
        raise ValueError("The milestone extensions have not been set.")
    # Get the start ids of the original milestones
    remove_ids = [token.i for span in spans for token in span]
    ms_start_ids = [span.start for span in spans]
    # Get the user data from the original doc; remove milestones
    new_user_data = {
        key: value
        for key, value in doc.user_data.items()
        if not isinstance(key, tuple)
        and key[1] not in ["milestone_iob", "milestone_label"]
    }
    # Make a new doc without the tokens in remove_ids
    words = [token.text for token in doc if token.i not in remove_ids]
    spaces = [token.whitespace_ == " " for token in doc if token.i not in remove_ids]
    new_doc = Doc(doc.vocab, words=words, spaces=spaces)
    # Replace the new doc's user data with the modified user data from the original doc
    new_doc.user_data = new_user_data
    # Set the milestone IOB to "B" for the tokens that were the start of milestones in the original doc
    new_doc.spans["milestones"] = []
    for token in new_doc:
        if token.i in ms_start_ids:
            token._.milestone_iob = "B"
            new_doc.spans["milestones"].append(new_doc[token.i : token.i + 1])
    return new_doc


def lowercase_spacy_rules(
    patterns: list[list[dict[str, Any]]],
    old_key: list[str] | str = ["TEXT", "ORTH"],
    new_key: str = "LOWER",
) -> list:
    """Convert spaCy Rule Matcher patterns to lowercase.

    Args:
        patterns: A list of spacy Rule Matcher patterns.
        old_key: A dictionary key or list of keys to rename.
        new_key: The new key name.

    Returns:
        A list of spaCy Rule Matcher patterns.
    """

    def convert(key):
        if key in old_key:
            return new_key
        else:
            return key

    if isinstance(patterns, dict):
        new_dict = {}
        for key, value in patterns.items():
            value = lowercase_spacy_rules(value)
            key = convert(key)
            new_dict[key] = value
        return new_dict
    if isinstance(patterns, list):
        new_list = []
        for value in patterns:
            new_list.append(lowercase_spacy_rules(value))
        return new_list
    return patterns


def move_milestone(doc: Doc, spans: list[Span], start: str) -> list[Span]:
    """Move the milestone start to a new token index.

    Args:
        doc: A spaCy doc.
        spans (list[Span]): The span(s) to use for identifying token attributes.
        start (str): Set milestone start to the token before or after the milestone span. May be "before" or "after".

    Returns:
        A list of new milestone spans.
    """
    # Do not process spans at the beginning or end of the doc
    if start == "before":
        spans = [span for span in spans if span.start > 0]
    if start == "after":
        spans = [span for span in spans if span.end < len(doc)]
    new_milestones = []
    for span in spans:
        # Reset the current milestone IOB and label
        for token in span:
            doc[token.i]._.milestone_iob = "O"
            doc[token.i]._.milestone_label = ""
        # Set the following token's IOB to "B" and add the following token to a list of new milestones
        try:
            if start == "after":
                doc[span.end]._.milestone_iob = "B"
                new_milestones.append(doc[span.end : span.end + 1])
            elif start == "before" and span.start > 0:
                doc[span.start - 1]._.milestone_iob = "B"
                new_milestones.append(doc[span.start - 1 : span.start])
        except IndexError:
            pass
    return new_milestones
