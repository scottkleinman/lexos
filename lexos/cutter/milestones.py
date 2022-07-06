"""milestones.py.

Usage:

    from lexos.cutter.milestones import Milestones
    milestones = "chapter"
    Milestones().set(docs, milestones)

Once milestones are set, they can be accessed with
`token._.is_milestone`. The cutter `split_on_milestones`
method can be used to split documents using this information.
"""

import re
from typing import List, Union

import spacy
from spacy.tokens import Token


class Milestones:
    """Milestones class."""

    def __init__(self, config: dict = None):
        """Initialise a Milestones object.

        Args:
            config (dict): Arbitrary configuration.
        """
        self.config = config

    def set(
        self,
        docs: Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]],
        milestone: Union[dict, str],
    ) -> Union[List[object], object]:
        """Set the milestones for a doc or a list of docs.

        Args:
            docs (Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]]): A spaCy doc or a list of spaCy docs.
            milestone (Union[dict, str]): The milestone token(s) to match.

        Returns:
            Union[List[spacy.tokens.doc.Doc], spacy.tokens.doc.Doc]: A spaCy doc or list of spacy docs with
                `doc._.is_milestone` set.
        """
        # Holder for processed docs
        result = []
        # Make sure single docs are in a list
        if not isinstance(docs, list):
            docs = [docs]
        # Set milestones on each doc
        for doc in docs:
            result.append(self._set_milestones(doc, milestone))
        # Return the processed docs list or a single processed doc
        if len(result) == 1:
            return result[0]
        else:
            return result

    def _set_milestones(
        self, doc: spacy.tokens.doc.Doc, milestone: str
    ) -> spacy.tokens.doc.Doc:
        """Set the milestones for a doc.

        Args:
            doc (spacy.tokens.doc.Doc): A spaCy doc.
            milestone (str): The milestone token(s) to match.

        Returns:
            object: A spaCy doc with `doc._.is_milestone` set.
        """
        # If the doc tokens do not have `is_milestone`, set them all to False
        if not doc[0].has_extension("is_milestone"):
            Token.set_extension("is_milestone", default=False, force=True)
        # Go through the doc, setting `is_milestone` for each match
        for token in doc:
            if self._matches_milestone(token, milestone):
                token._.is_milestone = True
            else:
                token._.is_milestone = False
        return doc

    def _matches_milestone(
        self, token: spacy.tokens.token.Token, milestone: Union[dict, list, str]
    ) -> bool:
        """Check if a token matches a milestone.

        Args:
            token (spacy.tokens.token.Token): The token to test.
            milestone (Union[dict, list, str]): The milestone token(s) to match.

        Returns:
            bool: Whether the token matches the milestone.
        """
        if isinstance(milestone, str):
            if token.text == milestone:
                return True
            else:
                return False
        elif isinstance(milestone, list):
            if token.text in milestone:
                return True
            else:
                return False
        elif isinstance(milestone, dict):
            return self._parse_milestone_dict(token, milestone)

    def _parse_milestone_dict(self, token, milestone_dict):
        """Parse a milestone dictionary and get results for each criterion.

        Key-value pairs in `milestone_dict` will be interpreted as token
        attributes and their values. If the value is given as a tuple, it
        must have the form `(pattern, operator)`, where the pattern is the
        string or regex pattern to match, and the operator is the matching
        method to use. Valid operators are "in", "not_in", "starts_with",
        "ends_with", "re_match", and "re_search". The prefix "re_" implies
        that the pattern is a regex, and either `re.match` or `re.search`
        will be used.

        Args:
            token (spacy.tokens.token.Token): The token to test.
            milestone_dict (dict): A dict in the format given above.

        Returns:
            bool: Whether the token matches the query.
        """
        # Get lists
        and_ = milestone_dict.get("and", {})
        or_ = milestone_dict.get("or", {})
        and_valid = True
        or_valid = False

        # Iterate through the and_ list
        for query_dict in and_:
            # Get the attribute and value
            attr, value = list(query_dict.items())[0]
            # The token fails to satisfy all criteria
            if self._get_milestone_result(attr, token, value):
                and_valid = True
            else:
                and_valid = False

        # Iterate through the or_ list
        for query_dict in or_:
            # Get the attribute and value
            attr, value = list(query_dict.items())[0]
            # The token satisfies at least one criterion
            if self._get_milestone_result(attr, token, value):
                or_valid = True

        # Determine if there is a match with "and" and "or"
        if and_valid and or_valid:
            is_match = True
        elif and_valid and not or_valid:
            is_match = True
        elif not and_valid and or_valid:
            is_match = True
        else:
            is_match = False

        # Handle keywords other than "and" and "or"
        for attr, value in milestone_dict.items():
            if attr not in ["and", "or"]:
                if self._get_milestone_result(attr, token, value):
                    is_match = True
                else:
                    is_match = False

        # Return the result
        return is_match

    def _get_milestone_result(
        self, attr: str, token: spacy.tokens.token.Token, value: Union[str, tuple]
    ) -> bool:
        """Test a token for a match.

        If value is a tuple, it must have the form `(pattern, operator)`,
        where pattern is the string or regex pattern to match, and
        operator is the method to use. Valid operators are "in", "not_in",
        "starts_with", "ends_with", "re_match", and "re_search".
        The prefix "re_" implies that the pattern is a regex, and either
        `re.match` or `re.search` will be used.

        Args:
            attr (str): The attribute to test.
            token (spacy.tokens.token.Token): The token to test.
            value (Union[str, tuple]): The value to test.

        Returns:
            bool: Whether the token matches the query.
        """
        if isinstance(value, str):
            if getattr(token, attr) == value:
                return True
            else:
                return False
        elif isinstance(value, tuple):
            pattern = value[0]
            operator = value[1]
            if operator == "in":
                if getattr(token, attr) in pattern:
                    return True
                else:
                    return False
            elif operator == "not_in":
                if getattr(token, attr) not in pattern:
                    return True
                else:
                    return False
            elif operator == "starts_with":
                if getattr(token, attr).startswith(pattern):
                    return True
                else:
                    return False
            elif operator == "ends_with":
                if getattr(token, attr).endswith(pattern):
                    return True
                else:
                    return False
            elif operator == "re_match":
                if re.match(pattern, getattr(token, attr)):
                    return True
                else:
                    return False
            elif operator == "re_search":
                if re.search(pattern, getattr(token, attr)):
                    return True
                else:
                    return False
