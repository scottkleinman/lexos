"""ginsu.py.

To do:
    - Document milestone usage.
"""

import re
from typing import Callable, List, Optional, Union

import spacy

# Deprecated: from pydantic import BaseModel, ValidationError, validator
from pydantic import BaseModel
from spacy.tokens import Doc

from lexos.exceptions import LexosException


class SplitMilestoneModel(BaseModel):
    """Validate the input for split functions."""

    docs: Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]]
    milestone: Union[dict, str]
    preserve_milestones: Optional[bool] = True

    class Config:
        """Config for SplitMilestoneModel."""

        arbitrary_types_allowed = True


class SplitModel(BaseModel):
    """Validate the input for split functions."""

    docs: Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]]
    n: Optional[int] = 1000
    merge_threshold: Optional[float] = 0.5
    overlap: Optional[int] = None

    class Config:
        """Config for SplitModel."""

        arbitrary_types_allowed = True


class Ginsu:
    """Codename Ginsu.

    https://www.youtube.com/watch?v=Sv_uL1Ar0oM.

    Note: Does not work on wood or watermelons.

    To do:
        - Allow the user to set token._.is_milestone on the fly.
    """

    def __init__(self, config: dict = None):
        """Initialize the class."""
        self.config = config

    def _chunk_doc(self, doc: list, n: int = 1000) -> Callable:
        """Yield successive n-sized chunks from a spaCy doc by a fixed number of tokens.

        Args:
            docs (list): A list of spaCy docs.
            n (int): The number of tokens to split on.

        Returns:
            list: A list of spaCy docs.
        """
        for i in range(0, len(doc), n):
            yield doc[i : i + n].as_doc()

    def _create_overlapping_segments(
        self,
        segments: List[spacy.tokens.doc.Doc],
        overlap: int,
    ) -> List[spacy.tokens.doc.Doc]:
        """Create overlapping segments.

        Args:
            segments (List[spacy.tokens.doc.Doc]): A list of spaCy docs.
            overlap (int): The number of tokens to overlap.

        Returns:
            List[spacy.tokens.doc.Doc]: A list of spaCy docs.
        """
        overlapped_segs = []
        for i, seg in enumerate(segments):
            if i == 0:
                # Get the first overlap tokens from the second segment
                overlapped_segs.append(
                    Doc.from_docs([seg, segments[i + 1][:overlap].as_doc()])
                )
            else:
                if i < len(segments) - 1:
                    # Get the last overlap tokens from the previous segment
                    # and the first from the next segment
                    overlapped_segs.append(
                        Doc.from_docs(
                            [
                                segments[i - 1][-overlap:].as_doc(),
                                seg,
                                segments[i + 1][:overlap].as_doc(),
                            ]
                        )
                    )
                else:
                    # Get the last overlap tokens from the previous segment
                    overlapped_segs.append(
                        Doc.from_docs([segments[i - 1][-overlap:].as_doc(), seg])
                    )
        return overlapped_segs

    def _get_milestone_result(
        self, attr: str, token: object, value: Union[str, tuple]
    ) -> bool:
        """Test a token for a match.

        If value is a tuple, it must have the form `(pattern, operator)`,
        where pattern is the string or regex pattern to match, and
        operator is the method to use. Valid operators are "in", "not_in",
        "starts_with", "ends_with", "re_match", and "re_search".
        The prefix "re_" implies that the pattern is a regex, and either
        `re.match` or `re.search` will be used.

        Args:
            attr (str): The spaCy token attribute to test.
            token (object): The token to test.
            value (Union[str, tuple]): The value to test.

        Returns:
            bool: Whether the token matches the query.
        """
        if attr == "is_milestone":
            if token._.is_milestone == True:
                return True
            else:
                return False
        elif isinstance(value, str) or isinstance(value, bool):
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

    def _matches_milestone(
        self, token: object, milestone: Union[dict, list, str]
    ) -> bool:
        """Test a token for a match.

        Args:
            token (object): The token to test.
            milestone (Union[dict, str]): A variable representing the value(s) to be matched.

        Returns:
            bool: Whether the token matches the query.
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
            token (object): The token to test.
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

    def _split_doc(
        self,
        doc: spacy.tokens.doc.Doc,
        n: int = 1000,
        merge_threshold: float = 0.5,
        overlap: int = None,
    ) -> list:
        """Split a spaCy doc into chunks by a fixed number of tokens.

        Args:
            doc (spacy.tokens.doc.Doc): A spaCy doc.
            n (int): The number of tokens to split on.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.

        Returns:
            list: A list of spaCy docs.
        """
        segments = list(self._chunk_doc(doc, n))
        # Apply the merge threshold
        if len(segments[-1]) < n * merge_threshold:
            last_seg = segments.pop(-1)
            # Combine the last two segments into a single doc
            segments[-1] = Doc.from_docs([segments[-1], last_seg])
        if overlap:
            return self._create_overlapping_segments(segments, overlap)
        else:
            return segments

    def _splitn_doc(
        self,
        doc: spacy.tokens.doc.Doc,
        n: int = 2,
        merge_threshold: float = 0.5,
        overlap: int = None,
    ) -> list:
        """Get a specific number of sequential segments from a spaCy doc.

        Args:
            doc (spacy.tokens.doc.Doc): A spaCy doc.
            n (int): The number of segments to create.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.

        Returns:
            list: A list of spaCy doc segments.

        Note:
            For this implementation, see https://stackoverflow.com/a/54802737.
            See `split()` for more information on the validation model.
        """
        # Validate input
        try:
            model = SplitModel(
                docs=doc, n=n, merge_threshold=merge_threshold, overlap=overlap
            )
        except Exception as e:
            raise LexosException(e)
        # Get the number of tokens per segment (d) and the remaining tokens (r)
        d, r = divmod(len(doc), model.n)

        # Get the segments
        segments = []
        for i in range(model.n):
            index = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
            segments.append(doc[index : index + (d + 1 if i < r else d)].as_doc())
            # Apply the merge threshold
            if len(segments[-1]) < model.n * model.merge_threshold:
                last_seg = segments.pop(-1)
                # Combine the last two segments into a single doc
                segments[-1] = Doc.from_docs([segments[-1], last_seg])
        if overlap:
            segments = [
                self._create_overlapping_segments(segment, model.overlap)
                for segment in segments
            ]
        # Convert the list of list segments to a list of spaCy doc segments
        segmented_doc = []
        for segment in segments:
            if isinstance(segment, spacy.tokens.doc.Doc):
                segmented_doc.append(segment)
            else:
                segmented_doc.append(segment.as_doc())
        return segmented_doc

    def _split_doc_on_milestones(
        self,
        doc: spacy.tokens.doc.Doc,
        milestone: Union[dict, str],
        preserve_milestones: bool = True,
    ):
        """Split document on a milestone.

        Args:
            doc (spacy.tokens.doc.Doc): The document to be split.
            milestone (Union[dict, str]): A variable representing the value(s) to be matched.
            preserve_milestones (bool): If True, the milestone token will be preserved at the
                beginning of every segment. Otherwise, it will be deleted.
        """
        segments = []
        indices = [
            i for i, x in enumerate(doc) if self._matches_milestone(x, milestone)
        ]
        for start, end in zip([0, *indices], [*indices, len(doc)]):
            if preserve_milestones:
                segments.append(doc[start:end].as_doc())
            else:
                segments.append(doc[start + 1 : end].as_doc())
        return segments

    def merge(self, segments: List[spacy.tokens.doc.Doc]) -> str:
        """Merge a list of segments into a single string.

        Args:
            segments (List[spacy.tokens.doc.Doc]): The list of segments to merge.

        Returns:
            spacy.tokens.doc.Doc: The merged doc.
        """
        return Doc.from_docs(segments)

    def split(
        self,
        docs: Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]],
        n: int = 1000,
        merge_threshold: float = 0.5,
        overlap: int = None,
    ) -> List[Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]]]:
        """Split spaCy docs into chunks by a fixed number of tokens.

        Args:
            docs (Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]]): A spaCy doc or list of spaCy docs.
            n (int): The number of tokens to split on.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.

        Returns:
            List[Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]]]: A list of spaCy docs (segments) for
            the input doc or a list of segment lists for multiple docs.

        Note:
            `n`, `merge_threshold`, and `overlap` are referenced from the validated
            model in case Pydantic has coerced them into the expected data types.
        """
        # Validate input
        try:
            model = SplitModel(
                docs=docs, n=n, merge_threshold=merge_threshold, overlap=overlap
            )
        except ValidationError as e:
            raise LexosException(e)

        # Handle single docs
        if isinstance(docs, spacy.tokens.doc.Doc):
            return self._split_doc(docs, model.n, model.merge_threshold, model.overlap)
        # Handle multiple docs
        else:
            all_segments = []
            for doc in docs:
                all_segments.append(
                    self._split_doc(doc, model.n, model.merge_threshold, model.overlap)
                )
            return all_segments

    def splitn(
        self,
        docs: Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]],
        n: int = 2,
        merge_threshold: float = 0.5,
        overlap: int = None,
    ) -> list:
        """Get a specific number of sequential segments from a spaCy doc or docs.

        Args:
            docs (Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]]): A spaCy doc or list of spaCy docs.
            n (int): The number of segments to create.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.

        Returns:
            list: A list of lists with where the inner list is the resulting segments
            for each doc.
        Note:
            For this implementation, see https://stackoverflow.com/a/54802737.
            See `split()` for more information on the validation model.
        """
        # Validate input
        try:
            model = SplitModel(
                docs=docs, n=n, merge_threshold=merge_threshold, overlap=overlap
            )
        except ValidationError as e:
            raise LexosException(e)

        # Handle single docs
        if isinstance(docs, spacy.tokens.doc.Doc):
            return self._splitn_doc(docs, model.n, model.merge_threshold, model.overlap)
        # Handle multiple docs
        else:
            all_segments = []
            for doc in docs:
                all_segments.append(
                    self._splitn_doc(doc, model.n, model.merge_threshold, model.overlap)
                )
            return all_segments

    def split_on_milestones(
        self,
        docs: Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]],
        milestone: Union[dict, str],
        preserve_milestones: bool = True,
    ):
        """Split document on a milestone.

        Args:
            docs (Union[spacy.tokens.doc.Doc, List[spacy.tokens.doc.Doc]]): The document(s) to be split.
            milestone (Union[dict, str]): A variable representing the value(s) to be matched.
            preserve_milestones (bool): If True, the milestone token will be preserved at the
                beginning of every segment. Otherwise, it will be deleted.
        """
        # Validate input
        try:
            _ = SplitMilestoneModel(
                docs=docs, milestone=milestone, preserve_milestones=preserve_milestones
            )
        except ValidationError as e:
            raise LexosException(e)
        # Handle single docs
        if isinstance(docs, spacy.tokens.doc.Doc):
            return self._split_doc_on_milestones(docs, milestone, preserve_milestones)
        # Handle multiple docs
        else:
            all_segments = []
            for doc in docs:
                all_segments.append(
                    self._split_doc_on_milestones(doc, milestone, preserve_milestones)
                )
            return all_segments
