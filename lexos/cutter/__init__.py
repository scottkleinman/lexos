"""__init.py."""

import re
from typing import Callable, List, Union
from lexos.cutter import registry


class Ginsu:
    """Codename Ginsu.

    https://www.youtube.com/watch?v=Sv_uL1Ar0oM.

    Note: Does not work on wood or watermelons.

    To do:
        - Allow the user to set token._.is_milestone on the fly.
        - StringSplit and FileSplit classes
    """

    def __init__(self, config: dict = None):
        """Initialize the class."""
        self.config = config

    def split_doc(
        self,
        doc: object,
        n: int = 1000,
        merge_threshold: float = 0.5,
        overlap: int = None,
    ) -> list:
        """Split a spaCy doc into chunks by a fixed number of tokens.

        Args:
            doc (object): A spaCy doc.
            n (int): The number of tokens to split on.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.

        Returns:
            list: A list of spaCy docs.
        """
        from spacy.tokens import Doc

        if overlap:
            segments = self.split_overlap(doc, n, merge_threshold, overlap)
        else:
            # Get the segments
            segments = list(self._chunk_doc(doc, n))
            # Apply the merge threshold
            if len(segments[-1]) < n * merge_threshold:
                last_seg = segments.pop(-1)
                # Combine the last two segments into a single doc
                segments[-1] = Doc.from_docs([segments[-1], last_seg])
        return segments

    def split(
        self, docs, n=1000, merge_threshold: float = 0.5, overlap: int = None
    ) -> list:
        """Split spaCy docs into chunks by a fixed number of tokens.

        Args:
            doc (object): A spaCy doc or list of spaCy docs.
            n (int): The number of tokens to split on.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.

        Returns:
            list: A list of lists spaCy docs (segments) for each input doc.
        """
        from spacy.tokens import Doc

        if not isinstance(docs, list):
            docs = [docs]
        all_segments = []
        for doc in docs:
            # Get segments with overlap
            if overlap:
                all_segments.append(
                    self._split_overlap(doc, n, merge_threshold, overlap)
                )
            # Get segments without overlap
            else:
                segments = list(self._chunk_doc(doc, n))
                # Apply the merge threshold
                if len(segments[-1]) < n * merge_threshold:
                    last_seg = segments.pop(-1)
                    # Combine the last two segments into a single doc
                    segments[-1] = Doc.from_docs([segments[-1], last_seg])
                all_segments.append(segments)
        return all_segments

    def splitn(
        self,
        docs: Union[list, object],
        n: int = 2,
        merge_threshold: float = 0.5,
        overlap: int = None,
        force_initial_overlap: bool = False,
    ) -> list:
        """Get a specific number of sequential segments from a spaCy doc or docs.

        Args:
            docs(Union[list, object]): A spaCy doc or list of spaCy docs.
            n (int): The number of segments to create. Calculated automatically.
            merge_threshold (float): ...
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.
            force_initial_overlap (bool): Force the first segment to contain
                an overlap, even if it is longer than the other segments.

        Returns:
            list: A list of lists with where the inner list is the resulting segments
            for each doc.
        Note:
            For this implementation, see https://stackoverflow.com/a/54802737.
        """
        if not isinstance(docs, list):
            docs = [docs]
        segments = []
        for doc in docs:
            # Get the number of tokens per segment (d) and the remaining tokens (r)
            d, r = divmod(len(doc), n)
            # Split with overlap
            if overlap:
                doc_segments = self._split_overlap(
                    doc,
                    d,
                    merge_threshold=merge_threshold,
                    overlap=overlap,
                    num_segments=n,
                    force_initial_overlap=force_initial_overlap,
                )
            # Split without overlap
            else:
                doc_segments = []
                for i in range(n):
                    index = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
                    doc_segments.append(doc[index : index + (d + 1 if i < r else d)])
            # Append the doc segments to the list for all docs
            segments.append(doc_segments)
        return segments

    def split_on_milestones(
        self, doc: object, milestone: Union[dict, str], preserve_milestone: bool = True
    ):
        """Split document on a milestone.

        Args:
            doc (object): The document to be split.
            milestone (Union[dict, str]): A variable representing the value(s) to be matched.
            preserve_milestone (bool): If True, the milestone token will be preserved at the
                beginning of every segment. Otherwise, it will be deleted.
        """
        segments = []
        indices = [
            i for i, x in enumerate(doc) if self._matches_milestone(x, milestone)
        ]
        for start, end in zip([0, *indices], [*indices, len(doc)]):
            if preserve_milestone:
                segments.append(doc[start:end])
            else:
                segments.append(doc[start + 1 : end])
        return segments

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
            attr (str): The attribute to test.
            token (object): The token to test.
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

    def _split_overlap(
        self,
        doc: object,
        segment_size: int,
        merge_threshold: float = None,
        overlap: int = None,
        num_segments: int = 1,
        force_initial_overlap: bool = False,
    ) -> List[list]:
        """Split a doc into segments.

        Calculates the number of segments with the overlap value;
        then uses it as indexing to capture all the sub-lists
        with the `get_single_seg` helper function.

        Args:
            doc (object): The input doc.
            segment_size (int): The size of the segment.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.
            num_segments (int): The number of segments to return, if known.
            force_initial_overlap (bool): Force the first segment to contain
                an overlap, even if it is longer than the other segments.

        Returns:
            list: A list of doc segments.
        """
        # Distance between starts of adjacent segments
        seg_start_distance = segment_size - overlap

        # Length of the token list except the last segment
        length_except_last = len(doc) - segment_size * merge_threshold

        # The total number of segments after cut, including the last
        if num_segments != 1:
            num_segments = int(length_except_last / seg_start_distance)
            num_segments += 1

            def get_single_seg(index: int, is_last_seg: bool) -> list:
                """Get a list of segments with index.

                Merge the last segment if it is within the threshold.

                Args:
                    is_last_seg (bool): Whether the segment is the last one.
                    index: The index of the segment in the final segment list.

                Returns:
                    A list of segments as spaCy docs.
                """
                # Define the current segment size
                if is_last_seg:
                    spans = doc[seg_start_distance * index :]
                else:
                    spans = doc[
                        seg_start_distance * index : seg_start_distance * index
                        + segment_size
                    ]
                return spans.as_doc()

        # Return the list of segments, evaluating for last segment
        segments = [
            get_single_seg(
                index=index, is_last_seg=True if index == num_segments - 1 else False
            )
            for index in range(num_segments - 1)
        ]
        if force_initial_overlap:
            segments[0] = doc[0 : segment_size + overlap]
        return segments


class Machete:
    """Codename Machete."""

    def __init__(self, tokenizer: str = "whitespace"):
        """Initialize the class."""
        self.tokenizer = tokenizer

    def tokenize(self, text: str, tokenizer: str = None) -> list:
        """Tokenize an input string without a language model.

        Loads a tokenizer function from the registry.

        Args:
            text (str): The input string.

        Returns:
            list: A list of tokens.
        """
        if not tokenizer:
            tokenizer = registry.load(self.tokenizer)
        else:
            try:
                tokenizer = registry.load(tokenizer)
            except ValueError:
                raise ValueError(
                    "The specified tokenizer could not be found in the tokenizer registry."
                    ""
                )
        return tokenizer(text)

    def split_list(
        self,
        token_list: list,
        n: int = 1000,
        merge_threshold: float = 0.5,
        overlap: int = None,
    ) -> list:
        """Split a list into chunks by a fixed number of tokens.

        Args:
            token_list (list): A list of tokens.
            n (int): The number of tokens to split on.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.

        Returns:
            list: A list of token lists.
        """
        if overlap:
            segments = self.split_overlap(token_list, n, merge_threshold, overlap)
        else:
            # Get the segments
            segments = list(self._chunk_doc(token_list, n))
            # Apply the merge threshold
            if len(segments[-1]) < n * merge_threshold:
                last_seg = segments.pop(-1)
                # Combine the last two segments into a single list
                segments[-1] = [segments[-1] + last_seg]
        return segments

    def split(
        self,
        texts: Union[List[str], str],
        n=1000,
        merge_threshold: float = 0.5,
        overlap: int = None,
        tokenizer: str = None,
        as_string: bool = True,
    ) -> list:
        """Split texts into chunks by a fixed number of tokens.

        Args:
            texts (Union[List[str], str]): A text string or list of text strings.
            n (int): The number of tokens to split on.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.
            tokenizer (str): The name of the tokenizer function to use.
            as_string (bool): Whether to return the segments as a list of strings.

        Returns:
            list: A list of lists or strings (segments) for each text.
        """
        # Ensure a list of texts as the starting point
        if not isinstance(texts, list):
            texts = [texts]

        # Process the texts into segments
        all_segments = []
        for text in texts:
            # Tokenise the text
            tokens = self.tokenize(text, tokenizer=tokenizer)

            # Get segments with overlap
            if overlap:
                all_segments.append(
                    self._split_overlap(tokens, n, merge_threshold, overlap)
                )
            # Get segments without overlap
            else:
                segments = list(self._chunk_tokens(tokens, n))
                # Apply the merge threshold
                if len(segments[-1]) < n * merge_threshold:
                    last_seg = segments.pop(-1)
                    # Combine the last two segments into a single list
                    segments[-1] = segments[-1] + last_seg
            all_segments.append(segments)

        # Return the segments as strings or lists
        if as_string:
            return [["".join(segment) for segment in text] for text in all_segments]
        else:
            return all_segments

    def splitn(
        self,
        texts: Union[List[str], str],
        n: int = 2,
        merge_threshold: float = 0.5,
        overlap: int = None,
        force_initial_overlap: bool = False,
        tokenizer: str = None,
        as_string: bool = True,
    ) -> list:
        """Get a specific number of sequential segments from a spaCy doc or docs.

        Args:
            texts (Union[List[str], str]): A text string or list of text strings.
            n (int): The number of segments to create. Calculated automatically.
            merge_threshold (float): ...
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.
            force_initial_overlap (bool): Force the first segment to contain
                an overlap, even if it is longer than the other segments.
            tokenizer (str): The name of the tokenizer function to use.
            as_string (bool): Whether to return the segments as a list of strings.

        Returns:
            list: A list of lists or strings (segments) for each text.

        Note:
            For this implementation, see https://stackoverflow.com/a/54802737.
        """
        # Ensure a list of texts as the starting point
        if not isinstance(texts, list):
            texts = [texts]

        # Process the texts into segments
        all_segments = []
        for text in texts:

            # Tokenise the text
            tokens = self.tokenize(text, tokenizer=tokenizer)

            # Get the number of tokens per segment (d) and the remaining tokens (r)
            d, r = divmod(len(tokens), n)

            # Split with overlap
            if overlap:
                segment_list = self._split_overlap(
                    tokens,
                    d,
                    merge_threshold=merge_threshold,
                    overlap=overlap,
                    num_segments=n,
                    force_initial_overlap=force_initial_overlap,
                )
            # Split without overlap
            else:
                segment_list = []
                for i in range(n):
                    index = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
                    segment_list.append(tokens[index : index + (d + 1 if i < r else d)])
            # Append the text segments to the list for all texts
            # all_segments.append(segment_list)
            all_segments = segment_list

        # Return the segments as strings or lists
        if as_string:
            return ["".join(segment) for segment in all_segments]
            # return [["".join(segment) for segment in text] for text in all_segments]
        else:
            return all_segments

    def _chunk_tokens(self, tokens: list, n: int = 1000) -> Callable:
        """Yield successive n-sized chunks from a list by a fixed number of tokens.

        Args:
            tokens (list): A list of tokens.
            n (int): The number of tokens to split on.

        Returns:
            list: A list of token lists (segments).
        """
        for i in range(0, len(tokens), n):
            yield tokens[i : i + n]

    def _split_overlap(
        self,
        tokens: list,
        segment_size: int,
        merge_threshold: float = None,
        overlap: int = None,
        num_segments: int = 1,
        force_initial_overlap: bool = False,
    ) -> List[list]:
        """Split a list of tokens into segments.

        Calculates the number of segments with the overlap value;
        then uses it as indexing to capture all the segments
        with the `get_single_seg` helper function.

        Args:
            tokens (list): The input tokens.
            segment_size (int): The size of the segment.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.
            num_segments (int): The number of segments to return, if known.
            force_initial_overlap (bool): Force the first segment to contain
                an overlap, even if it is longer than the other segments.

        Returns:
            list: A list of segments for each text.
        """
        # Distance between starts of adjacent segments
        seg_start_distance = segment_size - overlap

        # Length of the token list except the last segment
        length_except_last = len(tokens) - segment_size * merge_threshold

        # The total number of segments after cut, including the last
        if num_segments != 1:
            num_segments = int(length_except_last / seg_start_distance)
            num_segments += 1

            def get_single_seg(index: int, is_last_seg: bool) -> list:
                """Get a list of segments with index.

                Merge the last segment if it is within the threshold.

                Args:
                    is_last_seg (bool): Whether the segment is the last one.
                    index (int): The index of the segment in the final segment list.

                Returns:
                    A list of segments as spaCy docs.
                """
                # Define the current segment size
                if is_last_seg:
                    spans = tokens[seg_start_distance * index :]
                else:
                    spans = tokens[
                        seg_start_distance * index : seg_start_distance * index
                        + segment_size
                    ]
                return spans.as_doc()

        # Return the list of segments, evaluating for last segment
        segments = [
            get_single_seg(
                index=index, is_last_seg=True if index == num_segments - 1 else False
            )
            for index in range(num_segments - 1)
        ]
        if force_initial_overlap:
            segments[0] = tokens[0 : segment_size + overlap]

        return segments
