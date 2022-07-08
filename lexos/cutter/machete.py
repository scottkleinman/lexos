"""machete.py.

To do:
    - Add regex milestone support.
"""
import re
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, ValidationError, validator

from lexos.cutter import registry
from lexos.exceptions import LexosException


class SplitListModel(BaseModel):
    """Validate the input for split_list function."""

    text: List[str]
    n: Optional[int] = 1000
    merge_threshold: Optional[float] = 0.5
    overlap: Optional[int] = None


class SplitMilestoneModel(BaseModel):
    """Validate the input for split_on_miletone function."""

    texts: Union[List[str], str]
    milestone: Union[dict, str]
    preserve_milestones: Optional[bool] = False
    tokenizer: Optional[str] = None
    as_string: Optional[bool] = True


class SplitModel(BaseModel):
    """Validate the input for split functions."""

    texts: Union[List[str], str]
    n: Optional[int] = 1000
    merge_threshold: Optional[float] = 0.5
    overlap: Optional[int] = None
    tokenizer: Optional[str] = None
    as_string: Optional[bool] = True


class Machete:
    """Codename Machete."""

    def __init__(self, tokenizer: str = "whitespace"):
        """Initialize the class."""
        self.tokenizer = tokenizer

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

    def _create_overlapping_segments(
        self, segments: List[str], overlap: int
    ) -> List[str]:
        """Create overlapping segments.

        Args:
            segments (List[str]): A list of token strings.
            overlap (int): The number of tokens to overlap.

        Returns:
            List[str]: A list of token strings.
        """
        overlapped_segs = []
        for i, seg in enumerate(segments):
            if i == 0:
                # Get the first overlap tokens from the second segment
                overlapped_segs.append(seg + segments[i + 1][:overlap])
            else:
                if i < len(segments) - 1:
                    # Get the last overlap tokens from the previous segment
                    overlapped_segs.append(seg + segments[i + 1][:overlap])
                else:
                    # Get the last segment
                    overlapped_segs.append(seg)
        return overlapped_segs

    def _tokenize(self, text: str, tokenizer: str = None) -> list:
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
                raise LexosException(
                    "The specified tokenizer could not be found in the tokenizer registry."
                )
        return tokenizer(text)

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
        # Validate input
        try:
            model = SplitModel(
                texts=texts,
                n=n,
                merge_threshold=merge_threshold,
                overlap=overlap,
                tokenizer=tokenizer,
                as_string=as_string,
            )
        except Exception as e:
            raise LexosException(e)

        # Ensure a list of texts as the starting point
        if not isinstance(model.texts, list):
            model.texts = [model.texts]

        # Process the texts into segments
        all_segments = []
        for text in model.texts:
            # Tokenise the text
            tokens = self._tokenize(text, tokenizer=model.tokenizer)
            segments = list(self._chunk_tokens(tokens, model.n))
            # Apply the merge threshold
            if len(segments[-1]) < model.n * model.merge_threshold:
                last_seg = segments.pop(-1)
                # Combine the last two segments into a single list
                segments[-1] = segments[-1] + last_seg
            all_segments.append(segments)
        if overlap:
            all_segments = [
                self._create_overlapping_segments(segment, overlap)
                for segment in all_segments
            ]
        if as_string:
            all_segments = [
                ["".join(segment) for segment in text] for text in all_segments
            ]
        return all_segments

    def splitn(
        self,
        texts: Union[List[str], str],
        n: int = 2,
        merge_threshold: float = 0.5,
        overlap: int = None,
        tokenizer: str = None,
        as_string: bool = True,
    ) -> list:
        """Get a specific number of sequential segments from a string or list of strings.

        Args:
            texts (Union[List[str], str]): A text string or list of text strings.
            n (int): The number of segments to create. Calculated automatically.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.
            tokenizer (str): The name of the tokenizer function to use.
            as_string (bool): Whether to return the segments as a list of strings.

        Returns:
            list: A list of lists or strings (segments) for each text.

        Note:
            For this implementation, see https://stackoverflow.com/a/54802737.
        """
        # Validate input
        try:
            model = SplitModel(
                texts=texts,
                n=n,
                merge_threshold=merge_threshold,
                overlap=overlap,
                tokenizer=tokenizer,
                as_string=as_string,
            )
        except Exception as e:
            raise LexosException(e)

        # Ensure a list of texts as the starting point
        if not isinstance(model.texts, list):
            model.texts = [model.texts]

        # Process the texts into segments
        all_segments = []
        for text in model.texts:

            # Tokenise the text
            tokens = self._tokenize(text, tokenizer=model.tokenizer)

            # Get the number of tokens per segment (d) and the remaining tokens (r)
            d, r = divmod(len(tokens), model.n)

            # Get the segments
            segments = []
            for i in range(model.n):
                index = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
                segments.append(tokens[index : index + (d + 1 if i < r else d)])
                # Apply the merge threshold
                if len(segments[-1]) < model.n * model.merge_threshold:
                    last_seg = segments.pop(-1)
                    # Combine the last two segments into a single list
                    segments[-1] = segments[-1] + last_seg
            all_segments.append(segments)
            if overlap:
                all_segments = [
                    self._create_overlapping_segments(segment, model.overlap)
                    for segment in all_segments
                ]
            if as_string:
                all_segments = [
                    ["".join(segment) for segment in text] for text in all_segments
                ]
        return all_segments

    def split_list(
        self,
        text: List[str],
        n: int = 1000,
        merge_threshold: float = 0.5,
        overlap: int = None,
        as_string: bool = False,
    ) -> list:
        """Split a list into chunks by a fixed number of tokens.

        Args:
            text (List[str]): A list of tokens.
            n (int): The number of tokens to split on.
            merge_threshold (float): The threshold to merge the last segment.
            overlap (int): The number of tokens to overlap.
            as_string (bool): Whether to return the segments as a list of strings.

        Returns:
            list: A list of token lists, one token list for each segment.
        """
        # Validate input
        try:
            model = SplitListModel(
                text=text,
                n=n,
                merge_threshold=merge_threshold,
                overlap=overlap,
                as_string=as_string,
            )
        except Exception as e:
            raise LexosException(e)

        # Ensure a list of texts as the starting point
        if isinstance(model.text[0], str):
            model.text = [model.text]

        # Process the texts into segments
        all_segments = []
        for text in model.text:
            segments = list(self._chunk_tokens(text, model.n))
            # Apply the merge threshold
            if len(segments[-1]) < model.n * model.merge_threshold:
                last_seg = segments.pop(-1)
                # Combine the last two segments into a single list
                segments[-1] = [segments[-1] + last_seg]
            all_segments.append(segments)
            if overlap:
                all_segments = [
                    self._create_overlapping_segments(segment, model.overlap)
                    for segment in all_segments
                ]
            if as_string:
                all_segments = [
                    ["".join(segment) for segment in text] for text in all_segments
                ]
        return all_segments

    def split_on_milestones(
        self,
        texts: Union[List[str], str],
        milestone: Union[dict, str],
        preserve_milestones: bool = True,
        tokenizer: str = None,
        as_string: bool = True,
    ) -> list:
        """Split texts on milestones.

        Args:
            texts (Union[List[str], str]): A text string or list of text strings.
            milestone (Union[dict, str]): A variable representing the value(s) to be matched.
            preserve_milestones (bool): If True, the milestone token will be preserved at the
                beginning of every segment. Otherwise, it will be deleted.
            tokenizer (str): The name of the tokenizer function to use.
            as_string (bool): Whether to return the segments as a list of strings.

        Returns:
            list: A list of lists or strings (segments) for each text.

        Note:
            The choice of tokenizer can lead to some unexpected results with regard to spacing
            around the milestone. The default behaviour is to delete the milestone and any
            following whitespace. If milestones are preserved, the milestone will occur at the
            beginning of the following segment and will be followed by a single space. If the
            segments are returned with `as_string=False`, each token will have a following space
            and it will be up to the end user to remove the space if desired.
        """
        # Validate input
        try:
            model = SplitMilestoneModel(
                texts=texts,
                milestone=milestone,
                preserve_milestones=preserve_milestones,
                tokenizer=tokenizer,
                as_string=as_string,
            )
        except Exception as e:
            raise LexosException(e)

        # Ensure a list of texts as the starting point
        if not isinstance(model.texts, list):
            model.texts = [model.texts]

        # Process the texts into segments
        all_segments = []
        milestone_pat = re.compile(milestone)
        for text in model.texts:
            cut_on_milestone = []
            seg = []
            # Tokenise the text
            tokens = self._tokenize(text, tokenizer=model.tokenizer)
            for i, token in enumerate(tokens):
                if re.match(
                    milestone_pat, token.strip()
                ):  # token.strip() == milestone:
                    cut_on_milestone.append(seg)
                    j = i
                    if preserve_milestones:
                        seg = [f"{milestone} "]
                    else:
                        seg = []
                else:
                    seg.append(token)
            # Add the last segment
            cut_on_milestone.append(tokens[j + 1 :])
            all_segments.append(cut_on_milestone)
        if as_string:
            all_segments = [
                ["".join(segment) for segment in text] for text in all_segments
            ]
        # If no milestone was found, return the original texts
        if len(all_segments) == 1 and all_segments[0] == []:
            return [model.texts]
        else:
            return all_segments
