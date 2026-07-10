"""filters.py.

Last Updated: July 9, 2026
Last Tested: July 9, 2026
"""

import re
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.matcher import Matcher
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Token

from lexos.exceptions import LexosException
from lexos.util import ensure_list

ROMAN_PATTERN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


class BaseFilter(BaseModel):
    """BaseFilter class."""

    id: ClassVar[str] = "base_filter"
    doc: Optional[Doc] = Field(default=None, description="A spaCy doc.")
    matcher: Optional[Matcher] = Field(default=None, description="A spaCy matcher.")
    matches: Optional[list[tuple[int, int, int]]] = Field(
        default=None, description="List of matches."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra=DocJSONSchema.model_json_schema(),
    )

    @validate_call(config=model_config)
    def __call__(self, doc: Optional[Doc], matcher: Optional[Matcher] = None) -> Doc:
        """Call the filter function.

        Args:
            doc (Optional[Doc]): A spaCy doc.
            matcher (Optional[Matcher]): A spaCy matcher.

        Returns:
            Doc: The filtered doc.
        """
        # Validate the inputs
        if not doc and not self.doc:
            raise LexosException("No doc has been assigned to the filter.")
        if not matcher and not self.matcher:
            raise LexosException("No matcher has been assigned to the filter.")
        if doc:
            self.doc = doc
        if matcher:
            self.matcher = matcher
        # Get the matches
        self.matches = self.matcher(doc)
        return self.doc if self.doc is not None else doc

    @property
    def matched_token_ids(self) -> Optional[set[int]]:
        """A list of matched token ids after the filter was applied."""
        if self.matches is None or self.doc is None:
            return None
        return {i for _, start, end in self.matches for i in range(start, end)}

    @property
    def matched_tokens(self) -> list[Token]:
        """A list of matched tokens after the filter was applied."""
        if self.doc is None or self.matched_token_ids is None:
            return []
        return [self.doc[i] for i in self.matched_token_ids]

    @property
    def filtered_tokens(self) -> list[Token]:
        """A list of filtered tokens after the filter was applied."""
        if self.filtered_token_ids is None or self.doc is None:
            return []
        return [self.doc[i] for i in self.filtered_token_ids]

    @property
    def filtered_token_ids(self) -> Optional[set[int]]:
        """A list of filtered token ids after the filter was applied."""
        if self.matches is None or self.doc is None:
            return None
        matched_ids = self.matched_token_ids or set()
        return set(range(len(self.doc))) - matched_ids

    def _set_extensions(self, attr: str, default: Any):
        """Set the extensions."""
        if not Token.has_extension(attr):
            Token.set_extension(attr, default=default, force=True)

    def get_matched_doc(self, add_spaces: bool = False) -> Doc:
        """Get a new doc from the matched tokens.

        Args:
            add_spaces (bool): If True, add a space after every token.
                             If False, preserve original whitespace. Default is False.

        Returns:
            Doc: A new spaCy Doc containing only the matched tokens.
        """
        matched_tokens = self.matched_tokens
        words = [t.text for t in matched_tokens]
        if add_spaces:
            spaces = [" " for _ in matched_tokens]
        else:
            spaces = [t.whitespace_ for t in matched_tokens]
        return Doc(self.doc.vocab, words=words, spaces=spaces)

    def get_filtered_doc(self, add_spaces: bool = False) -> Doc:
        """Get a new doc from the filtered tokens.

        Args:
            add_spaces (bool): If True, add a space after every token.
                             If False, preserve original whitespace. Default is False.

        Returns:
            Doc: A new spaCy Doc containing only the filtered tokens.
        """
        filtered_tokens = self.filtered_tokens
        words = [t.text for t in filtered_tokens]
        if add_spaces:
            spaces = [" " for _ in filtered_tokens]
        else:
            spaces = [t.whitespace_ for t in filtered_tokens]
        return Doc(self.doc.vocab, words=words, spaces=spaces)


class IsRomanFilter(BaseFilter):
    """A filter for Roman numerals."""

    id: ClassVar[str] = "is_roman"
    ROMAN_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
    )
    doc: Optional[Doc] = Field(default=None, description="A spaCy doc.")
    attr: Optional[str] = Field(
        default="is_roman",
        description="The name of the attribute to add to the tokens.",
    )
    default: Optional[Any] = Field(
        default=None, description="The default value of the attribute."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra=DocJSONSchema.model_json_schema(),
    )

    def __init__(self, **data):
        """Initialise the filter object and set custom attribute extensions."""
        super().__init__(**data)
        if self.attr:
            self._set_extensions(self.attr, self.default)

    @validate_call(config=model_config)
    def __call__(
        self,
        doc: Optional[Doc] = Field(default=None, description="A spaCy doc."),
        attr: Optional[str] = Field(
            default=None, description="The name of the attribute to add to the tokens."
        ),
        default: Optional[Any] = Field(
            default=None, description="The default value of the attribute."
        ),
    ) -> Doc:
        """Apply the filter.

        Returns:
            Doc: The filtered doc.
        """
        # Validation
        if doc:
            self.doc = doc
        if attr:
            self.attr = attr
        if default is not None:
            self.default = default

        # Use instance attributes if we have them
        working_doc = self.doc if self.doc is not None else doc
        working_attr = (
            self.attr if hasattr(self, "attr") and self.attr is not None else attr
        )
        working_default = (
            self.default
            if hasattr(self, "default") and self.default is not None
            else default
        )

        # Set custom extensions
        if working_attr:
            self._set_extensions(working_attr, working_default)

        # Apply the filter only if we have a valid doc
        if working_doc is not None:
            # Store matches for tokens that are Roman numerals
            self.matches = []
            for i, token in enumerate(working_doc):
                is_roman_result = self.is_roman(token)
                setattr(working_doc[i]._, working_attr, is_roman_result)
                if is_roman_result:
                    self.matches.append((0, i, i + 1))

        return working_doc if working_doc is not None else doc

    @validate_call(config=model_config)
    def is_roman(self, token: Token) -> bool:
        """Detect Roman numerals (capitals only).

        Args:
            token (Token): A spaCy token.

        Returns:
            bool: True if the token is a Roman numeral.
        """
        if token.text == "":
            return False
        return bool(self.ROMAN_PATTERN.search(token.text))


class IsStopwordFilter(BaseFilter):
    """A filter to detect stop words in a spaCy doc."""

    id: ClassVar[str] = "is_stopword"
    doc: Optional[Doc] = Field(default=None, description="A spaCy doc.")
    stopwords: Optional[list | str] = Field(
        default=None,
        description="A list or string containing the stop word(s) to add or remove.",
    )
    remove: Optional[bool] = Field(
        default=False,
        description="If True, the stop word(s) will be removed from the model.",
    )
    case_sensitive: Optional[bool] = Field(
        default=False,
        description="If False (default), stop word changes apply to all case variations. If True, only the exact case provided is modified.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra=DocJSONSchema.model_json_schema(),
    )

    def __init__(self, **data: Any):
        """Initialise the filter object with configuration.

        Args:
            **data (Any): Configuration data
        """
        super().__init__(**data)
        self.stopwords = ensure_list(self.stopwords)

    @validate_call(config=model_config)
    def __call__(
        self,
        doc: Optional[Doc] = Field(default=None, description="A spaCy doc."),
        stopwords: Optional[list | str] = Field(
            default=None,
            description="A list or string containing the stop word(s) to add or remove.",
        ),
        remove: Optional[bool] = Field(
            default=None,
            description="If True, the stop word(s) will be removed from the model.",
        ),
        case_sensitive: Optional[bool] = Field(
            default=None,
            description="If False (default), stop word changes apply to all case variations. If True, only the exact case provided is modified.",
        ),
    ) -> Doc:
        """Apply the filter.

        Returns:
            Doc: The filtered doc.

        Note:
            This filter modifies the model defaults. If you need the model's original default stop words.
            you will need to re-load the model.
        """
        # Validation
        if doc:
            self.doc = doc
        if stopwords is not None:
            self.stopwords = ensure_list(stopwords)
        if remove is not None:
            self.remove = remove
        if case_sensitive is not None:
            self.case_sensitive = case_sensitive

        working_doc = self.doc if self.doc is not None else doc
        working_stopwords = (
            ensure_list(self.stopwords) if self.stopwords is not None else []
        )
        working_remove = self.remove if self.remove is not None else remove
        working_case_sensitive = (
            self.case_sensitive
            if self.case_sensitive is not None
            else case_sensitive
            if case_sensitive is not None
            else False
        )

        # Apply the filter only if we have valid inputs
        if working_doc is not None and working_stopwords is not None:
            # Ensure stopwords is iterable and properly formatted
            if not isinstance(working_stopwords, list):
                working_stopwords = ensure_list(working_stopwords)

            # Get the vocab (shared across all docs from the same model)
            vocab = working_doc.vocab

            stopword_items = {item for item in working_stopwords if item is not None}
            for item in stopword_items:
                if working_case_sensitive:
                    vocab[item].is_stop = not working_remove
                else:
                    vocab[item.lower()].is_stop = not working_remove
                    vocab[item].is_stop = not working_remove
                    vocab[item.capitalize()].is_stop = not working_remove

            # Store matches for stopwords
            self.matches = []
            for i, token in enumerate(working_doc):
                if token.is_stop:
                    self.matches.append((0, i, i + 1))

        return working_doc if working_doc is not None else doc


class IsWordFilter(BaseFilter):
    """A filter to detect words in a spaCy doc."""

    id: ClassVar[str] = "is_word"
    doc: Optional[Doc] = Field(default=None, description="A spaCy doc.")
    attr: Optional[str] = Field(
        default="is_word", description="The name of the attribute to add to the tokens."
    )
    default: Optional[bool] = Field(
        default=False, description="The default value of the attribute."
    )
    exclude: Optional[Optional[list[str] | str]] = Field(
        default_factory=lambda: [" ", "\n"],
        description="A string/regex or list of strings/regex patterns to exclude.",
    )
    exclude_digits: Optional[Optional[bool]] = Field(
        default=False, description="If True, digits will not be treated as words."
    )
    exclude_roman_numerals: Optional[Optional[bool]] = Field(
        default=False,
        description="Same as above for Roman numerals, but only works on capital letters.",
    )
    exclude_pattern: Optional[list[str] | str] = Field(
        default=None,
        description="Additional patterns to add to the default exclude list.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra=DocJSONSchema.model_json_schema(),
    )

    def __init__(self, **data: Any):
        """Initialise the filter object with configuration.

        Args:
            **data (Any): Configuration data
        """
        super().__init__(**data)
        if self.attr:
            self._set_extensions(self.attr, self.default)

    @validate_call(config=model_config)
    def __call__(
        self,
        doc: Optional[Doc] = Field(default=None, description="A spaCy doc."),
        attr: Optional[str] = Field(
            default=None,
            description="The name of the attribute to add to the tokens.",
        ),
        default: Optional[bool] = Field(
            default=None, description="The default value of the attribute."
        ),
        exclude: Optional[list[str] | str] = Field(
            default=None,
            description="A string/regex or list of strings/regex patterns to exclude.",
        ),
        exclude_digits: Optional[bool] = Field(
            default=None, description="If True, digits will not be treated as words."
        ),
        exclude_roman_numerals: Optional[bool] = Field(
            default=None,
            description="Same as above for Roman numerals, but only works on capital letters.",
        ),
        exclude_pattern: Optional[list[str] | str] = Field(
            default=None,
            description="Additional patterns to add to the default exclude list.",
        ),
    ) -> Doc:
        """Apply the filter.

        Returns:
            Doc: The filtered doc.
        """
        # Assign keyword variables to the instance attributes
        if doc:
            self.doc = doc
        if exclude is not None:
            self.exclude = ensure_list(exclude)
        if exclude_digits is not None:
            self.exclude_digits = exclude_digits
        if exclude_roman_numerals is not None:
            self.exclude_roman_numerals = exclude_roman_numerals
        if exclude_pattern is not None:
            self.exclude_pattern = ensure_list(exclude_pattern)
        if attr:
            self.attr = attr
        if default is not None:
            self.default = default

        # Use instance attributes if we have them
        working_doc = self.doc if self.doc is not None else doc
        working_attr = (
            self.attr if hasattr(self, "attr") and self.attr is not None else attr
        )
        working_default = (
            self.default
            if hasattr(self, "default") and self.default is not None
            else default
        )

        # Set ._is_word extension
        if working_attr:
            self._set_extensions(working_attr, working_default)

        # Apply the filter only if we have a valid doc
        if working_doc is not None:
            # Store matches for tokens that are words
            self.matches = []
            for i, token in enumerate(working_doc):
                is_word_result = self.is_word(token)
                setattr(working_doc[i]._, working_attr, is_word_result)
                if is_word_result:
                    self.matches.append((0, i, i + 1))

        return working_doc if working_doc is not None else doc

    def _is_roman_numeral(self, string: str) -> bool:
        """Check if a string is a Roman numeral.

        Args:
            string (str): A string.

        Returns:
            bool: True if the string is a Roman numeral.
        """
        if string == "":
            return False
        return bool(ROMAN_PATTERN.search(string))

    @validate_call(config=model_config)
    def is_word(self, token: Token) -> bool:
        """Detect words.

        Args:
            token (Token): A spaCy token.

        Returns:
            bool: True if the token is a word.
        """
        text = token.text
        if text == "":
            return False

        if self.exclude_digits:
            if not text.isalpha():
                return False
        else:
            if not (text.isalpha() or text.isdigit()):
                return False

        if self.exclude_roman_numerals and self._is_roman_numeral(text):
            return False

        exclude_list = list(ensure_list(self.exclude))
        if self.exclude_pattern is not None:
            exclude_list += ensure_list(self.exclude_pattern)
        if exclude_list:
            exclude_pat = re.compile("|".join(exclude_list))
            if exclude_pat.search(text):
                return False

        return True
