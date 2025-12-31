"""__init__.py.

Last Update: 8 September, 2025
Last Tested: 9 September, 2025

Credits:

    A preliminary version of this module was developed with a code review by DHTech
    (https://dhcodereview.github.io/), facilitated by Julia Damerow (Arizona State University).
    The code was reviewed by Cole  Crawford (Harvard University) and Ryan Muther (Harvard University).
"""

import itertools
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token

from lexos.exceptions import LexosException

doc_schema = DocJSONSchema.schema()
Tokenized = Doc | Span | list[Span | Token]

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=doc_schema
)


class Windows(BaseModel):
    """Basic model for windows."""

    input: Optional[str | list[str] | Tokenized] = Field(
        None, description="The input data to be windowed."
    )
    n: Optional[int] = Field(1000, gt=0, description="The size of the window.")
    window_type: Optional[str] = Field(
        "characters",
        description="The type of window to generate: `characters`, `spans`, or `tokens`.",
    )
    alignment_mode: Optional[str] = Field(
        "strict",
        description="The alignment mode for the window.",
    )
    output: Optional[str] = Field(
        "strings", description="The output type for the windows."
    )
    windows: Optional[Iterator] = Field(
        None, description="Container for the windows generator."
    )

    model_config = validation_config

    def __init__(self, **data):
        """Create the Windows instance."""
        super().__init__(**data)
        if self.window_type not in [None, "characters", "spans", "tokens"]:
            raise LexosException("Window type must be 'characters' or 'tokens'.")
        if self.output not in ["strings", "tokens"]:
            raise LexosException("Output must be 'strings' or 'tokens'.")

    def __iter__(self):
        """Iterate over the windows."""
        if self.windows is None:
            return iter([])
        return iter(self.windows)

    @validate_call(config=validation_config)
    def __call__(
        self,
        input: Optional[str | list[str] | Tokenized] = Field(
            None, description="The input data to be windowed."
        ),
        n: Optional[int] = Field(None, gt=0, description="The size of the window."),
        window_type: Optional[str] = Field(
            None,
            description="The type of window to generate: `characters`, `spans`, or `tokens`.",
        ),
        alignment_mode: Optional[str] = Field(
            None,
            description="The alignment mode for the window.",
        ),
        output: Optional[str] = Field(
            None,
            description="The output type for the windows.",
        ),
    ) -> Iterator:
        """Generate windows based on the input data type."""
        self._set_attrs(
            input=input,
            n=n,
            window_type=window_type,
            alignment_mode=alignment_mode,
            output=output,
            windows=None,
        )
        if self.window_type not in ["characters", "spans", "tokens"]:
            raise LexosException("Window type must be 'characters' or 'tokens'.")
        if self.output not in ["spans", "strings", "tokens"]:
            raise LexosException("Output must be 'spans', 'strings' or 'tokens'.")
        if isinstance(self.input, str):
            self.windows = self._get_string_windows(self.input)
        elif isinstance(self.input, list):
            if isinstance(self.input[0], str):
                self.windows = self._get_string_windows(self.input)
            elif isinstance(self.input[0], Token):
                self.windows = self._get_token_list_windows(self.input)
            else:
                self.windows = self._get_span_list_windows(self.input)
        else:
            self.windows = self._get_doc_windows(self.input)
        return self

    @property
    def length(self):
        """Create a temporary copy of the windows and calculate the length."""
        if self.windows is None:
            return 0

        temp_windows, self.windows = itertools.tee(self.windows)
        return sum(1 for _ in temp_windows)

    def _get_doc_windows(self, input: Doc | Span) -> Iterator[list[Span | str | Token]]:
        """Generate windows from a Doc or Span object with output as Spans, strings, or Tokens.

        Args:
            input (Doc | Span): A spaCy Doc or Span object.

        Yields:
            Iterator[list[Span | str | Token]]: A generator of windows.
        """
        if self.output not in ["spans", "strings", "tokens"]:
            raise LexosException("Output must be 'spans', 'strings', or 'tokens'.")
        # Process a Span as a Doc
        if isinstance(input, Span):
            input = input.as_doc()
        if self.window_type == "characters":
            length = len(input.text)
        else:
            length = len(input)
        length = len(input)
        boundaries = [(i, i + self.n) for i in range(length) if i + self.n <= length]
        for start, end in boundaries:
            if self.alignment_mode == "strict":
                span = input[start:end]
            else:
                span = input.char_span(start, end, self.alignment_mode)
            if span is not None:
                if self.output == "strings":
                    if span.text != "":
                        yield span.text
                elif self.output == "tokens":
                    yield [token for token in span if token.text != ""]
                else:
                    yield span

    def _get_span_list_windows(
        self, input: list[Span]
    ) -> Iterator[list[Span | str | Token]]:
        """Generate windows from a Doc or Span object with output as strings or Token objects.

        Args:
            input (list[Span]): A list of spaCy Span objects.

        Yields:
            Iterator[list[Span | str | Token]]: A generator of windows.
        """
        if self.output not in ["spans", "strings", "tokens"]:
            raise LexosException("Output must be 'strings', or 'tokens'.")
        if self.window_type != "characters":
            length = len(input)
            boundaries = [
                (i, i + self.n) for i in range(length) if i + self.n <= length
            ]
            for start, end in boundaries:
                slice = input[start:end]
                if slice is not None:
                    if self.output == "strings":
                        yield [token.text for token in slice]
                    elif self.output == "tokens":  # assuming self.output == "tokens"
                        yield [token for token in slice]
                    # appears to be unreachable code as output must be 'strings' or 'tokens'
                    else:
                        yield slice
        else:
            # Merge spans into a single Doc object
            input = Doc.from_docs([span.as_doc() for span in input])
            yield from self._get_doc_windows(input)

    def _get_string_windows(self, input: str | list[str]) -> Iterator[list[str]]:
        """Generate windows from a string or list of strings.

        Args:
            input (str | list[str]): A string or list of strings.

        Yields:
            Iterator[list[str]]: A generator of windows.
        """
        if self.output != "strings":
            raise LexosException("Output must be 'strings'.")
        length = len(input)
        boundaries = [(i, i + self.n) for i in range(length) if i + self.n <= length]
        for start_char, end_char in boundaries:
            yield input[start_char:end_char]

    def _get_token_list_windows(
        self, input: list[Token]
    ) -> Iterator[list[str | Token]]:
        """Generate windows from a Doc or Span object with output as strings or Token objects.

        Args:
            input (list[Token]): A list of spaCy Token objects.

        Yields:
            Iterator[list[str | Token]]: A generator of windows.
        """
        if self.output not in ["strings", "tokens"]:
            raise LexosException("Output must be 'strings' or 'tokens'.")
        if self.window_type != "characters":
            length = len(input)
            boundaries = [
                (i, i + self.n) for i in range(length) if i + self.n <= length
            ]
            for start, end in boundaries:
                slice = input[start:end]
                if slice is not None:
                    if self.output == "strings":
                        yield [token.text for token in slice]
                    else:
                        yield [token for token in slice]
        else:
            # Merge tokens into a single Doc object
            words = [token.text for token in input]
            spaces = [True if token.whitespace_ else False for token in input]
            input = Doc(input[0].vocab, words=words, spaces=spaces)
            yield from self._get_doc_windows(input)

    def _set_attrs(self, **kwargs) -> None:
        """Set instance attributes when public method is called."""
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
