"""test_windows.py.

Coverage: 100%
Last Update: September 9, 2025
"""

import pytest
import spacy
from spacy.tokens import Doc, Span, Token

from lexos.exceptions import LexosException
from lexos.rolling_windows import Windows

# Fixtures


@pytest.fixture
def basic_windows():
    """Create basic Windows instance.

    Returns:
        Windows: Windows instance with default settings
    """
    return Windows()


@pytest.fixture
def nlp():
    """Create spaCy English language model.

    Returns:
        spacy.Language: Blank English language model
    """
    return spacy.blank("en")


def test_windows_init(basic_windows):
    """Test Windows class initialization with default values."""
    assert basic_windows.input is None
    assert basic_windows.n == 1000
    assert basic_windows.window_type == "characters"
    assert basic_windows.alignment_mode == "strict"
    assert basic_windows.output == "strings"
    assert basic_windows.windows is None


def test_windows_custom_init():
    """Test Windows class initialization with custom values."""
    windows = Windows(
        n=5, window_type="tokens", alignment_mode="expand", output="tokens"
    )
    assert windows.n == 5
    assert windows.window_type == "tokens"
    assert windows.alignment_mode == "expand"
    assert windows.output == "tokens"


def test_windows_iterator_empty(basic_windows):
    """Test iterator with no windows generated."""
    assert list(basic_windows) == []


def test_windows_iterator_with_data(basic_windows):
    """Test iterator with generated windows."""
    test_input = "Hello world"
    basic_windows(input=test_input, n=5)
    windows_list = list(basic_windows)

    assert len(windows_list) > 0
    assert all(isinstance(w, str) for w in windows_list)
    assert len(windows_list[0]) == 5


@pytest.mark.parametrize(
    "n,expected_count",
    [
        (1, 11),  # Single character windows
        (5, 7),  # 5-character windows
        (11, 1),  # Full text window
        (12, 0),  # No windows (larger than text)
    ],
)
def test_windows_iterator_various_sizes(n, expected_count):
    """Test iterator with different window sizes.

    Args:
        n: Window size to test
        expected_count: Expected number of windows
    """
    windows = Windows()
    windows(input="Hello world", n=n)
    assert len(list(windows)) == expected_count


def test_windows_iterator_doc(nlp):
    """Test iterator with spaCy Doc input."""
    doc = nlp("Hello world")
    windows = Windows(n=2, window_type="tokens")
    windows(input=doc)

    windows_list = list(windows)
    assert len(windows_list) > 0
    assert isinstance(windows_list[0], str)


def test_windows_iterator_validation():
    """Test iterator with invalid window size."""
    with pytest.raises(ValueError):
        Windows(n=0)


def test_call_with_string(basic_windows):
    """Test window generation with string input."""
    result = basic_windows(input="Hello world", n=5)
    windows = list(result)

    assert len(windows) > 0
    assert all(isinstance(w, str) for w in windows)
    assert all(len(w) == 5 for w in windows)


def test_call_with_string_list(basic_windows):
    """Test window generation with list of strings."""
    result = basic_windows(input=["Hello", "world"], n=2)
    windows = list(result)
    assert len(windows) > 0
    for window in windows:
        assert all(isinstance(item, str) for item in window)


def test_call_with_token_list(nlp):
    """Test window generation with list of tokens."""
    doc = Doc(nlp.vocab, words=["Hello", "world"])
    tokens = [token for token in doc]
    windows = Windows()
    result = windows(input=tokens, n=2, window_type="tokens", output="strings")
    window_list = list(result)

    assert len(window_list) > 0
    assert all(isinstance(w, list) for w in window_list)
    assert all(isinstance(item, str) for w in window_list for item in w)


def test_call_with_span_list(nlp):
    """Test window generation with list of spans."""
    doc = Doc(nlp.vocab, words=["Hello", "world", "test"])
    spans = [doc[i : i + 1] for i in range(len(doc))]
    windows = Windows()
    result = windows(input=spans, n=2, window_type="spans", output="strings")
    window_list = list(result)

    assert len(window_list) > 0
    assert all(isinstance(w, list) for w in window_list)


def test_call_with_span_list_spans_output(nlp):
    """Test window generation with list of spans and spans output."""
    doc = Doc(nlp.vocab, words=["Hello", "world", "test"])
    spans = [doc[i : i + 1] for i in range(len(doc))]
    windows = Windows()
    result = windows(input=spans, n=2, window_type="spans", output="spans")
    window_list = list(result)
    for window in window_list:
        for item in window:
            assert isinstance(item, Span)


def test_call_with_doc(nlp):
    """Test window generation with spaCy Doc."""
    doc = Doc(nlp.vocab, words=["Hello", "world", "test"])
    windows = Windows()
    result = windows(input=doc, n=2, window_type="tokens", output="strings")
    window_list = list(result)

    assert len(window_list) > 0
    assert all(isinstance(w, str) for w in window_list)


@pytest.mark.parametrize(
    "input_type,window_type,output,expected_type",
    [
        ("Hello world", "characters", "strings", str),
        (["Hello", "world"], "characters", "strings", list),
        (["Hello", "world"], "tokens", "strings", list),
    ],
)
def test_call_various_configurations(
    basic_windows, input_type, window_type, output, expected_type
):
    """Test window generation with various configurations.

    Args:
        input_type: Input text or tokens
        window_type: Type of window to generate
        output: Desired output format
        expected_type: Expected type of window items
    """
    result = basic_windows(
        input=input_type, n=2, window_type=window_type, output=output
    )
    windows = list(result)
    print(windows)

    assert len(windows) > 0
    assert isinstance(windows[0], expected_type)


def test_call_invalid_window_type(basic_windows):
    """Test error handling for invalid window type."""
    with pytest.raises(LexosException):
        basic_windows(input="test", window_type="invalid")


def test_call_invalid_output_type(basic_windows):
    """Test error handling for invalid output type in call method."""
    with pytest.raises(
        LexosException, match="Output must be 'spans', 'strings' or 'tokens'."
    ):
        basic_windows(input="test", output="invalid_output")


def test_windows_invalid_output_type(nlp):
    """Test error handling for invalid output type during initialization."""
    doc = nlp("This is a test document.")

    with pytest.raises(LexosException, match="Output must be 'strings' or 'tokens'."):
        Windows(
            input=doc,
            output="not_valid",  # Invalid output to trigger the uncovered branch
            window_type="tokens",
        )


def test_windows_invalid_window_type_init():
    """Test error handling for invalid window type during initialization."""
    with pytest.raises(
        LexosException, match="Window type must be 'characters' or 'tokens'."
    ):
        Windows(window_type="invalid_type")


def test_call_attribute_persistence(basic_windows):
    """Test persistence of attributes after call."""
    input_text = "Hello world"
    n_size = 5

    basic_windows(input=input_text, n=n_size)

    assert basic_windows.input == input_text
    assert basic_windows.n == n_size
    assert basic_windows.windows is not None


@pytest.fixture
def sample_doc(nlp):
    """Create sample spaCy Doc.

    Returns:
        spacy.tokens.Doc: Sample document with test content
    """
    return nlp("Hello world test document")


@pytest.fixture
def sample_span(sample_doc):
    """Create sample spaCy Span.

    Returns:
        spacy.tokens.Span: Sample span from test document
    """
    return sample_doc[0:2]


@pytest.fixture
def sample_spans(nlp):
    """Create list of sample spaCy Spans.

    Returns:
        list[Span]: List of test spans
    """
    doc = nlp("Hello world test document")
    return [doc[i : i + 1] for i in range(len(doc))]


@pytest.fixture
def sample_tokens(nlp):
    """Create list of sample spaCy Tokens.

    Returns:
        list[Token]: List of test tokens
    """
    doc = Doc(nlp.vocab, words=["Hello", "world", "test", "document"])
    return [token for token in doc]


# Tests


def test_doc_windows_basic(sample_doc):
    """Test basic document windowing with default settings."""
    windows = Windows(n=2, output="strings")
    generator = windows._get_doc_windows(sample_doc)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, str) for w in results)


def test_doc_windows_span_input(sample_span):
    """Test window generation from Span input."""
    windows = Windows(n=2, output="strings")
    generator = windows._get_doc_windows(sample_span)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, str) for w in results)


@pytest.mark.parametrize(
    "output_type,expected_type",
    [
        ("strings", str),
        ("tokens", list),
    ],
)
def test_doc_windows_output_types(sample_doc, output_type, expected_type):
    """Test different output types for windows.

    Args:
        output_type: Type of output to generate
        expected_type: Expected type of window items
    """
    windows = Windows(n=2, output=output_type)
    generator = windows._get_doc_windows(sample_doc)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, expected_type) for w in results)


def test_doc_windows_invalid_output():
    """Test error handling for invalid output type."""
    with pytest.raises(LexosException):
        _ = Windows(n=2, output="invalid")


def test_get_doc_windows_invalid_output_direct(sample_doc):
    """Test _get_doc_windows with invalid output type by setting it directly."""
    windows = Windows(n=2, output="strings")  # Valid initialization
    windows.output = "invalid"  # Directly set invalid output after init

    with pytest.raises(
        LexosException, match="Output must be 'spans', 'strings', or 'tokens'."
    ):
        list(windows._get_doc_windows(sample_doc))


def test_get_doc_windows_spans_output_direct(sample_doc):
    """Test _get_doc_windows with spans output by setting it directly."""
    windows = Windows(
        n=2, output="strings", window_type="tokens"
    )  # Valid initialization
    windows.output = "spans"  # Directly set spans output after init

    generator = windows._get_doc_windows(sample_doc)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, spacy.tokens.Span) for w in results)


@pytest.mark.parametrize(
    "window_type,alignment_mode",
    [
        ("characters", "strict"),
        ("tokens", "strict"),
        ("characters", "contract"),
        ("characters", "expand"),
    ],
)
def test_doc_windows_configurations(sample_doc, window_type, alignment_mode):
    """Test window generation with different configurations."""
    windows = Windows(
        n=2, output="strings", window_type=window_type, alignment_mode=alignment_mode
    )
    generator = windows._get_doc_windows(sample_doc)
    results = list(generator)

    if alignment_mode == "strict":
        assert len(results) > 0
        assert all(isinstance(w, str) for w in results)
    else:
        assert len(results) == 0


def test_doc_windows_boundaries(sample_doc):
    """Test window boundary calculations."""
    windows = Windows(n=2, output="strings", window_type="tokens")
    generator = windows._get_doc_windows(sample_doc)
    results = list(generator)

    # For n=2 and 4 tokens, expect 3 windows
    assert len(results) == 3


def test_doc_windows_token_output(sample_doc):
    """Test window generation with token output."""
    windows = Windows(n=2, output="tokens")
    generator = windows._get_doc_windows(sample_doc)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, list) for w in results)
    assert all(isinstance(t, spacy.tokens.Token) for w in results for t in w)


def test_span_list_windows_basic(sample_spans):
    """Test basic span list windowing with default settings."""
    windows = Windows(n=2, output="strings", window_type="tokens")
    generator = windows._get_span_list_windows(sample_spans)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, list) for w in results)
    assert all(isinstance(t, str) for w in results for t in w)


@pytest.mark.parametrize(
    "output_type,expected_type", [("strings", str), ("tokens", Span | Token)]
)
def test_span_list_windows_output_types(sample_spans, output_type, expected_type):
    """Test different output types for span list windows.

    Args:
        output_type: Type of output to generate
        expected_type: Expected type of window items
    """
    windows = Windows(n=2, output=output_type, window_type="tokens")
    generator = windows._get_span_list_windows(sample_spans)
    results = list(generator)
    if expected_type is str:
        assert len(results) > 0
        assert all(isinstance(w, list) for w in results)
        assert all(isinstance(t, str) for w in results for t in w)
    else:
        assert len(results) > 0
        assert all(isinstance(w, list) for w in results)
        for window in results:
            assert all(isinstance(item, expected_type) for item in window)


def test_span_list_windows_invalid_output(sample_spans):
    """Test error handling for invalid output type."""
    with pytest.raises(LexosException):
        _ = Windows(input=sample_spans, n=2, output="invalid", window_type="tokens")


def test_get_span_list_windows_invalid_output_direct(sample_spans):
    """Test _get_span_list_windows with invalid output type by setting it directly."""
    windows = Windows(
        n=2, output="strings", window_type="tokens"
    )  # Valid initialization
    windows.output = "invalid"  # Directly set invalid output after init

    with pytest.raises(LexosException, match="Output must be 'strings', or 'tokens'."):
        list(windows._get_span_list_windows(sample_spans))


# Note: Line 179 (yeild slice) in _get_span_list_windows appears to be unreachable code
# since the validation at line 164 only allows "strings" or "tokens" as output types,
# and both cases are handled explicitly in the if/elif block above the else clause.


def test_span_list_windows_character_mode(sample_spans):
    """Test span list windows in character mode."""
    windows = Windows(n=2, output="strings", window_type="characters")
    generator = windows._get_span_list_windows(sample_spans)
    results = list(generator)

    assert len(results) == 3
    assert all(isinstance(w, str) for w in results)


def test_span_list_windows_boundaries(sample_spans):
    """Test window boundary calculations for span lists."""
    windows = Windows(n=2, output="strings", window_type="tokens")
    generator = windows._get_span_list_windows(sample_spans)
    results = list(generator)

    # For n=2 and 4 spans, expect 3 windows
    assert len(results) == 3
    assert all(len(w) == 2 for w in results)


def test_span_list_windows_empty_spans():
    """Test handling of empty span list."""
    windows = Windows(n=2, output="strings", window_type="tokens")
    generator = windows._get_span_list_windows([])
    results = list(generator)

    assert len(results) == 0


def test_span_list_windows_single_span(nlp):
    """Test handling of single span input."""
    doc = Doc(nlp.vocab, words=["Hello"])
    spans = [doc[0:1]]
    windows = Windows(n=1, output="strings", window_type="tokens")
    generator = windows._get_span_list_windows(spans)
    results = list(generator)

    assert len(results) == 1
    assert results[0] == ["Hello"]


def test_string_windows_basic():
    """Test basic string windowing."""
    windows = Windows(n=3, output="strings")
    generator = windows._get_string_windows("Hello")
    results = list(generator)

    assert len(results) == 3  # "Hel", "ell", "llo"
    assert all(isinstance(w, str) for w in results)
    assert all(len(w) == 3 for w in results)


def test_string_windows_list():
    """Test string windowing with list input."""
    windows = Windows(n=2, output="strings")
    generator = windows._get_string_windows(["a", "b", "c", "d"])
    results = list(generator)

    assert len(results) == 3  # ["a", "b"], ["b", "c"], ["c", "d"]
    assert all(isinstance(w, list) for w in results)
    assert all(len(w) == 2 for w in results)


def test_string_windows_invalid_output():
    """Test error handling for invalid output type."""
    windows = Windows(n=2, output="tokens")
    with pytest.raises(LexosException) as exc_info:
        next(windows._get_string_windows("test"))
    assert "Output must be 'strings'" in str(exc_info.value)


@pytest.mark.parametrize(
    "input_text,window_size,expected_count",
    [
        ("Hello", 2, 4),  # "He", "el", "ll", "lo"
        ("Test", 4, 1),  # "Test"
        ("A", 1, 1),  # "A"
        ("", 1, 0),  # Empty string
        ("Hi", 3, 0),  # Window larger than input
    ],
)
def test_string_windows_various_sizes(input_text, window_size, expected_count):
    """Test string windowing with various input sizes.

    Args:
        input_text: Input text to window
        window_size: Size of sliding window
        expected_count: Expected number of windows
    """
    windows = Windows(n=window_size, output="strings")
    generator = windows._get_string_windows(input_text)
    results = list(generator)

    assert len(results) == expected_count
    if results:
        assert all(len(w) == window_size for w in results)


def test_string_windows_boundaries():
    """Test window boundary handling."""
    windows = Windows(n=3, output="strings")
    generator = windows._get_string_windows("Hello")
    results = list(generator)

    assert results[0] == "Hel"  # First window
    assert results[-1] == "llo"  # Last window


def test_string_windows_unicode():
    """Test string windowing with Unicode characters."""
    windows = Windows(n=2, output="strings")
    generator = windows._get_string_windows("Hello世界")
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, str) for w in results)
    assert "世界" in results


def test_string_windows_list_types():
    """Test string windowing with different list element types."""
    windows = Windows(n=2, output="strings")
    input_list = ["Hello", "123", "世界"]
    generator = windows._get_string_windows(input_list)
    results = list(generator)

    assert len(results) == 2
    assert all(isinstance(w, list) for w in results)
    assert all(isinstance(item, str) for w in results for item in w)


def test_token_list_windows_basic(sample_tokens):
    """Test basic token list windowing with default settings."""
    windows = Windows(n=2, output="strings", window_type="tokens")
    generator = windows._get_token_list_windows(sample_tokens)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, list) for w in results)
    assert all(isinstance(t, str) for w in results for t in w)


@pytest.mark.parametrize(
    "output_type,expected_type", [("strings", str), ("tokens", Token)]
)
def test_token_list_windows_output_types(sample_tokens, output_type, expected_type):
    """Test different output types for token list windows.

    Args:
        output_type: Type of output to generate
        expected_type: Expected type of window items
    """
    windows = Windows(n=2, output=output_type, window_type="tokens")
    generator = windows._get_token_list_windows(sample_tokens)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, list) for w in results)
    assert all(isinstance(t, expected_type) for w in results for t in w)


def test_token_list_windows_invalid_output(sample_tokens):
    """Test error handling for invalid output type."""
    with pytest.raises(LexosException):
        _ = Windows(input=sample_tokens, n=2, output="invalid", window_type="tokens")


def test_get_token_list_windows_invalid_output_direct(sample_tokens):
    """Test _get_token_list_windows with invalid output type by setting it directly."""
    windows = Windows(
        n=2, output="strings", window_type="tokens"
    )  # Valid initialization
    windows.output = "invalid"  # Directly set invalid output after init

    with pytest.raises(LexosException, match="Output must be 'strings' or 'tokens'."):
        list(windows._get_token_list_windows(sample_tokens))


def test_token_list_windows_character_mode(sample_tokens):
    """Test token list windows in character mode."""
    windows = Windows(n=2, output="strings", window_type="characters")
    generator = windows._get_token_list_windows(sample_tokens)
    results = list(generator)

    assert len(results) == 3
    assert all(isinstance(w, str) for w in results)


def test_token_list_windows_boundaries(sample_tokens):
    """Test window boundary calculations for token lists."""
    windows = Windows(n=2, output="strings", window_type="tokens")
    generator = windows._get_token_list_windows(sample_tokens)
    results = list(generator)

    # For n=2 and 4 tokens, expect 3 windows
    assert len(results) == 3
    assert all(len(w) == 2 for w in results)


def test_token_list_windows_with_spaces(nlp):
    """Test token list windows with whitespace handling."""
    doc = Doc(nlp.vocab, words=["Hello", " ", "world"], spaces=[True, True, False])
    tokens = [token for token in doc]
    windows = Windows(n=2, output="strings", window_type="tokens")
    generator = windows._get_token_list_windows(tokens)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, list) for w in results)


def test_token_list_windows_empty_tokens():
    """Test handling of empty token list."""
    windows = Windows(n=2, output="strings", window_type="tokens")
    generator = windows._get_token_list_windows([])
    results = list(generator)

    assert len(results) == 0


def test_token_list_windows_doc_conversion(sample_tokens):
    """Test conversion of tokens to Doc in character mode."""
    windows = Windows(n=3, output="strings", window_type="characters")
    generator = windows._get_token_list_windows(sample_tokens)
    results = list(generator)

    assert len(results) > 0
    assert all(isinstance(w, str) for w in results)


def test_set_attrs_basic(basic_windows):
    """Test setting basic attributes."""
    attrs = {"n": 5, "window_type": "tokens", "output": "strings"}
    basic_windows._set_attrs(**attrs)

    assert basic_windows.n == 5
    assert basic_windows.window_type == "tokens"
    assert basic_windows.output == "strings"


def test_set_attrs_none_values(basic_windows):
    """Test setting attributes with None values."""
    original_n = basic_windows.n
    attrs = {"n": None, "window_type": "tokens"}
    basic_windows._set_attrs(**attrs)

    assert basic_windows.n == original_n  # Should not change
    assert basic_windows.window_type == "tokens"


def test_set_attrs_empty_dict(basic_windows):
    """Test setting attributes with empty dict."""
    original_state = {
        "n": basic_windows.n,
        "window_type": basic_windows.window_type,
        "output": basic_windows.output,
    }
    basic_windows._set_attrs()

    assert basic_windows.n == original_state["n"]
    assert basic_windows.window_type == original_state["window_type"]
    assert basic_windows.output == original_state["output"]


def test_set_attrs_multiple_calls(basic_windows):
    """Test multiple calls to set_attrs."""
    # First call
    basic_windows._set_attrs(n=5)
    assert basic_windows.n == 5

    # Second call
    basic_windows._set_attrs(n=10)
    assert basic_windows.n == 10


def test_set_attrs_with_input(basic_windows):
    """Test setting input attribute."""
    test_input = "Hello world"
    basic_windows._set_attrs(input=test_input)

    assert basic_windows.input == test_input


@pytest.mark.parametrize(
    "attr_name,attr_value",
    [
        ("n", 100),
        ("window_type", "characters"),
        ("alignment_mode", "strict"),
        ("output", "strings"),
    ],
)
def test_set_attrs_individual(basic_windows, attr_name, attr_value):
    """Test setting individual attributes.

    Args:
        attr_name: Name of attribute to set
        attr_value: Value to set for attribute
    """
    attrs = {attr_name: attr_value}
    basic_windows._set_attrs(**attrs)

    assert getattr(basic_windows, attr_name) == attr_value


def test_length_property(basic_windows):
    """Test length property with generated windows."""
    empty_windows = Windows()
    assert empty_windows.length == 0
    test_input = "Hello world"
    basic_windows(input=test_input, n=5)
    windows_list = list(basic_windows)
    num_windows = len(windows_list)
    basic_windows(input=test_input, n=5)
    assert basic_windows.length == num_windows
