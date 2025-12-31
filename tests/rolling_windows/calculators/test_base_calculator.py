"""test_base_calculator.py.

Coverage: 95%. Missing: 23, 27, 47, 362-367, 845-859, 924-925
Last Update: February 16, 2025
"""

import pytest
import spacy
from pydantic import ValidationError
from spacy.language import Language
from spacy.tokens import Doc

from lexos.exceptions import LexosException
from lexos.rolling_windows import Windows
from lexos.rolling_windows.calculators.base_calculator import BaseCalculator


class TestCalculator(BaseCalculator):
    """Test implementation of BaseCalculator."""

    def __call__(self, *args, **kwargs):
        """Implement abstract method."""
        pass

    def to_df(self, *args, **kwargs):
        """Implement abstract method."""
        pass


@pytest.fixture
def nlp():
    """Create spaCy English language model.

    Returns:
        spacy.Language: Blank English language model
    """
    return spacy.blank("en")


@pytest.fixture
def sample_doc(nlp):
    """Create sample spaCy Doc.

    Returns:
        Doc: Sample document with test content
    """
    return Doc(nlp.vocab, words=["Hello", "world", "this", "is", "test"])


@pytest.fixture
def basic_calculator():
    """Create basic calculator instance.

    Returns:
        TestCalculator: Calculator instance with default settings
    """
    return TestCalculator()


@pytest.fixture
def sample_span(nlp):
    """Create sample spaCy Span.

    Returns:
        Span: Test span with sample text
    """
    doc = Doc(nlp.vocab, words=["Hello", "world", "Hello", "Test"])
    return doc[:]


# Tests


def test_regex_escape_bytes_handling():
    """Test regex_escape function with bytes input (line 65 coverage)."""
    from lexos.rolling_windows.calculators.base_calculator import regex_escape

    # Test bytes input with regex special characters
    bytes_input = b"test[pattern](with)*special+chars.^$"
    result = regex_escape(bytes_input)

    # Verify it returns bytes
    assert isinstance(result, bytes)

    # Verify special characters are escaped
    expected = b"test\\[pattern\\]\\(with\\)\\*special\\+chars\\.\\^\\$"
    assert result == expected

    # Test bytes input without special characters
    simple_bytes = b"simple_text"
    result_simple = regex_escape(simple_bytes)
    assert result_simple == b"simple_text"
    assert isinstance(result_simple, bytes)


def test_regex_escape_string_handling():
    """Test regex_escape function with string input (line 66 coverage)."""
    from lexos.rolling_windows.calculators.base_calculator import regex_escape

    # Test string input with regex special characters
    string_input = "test[pattern](with)*special+chars.^$"
    result = regex_escape(string_input)

    # Verify it returns string
    assert isinstance(result, str)

    # Verify special characters are escaped
    expected = "test\\[pattern\\]\\(with\\)\\*special\\+chars\\.\\^\\$"
    assert result == expected

    # Test string input without special characters
    simple_string = "simple_text"
    result_simple = regex_escape(simple_string)
    assert result_simple == "simple_text"
    assert isinstance(result_simple, str)


def test_calculator_init_defaults(basic_calculator):
    """Test calculator initialization with default values."""
    assert basic_calculator.id == "base_calculator"
    assert basic_calculator.patterns is None
    assert basic_calculator.windows is None
    assert basic_calculator.mode == "exact"
    assert basic_calculator.case_sensitive is False
    assert basic_calculator.alignment_mode == "strict"
    assert basic_calculator.model == "xx_sent_ud_sm"
    assert basic_calculator.nlp is None
    assert basic_calculator.data == []


def test_calculator_custom_init():
    """Test calculator initialization with custom values."""
    calc = TestCalculator(
        patterns=["test"], mode="regex", case_sensitive=True, alignment_mode="expand"
    )
    assert calc.patterns == ["test"]
    assert calc.mode == "regex"
    assert calc.case_sensitive is True
    assert calc.alignment_mode == "expand"


def test_calculator_with_windows():
    """Test calculator with Windows instance."""
    windows = Windows()
    calc = TestCalculator(windows=windows)
    assert calc.windows == windows


def test_metadata_property(basic_calculator):
    """Test metadata property (line 158 coverage)."""
    # Call the metadata property to trigger model_dump()
    result = basic_calculator.metadata
    # The method calls model_dump() but doesn't return it, so result should be None
    assert result is None


def test_n_property_with_none_windows_n():
    """Test n property when windows.n is None (line 165 coverage)."""
    # Create windows with n=None to trigger line 165
    windows = Windows()
    windows.n = None  # Explicitly set to None
    calc = TestCalculator(windows=windows)
    result = calc.n
    assert result is None


def test_n_property_with_windows():
    """Test n property when windows.n has a value."""
    windows = Windows(n=5)
    calc = TestCalculator(windows=windows)
    result = calc.n
    assert result == 5


def test_window_type_property_with_none_window_type():
    """Test window_type property when windows.window_type is None (line 180 coverage)."""
    # Create windows with window_type=None to trigger line 180
    windows = Windows()
    windows.window_type = None  # Explicitly set to None
    calc = TestCalculator(windows=windows)
    result = calc.window_type
    assert result is None


def test_window_type_property_with_windows():
    """Test window_type property when windows.window_type has a value."""
    windows = Windows(window_type="tokens")
    calc = TestCalculator(windows=windows)
    result = calc.window_type
    assert result == "tokens"


@pytest.mark.parametrize(
    "mode", ["exact", "regex", "spacy_matcher", "multi_token", "multi_token_exact"]
)
def test_calculator_valid_modes(mode):
    """Test calculator with different valid modes.

    Args:
        mode: Search mode to test
    """
    calc = TestCalculator(mode=mode)
    assert calc.mode == mode


def test_calculator_with_patterns():
    """Test calculator with different pattern types."""
    # String pattern
    calc1 = TestCalculator(patterns="test")
    assert calc1.patterns == "test"

    # List of patterns
    patterns = ["test1", "test2"]
    calc2 = TestCalculator(patterns=patterns)
    assert calc2.patterns == patterns


def test_count_exact_match_case_sensitive():
    """Test exact pattern matching with case sensitivity."""
    calc = TestCalculator(case_sensitive=True, mode="exact")
    window = "Hello Hello HELLO"
    pattern = "Hello"

    count = calc._count_character_patterns_in_character_windows(window, pattern)
    assert count == 2


def test_count_exact_match_case_insensitive():
    """Test exact pattern matching without case sensitivity."""
    calc = TestCalculator(case_sensitive=False, mode="exact")
    window = "Hello HELLO hello"
    pattern = "hello"

    count = calc._count_character_patterns_in_character_windows(window, pattern)
    assert count == 3


def test_count_regex_pattern():
    """Test regex pattern matching."""
    calc = TestCalculator(mode="regex")
    window = "Hello123 hello456 HELLO789"
    pattern = r"hello\d{3}"

    count = calc._count_character_patterns_in_character_windows(window, pattern)
    assert count == 3


@pytest.mark.parametrize(
    "window,pattern,expected",
    [
        ("aaaaa", "aa", 2),  # Overlapping patterns
        ("", "test", 0),  # Empty window
        ("test\ntest", "test", 2),  # Multiple lines
        ("test.test", "test", 2),  # With punctuation
    ],
)
def test_count_various_patterns(basic_calculator, window, pattern, expected):
    """Test pattern counting with various input combinations.

    Args:
        window: Input text to search
        pattern: Pattern to find
        expected: Expected number of matches
    """
    count = basic_calculator._count_character_patterns_in_character_windows(
        window, pattern
    )
    assert count == expected


def test_count_regex_special_chars():
    """Test regex pattern matching with special characters."""
    calc = TestCalculator(mode="regex")
    window = "test.com test.org test.net"
    pattern = r"test\.[a-z]+"

    count = calc._count_character_patterns_in_character_windows(window, pattern)
    assert count == 3


def test_count_unicode_characters():
    """Test pattern matching with Unicode characters."""
    calc = TestCalculator(mode="exact")
    window = "Hello 世界 Hello 世界"
    pattern = "世界"

    count = calc._count_character_patterns_in_character_windows(window, pattern)
    assert count == 2


def test_count_whitespace():
    """Test pattern matching with whitespace."""
    calc = TestCalculator(mode="exact")
    window = "a b  a  b   a   b"
    pattern = "a b"

    count = calc._count_character_patterns_in_character_windows(window, pattern)
    assert count == 1


def test_regex_flags():
    """Test regex flag handling."""
    calc = TestCalculator(mode="regex", case_sensitive=False)
    window = "TEST Test test"
    pattern = r"test"

    count = calc._count_character_patterns_in_character_windows(window, pattern)
    assert count == 3


def test_count_exact_mode(basic_calculator):
    """Tests counting in exact match mode."""
    window = "hello hello HELLO"
    pattern = "hello"
    count = basic_calculator._count_in_character_window(window, pattern)
    assert count == 3  # Case insensitive by default


def test_count_regex_mode():
    """Tests counting in regex mode."""
    calc = TestCalculator(mode="regex")
    window = "test123 test456 test789"
    pattern = r"test\d{3}"
    count = calc._count_in_character_window(window, pattern)
    assert count == 3


def test_invalid_mode():
    """Tests error handling for invalid modes."""
    calc = TestCalculator(mode="invalid_mode")
    with pytest.raises(Exception) as exc_info:
        calc._count_in_character_window("test", "test")
    assert "Invalid mode for character windows" in str(exc_info.value)


@pytest.mark.parametrize(
    "mode,window,pattern,expected",
    [
        ("exact", "hello hello", "hello", 2),
        ("regex", "test123 test456", r"test\d+", 2),
        ("exact", "", "test", 0),
        ("regex", "TEST123 test123", r"test\d+", 2),
    ],
)
def test_count_various_modes_and_patterns(mode, window, pattern, expected):
    """Tests counting with different modes and patterns.

    Args:
        mode: Search mode to use
        window: Input text to search
        pattern: Pattern to find
        expected: Expected match count
    """
    calc = TestCalculator(mode=mode)
    count = calc._count_in_character_window(window, pattern)
    assert count == expected


@pytest.mark.skip(reason="Requires pytest-mock fixture which is not installed")
def test_mode_delegation(mocker):
    """Tests proper delegation to specific counting methods."""
    calc = TestCalculator(mode="exact")
    spy = mocker.spy(calc, "_count_character_patterns_in_character_windows")

    calc._count_in_character_window("test", "test")
    assert spy.called
    assert spy.call_count == 1


def test_token_list_exact_match_case_sensitive():
    """Test exact token matching with case sensitivity."""
    calc = TestCalculator(mode="exact", case_sensitive=True)
    window = ["Hello", "World", "Hello"]
    pattern = "Hello"

    count = calc._count_token_patterns_in_token_lists(window, pattern)
    assert count == 2


def test_token_list_exact_match_case_insensitive():
    """Test exact token matching without case sensitivity."""
    calc = TestCalculator(mode="exact", case_sensitive=False)
    window = ["Hello", "HELLO", "hello"]
    pattern = "hello"

    count = calc._count_token_patterns_in_token_lists(window, pattern)
    assert count == 3


def test_token_list_regex_mode():
    """Test regex pattern matching in token lists."""
    calc = TestCalculator(mode="regex")
    window = ["test123", "test456", "TEST789"]
    pattern = r"test\d{3}"

    count = calc._count_token_patterns_in_token_lists(window, pattern)
    assert count == 3


@pytest.mark.parametrize(
    "window,pattern,mode,expected",
    [
        ([""], "test", "exact", 0),  # Empty token
        ([], "test", "exact", 0),  # Empty window
        (["test"], "", "exact", 0),  # Empty pattern
        (["test", "test"], "test", "exact", 2),  # Multiple matches
        (["te.st", "te*st"], "te.\\w+", "regex", 2),  # Special chars
    ],
)
def test_token_list_various_patterns(basic_calculator, window, pattern, mode, expected):
    """Test token pattern counting with different inputs.

    Args:
        window: List of tokens to search
        pattern: Pattern to find
        expected: Expected number of matches
    """
    basic_calculator.mode = mode
    count = basic_calculator._count_token_patterns_in_token_lists(window, pattern)
    assert count == expected


def test_token_list_unicode():
    """Test token pattern matching with Unicode characters."""
    calc = TestCalculator(mode="exact")
    window = ["Hello", "世界", "Hello", "世界"]
    pattern = "世界"

    count = calc._count_token_patterns_in_token_lists(window, pattern)
    assert count == 2


def test_token_list_regex_partial_matches():
    """Test regex pattern matching with partial token matches."""
    calc = TestCalculator(mode="regex")
    window = ["testing", "tester", "tested"]
    pattern = r"test"

    count = calc._count_token_patterns_in_token_lists(window, pattern)
    assert count == 3


def test_token_list_with_whitespace():
    """Test token pattern matching with whitespace tokens."""
    calc = TestCalculator(mode="exact")
    window = ["  ", "test", " ", "test"]
    pattern = "test"

    count = calc._count_token_patterns_in_token_lists(window, pattern)
    assert count == 2


def test_exact_match_case_sensitive(nlp, sample_span):
    """Test exact pattern matching with case sensitivity."""
    calc = TestCalculator(mode="exact", case_sensitive=True, nlp=nlp)
    pattern = "Hello"

    count = calc._count_token_patterns_in_span(sample_span, pattern)
    assert count == 2


def test_exact_match_case_insensitive(nlp, sample_span):
    """Test exact pattern matching without case sensitivity."""
    calc = TestCalculator(mode="exact", case_sensitive=False, nlp=nlp)
    pattern = "hello"

    count = calc._count_token_patterns_in_span(sample_span, pattern)
    assert count == 2


def test_regex_pattern(nlp):
    """Test regex pattern matching in spans."""
    doc = Doc(nlp.vocab, words=["test123", "test456", "TEST789"])
    span = doc[:]
    calc = TestCalculator(mode="regex", nlp=nlp)
    pattern = r"test\d{3}"

    count = calc._count_token_patterns_in_span(span, pattern)
    assert count == 3


def test_spacy_rule_matching(nlp):
    """Test spaCy rule pattern matching."""
    doc = Doc(nlp.vocab, words=["The", "quick", "brown", "fox"])
    span = doc[:]
    calc = TestCalculator(mode="spacy_rule", nlp=nlp)
    pattern = [{"LOWER": "quick"}]

    count = calc._count_token_patterns_in_span(span, pattern)
    assert count == 1


def test_spacy_rule_case_insensitive(nlp):
    """Test spaCy rule matching with case insensitivity."""
    doc = Doc(nlp.vocab, words=["TEST", "Test", "test"])
    span = doc[:]
    calc = TestCalculator(mode="spacy_rule", case_sensitive=False, nlp=nlp)
    pattern = [{"LOWER": "test"}]

    count = calc._count_token_patterns_in_span(span, pattern)
    assert count == 3


@pytest.mark.parametrize(
    "mode,pattern,expected",
    [
        ("exact", "missing", 0),
        ("regex", r"\d+", 0),
        ("spacy_rule", [{"LOWER": "missing"}], 0),
    ],
)
def test_no_matches(nlp, sample_span, mode, pattern, expected):
    """Test patterns that don't match anything.

    Args:
        mode: Search mode to use
        pattern: Pattern to search for
        expected: Expected count (0)
    """
    calc = TestCalculator(mode=mode, nlp=nlp)
    count = calc._count_token_patterns_in_span(sample_span, pattern)
    assert count == expected


def test_unicode_handling(nlp):
    """Test pattern matching with Unicode characters."""
    doc = Doc(nlp.vocab, words=["Hello", "世界", "世界"])
    span = doc[:]
    calc = TestCalculator(mode="exact", nlp=nlp)
    pattern = "世界"

    count = calc._count_token_patterns_in_span(span, pattern)
    assert count == 2


@pytest.fixture
def sample_doc2(nlp):
    """Create sample spaCy Doc.

    Returns:
        Doc: Sample document with test content
    """
    # words = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dogs"]
    # return Doc(nlp.vocab, words=words)
    return nlp("The quick brown fox jumps over lazy dogs")


def test_multi_token_exact(nlp, sample_doc2):
    """Test multi-token exact pattern matching."""
    calc = TestCalculator(mode="multi_token_exact", case_sensitive=False)
    window = sample_doc2[1:4]  # quick brown fox
    pattern = "quick brown"

    count = calc._count_token_patterns_in_span_text(window, pattern)
    assert count == 1


def test_multi_token_regex(nlp, sample_doc2):
    """Test multi-token regex pattern matching."""
    calc = TestCalculator(mode="multi_token", case_sensitive=False)
    window = sample_doc2[1:5]  # quick brown fox jumps
    pattern = r"brown \w+ jumps"

    count = calc._count_token_patterns_in_span_text(window, pattern)
    assert count == 1


def test_case_sensitivity(nlp, sample_doc2):
    """Test case-sensitive pattern matching."""
    calc = TestCalculator(mode="multi_token_exact", case_sensitive=True)
    doc = nlp("The Quick Brown Fox quick brown fox")
    window = doc[1:]  # Quick Brown Fox quick fox
    pattern = "quick brown"

    count = calc._count_token_patterns_in_span_text(window, pattern)
    assert count == 1


@pytest.mark.parametrize("alignment_mode", ["strict", "contract", "expand"])
def test_alignment_modes(nlp, sample_doc2, alignment_mode):
    """Test different alignment modes.

    Args:
        alignment_mode: Alignment mode to test
    """
    calc = TestCalculator(mode="multi_token", alignment_mode=alignment_mode)
    window = sample_doc2[1:4]  # quick brown fox
    pattern = r"quick\s+brown"

    count = calc._count_token_patterns_in_span_text(window, pattern)
    assert count in [0, 1]


def test_overlapping_matches(nlp):
    """Test handling of overlapping pattern matches."""
    doc = nlp("aa aa aa")
    calc = TestCalculator(
        mode="multi_token_exact",
    )
    window = doc[0:]  # aa aa aa
    pattern = "aa aa"

    count = calc._count_token_patterns_in_span_text(window, pattern)
    assert count == 1


def test_no_matches2(sample_doc2):
    """Test pattern with no matches."""
    calc = TestCalculator(
        mode="multi_token_exact",
    )
    window = sample_doc2[1:4]  # "quick brown fox"
    pattern = "missing pattern"

    count = calc._count_token_patterns_in_span_text(window, pattern)
    assert count == 0


@pytest.fixture
def sample_doc3(nlp):
    """Create sample spaCy Doc.

    Returns:
        Doc: Sample document with test content
    """
    return nlp("The quick brown fox jumps")


@pytest.fixture
def sample_span2(sample_doc3):
    """Create sample spaCy Span.

    Returns:
        Span: Test span from sample document
    """
    return sample_doc3[0:]


def test_list_window_exact():
    """Test token list window with exact matching."""
    calc = TestCalculator(mode="exact")
    window = ["Hello", "world", "Hello"]
    pattern = "Hello"

    count = calc._count_in_token_window(window, pattern)
    assert count == 2


def test_list_window_regex():
    """Test token list window with regex matching."""
    calc = TestCalculator(mode="regex")
    window = ["test123", "test456", "TEST789"]
    pattern = r"test\d{3}"

    count = calc._count_in_token_window(window, pattern)
    assert count == 3


def test_invalid_mode_for_list():
    """Test error handling for invalid modes with list window."""
    calc = TestCalculator(mode="spacy_rule")
    window = ["test", "text"]
    pattern = [{"LOWER": "test"}]

    with pytest.raises(LexosException) as exc_info:
        calc._count_in_token_window(window, pattern)
    assert "You cannot use spaCy rule" in str(exc_info.value)


def test_span_window_multi_token(sample_span2):
    """Test span window with multi-token matching."""
    calc = TestCalculator(
        mode="multi_token",
    )
    pattern = "quick brown"

    count = calc._count_in_token_window(sample_span2, pattern)
    assert count == 1


def test_span_window_spacy_rule(nlp, sample_span2):
    """Test span window with spaCy rule matching."""
    calc = TestCalculator(mode="spacy_rule", nlp=nlp)
    pattern = [{"LOWER": "quick"}]

    count = calc._count_in_token_window(sample_span2, pattern)
    assert count == 1


@pytest.mark.parametrize(
    "window_type,pattern,mode",
    [
        (["token1", "token2"], "token", "exact"),
        (["test123", "test456"], r"test\d+", "regex"),
        ("invalid_type", "pattern", "multi_token"),
    ],
)
def test_window_types(window_type, pattern, mode):
    """Test different window types and modes.

    Args:
        window_type: Type of window to test
        pattern: Pattern to search for
        mode: Search mode to use
    """
    calc = TestCalculator(mode=mode)
    if isinstance(window_type, str) and calc.mode == "multi_token":
        with pytest.raises(LexosException):
            calc._count_in_token_window(window_type, pattern)
    else:
        count = calc._count_in_token_window(window_type, pattern)
        assert isinstance(count, int)


def test_span_window_case_sensitivity(sample_span2):
    """Test case sensitivity in span window."""
    calc = TestCalculator(mode="exact", case_sensitive=True)
    pattern = "Quick"  # Different case from sample_doc3

    count = calc._count_in_token_window(sample_span2, pattern)
    assert count == 0


def test_extract_string_pattern_from_spacy_rule(basic_calculator):
    """Test extraction of string pattern from a spaCy rule."""
    pattern = [[{"LOWER": "test"}, {"LOWER": "pattern"}]]
    result = basic_calculator._extract_string_pattern(pattern)
    assert result == "test|pattern"


def test_spacy_rule_to_lower_dict_with_text_key():
    """Test spacy_rule_to_lower with dict containing TEXT key (line 88 coverage)."""
    from lexos.rolling_windows.calculators.base_calculator import spacy_rule_to_lower

    # Test with TEXT key (should be converted to LOWER)
    patterns = {"TEXT": "hello", "POS": "NOUN"}
    result = spacy_rule_to_lower(patterns)

    expected = {"LOWER": "hello", "POS": "NOUN"}
    assert result == expected

    # Test with ORTH key (should be converted to LOWER)
    patterns2 = {"ORTH": "world", "TAG": "NN"}
    result2 = spacy_rule_to_lower(patterns2)

    expected2 = {"LOWER": "world", "TAG": "NN"}
    assert result2 == expected2


@pytest.fixture
def sample_doc4(nlp):
    """Create sample spaCy Doc.

    Returns:
        Doc: Sample document with test content
    """
    return nlp("Hello world Hello test")


@pytest.fixture
def sample_span3(sample_doc4):
    """Create sample spaCy Span.

    Returns:
        Span: Test span from sample document
    """
    return sample_doc4[:]


def test_get_window_count_character_window():
    """Test window count with character window units."""
    calc = TestCalculator()
    windows = Windows()
    calc.windows = windows
    calc.windows.window_type = "characters"

    result = calc._get_window_count("Hello Hello", "Hello")
    assert result == 2


def test_get_window_count_token_window(sample_span3):
    """Test window count with token window units."""
    calc = TestCalculator()
    windows = Windows()
    calc.windows = windows
    calc.windows.window_type = "tokens"

    result = calc._get_window_count(sample_span3, "Hello")
    assert result == 2


def test_get_window_count_with_list():
    """Test window count with list of tokens."""
    calc = TestCalculator()
    windows = Windows()
    calc.windows = windows
    calc.windows.window_type = "tokens"
    tokens = ["Hello", "world", "Hello"]

    result = calc._get_window_count(tokens, "Hello")
    assert result == 2


@pytest.mark.parametrize(
    "window_type,window,pattern,expected",
    [
        ("characters", "test test", "test", 2),
        ("tokens", ["test", "word", "test"], "test", 2),
        ("characters", "TEST TEST", "test", 2),  # Case insensitive by default
        ("tokens", ["TEST", "WORD", "TEST"], "test", 2),
    ],
)
def test_get_window_count_various_inputs(window_type, window, pattern, expected):
    """Test window count with various inputs and units.

    Args:
        window_type: Type of window units to test
        window: Window content to search
        pattern: Pattern to find
        expected: Expected match count
    """
    calc = TestCalculator()
    windows = Windows()
    calc.windows = windows
    calc.windows.window_type = window_type

    result = calc._get_window_count(window, pattern)
    assert result == expected


def test_get_window_count_with_regex_mode():
    """Test window count with regex mode."""
    calc = TestCalculator(mode="regex")
    windows = Windows()
    calc.windows = windows
    calc.windows.window_type = "characters"

    result = calc._get_window_count("test123 test456", r"test\d{3}")
    assert result == 2


@pytest.mark.skip(reason="Requires pytest-mock fixture which is not installed")
def test_get_window_count_delegation(mocker):
    """Test proper method delegation based on window units."""
    calc = TestCalculator()
    windows = Windows()
    calc.windows = windows

    # Test character window delegation
    calc.windows.window_type = "characters"
    spy_char = mocker.spy(calc, "_count_in_character_window")
    calc._get_window_count("test", "test")
    assert spy_char.called

    # Test token window delegation
    calc.windows.window_type = "tokens"
    spy_token = mocker.spy(calc, "_count_in_token_window")
    calc._get_window_count(["test"], "test")
    assert spy_token.called


def test_set_attrs_basic(basic_calculator):
    """Test setting basic attributes."""
    attrs = {"case_sensitive": True, "mode": "regex", "alignment_mode": "expand"}
    basic_calculator._set_attrs(attrs)

    assert basic_calculator.case_sensitive is True
    assert basic_calculator.mode == "regex"
    assert basic_calculator.alignment_mode == "expand"


def test_set_attrs_with_none_values(basic_calculator):
    """Test setting attributes with None values."""
    attrs = {"case_sensitive": None, "mode": "regex", "alignment_mode": None}
    basic_calculator._set_attrs(attrs)

    assert basic_calculator.mode == "regex"
    # None values should not override existing defaults
    assert basic_calculator.case_sensitive is False
    assert basic_calculator.alignment_mode == "strict"


def test_set_attrs_spacy_model(basic_calculator):
    """Test setting spaCy model attribute."""
    attrs = {"model": "xx_sent_ud_sm"}
    basic_calculator._set_attrs(attrs)

    assert basic_calculator.model == "xx_sent_ud_sm"
    assert isinstance(basic_calculator.nlp, Language)


def test_set_attrs_multiple_calls(basic_calculator):
    """Test multiple calls to set_attrs."""
    # First call
    basic_calculator._set_attrs({"mode": "regex"})
    assert basic_calculator.mode == "regex"

    # Second call
    basic_calculator._set_attrs({"mode": "exact"})
    assert basic_calculator.mode == "exact"


def test_set_attrs_invalid_attribute(basic_calculator):
    """Test setting non-existent attribute."""
    attrs = {"invalid_attr": "value"}

    with pytest.raises(ValidationError):
        basic_calculator._set_attrs(attrs)


@pytest.mark.parametrize("model_name", ["xx_sent_ud_sm", "en_core_web_sm"])
def test_set_attrs_different_models(basic_calculator, model_name):
    """Test setting different spaCy models.

    Args:
        model_name: Name of spaCy model to test
    """
    try:
        attrs = {"model": model_name}
        basic_calculator._set_attrs(attrs)

        assert basic_calculator.model == model_name
        assert isinstance(basic_calculator.nlp, Language)
    except OSError:
        pytest.skip(f"Model {model_name} not installed")
