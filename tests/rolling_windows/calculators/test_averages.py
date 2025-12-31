"""test_averages.py.

Coverage: 100%
Last Update: February 16, 2025
"""

import pandas as pd
import pytest

from lexos.exceptions import LexosException
from lexos.rolling_windows import Windows
from lexos.rolling_windows.calculators.averages import Averages

# Fixtures


@pytest.fixture
def basic_averages():
    """Create basic Averages calculator instance.

    Returns:
        Averages: Calculator instance with default settings
    """
    return Averages()


@pytest.fixture
def sample_windows():
    """Create Windows instance with sample data.

    Returns:
        Windows: Windows instance with test content and n=5
    """
    w = Windows()
    w(input="Hello world Hello test", n=5)
    return w


@pytest.fixture
def sample_data(basic_averages):
    """Create sample averaging data.

    Returns:
        Averages: Calculator instance with sample data
    """
    w = Windows()
    w(input="Hello world Hello test world", n=10)
    basic_averages(patterns=["Hello", "world"], windows=w)
    return basic_averages


# Tests


def test_averages_init(basic_averages):
    """Test Averages calculator initialization."""
    assert basic_averages._id == "averages"
    assert basic_averages.patterns is None
    assert basic_averages.windows is None


def test_averages_call_basic(basic_averages, sample_windows):
    """Test basic Averages calculator call."""
    result = basic_averages(patterns=["Hello"], windows=sample_windows)

    assert isinstance(result, list)
    assert all(isinstance(sublist, list) for sublist in result)
    assert all(isinstance(item, float) for sublist in result for item in sublist)
    # Check that values are normalized by window size
    assert all(0 <= item <= 1 for sublist in result for item in sublist)


def test_averages_call_multiple_patterns(basic_averages, sample_windows):
    """Test Averages calculator with multiple patterns."""
    result = basic_averages(patterns=["Hello", "world"], windows=sample_windows)

    assert len(result) == 18  # The window size is 5, so 18 windows
    assert all(len(sublist) == 2 for sublist in result)  # Two patterns
    assert all(0 <= item <= 1 for sublist in result for item in sublist)


def test_averages_call_no_windows(basic_averages):
    """Test error handling when no windows provided."""
    with pytest.raises(LexosException) as exc_info:
        basic_averages(patterns=["test"])
    assert "Calculator `windows` attribute is empty" in str(exc_info.value)


@pytest.mark.parametrize(
    "pattern,window_text,window_size,expected",
    [
        (["test"], "test test", 4, [0.25]),  # 2 occurrences / 4 length
        (["a", "b"], "a b a b", 4, [0.25, 0.25]),  # 2 occurrences each / 4 length
        (["missing"], "test text", 4, [0.0]),  # 0 occurrences / 4 length
    ],
)
def test_averages_various_patterns(
    basic_averages, pattern, window_text, window_size, expected
):
    """Test Averages calculator with different patterns and texts.

    Args:
        pattern: Pattern(s) to search for
        window_text: Text to search in
        window_size: Size of window
        expected: Expected average results
    """
    w = Windows()
    w(input=window_text, n=window_size)
    result = basic_averages(patterns=pattern, windows=w)
    assert result[0] == expected


def test_averages_spacy_patterns(basic_averages):
    """Test Averages calculator with spaCy patterns."""
    w = Windows()
    w(input="The quick brown fox", n=4)
    patterns = [[{"LOWER": "quick"}]]

    with pytest.raises(LexosException) as exc_info:
        _ = basic_averages(
            patterns=patterns, windows=w, mode="spacy_rule", model="xx_sent_ud_sm"
        )
    assert "Invalid mode for character windows" in str(exc_info.value)

    # TODO: Add test for valid spaCy patterns once spans are accepted
    # assert isinstance(result, list)
    # assert all(isinstance(val, float) for sublist in result for val in sublist)
    # assert all(0 <= val <= 1 for sublist in result for val in sublist)


def test_averages_case_sensitivity(basic_averages, sample_windows):
    """Test case sensitivity in pattern matching."""
    # Case sensitive
    result_sensitive = basic_averages(
        patterns=["HELLO"], windows=sample_windows, case_sensitive=True
    )

    # Case insensitive
    result_insensitive = basic_averages(
        patterns=["HELLO"], windows=sample_windows, case_sensitive=False
    )

    assert result_sensitive != result_insensitive


def test_to_df_basic(sample_data):
    """Test basic DataFrame conversion."""
    df = sample_data.to_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2  # Two pattern columns
    assert all(isinstance(val, float) for col in df.columns for val in df[col])
    assert all(0 <= val <= 1 for col in df.columns for val in df[col])


def test_to_df_spacy_rules(basic_averages):
    """Test DataFrame conversion with spaCy rules."""
    w = Windows()
    w(input="The quick brown fox", n=20)
    patterns = [[{"LOWER": "quick"}], [{"LOWER": "brown"}]]
    basic_averages(
        patterns=patterns, windows=w, mode="spacy_rule", model="xx_sent_ud_sm"
    )

    # Test with show_spacy_rules=True
    df_rules = basic_averages.to_df(show_spacy_rules=True)
    assert len(df_rules.columns) == 2
    assert all(isinstance(col, str) for col in df_rules.columns)

    # Test with show_spacy_rules=False
    df_simple = basic_averages.to_df(show_spacy_rules=False)
    assert len(df_simple.columns) == 2
    assert "quick" in df_simple.columns[0].lower()
    assert "brown" in df_simple.columns[1].lower()


def test_to_df_case_sensitivity(basic_averages):
    """Test DataFrame conversion with case sensitivity."""
    w = Windows()
    w(input="HELLO world", n=10)
    patterns = ["HELLO", "WORLD"]

    # Case sensitive
    basic_averages(patterns=patterns, windows=w, case_sensitive=True)
    df_sensitive = basic_averages.to_df()
    assert "HELLO" in df_sensitive.columns
    assert "WORLD" in df_sensitive.columns

    # Case insensitive
    basic_averages(patterns=patterns, windows=w, case_sensitive=False)
    df_insensitive = basic_averages.to_df()
    assert "hello" in df_insensitive.columns
    assert "world" in df_insensitive.columns


def test_to_df_mixed_patterns(basic_averages):
    """Test DataFrame conversion with mixed pattern types."""
    w = Windows()
    w(input="The quick brown fox", n=20)
    patterns = ["quick", [{"LOWER": "brown"}]]

    basic_averages(
        patterns=patterns, windows=w, mode="spacy_rule", model="xx_sent_ud_sm"
    )

    df = basic_averages.to_df()
    assert len(df.columns) == 2
    assert "quick" in df.columns[0].lower()
    assert "brown" in df.columns[1].lower()


def test_to_df_empty_data(basic_averages):
    """Test DataFrame conversion with empty data."""
    basic_averages.data = []
    basic_averages.patterns = ["test1", "test2"]
    df = basic_averages.to_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert len(df.columns) == 2
    assert list(df.columns) == ["test1", "test2"]


@pytest.mark.parametrize(
    "patterns,expected_columns",
    [
        (["a", "b"], ["a", "b"]),
        (["test1", "test2"], ["test1", "test2"]),
        ([["quick"], ["brown"]], ["quick", "brown"]),
    ],
)
def test_to_df_column_naming(basic_averages, patterns, expected_columns):
    """Test DataFrame column naming with different patterns.

    Args:
        patterns: List of patterns to use as columns
        expected_columns: Expected column names in result
    """
    basic_averages.data = [[0.5, 0.5]]
    basic_averages.patterns = patterns
    df = basic_averages.to_df()

    assert list(df.columns) == expected_columns
