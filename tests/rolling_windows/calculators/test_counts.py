"""test_counts.py.

Coverage: 100%
Last Update: February 16, 2025
"""

import pandas as pd
import pytest

from lexos.exceptions import LexosException
from lexos.rolling_windows import Windows
from lexos.rolling_windows.calculators.counts import Counts


@pytest.fixture
def basic_counts():
    """Create basic Counts calculator instance.

    Returns:
        Counts: Calculator instance with default settings
    """
    return Counts()


@pytest.fixture
def windows():
    """Create Windows instance with sample data.

    Returns:
        Windows: Configured Windows instance
    """
    w = Windows()
    w(input="Hello world Hello test", n=5)
    return w


def test_counts_init(basic_counts):
    """Test Counts calculator initialization."""
    assert basic_counts._id == "counts"
    assert basic_counts.patterns is None
    assert basic_counts.windows is None


def test_counts_call_basic(basic_counts, windows):
    """Test basic Counts calculator call."""
    result = basic_counts(patterns=["Hello"], windows=windows)
    assert isinstance(result, list)
    assert all(isinstance(sublist, list) for sublist in result)
    assert all(isinstance(item, int) for sublist in result for item in sublist)


def test_counts_call_multiple_patterns(basic_counts, windows):
    """Test Counts calculator with multiple patterns."""
    result = basic_counts(patterns=["Hello", "world"], windows=windows)
    pattern1_count = sum([x[0] for x in result])
    pattern2_count = sum([x[1] for x in result])
    assert pattern1_count == 2
    assert pattern2_count == 1


def test_counts_call_with_options(basic_counts, windows):
    """Test Counts calculator with various options."""
    result = basic_counts(
        patterns=["HELLO"], windows=windows, case_sensitive=False, mode="exact"
    )
    assert isinstance(result, list)
    assert len(result) > 0


def test_counts_call_no_windows(basic_counts):
    """Test error handling when no windows provided."""
    with pytest.raises(LexosException) as exc_info:
        basic_counts(patterns=["test"])
    assert "Calculator `windows` attribute is empty" in str(exc_info.value)


@pytest.mark.parametrize(
    "pattern,window_text,expected",
    [
        (["test"], "test test", [2]),
        (["a", "b"], "a b a b", [2, 2]),
        (["missing"], "test text", [0]),
    ],
)
def test_counts_various_patterns(basic_counts, pattern, window_text, expected):
    """Test Counts calculator with different patterns and texts.

    Args:
        pattern: Pattern(s) to search for
        window_text: Text to search in
        expected: Expected count results
    """
    w = Windows()
    w(input=window_text, n=len(window_text))
    result = basic_counts(patterns=pattern, windows=w)
    assert result[0] == expected


def test_counts_spacy_patterns(basic_counts):
    """Test Counts calculator with spaCy patterns."""
    w = Windows()
    w(input="The quick brown fox", n=12)
    patterns = [[{"LOWER": "quick"}]]
    with pytest.raises(LexosException):
        _ = basic_counts(
            patterns=patterns, windows=w, mode="spacy_rule", model="xx_sent_ud_sm"
        )


def test_counts_attribute_persistence(basic_counts, windows):
    """Test persistence of calculator attributes after call."""
    patterns = ["test"]
    basic_counts(patterns=patterns, windows=windows)

    assert basic_counts.patterns == patterns
    assert basic_counts.windows == windows


@pytest.fixture
def sample_data():
    """Create sample counting data.

    Returns:
        tuple: (Windows instance, patterns list)
    """
    w = Windows()
    w(input="Hello world Hello test", n=5)
    patterns = ["Hello", "world"]
    return w, patterns


def test_to_df_basic(basic_counts, sample_data):
    """Test basic DataFrame conversion."""
    windows, patterns = sample_data
    basic_counts(patterns=patterns, windows=windows)
    df = basic_counts.to_df()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [pattern.lower() for pattern in patterns]
    assert len(df) == 18  # The number of windows in the sample data


def test_to_df_spacy_rules(basic_counts):
    """Test DataFrame conversion with spaCy rules."""
    windows = Windows()
    windows(input="The quick brown fox", n=20)
    patterns = [[{"LOWER": "quick"}], [{"LOWER": "brown"}]]

    basic_counts(
        patterns=patterns, windows=windows, mode="spacy_rule", model="xx_sent_ud_sm"
    )

    # Test with show_spacy_rules=True
    df_rules = basic_counts.to_df(show_spacy_rules=True)
    assert len(df_rules.columns) == len(patterns)
    assert all(isinstance(col, str) for col in df_rules.columns)

    # Test with show_spacy_rules=False
    df_simple = basic_counts.to_df(show_spacy_rules=False)
    assert len(df_simple.columns) == len(patterns)
    assert all("quick" in col or "brown" in col for col in df_simple.columns)


def test_to_df_case_sensitivity(basic_counts, sample_data):
    """Test DataFrame conversion with case sensitivity settings."""
    windows, _ = sample_data
    patterns = ["HELLO", "WORLD"]

    # Case sensitive
    basic_counts(patterns=patterns, windows=windows, case_sensitive=True)
    df_sensitive = basic_counts.to_df()
    assert list(df_sensitive.columns) == patterns

    # Case insensitive
    basic_counts(patterns=patterns, windows=windows, case_sensitive=False)
    df_insensitive = basic_counts.to_df()
    assert list(df_insensitive.columns) == [p.lower() for p in patterns]


def test_to_df_mixed_patterns(basic_counts):
    """Test DataFrame conversion with mixed pattern types."""
    windows = Windows()
    windows(input="The quick brown fox", n=20)
    patterns = ["quick", [{"LOWER": "brown"}]]

    basic_counts(
        patterns=patterns, windows=windows, mode="spacy_rule", model="xx_sent_ud_sm"
    )

    df = basic_counts.to_df()
    assert len(df.columns) == len(patterns)
    assert "quick" in df.columns[0]


@pytest.mark.parametrize(
    "data,patterns,expected_cols",
    [
        ([[1, 2]], ["a", "b"], ["a", "b"]),
        ([[0, 0]], ["test", "example"], ["test", "example"]),
        ([], ["pattern"], ["pattern"]),
    ],
)
def test_to_df_various_data(basic_counts, data, patterns, expected_cols):
    """Test DataFrame conversion with various data patterns.

    Args:
        data: Count data to test
        patterns: Patterns to use as column names
        expected_cols: Expected column names
    """
    basic_counts.data = data
    basic_counts.patterns = patterns
    df = basic_counts.to_df()

    assert list(df.columns) == expected_cols
    assert len(df) == len(data)


def test_to_df_empty_data(basic_counts):
    """Test DataFrame conversion with empty data."""
    basic_counts.data = []
    basic_counts.patterns = ["test"]
    df = basic_counts.to_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["test"]
