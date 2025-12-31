"""test_ratios.py.

Coverage: 100%
Last Update: February 16, 2025
"""

import pandas as pd
import pytest

from lexos.exceptions import LexosException
from lexos.rolling_windows import Windows
from lexos.rolling_windows.calculators.ratios import Ratios

# Fixtures


@pytest.fixture
def basic_ratios():
    """Create basic Ratios calculator instance.

    Returns:
        Ratios: Calculator instance with default settings
    """
    return Ratios()


@pytest.fixture
def sample_windows():
    """Create Windows instance with sample data.

    Returns:
        Windows: Windows instance with test content
    """
    w = Windows()
    w(input="Hello world Hello test world", n=10)
    return w


def test_ratios_init(basic_ratios):
    """Test Ratios calculator initialization."""
    assert basic_ratios._id == "ratios"
    assert basic_ratios.patterns is None
    assert basic_ratios.windows is None


def test_ratios_call_basic(basic_ratios, sample_windows):
    """Test basic Ratios calculator call."""
    result = basic_ratios(patterns=["Hello", "world"], windows=sample_windows)

    assert isinstance(result, list)
    assert all(isinstance(item, float) for item in result)


def test_ratios_single_pattern(basic_ratios, sample_windows):
    """Test error handling with single pattern."""
    with pytest.raises(LexosException) as exc_info:
        basic_ratios(patterns="single_pattern", windows=sample_windows)
    assert "supply a list of two patterns" in str(exc_info.value)


def test_ratios_wrong_pattern_count(basic_ratios, sample_windows):
    """Test error handling with wrong number of patterns."""
    with pytest.raises(LexosException) as exc_info:
        basic_ratios(patterns=["one", "two", "three"], windows=sample_windows)
    assert "can only calculate ratios for two patterns" in str(exc_info.value)


def test_ratios_no_windows(basic_ratios):
    """Test error handling when no windows provided."""
    with pytest.raises(LexosException) as exc_info:
        basic_ratios(patterns=["test1", "test2"])
    assert "Calculator `windows` attribute is empty" in str(exc_info.value)


@pytest.mark.parametrize(
    "patterns,text,expected_ratio",
    [
        (["a", "b"], "a b a b", 0.5),  # Equal occurrences
        (["a", "b"], "aabab", 0.6),  # Unequal occurrences
    ],
)
def test_ratios_various_patterns(basic_ratios, patterns, text, expected_ratio):
    """Test Ratios calculator with different pattern combinations.

    Args:
        patterns: List of two patterns to compare
        text: Text to search in
        expected_ratio: Expected ratio result
    """
    w = Windows()
    w(input=text, n=len(text))
    result = basic_ratios(patterns=patterns, windows=w)
    assert result[0] == expected_ratio


def test_ratios_spacy_patterns(basic_ratios):
    """Test Ratios calculator with spaCy patterns."""
    w = Windows()
    w(input="The quick brown fox", n=20)
    patterns = [[{"LOWER": "quick"}], [{"LOWER": "brown"}]]

    result = basic_ratios(
        patterns=patterns, windows=w, mode="spacy_rule", model="xx_sent_ud_sm"
    )

    assert isinstance(result, list)
    assert all(isinstance(val, float) for val in result)


def test_ratios_case_sensitivity(basic_ratios, sample_windows):
    """Test case sensitivity in ratio calculations."""
    # Case sensitive
    result_sensitive = basic_ratios(
        patterns=["HELLO", "world"], windows=sample_windows, case_sensitive=True
    )

    # Case insensitive
    result_insensitive = basic_ratios(
        patterns=["HELLO", "world"], windows=sample_windows, case_sensitive=False
    )

    assert result_sensitive != result_insensitive


def test_get_ratio_basic(basic_ratios):
    """Test basic ratio calculation."""
    counts = [1, 1]  # Equal counts
    result = basic_ratios._get_ratio(counts)
    assert result == 0.5  # Equal distribution


def test_get_ratio_zero_denominator(basic_ratios):
    """Test ratio calculation with zero denominator."""
    counts = [1, 0]  # Only numerator has count
    result = basic_ratios._get_ratio(counts)
    assert result == 1.0  # All numerator


def test_get_ratio_zero_numerator(basic_ratios):
    """Test ratio calculation with zero numerator."""
    counts = [0, 1]  # Only denominator has count
    result = basic_ratios._get_ratio(counts)
    assert result == 0.0  # All denominator


def test_get_ratio_both_zero(basic_ratios):
    """Test ratio calculation with both counts zero."""
    counts = [0, 0]  # No counts
    result = basic_ratios._get_ratio(counts)
    assert result == 0.0  # Default to zero


@pytest.mark.parametrize(
    "counts,expected",
    [
        ([2, 2], 0.5),  # Equal counts
        ([3, 1], 0.75),  # More numerator
        ([1, 3], 0.25),  # More denominator
        ([10, 0], 1.0),  # Only numerator
        ([0, 10], 0.0),  # Only denominator
        ([0, 0], 0.0),  # Both zero
    ],
)
def test_get_ratio_various_counts(basic_ratios, counts, expected):
    """Test ratio calculation with various count combinations.

    Args:
        counts: List of two counts [numerator, denominator]
        expected: Expected ratio result
    """
    result = basic_ratios._get_ratio(counts)
    assert result == expected


def test_get_ratio_floating_point_precision(basic_ratios):
    """Test ratio calculation floating point precision."""
    counts = [1, 999]  # Small numerator, large denominator
    result = basic_ratios._get_ratio(counts)
    assert 0 <= result <= 1  # Result should be between 0 and 1
    assert isinstance(result, float)  # Result should be float


#


@pytest.fixture
def sample_data(basic_ratios):
    """Create sample ratio data.

    Returns:
        Ratios: Calculator instance with sample data
    """
    w = Windows()
    w(input="Hello world Hello test world", n=10)
    basic_ratios(patterns=["Hello", "world"], windows=w)
    return basic_ratios


def test_to_df_basic(sample_data):
    """Test basic DataFrame conversion."""
    df = sample_data.to_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1  # Single ratio column
    assert ":" in df.columns[0]  # Column name contains ratio separator
    assert all(isinstance(val, float) for val in df[df.columns[0]])


def test_to_df_spacy_rules(basic_ratios):
    """Test DataFrame conversion with spaCy rules."""
    w = Windows()
    w(input="The quick brown fox", n=20)
    patterns = [[{"LOWER": "quick"}], [{"LOWER": "brown"}]]
    basic_ratios(patterns=patterns, windows=w, mode="spacy_rule", model="xx_sent_ud_sm")

    # Test with show_spacy_rules=True
    df_rules = basic_ratios.to_df(show_spacy_rules=True)
    assert isinstance(df_rules, pd.DataFrame)
    assert len(df_rules.columns) == 1

    # Test with show_spacy_rules=False
    df_simple = basic_ratios.to_df(show_spacy_rules=False)
    assert isinstance(df_simple, pd.DataFrame)
    assert len(df_simple.columns) == 1
    assert "quick:brown" in df_simple.columns[0].lower()


def test_to_df_case_sensitivity(basic_ratios):
    """Test DataFrame conversion with case sensitivity."""
    w = Windows()
    w(input="HELLO world", n=10)
    patterns = ["HELLO", "world"]

    # Case sensitive
    basic_ratios(patterns=patterns, windows=w, case_sensitive=True)
    df_sensitive = basic_ratios.to_df()
    assert "HELLO:world" in df_sensitive.columns[0]

    # Case insensitive
    basic_ratios(patterns=patterns, windows=w, case_sensitive=False)
    df_insensitive = basic_ratios.to_df()
    assert "hello:world" in df_insensitive.columns[0]


def test_to_df_mixed_patterns(basic_ratios):
    """Test DataFrame conversion with mixed pattern types."""
    w = Windows()
    w(input="The quick brown fox", n=20)
    patterns = ["quick", [{"LOWER": "brown"}]]

    basic_ratios(patterns=patterns, windows=w, mode="spacy_rule", model="xx_sent_ud_sm")

    df = basic_ratios.to_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1
    assert "quick" in df.columns[0].lower()
    assert "brown" in df.columns[0].lower()


def test_to_df_empty_data(basic_ratios):
    """Test DataFrame conversion with empty data."""
    basic_ratios.data = []
    basic_ratios.patterns = ["test1", "test2"]
    df = basic_ratios.to_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert len(df.columns) == 1
    assert "test1:test2" in df.columns[0]


@pytest.mark.parametrize(
    "patterns,expected_column",
    [
        (["a", "b"], "a:b"),
        (["test1", "test2"], "test1:test2"),
        ([["quick"], ["brown"]], "quick:brown"),
    ],
)
def test_to_df_column_naming(basic_ratios, patterns, expected_column):
    """Test DataFrame column naming with different patterns.

    Args:
        patterns: List of two patterns to use
        expected_column: Expected column name in result
    """
    basic_ratios.data = [[0.5]]
    basic_ratios.patterns = patterns
    df = basic_ratios.to_df()

    assert expected_column in df.columns[0]
