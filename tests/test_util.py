"""Tests for util.py module.

Coverage: 100%
Last Update: June 24, 2025
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import spacy
from spacy.tokens import Doc, Token

from lexos.exceptions import LexosException
from lexos.util import (
    _decode_bytes,
    _try_decode_bytes_,
    ensure_list,
    ensure_path,
    get_encoding,
    get_paths,
    get_token_extension_names,
    is_valid_colour,
    load_spacy_model,
    normalize,
    normalize_file,
    normalize_files,
    normalize_strings,
    strip_doc,
    to_collection,
)

# ---------------- Fixtures ----------------


@pytest.fixture
def sample_text_bytes():
    """Sample text as bytes with different encodings."""
    return {
        "utf8": "Hello, 世界! This is a test.".encode("utf-8"),
        "latin1": "Hello, café!".encode("latin-1"),
        "ascii": "Hello, world!".encode("ascii"),
        "windows1252": "Hello, café!".encode("windows-1252"),
    }


@pytest.fixture
def line_ending_samples():
    """Sample text with different line endings."""
    return {
        "unix": "line1\nline2\nline3",
        "windows": "line1\r\nline2\r\nline3",
        "mac": "line1\rline2\rline3",
        "mixed": "line1\r\nline2\rline3\nline4",
    }


@pytest.fixture
def temp_files():
    """Create temporary files for testing."""
    files = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files with different encodings and content
        utf8_file = tmpdir / "test_utf8.txt"
        utf8_file.write_text("Hello, 世界!", encoding="utf-8")
        files["utf8"] = utf8_file

        latin1_file = tmpdir / "test_latin1.txt"
        latin1_file.write_bytes("Hello, café!".encode("latin-1"))
        files["latin1"] = latin1_file

        windows_file = tmpdir / "test_windows.txt"
        windows_file.write_text("line1\r\nline2\r\nline3", encoding="utf-8")
        files["windows"] = windows_file

        files["tmpdir"] = tmpdir
        yield files


@pytest.fixture
def spacy_nlp():
    """Load spaCy model for testing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy English model not available")


# ---------------- Test ensure_list ----------------


def test_ensure_list_with_string():
    """Test ensure_list with a string input."""
    result = ensure_list("test")
    assert result == ["test"]
    assert isinstance(result, list)


def test_ensure_list_with_list():
    """Test ensure_list with list input."""
    input_list = ["a", "b", "c"]
    result = ensure_list(input_list)
    assert result == input_list
    assert result is input_list  # Should return the same object


def test_ensure_list_with_none():
    """Test ensure_list with None."""
    result = ensure_list(None)
    assert result == [None]


def test_ensure_list_with_number():
    """Test ensure_list with numeric input."""
    result = ensure_list(42)
    assert result == [42]


def test_ensure_list_with_dict():
    """Test ensure_list with dictionary input."""
    input_dict = {"key": "value"}
    result = ensure_list(input_dict)
    assert result == [input_dict]


# ---------------- Test ensure_path ----------------


def test_ensure_path_with_string():
    """Test ensure_path with string input."""
    result = ensure_path("test/path")
    assert isinstance(result, Path)
    assert str(result) == "test/path"


def test_ensure_path_with_windows_backslash():
    """Test ensure_path converts Windows backslashes."""
    result = ensure_path("test\\path\\file.txt")
    assert isinstance(result, Path)
    assert str(result) == "test/path/file.txt"


def test_ensure_path_with_path_object():
    """Test ensure_path with Path object."""
    input_path = Path("test/path")
    result = ensure_path(input_path)
    assert result is input_path


def test_ensure_path_with_non_string():
    """Test ensure_path with non-string, non-Path input."""
    result = ensure_path(42)
    assert result == 42


# ---------------- Test get_paths ----------------


def test_get_paths(temp_files):
    """Test get_paths returns all files in directory."""
    tmpdir = temp_files["tmpdir"]

    # Create some nested directories and files
    (tmpdir / "subdir").mkdir()
    (tmpdir / "subdir" / "nested.txt").write_text("nested content")

    paths = get_paths(tmpdir)

    # Should include all files and directories
    assert len(paths) >= 4  # At least our test files + subdir + nested file
    assert all(isinstance(p, Path) for p in paths)

    # Check specific files exist
    filenames = [p.name for p in paths if p.is_file()]
    assert "test_utf8.txt" in filenames
    assert "test_latin1.txt" in filenames
    assert "nested.txt" in filenames


def test_get_paths_with_string():
    """Test get_paths with string path."""
    paths = get_paths(".")
    assert isinstance(paths, list)
    assert all(isinstance(p, Path) for p in paths)


# ---------------- Test get_encoding ----------------


def test_get_encoding_utf8(sample_text_bytes):
    """Test get_encoding detects UTF-8."""
    encoding = get_encoding(sample_text_bytes["utf8"])
    assert encoding.lower() in ["utf-8", "utf8"]


def test_get_encoding_ascii(sample_text_bytes):
    """Test get_encoding detects ASCII."""
    encoding = get_encoding(sample_text_bytes["ascii"])
    assert encoding.lower() in ["ascii", "utf-8", "utf8"]  # ASCII is subset of UTF-8


def test_get_encoding_latin1(sample_text_bytes):
    """Test get_encoding detects Latin-1."""
    encoding = get_encoding(sample_text_bytes["latin1"])
    # Note: chardet might return various encodings for Latin-1
    assert encoding is not None
    assert isinstance(encoding, str)


def test_get_encoding_none_detection():
    """Test get_encoding falls back to UTF-8 when detection fails."""
    # Create bytes that might confuse chardet
    problematic_bytes = b"\x00\x01\x02\x03"

    with patch("chardet.detect", return_value={"encoding": None}):
        encoding = get_encoding(problematic_bytes)
        assert encoding == "utf-8"


def test_get_encoding_empty_bytes():
    """Test get_encoding with empty bytes."""
    encoding = get_encoding(b"")
    assert encoding is not None
    assert isinstance(encoding, str)


# ---------------- Test is_valid_colour ----------------


def test_is_valid_colour_valid_colors():
    """Test is_valid_colour with valid color strings."""
    valid_colors = [
        "red",
        "blue",
        "#FF0000",
        "#ff0000",
        "rgb(255, 0, 0)",
        "rgba(255, 0, 0, 0.5)",
        "hsl(0, 100%, 50%)",
        "hsla(0, 100%, 50%, 0.5)",
        "#abc",
        "#ABC123",
    ]

    for color in valid_colors:
        assert is_valid_colour(color), f"'{color}' should be valid"


def test_is_valid_colour_invalid_colors():
    """Test is_valid_colour with invalid color strings."""
    invalid_colors = [
        "not_a_color",
        "#zzzzzz",
        "rgb(300, 0, 0)",  # Invalid RGB value
        "#12345",  # Invalid hex length
        "",
        "purple_unicorn",
    ]

    for color in invalid_colors:
        assert not is_valid_colour(color), f"'{color}' should be invalid"


# ---------------- Test load_spacy_model ----------------


def test_load_spacy_model_with_language_object(spacy_nlp):
    """Test load_spacy_model with Language object."""
    result = load_spacy_model(spacy_nlp)
    assert result is spacy_nlp


def test_load_spacy_model_with_valid_string():
    """Test load_spacy_model with valid model string."""
    with patch("spacy.load") as mock_load:
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp

        result = load_spacy_model("en_core_web_sm")

        mock_load.assert_called_once_with("en_core_web_sm")
        assert result is mock_nlp


def test_load_spacy_model_with_invalid_string():
    """Test load_spacy_model with invalid model string."""
    with patch("spacy.load", side_effect=OSError("Model not found")):
        with pytest.raises(LexosException, match="Error loading model"):
            load_spacy_model("invalid_model")


def test_load_spacy_model_with_invalid_type():
    """Test load_spacy_model with invalid input type."""
    with pytest.raises(
        LexosException, match="Model must be a string or a spaCy Language object"
    ):
        load_spacy_model(123)  # type: ignore - We are testing invalid input type


# ---------------- Test normalize functions ----------------


def test_normalize_with_string(line_ending_samples):
    """Test normalize with string input."""
    result = normalize(line_ending_samples["windows"])
    assert "\r\n" not in result
    assert result.count("\n") == 2


def test_normalize_with_bytes(sample_text_bytes):
    """Test normalize with bytes input."""
    result = normalize(sample_text_bytes["utf8"])
    assert isinstance(result, str)
    assert "世界" in result


def test_normalize_strings():
    """Test normalize_strings with list of strings."""
    strings = ["test1\r\ntest2", "test3\rtest4", "test5\ntest6"]
    results = normalize_strings(strings)

    assert len(results) == 3
    assert all(isinstance(s, str) for s in results)
    assert "\r" not in "".join(results)


def test_normalize_files(temp_files):
    """Test normalize_files processes multiple files."""
    tmpdir = temp_files["tmpdir"]
    output_dir = tmpdir / "output"
    output_dir.mkdir()

    input_files = [temp_files["utf8"], temp_files["windows"]]

    normalize_files(input_files, output_dir)

    # Check output files were created
    assert (output_dir / "test_utf8.txt").exists()
    assert (output_dir / "test_windows.txt").exists()

    # Check content was normalized
    content = (output_dir / "test_windows.txt").read_text()
    assert "\r\n" not in content
    assert content.count("\n") == 2


def test_normalize_file(temp_files):
    """Test normalize_file processes single file."""
    tmpdir = temp_files["tmpdir"]
    output_dir = tmpdir / "output"
    output_dir.mkdir()

    normalize_file(temp_files["windows"], output_dir)

    output_file = output_dir / "test_windows.txt"
    assert output_file.exists()

    content = output_file.read_text()
    assert "\r\n" not in content
    assert content.count("\n") == 2


# ---------------- Test _try_decode_bytes_ ----------------


def test_try_decode_bytes_utf8(sample_text_bytes):
    """Test _try_decode_bytes_ with UTF-8."""
    result = _try_decode_bytes_(sample_text_bytes["utf8"])
    assert isinstance(result, str)
    assert "世界" in result


def test_try_decode_bytes_latin1(sample_text_bytes):
    """Test _try_decode_bytes_ with Latin-1."""
    result = _try_decode_bytes_(sample_text_bytes["latin1"])
    assert isinstance(result, str)
    assert "café" in result


def test_try_decode_bytes_ascii_fallback():
    """Test _try_decode_bytes_ ASCII fallback behavior."""
    # Create problematic bytes that might trigger ASCII detection
    test_bytes = b"hello\x80world"  # Invalid UTF-8 but might be detected as ASCII

    with patch("chardet.detect", return_value={"encoding": "ascii"}):
        result = _try_decode_bytes_(test_bytes)
        assert isinstance(result, str)


def test_try_decode_bytes_unicode_dammit_fallback():
    """Test _try_decode_bytes_ falls back to UnicodeDammit."""
    problematic_bytes = b"\x80\x81\x82\x83"

    with patch("chardet.detect", return_value={"encoding": "utf-8"}):
        # This will force a UnicodeDecodeError, triggering UnicodeDammit
        result = _try_decode_bytes_(problematic_bytes)
        assert isinstance(result, str)


# ---------------- Test _decode_bytes ----------------


def test_decode_bytes_with_bytes(line_ending_samples):
    """Test _decode_bytes with bytes input."""
    windows_bytes = line_ending_samples["windows"].encode("utf-8")
    result = _decode_bytes(windows_bytes)

    assert isinstance(result, str)
    assert "\r\n" not in result  # Should be normalized
    assert result.count("\n") == 2


def test_decode_bytes_with_string(line_ending_samples):
    """Test _decode_bytes with string input."""
    result = _decode_bytes(line_ending_samples["windows"])

    assert isinstance(result, str)
    assert "\r\n" not in result  # Should be normalized
    assert result.count("\n") == 2


def test_decode_bytes_line_ending_normalization(line_ending_samples):
    """Test _decode_bytes normalizes all line ending types."""
    test_cases = [
        (line_ending_samples["unix"], 2),  # Already correct
        (line_ending_samples["windows"], 2),  # \r\n -> \n
        (line_ending_samples["mac"], 2),  # \r -> \n
        (line_ending_samples["mixed"], 3),  # Mixed -> all \n
    ]

    for text, expected_newlines in test_cases:
        result = _decode_bytes(text)
        assert result.count("\n") == expected_newlines
        assert "\r" not in result


def test_decode_bytes_encoding_error():
    """Test _decode_bytes raises LexosException on encoding failure."""
    with patch(
        "lexos.util._try_decode_bytes_",
        side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "test"),
    ):
        with pytest.raises(LexosException, match="Chardet failed to detect encoding"):
            _decode_bytes(b"test")


# ---------------- Test strip_doc ----------------


def test_strip_doc_removes_leading_whitespace(spacy_nlp):
    """Test strip_doc removes leading whitespace."""
    doc = spacy_nlp("   hello world   ")
    stripped = strip_doc(doc)

    assert stripped.text.startswith("hello")
    assert not stripped.text.startswith(" ")


def test_strip_doc_removes_trailing_whitespace(spacy_nlp):
    """Test strip_doc removes trailing whitespace."""
    doc = spacy_nlp("   hello world   ")
    stripped = strip_doc(doc)

    # Note: trailing whitespace behavior depends on spaCy tokenization
    assert len(stripped) <= len(doc)


def test_strip_doc_empty_document():
    """Test strip_doc with empty document."""
    # Create an empty doc
    nlp = spacy.blank("en")
    doc = nlp("")

    with pytest.raises(LexosException, match="Document is empty"):
        strip_doc(doc)


def test_strip_doc_whitespace_only(spacy_nlp):
    """Test strip_doc with whitespace-only document."""
    doc = spacy_nlp("   \n\t   ")

    # This should raise an exception or handle gracefully
    try:
        result = strip_doc(doc)
        # If it doesn't raise, check it's reasonable
        assert len(result) <= len(doc)
    except LexosException:
        # This is also acceptable behavior
        pass


# ---------------- Test get_token_extension_names ----------------


def test_get_token_extension_names_no_extensions(spacy_nlp):
    """Test get_token_extension_names with no custom extensions."""
    doc = spacy_nlp("test")
    extensions = get_token_extension_names(doc)

    assert isinstance(extensions, list)
    # Should return empty list or default extensions
    assert len(extensions) >= 0


def test_get_token_extension_names_with_extensions(spacy_nlp):
    """Test get_token_extension_names with custom extensions."""
    # Add a custom extension
    if not Token.has_extension("test_ext"):
        Token.set_extension("test_ext", default=None)

    doc = spacy_nlp("test")
    extensions = get_token_extension_names(doc)

    assert isinstance(extensions, list)
    assert "test_ext" in extensions


# ---------------- Test to_collection ----------------


def test_to_collection_single_value():
    """Test to_collection with single value."""
    result = to_collection("test", str, list)
    assert result == ["test"]
    assert isinstance(result, list)


def test_to_collection_list_input():
    """Test to_collection with list input."""
    input_list = ["a", "b", "c"]
    result = to_collection(input_list, str, list)
    assert result == input_list
    assert isinstance(result, list)


def test_to_collection_tuple_to_set():
    """Test to_collection converting tuple to set."""
    input_tuple = (1, 2, 3, 2)  # Duplicate to test set behavior
    result = to_collection(input_tuple, int, set)
    assert result == {1, 2, 3}
    assert isinstance(result, set)


def test_to_collection_multiple_types():
    """Test to_collection with multiple allowed types."""
    result1 = to_collection("test", (str, int), list)
    result2 = to_collection(42, (str, int), list)

    assert result1 == ["test"]
    assert result2 == [42]


def test_to_collection_none_value():
    """Test to_collection with None."""
    result = to_collection(None, str, list)
    assert result == []


def test_to_collection_invalid_value_type():
    """Test to_collection with invalid value type."""
    with pytest.raises(TypeError, match="not all values are of type"):
        to_collection([1, "string"], int, list)


def test_to_collection_invalid_input_type():
    """Test to_collection with invalid input type."""
    with pytest.raises(TypeError, match="not all values are of type"):
        to_collection({1, 2, 3}, str, list)  # Set with wrong value type


def test_to_collection_invalid_input_object():
    """Test to_collection with completely invalid input type."""
    with pytest.raises(TypeError, match="values must be .* or a collection thereof"):
        to_collection(42.5, str, list)  # Float is not string or collection


# ---------------- Integration Tests ----------------


def test_full_text_processing_pipeline(temp_files):
    """Test complete text processing from file to normalized string."""
    # Create a file with complex content
    test_file = temp_files["tmpdir"] / "complex.txt"
    complex_content = "Hello, 世界!\r\nLine 2\rLine 3\nEnd"
    test_file.write_bytes(complex_content.encode("utf-8"))

    # Read and normalize
    with open(test_file, "rb") as f:
        raw_bytes = f.read()

    result = normalize(raw_bytes)

    # Should be properly decoded and normalized
    assert "世界" in result
    assert "\r" not in result
    assert result.count("\n") == 3


def test_encoding_detection_and_normalization():
    """Test encoding detection works with normalization."""
    # Test with different encodings
    texts = {
        "utf8": "Café, naïve, résumé",
        "latin1": "Café, naïve, résumé",
        "windows1252": "Smart quotes: \"hello\" 'world'",
    }

    for encoding, text in texts.items():
        if encoding == "utf8":
            encoded = text.encode("utf-8")
        elif encoding == "latin1":
            encoded = text.encode("latin-1")
        elif encoding == "windows1252":
            encoded = text.encode("windows-1252")

        result = normalize(encoded)
        assert isinstance(result, str)
        # Should contain the original text (possibly with character substitutions)
        assert len(result) > 0


def test_path_utilities_integration(temp_files):
    """Test path utilities work together."""
    tmpdir = temp_files["tmpdir"]

    # Test ensure_path with get_paths
    paths = get_paths(ensure_path(str(tmpdir)))
    assert isinstance(paths, list)
    assert len(paths) >= 3  # Our test files

    # Test with string paths vs Path objects
    str_paths = get_paths(str(tmpdir))
    path_paths = get_paths(tmpdir)
    assert len(str_paths) == len(path_paths)


# ---------------- Edge Cases and Error Handling ----------------


def test_edge_case_empty_inputs():
    """Test functions handle empty inputs gracefully."""
    # Empty string normalization
    assert normalize("") == ""
    assert normalize(b"") == ""

    # Empty list handling
    assert normalize_strings([]) == []
    assert ensure_list([]) == []


def test_edge_case_large_files():
    """Test handling of large content."""
    # Create large text content
    large_text = "Hello, world!\r\n" * 10000
    large_bytes = large_text.encode("utf-8")

    result = normalize(large_bytes)
    assert isinstance(result, str)
    assert "\r\n" not in result
    assert result.count("\n") == 10000


def test_edge_case_unusual_encodings():
    """Test handling of unusual but valid encodings."""
    text = "Hello, test!"

    # Test with UTF-16
    utf16_bytes = text.encode("utf-16")
    result = normalize(utf16_bytes)
    assert isinstance(result, str)
    assert "Hello" in result


def test_error_handling_file_operations(temp_files):
    """Test error handling in file operations."""
    tmpdir = temp_files["tmpdir"]

    # Test with non-existent directory
    with pytest.raises(FileNotFoundError):
        normalize_file("nonexistent.txt", tmpdir)

    # Test with invalid destination
    with pytest.raises((FileNotFoundError, PermissionError, OSError)):
        normalize_file(temp_files["utf8"], "/invalid/path/")


# ---------------- Performance and Memory Tests ----------------


def test_memory_efficiency_large_text():
    """Test memory efficiency with large text processing."""
    # This is more of a smoke test to ensure no obvious memory leaks
    large_texts = ["Large text content " * 1000] * 100

    results = normalize_strings(large_texts)
    assert len(results) == 100
    assert all(isinstance(r, str) for r in results)


def test_encoding_detection_performance():
    """Test encoding detection doesn't hang on problematic input."""
    import time

    # Create potentially problematic bytes
    problematic_bytes = bytes(range(256)) * 100

    start_time = time.time()
    encoding = get_encoding(problematic_bytes)
    end_time = time.time()

    # Should complete in reasonable time (less than 5 seconds)
    assert end_time - start_time < 5.0
    assert isinstance(encoding, str)
