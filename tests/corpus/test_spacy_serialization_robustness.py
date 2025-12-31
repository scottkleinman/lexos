"""test_spacy_serialization_robustness.py.

Comprehensive test suite for spaCy document serialization/deserialization robustness
across the entire corpus module. Validates reliable persistent storage of spaCy Doc objects.

Tests cover:
- Basic serialization roundtrip integrity
- Token-level attribute preservation
- Custom extension handling
- Error conditions and edge cases
- Performance with large documents
- Corpus-level serialization robustness

Coverage: 96%. Missing: 34-38, 47-49, 56-57, 70, 73-75
Last Update: 2025-06-20.
"""

import io
import tempfile
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Try to import spacy, skip tests if not available
try:
    import spacy
    from spacy.tokens import Doc, Token

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Doc = None
    Token = None

# Try to import the corpus modules
try:
    from lexos.corpus.record import Record
    from lexos.corpus.utils import LexosModelCache
    from lexos.exceptions import LexosException

    CORPUS_AVAILABLE = True
except ImportError as e:
    CORPUS_AVAILABLE = False
    print(f"Corpus module import failed: {e}")

# Try to import Corpus class separately since it has known issues
try:
    from lexos.corpus.corpus import Corpus

    CORPUS_CLASS_AVAILABLE = True
except ImportError:
    CORPUS_CLASS_AVAILABLE = False

# Mark all tests to skip if dependencies not available
pytestmark = pytest.mark.skipif(
    not CORPUS_AVAILABLE or not SPACY_AVAILABLE,
    reason="Required dependencies not available",
)


@pytest.fixture
def nlp():
    """SpaCy English model fixture."""
    if not SPACY_AVAILABLE:
        pytest.skip("SpaCy not available")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Use blank model if full model not available
        return spacy.blank("en")


@pytest.fixture
def simple_text():
    """Simple text for basic testing."""
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def complex_text():
    """Complex text with various linguistic features."""
    return """This is a complex document with multiple sentences! It contains:
    - Punctuation marks (like these parentheses)
    - Numbers: 123, 456.78, and dates like 2023-12-25
    - Special characters: café, naïve, résumé
    - Unicode symbols: ©, ™, →, ∞
    - Contractions: don't, won't, can't
    - Mixed case: CamelCase, UPPERCASE, lowercase
    """


@pytest.fixture
def large_text():
    """Large text for performance testing."""
    base_text = (
        "This is sentence number {}. It contains various words and punctuation! "
    )
    return "".join(
        base_text.format(i) for i in range(100)
    )  # Reduced for faster testing


@pytest.fixture
def empty_text():
    """Empty text for edge case testing."""
    return ""


@pytest.fixture
def whitespace_text():
    """Text with only whitespace."""
    return "   \n\t   \r\n   "


class TestSpacyDocRoundtripIntegrity:
    """Test basic spaCy document serialization roundtrip integrity."""

    def test_simple_text_roundtrip(self, nlp, simple_text):
        """Test that simple text content matches exactly after serialize/deserialize."""
        # Create original doc
        original_doc = nlp(simple_text)

        # Create record and serialize
        record = Record(
            id=str(uuid.uuid4()),
            name="simple_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Serialize to bytes
        serialized_bytes = record.to_bytes()
        assert isinstance(serialized_bytes, bytes)
        assert len(serialized_bytes) > 0

        # Deserialize
        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

        # Verify text content is identical
        assert new_record.text == original_doc.text
        assert new_record.text == simple_text
        assert new_record.is_parsed is True

    def test_complex_text_roundtrip(self, nlp, complex_text):
        """Test complex text with special characters survives roundtrip."""
        original_doc = nlp(complex_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="complex_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        serialized_bytes = record.to_bytes()

        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

        # Verify exact text match including special characters
        assert new_record.text == complex_text
        assert "café" in new_record.text
        assert "naïve" in new_record.text
        assert "©" in new_record.text
        assert "→" in new_record.text

    def test_empty_doc_roundtrip(self, nlp, empty_text):
        """Test empty document serialization."""
        original_doc = nlp(empty_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="empty_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        serialized_bytes = record.to_bytes()

        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

        assert new_record.text == ""
        assert new_record.is_parsed is True
        assert len(new_record.tokens) == 0

    def test_whitespace_only_doc_roundtrip(self, nlp, whitespace_text):
        """Test document with only whitespace."""
        original_doc = nlp(whitespace_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="whitespace_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        serialized_bytes = record.to_bytes()

        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

        assert new_record.text == whitespace_text
        assert new_record.is_parsed is True


class TestSpacyDocTokenAttributesPreserved:
    """Test that token-level attributes are preserved during serialization."""

    def test_pos_tags_preserved(self, nlp, simple_text):
        """Test POS tags are maintained after serialization."""
        original_doc = nlp(simple_text)

        # Collect original POS tags
        original_pos = [token.pos_ for token in original_doc]

        record = Record(
            id=str(uuid.uuid4()),
            name="pos_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        serialized_bytes = record.to_bytes()

        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

        # Verify POS tags match
        new_pos = [token.pos_ for token in new_record.content]
        assert new_pos == original_pos

    def test_lemmas_preserved(self, nlp, simple_text):
        """Test lemmas are maintained after serialization."""
        original_doc = nlp(simple_text)

        # Collect original lemmas
        original_lemmas = [token.lemma_ for token in original_doc]

        record = Record(
            id=str(uuid.uuid4()),
            name="lemma_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        serialized_bytes = record.to_bytes()

        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

        # Verify lemmas match
        new_lemmas = [token.lemma_ for token in new_record.content]
        assert new_lemmas == original_lemmas

    def test_dependencies_preserved(self, nlp, simple_text):
        """Test dependency relations are maintained."""
        original_doc = nlp(simple_text)

        # Collect original dependencies
        original_deps = [(token.dep_, token.head.i) for token in original_doc]

        record = Record(
            id=str(uuid.uuid4()),
            name="dep_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        serialized_bytes = record.to_bytes()

        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

        # Verify dependencies match
        new_deps = [(token.dep_, token.head.i) for token in new_record.content]
        assert new_deps == original_deps


class TestSpacyDocExtensionsRoundtrip:
    """Test custom Token extensions are preserved during serialization."""

    def test_simple_extension_preservation(self, nlp, simple_text):
        """Test simple custom extension is preserved."""
        # Add custom extension
        if not Token.has_extension("test_flag"):
            Token.set_extension("test_flag", default=False)

        original_doc = nlp(simple_text)

        # Set extension values
        for i, token in enumerate(original_doc):
            token._.test_flag = i % 2 == 0  # Alternate True/False

        record = Record(
            id=str(uuid.uuid4()),
            name="ext_test",
            content=original_doc,
            extensions=["test_flag"],
        )

        serialized_bytes = record.to_bytes()

        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

        # Verify extension values preserved
        for i, token in enumerate(new_record.content):
            expected_value = i % 2 == 0
            assert token._.test_flag == expected_value

    def test_multiple_extensions_preservation(self, nlp, simple_text):
        """Test multiple custom extensions are preserved."""
        # Add multiple extensions
        extensions = ["sentiment", "confidence", "category"]
        for ext in extensions:
            if not Token.has_extension(ext):
                Token.set_extension(ext, default=None)

        original_doc = nlp(simple_text)

        # Set extension values
        for i, token in enumerate(original_doc):
            token._.sentiment = ["positive", "negative", "neutral"][i % 3]
            token._.confidence = round(0.1 * (i + 1), 2)
            token._.category = f"cat_{i}"

        record = Record(
            id=str(uuid.uuid4()),
            name="multi_ext_test",
            content=original_doc,
            extensions=extensions,
        )

        serialized_bytes = record.to_bytes()

        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

        # Verify all extension values preserved
        for i, token in enumerate(new_record.content):
            assert token._.sentiment == ["positive", "negative", "neutral"][i % 3]
            assert token._.confidence == round(0.1 * (i + 1), 2)
            assert token._.category == f"cat_{i}"


class TestLargeSpacyDocSerialization:
    """Test performance and reliability with large spaCy documents."""

    def test_large_doc_serialization_performance(self, nlp, large_text):
        """Test serialization performance with large documents."""
        original_doc = nlp(large_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="large_doc_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Time serialization
        start_time = time.time()
        serialized_bytes = record.to_bytes()
        serialize_time = time.time() - start_time

        # Time deserialization
        start_time = time.time()
        new_record = Record()
        new_record.from_bytes(serialized_bytes, model="en_core_web_sm")
        deserialize_time = time.time() - start_time

        # Verify correctness
        assert new_record.text == large_text
        assert len(new_record.content) == len(original_doc)

        # Performance assertions (reasonable bounds)
        assert serialize_time < 10.0  # Should complete in under 10 seconds
        assert deserialize_time < 10.0
        assert len(serialized_bytes) > 0

        print(f"Large doc serialization: {serialize_time:.3f}s")
        print(f"Large doc deserialization: {deserialize_time:.3f}s")
        print(f"Serialized size: {len(serialized_bytes)} bytes")


class TestCorruptedSerializedDocHandling:
    """Test graceful handling of corrupted serialized data."""

    def test_corrupted_bytes_handling(self, nlp, simple_text):
        """Test handling of corrupted byte data."""
        original_doc = nlp(simple_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="corrupt_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Get valid serialized bytes
        valid_bytes = record.to_bytes()

        # More severely corrupt the bytes - corrupt the msgpack header
        corrupted_bytes = bytearray(valid_bytes)
        # Corrupt the first few bytes which contain msgpack format info
        for i in range(min(5, len(corrupted_bytes))):
            corrupted_bytes[i] = (corrupted_bytes[i] + 100) % 256

        # Attempt to deserialize corrupted data
        new_record = Record()
        with pytest.raises(Exception):  # Should raise msgpack or other exception
            new_record.from_bytes(bytes(corrupted_bytes), model="en_core_web_sm")

    def test_truncated_bytes_handling(self, nlp, simple_text):
        """Test handling of truncated byte data."""
        original_doc = nlp(simple_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="truncated_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        valid_bytes = record.to_bytes()

        # Truncate the bytes
        truncated_bytes = valid_bytes[: len(valid_bytes) // 2]

        new_record = Record()
        with pytest.raises(Exception):
            new_record.from_bytes(truncated_bytes, model="en_core_web_sm")

    def test_invalid_msgpack_data(self):
        """Test handling of invalid msgpack data."""
        invalid_data = b"this is not valid msgpack data"

        new_record = Record()
        with pytest.raises(Exception):
            new_record.from_bytes(invalid_data, model="en_core_web_sm")

    def test_completely_empty_bytes(self):
        """Test handling of completely empty byte data."""
        empty_data = b""

        new_record = Record()
        with pytest.raises(Exception):
            new_record.from_bytes(empty_data, model="en_core_web_sm")


class TestMissingModelDuringDeserialization:
    """Test behavior when spaCy model is unavailable during deserialization."""

    def test_missing_model_error(self, nlp, simple_text):
        """Test error when specified model is not available."""
        original_doc = nlp(simple_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="missing_model_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        serialized_bytes = record.to_bytes()

        # Try to deserialize with non-existent model
        new_record = Record()
        with pytest.raises(Exception):  # Should raise OSError or similar
            new_record.from_bytes(serialized_bytes, model="non_existent_model")

    def test_no_model_specified_error(self, nlp, simple_text):
        """Test error when no model is specified for deserialization."""
        original_doc = nlp(simple_text)

        # Create record without model info stored
        record = Record(
            id=str(uuid.uuid4()),
            name="no_model_test",
            content=original_doc,
            model=None,  # No model stored in record
        )

        serialized_bytes = record.to_bytes()

        # Try to deserialize without specifying model and no model in data
        new_record = Record()
        new_record.model = None  # Ensure no model is set

        with pytest.raises(LexosException, match="No model specified"):
            new_record.from_bytes(
                serialized_bytes
            )  # No model provided and none in data


class TestDiskSerializationRobustness:
    """Test disk serialization/deserialization robustness."""

    def test_disk_roundtrip_integrity(self, nlp, complex_text):
        """Test to_disk/from_disk maintains complete integrity."""
        original_doc = nlp(complex_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="disk_integrity_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_record.bin"

            # Save to disk
            record.to_disk(file_path)
            assert file_path.exists()

            # Load from disk
            new_record = Record()
            new_record.from_disk(file_path, model="en_core_web_sm")

            # Verify complete integrity
            assert new_record.text == complex_text
            assert new_record.name == record.name
            assert new_record.is_active == record.is_active
            assert new_record.is_parsed is True

    def test_corrupted_disk_file_handling(self, nlp, simple_text):
        """Test handling of corrupted disk files."""
        original_doc = nlp(simple_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="corrupted_disk_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "corrupted_record.bin"

            # Save valid file first
            record.to_disk(file_path)

            # Corrupt the file
            with open(file_path, "wb") as f:
                f.write(b"corrupted data")

            # Try to load corrupted file
            new_record = Record()
            with pytest.raises(Exception):
                new_record.from_disk(file_path, model="en_core_web_sm")


@pytest.mark.skipif(not CORPUS_CLASS_AVAILABLE, reason="Corpus class not available")
class TestCorpusLevelSerializationRobustness:
    """Test corpus-level serialization robustness when Corpus class is available."""

    def test_corpus_with_multiple_spacy_docs_integrity(self, nlp):
        """Test corpus containing multiple spaCy docs maintains integrity."""
        texts = [
            "First document with some content.",
            "Second document with different content.",
            "Third document with special characters: café, naïve.",
            "Fourth document with numbers: 123, 456.78.",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Corpus(name="MultiDoc Test", corpus_dir=temp_dir)

            # Add multiple spaCy docs
            for i, text in enumerate(texts):
                doc = nlp(text)
                corpus.add(content=doc, name=f"doc_{i}", model="en_core_web_sm")

            assert corpus.num_docs == len(texts)

            # Verify all documents can be retrieved and have correct content
            for i, original_text in enumerate(texts):
                record = corpus.get(name=f"doc_{i}")
                assert record.text == original_text
                assert record.is_parsed is True


class TestSerializationComprehensiveValidation:
    """Comprehensive validation of serialization robustness."""

    def test_all_serialization_methods_consistency(self, nlp, complex_text):
        """Test that all serialization methods produce consistent results."""
        if not Token.has_extension("consistency_ext"):
            Token.set_extension("consistency_ext", default=None)

        original_doc = nlp(complex_text)

        # Set extension data
        for i, token in enumerate(original_doc):
            token._.consistency_ext = f"consistent_{i}"

        record = Record(
            id=str(uuid.uuid4()),
            name="consistency_test",
            content=original_doc,
            model="en_core_web_sm",
            extensions=["consistency_ext"],
        )

        # Test to_bytes/from_bytes
        bytes_data = record.to_bytes()
        bytes_record = Record()
        bytes_record.from_bytes(bytes_data, model="en_core_web_sm")

        # Test to_disk/from_disk
        with tempfile.TemporaryDirectory() as temp_dir:
            disk_path = Path(temp_dir) / "consistency.bin"
            record.to_disk(disk_path)
            disk_record = Record()
            disk_record.from_disk(disk_path, model="en_core_web_sm")

            # All methods should produce identical results
            assert bytes_record.text == disk_record.text == complex_text
            assert bytes_record.name == disk_record.name == record.name

            # Check extension consistency
            for i, (bytes_token, disk_token) in enumerate(
                zip(bytes_record.content, disk_record.content)
            ):
                expected_value = f"consistent_{i}"
                assert bytes_token._.consistency_ext == expected_value
                assert disk_token._.consistency_ext == expected_value

    def test_serialization_idempotency(self, nlp, simple_text):
        """Test that multiple serialization cycles don't change data."""
        original_doc = nlp(simple_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="idempotency_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        current_record = record

        # Perform multiple serialization cycles
        for cycle in range(3):  # Reduced for faster testing
            # Serialize
            serialized_bytes = current_record.to_bytes()

            # Deserialize
            new_record = Record()
            new_record.from_bytes(serialized_bytes, model="en_core_web_sm")

            # Verify content hasn't changed
            assert new_record.text == simple_text
            assert new_record.name == record.name
            assert len(new_record.content) == len(original_doc)

            current_record = new_record
