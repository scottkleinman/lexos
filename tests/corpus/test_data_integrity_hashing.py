"""test_data_integrity_hashing.py.

Test suite for data integrity hashing functionality in spaCy document serialization.
Validates that hash-based corruption detection works correctly.

Coverage: 92%. Missing: 22-26, 35-37, 50, 53-55, 189
Last Update: 2025-06-20.
"""

import hashlib
import tempfile
import uuid
from pathlib import Path

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
def sample_text():
    """Sample text for testing."""
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def complex_text():
    """Complex text with special characters."""
    return "This document contains special characters: café, naïve, résumé. Numbers: 123.45. Symbols: ©, ™."


class TestDataIntegrityHashing:
    """Test data integrity hashing functionality."""

    def test_hash_included_by_default(self, nlp, sample_text):
        """Test that data integrity hash is included by default."""
        original_doc = nlp(sample_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="hash_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Serialize with default settings (should include hash)
        serialized_bytes = record.to_bytes()

        # Verify hash is in serialized data
        import msgpack

        data = msgpack.unpackb(serialized_bytes)
        assert "data_integrity_hash" in data
        assert isinstance(data["data_integrity_hash"], str)
        assert len(data["data_integrity_hash"]) == 64  # SHA256 hex digest length

    def test_hash_can_be_disabled(self, nlp, sample_text):
        """Test that data integrity hash can be disabled."""
        original_doc = nlp(sample_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="no_hash_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Serialize without hash
        serialized_bytes = record.to_bytes(include_hash=False)

        # Verify hash is not in serialized data
        import msgpack

        data = msgpack.unpackb(serialized_bytes)
        assert "data_integrity_hash" not in data

    def test_successful_hash_verification(self, nlp, sample_text):
        """Test successful hash verification during deserialization."""
        original_doc = nlp(sample_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="verify_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Serialize with hash
        serialized_bytes = record.to_bytes()

        # Deserialize with verification (should succeed)
        new_record = Record()
        new_record.from_bytes(
            serialized_bytes, model="en_core_web_sm", verify_hash=True
        )

        # Verify content is correct
        assert new_record.text == sample_text
        assert new_record.name == record.name
        assert new_record.is_parsed is True

    def test_hash_verification_can_be_disabled(self, nlp, sample_text):
        """Test that hash verification can be disabled."""
        original_doc = nlp(sample_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="no_verify_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Serialize with hash
        serialized_bytes = record.to_bytes()

        # Deserialize without verification
        new_record = Record()
        new_record.from_bytes(
            serialized_bytes, model="en_core_web_sm", verify_hash=False
        )

        # Should work fine
        assert new_record.text == sample_text

    def test_corrupted_data_detected_by_hash(self, nlp, sample_text):
        """Test that corrupted data is detected by hash verification."""
        original_doc = nlp(sample_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="corrupt_detect_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Serialize with hash
        serialized_bytes = record.to_bytes()

        # Corrupt the data (modify content but not hash)
        import msgpack

        data = msgpack.unpackb(serialized_bytes)

        # Corrupt the content slightly
        if isinstance(data["content"], bytes):
            corrupted_content = bytearray(data["content"])
            if len(corrupted_content) > 10:
                corrupted_content[10] = (corrupted_content[10] + 1) % 256
            data["content"] = bytes(corrupted_content)
        else:
            data["content"] = "corrupted content"

        corrupted_bytes = msgpack.dumps(data)

        # Try to deserialize corrupted data with verification
        new_record = Record()
        with pytest.raises(LexosException, match="Data integrity check failed"):
            new_record.from_bytes(
                corrupted_bytes, model="en_core_web_sm", verify_hash=True
            )

    def test_hash_survives_roundtrip_with_extensions(self, nlp, sample_text):
        """Test that hash works correctly with custom extensions."""
        # Add custom extension
        if not Token.has_extension("integrity_test"):
            Token.set_extension("integrity_test", default=None)

        original_doc = nlp(sample_text)

        # Set extension values
        for i, token in enumerate(original_doc):
            token._.integrity_test = f"value_{i}"

        record = Record(
            id=str(uuid.uuid4()),
            name="extension_hash_test",
            content=original_doc,
            extensions=["integrity_test"],
        )

        # Serialize with hash
        serialized_bytes = record.to_bytes()

        # Deserialize with verification
        new_record = Record()
        new_record.from_bytes(
            serialized_bytes, model="en_core_web_sm", verify_hash=True
        )

        # Verify content and extensions
        assert new_record.text == sample_text
        for i, token in enumerate(new_record.content):
            assert token._.integrity_test == f"value_{i}"

    def test_complex_document_hash_integrity(self, nlp, complex_text):
        """Test hash integrity with complex documents containing special characters."""
        original_doc = nlp(complex_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="complex_hash_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Serialize with hash
        serialized_bytes = record.to_bytes()

        # Deserialize with verification
        new_record = Record()
        new_record.from_bytes(
            serialized_bytes, model="en_core_web_sm", verify_hash=True
        )

        # Verify complex content is preserved
        assert new_record.text == complex_text
        assert "café" in new_record.text
        assert "naïve" in new_record.text
        assert "©" in new_record.text
        assert "123.45" in new_record.text

    def test_hash_works_with_disk_serialization(self, nlp, sample_text):
        """Test that hash verification works with disk serialization."""
        original_doc = nlp(sample_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="disk_hash_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "hash_test.bin"

            # Save to disk (uses to_bytes internally)
            record.to_disk(file_path)

            # Load from disk with verification
            new_record = Record()
            new_record.from_disk(file_path, model="en_core_web_sm")

            # Should verify hash automatically and succeed
            assert new_record.text == sample_text
            assert new_record.name == record.name

    def test_old_data_without_hash_works(self, nlp, sample_text):
        """Test that old serialized data without hash still works."""
        original_doc = nlp(sample_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="legacy_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Serialize without hash (simulating old data)
        serialized_bytes = record.to_bytes(include_hash=False)

        # Should deserialize successfully even with verify_hash=True
        # (because no hash is present to verify)
        new_record = Record()
        new_record.from_bytes(
            serialized_bytes, model="en_core_web_sm", verify_hash=True
        )

        assert new_record.text == sample_text

    def test_multiple_serialization_cycles_preserve_integrity(self, nlp, sample_text):
        """Test that multiple serialization cycles maintain hash integrity."""
        original_doc = nlp(sample_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="multi_cycle_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        current_record = record

        # Perform multiple serialization cycles
        for cycle in range(3):
            # Serialize with hash
            serialized_bytes = current_record.to_bytes()

            # Deserialize with verification
            new_record = Record()
            new_record.from_bytes(
                serialized_bytes, model="en_core_web_sm", verify_hash=True
            )

            # Verify content hasn't changed
            assert new_record.text == sample_text
            assert new_record.name == record.name

            current_record = new_record

    def test_hash_error_message_helpful(self, nlp, sample_text):
        """Test that hash verification error messages are helpful."""
        original_doc = nlp(sample_text)

        record = Record(
            id=str(uuid.uuid4()),
            name="error_message_test",
            content=original_doc,
            model="en_core_web_sm",
        )

        # Serialize with hash
        serialized_bytes = record.to_bytes()

        # Manually corrupt the hash
        import msgpack

        data = msgpack.unpackb(serialized_bytes)
        data["data_integrity_hash"] = "deliberately_corrupted_hash_value_12345"
        corrupted_bytes = msgpack.dumps(data)

        # Try to deserialize
        new_record = Record()

        with pytest.raises(LexosException) as exc_info:
            new_record.from_bytes(
                corrupted_bytes, model="en_core_web_sm", verify_hash=True
            )

        error_message = str(exc_info.value)
        assert "Data integrity check failed" in error_message
        assert "Hash mismatch detected" in error_message
        assert "Expected:" in error_message
        assert "Got:" in error_message
        assert "corrupted during storage or transmission" in error_message
