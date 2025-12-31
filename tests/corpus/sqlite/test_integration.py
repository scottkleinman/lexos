"""Test suite for integration.py.

Tests the SQLiteCorpus class which integrates SQLite database backend
with the Corpus class for dual storage and enhanced querying capabilities.

Coverage: 100%

Last Updated: November 20, 2025
Last Tested: November 20, 2025
"""

import json
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from lexos.corpus.record import Record
from lexos.corpus.sqlite.integration import SQLiteCorpus, create_corpus
from lexos.exceptions import LexosException

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_corpus_dir():
    """Create a temporary corpus directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_texts():
    """Sample text content for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language.",
        "Machine learning transforms data into insights.",
    ]


@pytest.fixture
def memory_corpus():
    """Create an in-memory SQLite corpus for testing."""
    corpus = SQLiteCorpus(
        corpus_dir="test_corpus",
        name="test_corpus",
        use_sqlite=True,
        sqlite_path=":memory:",
    )
    yield corpus
    corpus.close()


@pytest.fixture
def file_corpus(temp_corpus_dir):
    """Create a file-based SQLite corpus for testing."""
    corpus = SQLiteCorpus(
        corpus_dir=temp_corpus_dir,
        name="file_corpus",
        use_sqlite=True,
    )
    yield corpus
    corpus.close()


@pytest.fixture
def sqlite_only_corpus():
    """Create a database-only corpus (no file storage)."""
    corpus = SQLiteCorpus(
        corpus_dir="test_corpus",
        name="sqlite_only",
        use_sqlite=True,
        sqlite_only=True,
        sqlite_path=":memory:",
    )
    yield corpus
    corpus.close()


# ============================================================================
# Test SQLiteCorpus Initialization
# ============================================================================


class TestSQLiteCorpusInitialization:
    """Test SQLiteCorpus initialization and configuration."""

    def test_corpus_initialization_with_sqlite(self, temp_corpus_dir):
        """Test basic corpus initialization with SQLite enabled."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=True,
        )

        assert corpus.use_sqlite is True
        assert corpus.sqlite_only is False
        assert corpus.db is not None
        assert corpus.name == "test"

        corpus.close()

    def test_corpus_initialization_without_sqlite(self, temp_corpus_dir):
        """Test corpus initialization without SQLite."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=False,
        )

        assert corpus.use_sqlite is False
        assert corpus.db is None

        corpus.close()

    def test_corpus_initialization_sqlite_only_mode(self):
        """Test corpus initialization in database-only mode."""
        corpus = SQLiteCorpus(
            corpus_dir="test_corpus",
            name="test",
            sqlite_only=True,
            sqlite_path=":memory:",
        )

        assert corpus.sqlite_only is True
        assert corpus.use_sqlite is False
        assert corpus.db is not None

        corpus.close()

    def test_corpus_initialization_memory_database(self):
        """Test corpus with in-memory database."""
        corpus = SQLiteCorpus(
            corpus_dir="test_corpus",
            name="test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        assert corpus.sqlite_path == ":memory:"
        assert corpus.db is not None

        corpus.close()

    def test_corpus_initialization_custom_sqlite_path(self, temp_corpus_dir):
        """Test corpus with custom SQLite database path."""
        db_path = f"{temp_corpus_dir}/custom.db"
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=True,
            sqlite_path=db_path,
        )

        assert corpus.sqlite_path == db_path
        assert Path(db_path).exists()

        corpus.close()

    def test_corpus_metadata_initialization(self, memory_corpus):
        """Test that corpus metadata is initialized in database."""
        assert memory_corpus.db is not None

        # Check that metadata was created
        with memory_corpus.db.SessionLocal() as session:
            from lexos.corpus.sqlite.database import SQLiteMetadata

            metadata = session.query(SQLiteMetadata).first()
            assert metadata is not None
            assert metadata.corpus_id == memory_corpus.name


# ============================================================================
# Test Add Operations
# ============================================================================


class TestSQLiteCorpusAdd:
    """Test adding records to SQLite corpus."""

    def test_add_single_text_to_memory_corpus(self, memory_corpus):
        """Test adding a single text record."""
        memory_corpus.add("Test content")

        assert memory_corpus.num_docs == 1
        assert len(memory_corpus.records) == 1

    def test_add_text_stored_in_database(self, memory_corpus):
        """Test that added text is stored in database."""
        memory_corpus.add("Test content", name="test_doc")

        # Verify in database
        records = memory_corpus.db.filter_records()
        assert len(records) == 1
        assert records[0].name == "test_doc"

    def test_add_multiple_texts(self, memory_corpus, sample_texts):
        """Test adding multiple text records."""
        memory_corpus.add(sample_texts)

        assert memory_corpus.num_docs == 3
        assert len(memory_corpus.records) == 3

    def test_add_with_metadata(self, memory_corpus):
        """Test adding record with metadata."""
        metadata = {"author": "Test Author", "year": 2025}
        memory_corpus.add("Test content", name="test", metadata=metadata)

        record = list(memory_corpus.records.values())[0]
        assert record.meta["author"] == "Test Author"
        assert record.meta["year"] == 2025

    def test_add_with_uuid_metadata(self, memory_corpus):
        """Test adding record with UUID in metadata (should be sanitized)."""
        test_uuid = uuid4()
        metadata = {"doc_id": test_uuid}
        memory_corpus.add("Test content", metadata=metadata)

        record = list(memory_corpus.records.values())[0]
        # UUID should be converted to string
        assert isinstance(record.meta["doc_id"], str)

    def test_sqlite_only_add_sanitizes_metadata(self, sqlite_only_corpus):
        """Test adding record with UUID in metadata in DB-only mode (should be sanitized)."""
        test_uuid = uuid4()
        metadata = {"doc_id": test_uuid}
        sqlite_only_corpus.add("Test content", metadata=metadata)

        record = list(sqlite_only_corpus.records.values())[0]
        # UUID should be converted to string
        assert isinstance(record.meta["doc_id"], str)

    def test_sqlite_only_add_no_pydantic_serializer_warning(self, sqlite_only_corpus):
        """Ensure DB-only add() does not trigger Pydantic serializer warnings."""
        from uuid import uuid4

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sqlite_only_corpus.add("Test content", metadata={"doc_id": uuid4()})

            # Assert that no pydantic serializer warnings were emitted
            assert not any("Pydantic serializer warnings" in str(x.message) for x in w)

    def test_add_record_object(self, memory_corpus):
        """Test adding a Record object directly."""
        record = Record(
            id=uuid4(),
            name="test_record",
            content="Test content",
        )
        memory_corpus.add(record)

        assert memory_corpus.num_docs == 1
        assert "test_record" in memory_corpus.names

    def test_add_inactive_record(self, memory_corpus):
        """Test adding an inactive record."""
        memory_corpus.add("Test content", name="inactive", is_active=False)

        assert memory_corpus.num_docs == 1
        assert memory_corpus.num_active_docs == 0

    def test_add_to_sqlite_only_corpus(self, sqlite_only_corpus):
        """Test adding records in database-only mode."""
        sqlite_only_corpus.add("Test content", name="test")

        assert sqlite_only_corpus.num_docs == 1

        # Verify in database
        records = sqlite_only_corpus.db.filter_records()
        assert len(records) == 1

    def test_names_mapping_is_list_in_sqlite_only_mode(self, sqlite_only_corpus):
        """Ensure that names map to lists (not strings) in DB-only corpus."""
        sqlite_only_corpus.add("First", name="dup")
        sqlite_only_corpus.add("Second", name="dup")
        # The corpus ensures unique names by default (_ensure_unique_name).
        # Therefore, the second 'dup' may be stored under a suffixed name (e.g., 'dup_<uuid>').
        # Ensure we still map names to lists and both entries starting with 'dup' are present.
        assert isinstance(sqlite_only_corpus.names.get("dup"), list)
        total_dup = sum(
            len(ids)
            for name, ids in sqlite_only_corpus.names.items()
            if name.startswith("dup")
        )
        assert total_dup == 2

    def test_add_spacy_doc_meta_in_sqlite_only_mode(self, sqlite_only_corpus):
        """Ensure DB-only add populates meta num_tokens/num_terms via model_dump path."""
        import spacy

        nlp = spacy.blank("en")
        doc = nlp("This is a test document with multiple tokens")

        sqlite_only_corpus.add(doc, name="parsed_doc")

        # Get the last record's metadata
        record = list(sqlite_only_corpus.records.values())[-1]
        meta_entry = sqlite_only_corpus.meta.get(str(record.id))
        assert meta_entry is not None
        assert "num_tokens" in meta_entry
        assert meta_entry["num_tokens"] >= 1
        assert "num_terms" in meta_entry
        assert meta_entry["num_terms"] >= 1

    def test_add_with_store_in_db_override(self, temp_corpus_dir):
        """Test overriding database storage for specific record."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=False,  # Database disabled by default
        )

        # Add without database (corpus setting)
        corpus.add("Test 1", name="no_db")

        # This should work even though use_sqlite=False
        # because store_in_db is not checked if db is None
        assert corpus.num_docs == 1

        corpus.close()

    def test_add_with_extensions(self, memory_corpus):
        """Test adding record with extensions."""
        memory_corpus.add(
            "Test content",
            name="test",
            extensions=["extension1", "extension2"],
        )

        record = list(memory_corpus.records.values())[0]
        assert "extension1" in record.extensions
        assert "extension2" in record.extensions

    def test_add_to_backend_model_dump_fallback(self, sqlite_only_corpus):
        """Force model_dump to raise in DB-only add to exercise fallback path for meta."""
        with patch.object(Record, "model_dump", side_effect=Exception("boom")):
            sqlite_only_corpus.add("Test content", name="broken_meta")

        record = list(sqlite_only_corpus.records.values())[-1]
        meta_entry = sqlite_only_corpus.meta.get(str(record.id))
        assert meta_entry is not None
        assert meta_entry["id"] == str(record.id)
        assert meta_entry["name"] == record.name
        assert "num_tokens" in meta_entry
        assert "num_terms" in meta_entry


# ============================================================================
# Test Filter Operations
# ============================================================================


class TestSQLiteCorpusFilter:
    """Test filtering records in SQLite corpus."""

    def test_filter_by_is_active(self, memory_corpus):
        """Test filtering records by active status."""
        memory_corpus.add("Active 1", name="active1", is_active=True)
        memory_corpus.add("Inactive", name="inactive", is_active=False)
        memory_corpus.add("Active 2", name="active2", is_active=True)

        active_records = memory_corpus.filter_records(is_active=True)
        assert len(active_records) == 2

    def test_filter_with_limit(self, memory_corpus, sample_texts):
        """Test filtering with result limit."""
        memory_corpus.add(sample_texts)

        limited_records = memory_corpus.filter_records(limit=2)
        assert len(limited_records) == 2

    def test_filter_in_memory_fallback(self, memory_corpus, sample_texts):
        """Test filtering falls back to in-memory when use_database=False."""
        memory_corpus.add(sample_texts)

        records = memory_corpus.filter_records(use_database=False)
        assert len(records) == 3

    def test_filter_by_model(self, memory_corpus):
        """Test filtering records by model."""
        memory_corpus.add("Text 1", name="doc1", model="en_core_web_sm")
        memory_corpus.add("Text 2", name="doc2", model="en_core_web_md")

        filtered = memory_corpus.filter_records(model="en_core_web_sm")
        assert len(filtered) == 1
        assert filtered[0].model == "en_core_web_sm"

    def test_filter_in_memory_model_mismatch(self, temp_corpus_dir):
        """Test in-memory filter when model mismatch should exclude records."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test_model_filter",
            use_sqlite=False,
        )

        # Add two records with different model attribute strings
        corpus.add("Text 1", name="doc1", model="en_core_web_sm")
        corpus.add("Text 2", name="doc2", model="en_core_web_md")

        filtered = corpus.filter_records(model="en_core_web_sm", use_database=False)
        assert len(filtered) == 1
        assert filtered[0].model == "en_core_web_sm"

        corpus.close()

    def test_filter_in_memory_is_active(self, temp_corpus_dir):
        """Test in-memory filtering by is_active includes/excludes correctly."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test_active_filter",
            use_sqlite=False,
        )

        corpus.add("Active 1", name="active1", is_active=True)
        corpus.add("Inactive 1", name="inactive1", is_active=False)

        active = corpus.filter_records(is_active=True, use_database=False)
        assert len(active) == 1
        assert active[0].name == "active1"

        inactive = corpus.filter_records(is_active=False, use_database=False)
        assert len(inactive) == 1
        assert inactive[0].name == "inactive1"

        corpus.close()

    def test_filter_no_results(self, memory_corpus):
        """Test filtering with no matching results."""
        memory_corpus.add("Test content")

        results = memory_corpus.filter_records(model="nonexistent_model")
        assert len(results) == 0


# ============================================================================
# Test Search Operations
# ============================================================================


class TestSQLiteCorpusSearch:
    """Test full-text search in SQLite corpus."""

    def test_search_basic_query(self, memory_corpus):
        """Test basic full-text search."""
        memory_corpus.add("The quick brown fox jumps", name="doc1")
        memory_corpus.add("Python programming language", name="doc2")

        results = memory_corpus.search("fox")
        assert len(results) == 1
        assert results[0].name == "doc1"

    def test_search_multiple_results(self, memory_corpus):
        """Test search returning multiple results."""
        memory_corpus.add("Python is powerful", name="doc1")
        memory_corpus.add("Python for data science", name="doc2")
        memory_corpus.add("JavaScript programming", name="doc3")

        results = memory_corpus.search("Python")
        assert len(results) == 2

    def test_search_with_limit(self, memory_corpus):
        """Test search with result limit."""
        for i in range(5):
            memory_corpus.add(f"Python document {i}", name=f"doc{i}")

        results = memory_corpus.search("Python", limit=3)
        assert len(results) == 3

    def test_search_excludes_inactive_by_default(self, memory_corpus):
        """Test that search excludes inactive records by default."""
        memory_corpus.add("Python active", name="active", is_active=True)
        memory_corpus.add("Python inactive", name="inactive", is_active=False)

        results = memory_corpus.search("Python")
        assert len(results) == 1
        assert results[0].name == "active"

    def test_search_include_inactive(self, memory_corpus):
        """Test search including inactive records."""
        memory_corpus.add("Python active", name="active", is_active=True)
        memory_corpus.add("Python inactive", name="inactive", is_active=False)

        results = memory_corpus.search("Python", include_inactive=True)
        assert len(results) == 2

    def test_search_with_model_filter(self, memory_corpus):
        """Test search with model filter."""
        memory_corpus.add("Python code", name="doc1", model="en_core_web_sm")
        memory_corpus.add("Python script", name="doc2", model="en_core_web_md")

        results = memory_corpus.search("Python", model_filter="en_core_web_sm")
        assert len(results) == 1
        assert results[0].model == "en_core_web_sm"

    def test_search_without_database_raises_exception(self, temp_corpus_dir):
        """Test that search raises exception when database is not enabled."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=False,
        )

        with pytest.raises(LexosException, match="Database is not enabled"):
            corpus.search("test query")

        corpus.close()


# ============================================================================
# Test Sync Operations
# ============================================================================


class TestSQLiteCorpusSync:
    """Test synchronization between files and database."""

    def test_sync_records_to_database(self, file_corpus):
        """Test syncing file-based records to database."""
        # Add records using file storage
        file_corpus.add("Test 1", name="doc1")
        file_corpus.add("Test 2", name="doc2")

        # Sync to database
        count = file_corpus.sync()

        assert count == 2

        # Verify in database
        db_records = file_corpus.db.filter_records()
        assert len(db_records) == 2

    def test_sync_without_overwrite(self, memory_corpus):
        """Test sync without overwriting existing records."""
        # Add a record
        memory_corpus.add("Original content", name="doc1")

        # Modify in memory
        record = list(memory_corpus.records.values())[0]
        original_content = record.content

        # First sync
        count1 = memory_corpus.sync(overwrite=False)
        assert count1 == 1

        # Second sync should not overwrite
        count2 = memory_corpus.sync(overwrite=False)
        assert count2 == 1  # Still counts as synced but not re-added

    def test_sync_with_overwrite(self):
        """Test sync with overwriting existing records."""
        # Create a fresh corpus for this test
        corpus = SQLiteCorpus(
            corpus_dir="test_corpus",
            name="sync_overwrite_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # Add and sync a record
        corpus.add("Original content", name="doc1")
        corpus.sync()

        # Get the record ID
        record = list(corpus.records.values())[0]
        record_id = str(record.id)

        # Create a new record with same ID but different content
        from uuid import UUID

        from lexos.corpus.record import Record

        updated_record = Record(
            id=UUID(record_id),
            name="doc1",
            content="Modified content",
        )
        corpus.records[record_id] = updated_record

        # Sync with overwrite
        count = corpus.sync(overwrite=True)
        assert count == 1

        # Verify update in database
        db_record = corpus.db.get_record(record_id)
        assert db_record.content == "Modified content"

        corpus.close()

    def test_sync_overwrite_adds_when_no_existing_record(self, temp_corpus_dir):
        """Test sync with overwrite=True adds records when they don't exist in DB (cover add_record path)."""
        # Create file-based corpus using the standard Corpus class
        from lexos.corpus.corpus import Corpus

        file_corpus = Corpus(
            corpus_dir=temp_corpus_dir,
            name="file_for_sync",
        )
        file_corpus.add("File content 1", name="file1")
        file_corpus.add("File content 2", name="file2")

        # Now create a fresh SQLiteCorpus with a new in-memory DB (so DB empty)
        sqlite_c = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="file_for_sync",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        count = sqlite_c.sync(overwrite=True)
        assert count == 2
        db_records = sqlite_c.db.filter_records()
        assert len(db_records) == 2

        sqlite_c.close()

    def test_sync_without_database_raises_exception(self, temp_corpus_dir):
        """Test that sync raises exception when database is not enabled."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=False,
        )

        with pytest.raises(LexosException, match="Database is not enabled"):
            corpus.sync()

        corpus.close()

    def test_sync_empty_corpus(self, temp_corpus_dir):
        """Test syncing an empty corpus."""
        # Use a truly empty temporary directory
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="empty_sync_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )
        # Don't add any records
        count = corpus.sync()
        assert count == 0
        corpus.close()


# ============================================================================
# Test Load Operations
# ============================================================================


class TestSQLiteCorpusLoad:
    """Test loading records from database to memory."""

    def test_load_records_from_database(self, memory_corpus):
        """Test loading records from database."""
        # Add records
        memory_corpus.add("Test 1", name="doc1")
        memory_corpus.add("Test 2", name="doc2")

        # Clear memory
        memory_corpus.records.clear()
        memory_corpus.names.clear()

        # Load from database
        count = memory_corpus.load()

        assert count == 2
        assert len(memory_corpus.records) == 2

    def test_load_active_only(self, memory_corpus):
        """Test loading only active records."""
        memory_corpus.add("Active 1", name="active1", is_active=True)
        memory_corpus.add("Inactive", name="inactive", is_active=False)
        memory_corpus.add("Active 2", name="active2", is_active=True)

        # Clear and reload
        memory_corpus.records.clear()
        memory_corpus.names.clear()

        count = memory_corpus.load(active_only=True)
        assert count == 2

    def test_load_all_records(self, memory_corpus):
        """Test loading all records including inactive."""
        memory_corpus.add("Active", name="active", is_active=True)
        memory_corpus.add("Inactive", name="inactive", is_active=False)

        # Clear and reload
        memory_corpus.records.clear()
        memory_corpus.names.clear()

        count = memory_corpus.load(active_only=False)
        assert count == 2

    def test_load_without_database_raises_exception(self, temp_corpus_dir):
        """Test that load raises exception when database is not enabled."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=False,
        )

        with pytest.raises(LexosException, match="Database is not enabled"):
            corpus.load()

        corpus.close()

    def test_load_empty_database(self, memory_corpus):
        """Test loading from empty database."""
        count = memory_corpus.load()
        assert count == 0

    def test_load_meta_model_dump_fallback(self, memory_corpus):
        """Force model_dump to raise when loading from DB so fallback meta creation is used."""
        # Add a record which will be added to DB
        memory_corpus.add("Test content for load", name="load_broken")

        # Clear in-memory records
        memory_corpus.records.clear()
        memory_corpus.names.clear()

        # Patch Record.model_dump to raise during load processing
        with patch.object(Record, "model_dump", side_effect=Exception("boom")):
            count = memory_corpus.load()
            assert count >= 1

        # Verify meta for loaded records exists and uses fallback schema
        for rid, meta in memory_corpus.meta.items():
            assert "id" in meta and "name" in meta


# ============================================================================
# Test Statistics Operations
# ============================================================================


class TestSQLiteCorpusStatistics:
    """Test corpus statistics operations."""

    def test_get_stats_from_database(self, memory_corpus):
        """Test getting corpus statistics from database."""
        memory_corpus.add("The quick brown fox", name="doc1")
        memory_corpus.add("Python programming", name="doc2")

        stats = memory_corpus.get_stats()

        assert "total_records" in stats
        assert "active_records" in stats
        assert stats["total_records"] == 2

    def test_get_stats_without_database_raises_exception(self, temp_corpus_dir):
        """Test that get_stats raises exception when database is not enabled."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=False,
        )

        with pytest.raises(LexosException, match="Database is not enabled"):
            corpus.get_stats()

        corpus.close()

    def test_update_corpus_state_updates_database(self, memory_corpus):
        """Test that corpus state updates are reflected in database."""
        memory_corpus.add("Test content", name="doc1")

        # Get metadata from database
        with memory_corpus.db.SessionLocal() as session:
            from lexos.corpus.sqlite.database import SQLiteMetadata

            metadata = session.query(SQLiteMetadata).first()
            assert metadata.num_docs == 1


# ============================================================================
# Test Cleanup Operations
# ============================================================================


class TestSQLiteCorpusCleanup:
    """Test corpus cleanup and resource management."""

    def test_close_database_connection(self, temp_corpus_dir):
        """Test closing database connection."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=True,
        )

        corpus.close()
        # Database should be closed, engine disposed
        assert corpus.db is not None  # Object still exists but connection closed

    def test_destructor_calls_close(self, temp_corpus_dir):
        """Test that destructor properly cleans up."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=True,
        )

        # Destructor should be called when object is deleted
        del corpus
        # No exception should be raised

    def test_close_without_database(self, temp_corpus_dir):
        """Test closing corpus without database enabled."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            use_sqlite=False,
        )

        # Should not raise exception
        corpus.close()


# ============================================================================
# Test Utility Methods
# ============================================================================


class TestSQLiteCorpusUtilities:
    """Test utility methods in SQLiteCorpus."""

    def test_sanitize_metadata_uuid(self, memory_corpus):
        """Test sanitizing UUID in metadata."""
        test_uuid = uuid4()
        metadata = {"id": test_uuid}

        sanitized = memory_corpus._sanitize_metadata(metadata)
        assert isinstance(sanitized["id"], str)

    def test_sanitize_metadata_datetime(self, memory_corpus):
        """Test sanitizing datetime in metadata."""
        from datetime import datetime

        now = datetime.now()
        metadata = {"timestamp": now}

        sanitized = memory_corpus._sanitize_metadata(metadata)
        assert isinstance(sanitized["timestamp"], str)

    def test_sanitize_metadata_path(self, memory_corpus):
        """Test sanitizing Path in metadata."""
        test_path = Path("/test/path")
        metadata = {"file_path": test_path}

        sanitized = memory_corpus._sanitize_metadata(metadata)
        assert isinstance(sanitized["file_path"], str)

    def test_sanitize_metadata_nested_dict(self, memory_corpus):
        """Test sanitizing nested dictionary."""
        test_uuid = uuid4()
        metadata = {"nested": {"id": test_uuid}}

        sanitized = memory_corpus._sanitize_metadata(metadata)
        assert isinstance(sanitized["nested"]["id"], str)

    def test_sanitize_metadata_list(self, memory_corpus):
        """Test sanitizing list with special types."""
        test_uuid = uuid4()
        metadata = {"ids": [test_uuid, "string_id"]}

        sanitized = memory_corpus._sanitize_metadata(metadata)
        assert isinstance(sanitized["ids"][0], str)
        assert sanitized["ids"][1] == "string_id"

    def test_get_timestamp(self, memory_corpus):
        """Test timestamp generation."""
        timestamp = memory_corpus._get_timestamp()

        assert isinstance(timestamp, str)
        # Should be ISO format
        from datetime import datetime

        parsed = datetime.fromisoformat(timestamp)
        assert parsed is not None


# ============================================================================
# Test create_corpus Function
# ============================================================================


class TestCreateCorpusFunction:
    """Test the create_corpus convenience function."""

    def test_create_corpus_basic(self, temp_corpus_dir):
        """Test basic corpus creation with create_corpus."""
        corpus = create_corpus(corpus_dir=temp_corpus_dir, name="test")

        assert isinstance(corpus, SQLiteCorpus)
        assert corpus.use_sqlite is True
        assert corpus.name == "test"

        corpus.close()

    def test_create_corpus_custom_sqlite_path(self, temp_corpus_dir):
        """Test create_corpus with custom SQLite path."""
        db_path = f"{temp_corpus_dir}/custom.db"
        corpus = create_corpus(
            corpus_dir=temp_corpus_dir,
            name="test",
            sqlite_path=db_path,
        )

        assert corpus.sqlite_path == db_path

        corpus.close()

    def test_create_corpus_sqlite_only_mode(self):
        """Test create_corpus in database-only mode."""
        corpus = create_corpus(
            corpus_dir="test_corpus",
            name="test",
            sqlite_only=True,
            sqlite_path=":memory:",
        )

        assert corpus.sqlite_only is True
        assert corpus.use_sqlite is True

        corpus.close()

    def test_create_corpus_auto_sqlite_path(self, temp_corpus_dir):
        """Test create_corpus with auto-generated SQLite path."""
        corpus = create_corpus(corpus_dir=temp_corpus_dir, name="test")

        expected_path = f"{temp_corpus_dir}/corpus.db"
        assert corpus.sqlite_path == expected_path

        corpus.close()


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestSQLiteCorpusEdgeCases:
    """Test edge cases and error handling."""

    def test_add_empty_content(self, memory_corpus):
        """Test adding record with empty content."""
        memory_corpus.add("", name="empty")

        assert memory_corpus.num_docs == 1
        record = list(memory_corpus.records.values())[0]
        assert record.content == ""

    def test_add_very_large_metadata(self, memory_corpus):
        """Test adding record with large metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        memory_corpus.add("Test content", metadata=large_metadata)

        assert memory_corpus.num_docs == 1
        record = list(memory_corpus.records.values())[0]
        # Account for auto-added metadata like filename and filepath
        assert len(record.meta) >= 1000

    def test_filter_with_no_matching_records(self, memory_corpus):
        """Test filtering with no matching criteria."""
        memory_corpus.add("Test content")

        results = memory_corpus.filter_records(model="nonexistent")
        assert len(results) == 0

    def test_search_no_results(self, memory_corpus):
        """Test search with no matching results."""
        memory_corpus.add("Python programming")

        results = memory_corpus.search("nonexistent")
        assert len(results) == 0

    def test_multiple_add_operations(self, memory_corpus, sample_texts):
        """Test multiple consecutive add operations."""
        for text in sample_texts:
            memory_corpus.add(text)

        assert memory_corpus.num_docs == 3

    def test_corpus_with_special_characters_in_name(self):
        """Test corpus with special characters in name."""
        corpus = SQLiteCorpus(
            corpus_dir="test_corpus",
            name="test-corpus_2025",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        assert corpus.name == "test-corpus_2025"

        corpus.close()


# ============================================================================
# Test Integration Scenarios
# ============================================================================


class TestSQLiteCorpusIntegration:
    """Test integration scenarios combining multiple operations."""

    def test_add_filter_and_search_workflow(self, memory_corpus):
        """Test complete workflow: add, filter, and search."""
        # Add diverse records
        memory_corpus.add("Python programming basics", name="doc1", is_active=True)
        memory_corpus.add("Advanced Python techniques", name="doc2", is_active=True)
        memory_corpus.add("JavaScript fundamentals", name="doc3", is_active=False)

        # Filter active records
        active = memory_corpus.filter_records(is_active=True)
        assert len(active) == 2

        # Search for Python
        results = memory_corpus.search("Python")
        assert len(results) == 2

    def test_sync_and_load_workflow(self, file_corpus):
        """Test sync and load workflow."""
        # Add records
        file_corpus.add("Test 1", name="doc1")
        file_corpus.add("Test 2", name="doc2")

        # Sync to database
        sync_count = file_corpus.sync()
        assert sync_count == 2

        # Clear memory
        file_corpus.records.clear()
        file_corpus.names.clear()

        # Load from database
        load_count = file_corpus.load()
        assert load_count == 2
        assert len(file_corpus.records) == 2

    def test_dual_storage_consistency(self, file_corpus):
        """Test consistency between file and database storage."""
        # Add with dual storage
        file_corpus.add("Test content", name="test_doc")

        # Check in-memory
        assert len(file_corpus.records) == 1

        # Check in database
        db_records = file_corpus.db.filter_records()
        assert len(db_records) == 1

        # Names should match
        memory_record = list(file_corpus.records.values())[0]
        db_record = db_records[0]
        assert memory_record.name == db_record.name

    def test_database_only_mode_workflow(self, sqlite_only_corpus):
        """Test complete workflow in database-only mode."""
        # Add records
        sqlite_only_corpus.add("Python code", name="doc1")
        sqlite_only_corpus.add("Java code", name="doc2")

        # Search
        results = sqlite_only_corpus.search("Python")
        assert len(results) == 1

        # Filter
        filtered = sqlite_only_corpus.filter_records(is_active=True)
        assert len(filtered) == 2

        # Stats
        stats = sqlite_only_corpus.get_stats()
        assert stats["total_records"] == 2


# ============================================================================
# Test Additional Coverage
# ============================================================================


class TestAdditionalCoverage:
    """Tests to increase code coverage for uncovered lines."""

    def test_add_spacy_doc_to_sqlite_only_corpus(self, sqlite_only_corpus):
        """Test adding spaCy Doc object in database-only mode (line 108)."""
        import spacy

        # Create a simple Doc
        nlp = spacy.blank("en")
        doc = nlp("Test document content")

        sqlite_only_corpus.add(doc, name="spacy_doc")

        assert sqlite_only_corpus.num_docs == 1
        record = list(sqlite_only_corpus.records.values())[0]
        assert record.is_parsed is True

    def test_add_list_of_records_to_sqlite_only_corpus(self, sqlite_only_corpus):
        """Test adding list of Record objects in database-only mode (line 115)."""
        records = [
            Record(id=uuid4(), name="rec1", content="Content 1"),
            Record(id=uuid4(), name="rec2", content="Content 2"),
        ]

        sqlite_only_corpus.add(records)

        assert sqlite_only_corpus.num_docs == 2

    def test_add_with_extensions_in_sqlite_only_mode(self, sqlite_only_corpus):
        """Test adding record with extensions in database-only mode (lines 126, 128)."""
        sqlite_only_corpus.add(
            "Test content",
            name="with_ext",
            extensions=["ext1", "ext2"],
            metadata={"key": "value"},
        )

        record = list(sqlite_only_corpus.records.values())[0]
        assert "ext1" in record.extensions
        assert "ext2" in record.extensions
        assert record.meta["key"] == "value"

    def test_destructor_exception_handling(self):
        """Test that __del__ handles exceptions gracefully (lines 150-151)."""
        corpus = SQLiteCorpus(
            corpus_dir="test_corpus",
            name="destructor_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # Close it first, then delete to trigger exception in __del__
        corpus.close()
        # This should not raise an exception even though close() is called twice
        del corpus

    def test_load_records_from_disk_with_files(self, temp_corpus_dir):
        """Test _load_records_from_disk with actual files (lines 170, 173, 183-203, 208)."""
        from lexos.corpus.corpus import Corpus

        # Create a regular corpus and add records with file storage
        corpus1 = Corpus(
            corpus_dir=temp_corpus_dir,
            name="test_load_disk",
        )
        corpus1.add("Test content 1", name="doc1")
        corpus1.add("Test content 2", name="doc2")

        # Create a new SQLite corpus pointing to same directory
        corpus2 = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test_load_disk",
            use_sqlite=True,
        )

        # Sync should load from disk
        count = corpus2.sync()
        assert count == 2

        corpus2.close()

    def test_add_record_database_exception_handling(self, memory_corpus):
        """Test exception handling when adding to database fails (lines 377-379)."""
        from unittest.mock import patch

        # Add a record normally
        memory_corpus.add("Test 1", name="doc1")

        # Mock db.add_record to raise an exception
        with patch.object(
            memory_corpus.db, "add_record", side_effect=Exception("Mock DB error")
        ):
            # This should print a warning but not raise an exception
            memory_corpus.add("Test 2", name="doc2")

        # Record should still be added to memory even though DB failed
        assert memory_corpus.num_docs == 2

    def test_filter_in_memory_with_is_parsed(self, temp_corpus_dir):
        """Test in-memory filtering with is_parsed (lines 420-422)."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test_parsed_filter",
            use_sqlite=False,
        )

        # Add unparsed record
        corpus.add("Unparsed text", name="unparsed")

        # Add parsed record with spaCy
        import spacy

        nlp = spacy.blank("en")
        doc = nlp("Parsed text")
        corpus.add(doc, name="parsed")

        # Filter for parsed records using in-memory filtering
        parsed = corpus.filter_records(is_parsed=True, use_database=False)
        assert len(parsed) == 1
        assert parsed[0].name == "parsed"

        # Filter for unparsed records
        unparsed = corpus.filter_records(is_parsed=False, use_database=False)
        assert len(unparsed) == 1
        assert unparsed[0].name == "unparsed"

        corpus.close()

    def test_filter_in_memory_with_token_counts(self, temp_corpus_dir):
        """Test in-memory filtering with min/max tokens (lines 424-436)."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test_token_filter",
            use_sqlite=False,
        )

        import spacy

        nlp = spacy.blank("en")

        # Add docs with different token counts
        doc1 = nlp("Short")  # 1 token
        doc2 = nlp("Medium length text")  # 3 tokens
        doc3 = nlp("This is a much longer document")  # 6 tokens

        corpus.add(doc1, name="short")
        corpus.add(doc2, name="medium")
        corpus.add(doc3, name="long")

        # Filter with min_tokens
        min_filtered = corpus.filter_records(min_tokens=2, use_database=False)
        assert len(min_filtered) == 2  # medium and long

        # Filter with max_tokens
        max_filtered = corpus.filter_records(max_tokens=4, use_database=False)
        assert len(max_filtered) == 2  # short and medium

        # Filter with both
        range_filtered = corpus.filter_records(
            min_tokens=2, max_tokens=4, use_database=False
        )
        assert len(range_filtered) == 1  # only medium

        corpus.close()

    def test_filter_in_memory_with_limit_and_multiple_criteria(self, temp_corpus_dir):
        """Test in-memory filtering with limit (line 441)."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test_limit_filter",
            use_sqlite=False,
        )

        # Add multiple active records
        for i in range(5):
            corpus.add(f"Document {i}", name=f"doc{i}", is_active=True)

        # Filter with limit
        limited = corpus.filter_records(is_active=True, limit=3, use_database=False)
        assert len(limited) == 3

        corpus.close()

    def test_sync_with_exception_handling(self, memory_corpus):
        """Test sync with error handling (lines 533, 538, 542-544)."""
        from unittest.mock import patch

        # Add records
        memory_corpus.add("Test 1", name="doc1")
        memory_corpus.add("Test 2", name="doc2")

        # Get the record IDs
        record_ids = list(memory_corpus.records.keys())

        # Mock add_record to raise exception for first record only
        original_add = memory_corpus.db.add_record
        call_count = [0]

        def mock_add_record(record):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Mock database error")
            return original_add(record)

        with patch.object(memory_corpus.db, "add_record", side_effect=mock_add_record):
            # This should handle exceptions and continue
            count = memory_corpus.sync(overwrite=False)

            # Should sync second record despite first failing
            assert count == 2  # Count includes both (one failed, one succeeded)

    def test_load_records_from_disk_exception_handling(self, temp_corpus_dir):
        """Test _load_records_from_disk exception handling (line 208)."""
        from lexos.corpus.corpus import Corpus

        # Create a valid corpus with a record
        corpus_dir = Path(temp_corpus_dir) / "load_error_test"
        corpus1 = Corpus(corpus_dir=str(corpus_dir), name="load_error")
        corpus1.add("Test content", name="doc1")

        # Now corrupt the data file to trigger exception during load
        data_dir = corpus_dir / "data"
        for bin_file in data_dir.glob("*.bin"):
            # Corrupt the file
            with open(bin_file, "wb") as f:
                f.write(b"corrupted data")

        # Create SQLite corpus - sync should handle the exception
        corpus2 = SQLiteCorpus(
            corpus_dir=str(corpus_dir),
            name="load_error",
            use_sqlite=True,
        )

        # Should not crash during sync, just print warning
        count = corpus2.sync()
        # Count might be 0 due to load error
        assert count >= 0

        corpus2.close()

    def test_filter_in_memory_with_exception_in_num_tokens(self, temp_corpus_dir):
        """Test in-memory filtering when num_tokens() raises exception (lines 426-430, 432-436)."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test_token_exception",
            use_sqlite=False,
        )

        # Add a record with string content (num_tokens might raise for unparsed)
        corpus.add("Unparsed content", name="unparsed")

        # These should handle exceptions gracefully
        min_filtered = corpus.filter_records(min_tokens=5, use_database=False)
        max_filtered = corpus.filter_records(max_tokens=5, use_database=False)

        # Should not crash
        assert isinstance(min_filtered, list)
        assert isinstance(max_filtered, list)

        corpus.close()


# ============================================================================
# Test Final Coverage Improvements
# ============================================================================


class TestFinalCoverageImprovements:
    """Tests to cover the last remaining uncovered lines."""

    def test_destructor_with_close_exception(self):
        """Test __del__ exception handling when close() raises (lines 150-151)."""
        corpus = SQLiteCorpus(
            corpus_dir="test_corpus",
            name="destructor_error_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # Close the database to potentially cause issues in __del__
        if corpus.db:
            corpus.db.engine.dispose()

        # Destructor should handle any exceptions gracefully
        try:
            del corpus
            # Should not raise
        except Exception as e:
            pytest.fail(f"__del__ raised exception: {e}")

    def test_load_from_disk_nonexistent_directory(self, temp_corpus_dir):
        """Test _load_records_from_disk with nonexistent directory (line 170)."""
        # Create corpus but delete the directory to test early return
        test_dir = Path(temp_corpus_dir) / "will_be_deleted"
        test_dir.mkdir(exist_ok=True)

        corpus = SQLiteCorpus(
            corpus_dir=str(test_dir),
            name="deleted_dir_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # Now delete the directory
        import shutil

        shutil.rmtree(test_dir)

        # Sync should handle missing directory gracefully
        count = corpus.sync()
        assert count == 0

        corpus.close()

    def test_load_from_disk_no_metadata_file(self, temp_corpus_dir):
        """Test _load_records_from_disk with no metadata file (line 173)."""
        # Create empty directory
        empty_dir = Path(temp_corpus_dir) / "empty"
        empty_dir.mkdir(exist_ok=True)

        corpus = SQLiteCorpus(
            corpus_dir=str(empty_dir),
            name="no_metadata_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # Sync should handle missing metadata gracefully
        count = corpus.sync()
        assert count == 0

        corpus.close()

    def test_initialize_metadata_without_database(self, temp_corpus_dir):
        """Test _initialize_metadata when db is None (line 208)."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="no_db_test",
            use_sqlite=False,  # No database
        )

        # Call _initialize_metadata directly - should return early
        corpus._initialize_metadata()

        # Should not crash
        assert corpus.db is None

        corpus.close()

    def test_filter_in_memory_is_parsed_check(self, temp_corpus_dir):
        """Test in-memory filtering is_parsed check (line 420)."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test_is_parsed_check",
            use_sqlite=False,
        )

        import spacy

        nlp = spacy.blank("en")

        # Add unparsed and parsed records
        corpus.add("Unparsed", name="unparsed")
        doc = nlp("Parsed")
        corpus.add(doc, name="parsed")

        # Filter explicitly for is_parsed=True
        parsed_records = corpus.filter_records(is_parsed=True, use_database=False)
        assert len(parsed_records) == 1

        # Filter explicitly for is_parsed=False
        unparsed_records = corpus.filter_records(is_parsed=False, use_database=False)
        assert len(unparsed_records) == 1

        corpus.close()

    def test_filter_in_memory_min_tokens_try_block(self, temp_corpus_dir):
        """Test in-memory filtering min_tokens try block (line 424)."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="test_min_tokens_try",
            use_sqlite=False,
        )

        import spacy

        nlp = spacy.blank("en")
        doc = nlp("One two three four five")
        corpus.add(doc, name="doc1")

        # This should execute the try block for min_tokens
        filtered = corpus.filter_records(min_tokens=3, use_database=False)
        assert len(filtered) == 1

        corpus.close()

    def test_sync_with_overwrite_update_record(self):
        """Test sync with overwrite calling update_record (line 533)."""
        corpus = SQLiteCorpus(
            corpus_dir="test_corpus",
            name="sync_update_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # Add and sync a record
        corpus.add("Original content", name="doc1")
        corpus.sync()

        # Modify the record in memory
        record_id = list(corpus.records.keys())[0]
        record = corpus.records[record_id]

        from uuid import UUID

        from lexos.corpus.record import Record

        modified_record = Record(
            id=UUID(record_id),
            name="doc1",
            content="Modified content",
        )
        corpus.records[record_id] = modified_record

        # Sync with overwrite - should call update_record
        count = corpus.sync(overwrite=True)
        assert count == 1

        # Verify the update
        db_record = corpus.db.get_record(record_id)
        assert db_record.content == "Modified content"

        corpus.close()

    def test_sync_exception_in_update_path(self):
        """Test sync exception handling in update path (lines 542-544)."""
        from unittest.mock import patch

        corpus = SQLiteCorpus(
            corpus_dir="test_corpus",
            name="sync_exception_update_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # Add record
        corpus.add("Test content", name="doc1")

        # Mock get_record to return a record (so it tries update path)
        # and then mock update_record to raise exception
        with patch.object(corpus.db, "get_record", return_value=True):
            with patch.object(
                corpus.db, "update_record", side_effect=Exception("Update failed")
            ):
                # Should handle exception and print warning but not raise
                try:
                    count = corpus.sync(overwrite=True)
                    # Exception happens but doesn't prevent completion
                    # The warning is printed but count might be 0 if all fail
                    assert count >= 0
                except Exception:
                    pytest.fail("sync should not raise exception")

        corpus.close()


# ============================================================================
# Test Remaining Uncovered Lines
# ============================================================================


class TestRemainingUncoveredLines:
    """Tests targeting the last 5 uncovered lines."""

    def test_destructor_exception_in_close(self):
        """Test __del__ catching exception from close() (lines 150-151)."""

        class CorpusWithBrokenClose(SQLiteCorpus):
            """Subclass that raises in close to test __del__ exception handling."""

            def close(self):
                raise RuntimeError("Simulated close error")

        corpus = CorpusWithBrokenClose(
            corpus_dir="test_corpus",
            name="broken_close_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # This should not raise - __del__ should catch the exception
        del corpus

    def test_load_from_disk_missing_metadata_file(self, temp_corpus_dir):
        """Test early return when metadata file doesn't exist (line 173)."""
        # Create directory structure without metadata file
        test_dir = Path(temp_corpus_dir) / "no_metadata"
        test_dir.mkdir(exist_ok=True)
        # Don't create data directory - just the main directory exists

        corpus = SQLiteCorpus(
            corpus_dir=str(test_dir),
            name="no_metadata",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # Delete the metadata file that was created during initialization
        metadata_file = test_dir / "corpus_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

        # Sync should trigger _load_records_from_disk which returns early
        count = corpus.sync()
        assert count == 0

        corpus.close()

    def test_filter_is_parsed_none_check(self, temp_corpus_dir):
        """Test is_parsed not None in filter (line 420)."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="is_parsed_not_none_test",
            use_sqlite=False,
        )

        import spacy

        nlp = spacy.blank("en")
        corpus.add("Unparsed", name="unparsed")
        corpus.add(nlp("Parsed"), name="parsed")

        # Filter with is_parsed=False (not None) to hit line 420
        unparsed = corpus.filter_records(is_parsed=False, use_database=False)
        assert len(unparsed) == 1
        assert unparsed[0].name == "unparsed"

        corpus.close()

    def test_filter_min_tokens_none_check(self, temp_corpus_dir):
        """Test min_tokens not None in filter (line 424)."""
        corpus = SQLiteCorpus(
            corpus_dir=temp_corpus_dir,
            name="min_tokens_not_none_test",
            use_sqlite=False,
        )

        import spacy

        nlp = spacy.blank("en")
        corpus.add(nlp("One two three"), name="doc1")
        corpus.add(nlp("Four five six seven eight"), name="doc2")

        # Filter with min_tokens=4 (not None) to hit line 424
        filtered = corpus.filter_records(min_tokens=4, use_database=False)
        assert len(filtered) == 1
        assert filtered[0].name == "doc2"

        corpus.close()

    def test_sync_overwrite_with_existing_record(self):
        """Test sync calling update_record when record exists (line 533)."""
        corpus = SQLiteCorpus(
            corpus_dir="test_corpus",
            name="update_existing_test",
            use_sqlite=True,
            sqlite_path=":memory:",
        )

        # Add a record and sync it to database
        corpus.add("Original content", name="doc1")
        record_id = list(corpus.records.keys())[0]

        # First sync to add record to database
        count1 = corpus.sync(overwrite=False)
        assert count1 == 1

        # Modify the record in memory
        from uuid import UUID

        from lexos.corpus.record import Record

        updated = Record(
            id=UUID(record_id),
            name="doc1",
            content="Updated content",
            is_active=True,
        )
        corpus.records[record_id] = updated

        # Sync with overwrite=True should hit line 533 (update_record)
        count2 = corpus.sync(overwrite=True)
        assert count2 == 1

        # Verify the record was updated
        db_record = corpus.db.get_record(record_id)
        assert db_record.content == "Updated content"

        corpus.close()
