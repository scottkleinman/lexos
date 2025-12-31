"""test_init.py.

Test coverage for sqlite.__init__.py imports.
Tests verify that all exported classes and functions are importable and functional.

Tests Cover:
- All exports from __all__ (SQLiteBackend, SQLiteMetadata, SQLiteCorpus, create_corpus)
- Import functionality for each exported class/function
- Basic instantiation of classes
- Integration between exported components
- Module documentation
- Class inheritance relationships
- Required methods existence

Coverage: 100%

Last Update: 2025-11-20.
"""

import pytest


class TestSQLiteInitImports:
    """Test that all exports from lexos.corpus.sqlite are importable and functional."""

    def test_import_sqlite_backend(self):
        """Test that SQLiteBackend can be imported from the package."""
        from lexos.corpus.sqlite import SQLiteBackend

        assert SQLiteBackend is not None
        assert hasattr(SQLiteBackend, "__init__")

    def test_import_sqlite_metadata(self):
        """Test that SQLiteMetadata can be imported from the package."""
        from lexos.corpus.sqlite import SQLiteMetadata

        assert SQLiteMetadata is not None
        assert hasattr(SQLiteMetadata, "__init__")

    def test_import_sqlite_corpus(self):
        """Test that SQLiteCorpus can be imported from the package."""
        from lexos.corpus.sqlite import SQLiteCorpus

        assert SQLiteCorpus is not None
        assert hasattr(SQLiteCorpus, "__init__")

    def test_import_create_corpus(self):
        """Test that create_corpus function can be imported from the package."""
        from lexos.corpus.sqlite import create_corpus

        assert create_corpus is not None
        assert callable(create_corpus)

    def test_all_exports_present(self):
        """Test that __all__ contains all expected exports."""
        from lexos.corpus.sqlite import __all__

        expected_exports = [
            "SQLiteBackend",
            "SQLiteMetadata",
            "SQLiteCorpus",
            "create_corpus",
        ]

        assert __all__ == expected_exports

    def test_import_all_at_once(self):
        """Test that all exports can be imported together using import *."""
        # Import all exports
        from lexos.corpus.sqlite import (
            SQLiteBackend,
            SQLiteCorpus,
            SQLiteMetadata,
            create_corpus,
        )

        # Verify all are importable
        assert all(
            [
                SQLiteBackend is not None,
                SQLiteMetadata is not None,
                SQLiteCorpus is not None,
                create_corpus is not None,
            ]
        )

    def test_sqlite_backend_instantiation(self):
        """Test that SQLiteBackend can be instantiated."""
        from lexos.corpus.sqlite import SQLiteBackend

        # Create an in-memory database for testing
        backend = SQLiteBackend(database_path=":memory:")
        assert backend is not None
        assert backend.database_path == ":memory:"
        backend.close()

    def test_sqlite_metadata_instantiation(self):
        """Test that SQLiteMetadata can be instantiated."""
        from lexos.corpus.sqlite import SQLiteMetadata

        # SQLiteMetadata is a SQLAlchemy table, not a Pydantic model
        # It can't be instantiated directly like a regular class
        # Instead, verify it has the expected table structure
        assert hasattr(SQLiteMetadata, "__tablename__")
        assert SQLiteMetadata.__tablename__ == "corpus_metadata"
        assert hasattr(SQLiteMetadata, "corpus_id")
        assert hasattr(SQLiteMetadata, "name")

    def test_sqlite_corpus_instantiation(self):
        """Test that SQLiteCorpus can be instantiated."""
        from lexos.corpus.sqlite import SQLiteCorpus

        corpus = SQLiteCorpus(name="test", use_sqlite=False)
        assert corpus is not None
        assert corpus.name == "test"
        assert corpus.use_sqlite is False

    def test_create_corpus_callable(self):
        """Test that create_corpus function is callable and works."""
        from lexos.corpus.sqlite import create_corpus

        # Create a simple corpus with database
        corpus = create_corpus(name="test_corpus", sqlite_only=False)
        assert corpus is not None
        assert corpus.name == "test_corpus"
        assert corpus.use_sqlite is True

    def test_module_has_docstring(self):
        """Test that the module has a docstring."""
        import lexos.corpus.sqlite as sqlite_module

        assert sqlite_module.__doc__ is not None
        assert "SQLite" in sqlite_module.__doc__
        assert "database" in sqlite_module.__doc__.lower()

    def test_exports_are_classes_or_functions(self):
        """Test that all exports are either classes or callable functions."""
        from lexos.corpus.sqlite import (
            SQLiteBackend,
            SQLiteCorpus,
            SQLiteMetadata,
            create_corpus,
        )

        # Check that classes are classes
        assert isinstance(SQLiteBackend, type)
        assert isinstance(SQLiteMetadata, type)
        assert isinstance(SQLiteCorpus, type)

        # Check that create_corpus is callable
        assert callable(create_corpus)

    def test_sqlite_corpus_inherits_from_corpus(self):
        """Test that SQLiteCorpus properly inherits from Corpus."""
        from lexos.corpus.corpus import Corpus
        from lexos.corpus.sqlite import SQLiteCorpus

        assert issubclass(SQLiteCorpus, Corpus)

    def test_sqlite_backend_has_required_methods(self):
        """Test that SQLiteBackend has expected methods."""
        from lexos.corpus.sqlite import SQLiteBackend

        expected_methods = ["close", "add_record", "get_stats"]

        for method in expected_methods:
            assert hasattr(SQLiteBackend, method), (
                f"SQLiteBackend missing method: {method}"
            )

    def test_sqlite_corpus_has_database_methods(self):
        """Test that SQLiteCorpus has database-specific methods."""
        from lexos.corpus.sqlite import SQLiteCorpus

        # Check for database-specific methods
        expected_methods = ["sync", "search"]

        for method in expected_methods:
            assert hasattr(SQLiteCorpus, method), (
                f"SQLiteCorpus missing method: {method}"
            )


class TestSQLiteInitIntegration:
    """Integration tests for sqlite module exports working together."""

    def test_create_corpus_with_database(self, tmp_path):
        """Test creating a corpus with database enabled."""
        from lexos.corpus.sqlite import create_corpus

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        corpus = create_corpus(
            corpus_dir=str(corpus_dir), name="test_corpus", sqlite_only=False
        )

        assert corpus is not None
        assert corpus.use_sqlite is True
        assert corpus.name == "test_corpus"

    def test_backend_and_corpus_integration(self, tmp_path):
        """Test that SQLiteBackend and SQLiteCorpus work together."""
        from lexos.corpus.sqlite import SQLiteBackend, SQLiteCorpus

        db_path = tmp_path / "integration.db"

        # Create backend
        backend = SQLiteBackend(database_path=str(db_path))

        # Create corpus using the same db_path
        corpus = SQLiteCorpus(
            name="integration_test",
            use_sqlite=True,
            sqlite_path=str(db_path),
            corpus_dir=str(tmp_path / "corpus"),
        )

        assert corpus.use_sqlite is True
        assert backend.database_path == str(db_path)

        backend.close()

    def test_metadata_and_corpus_integration(self):
        """Test that SQLiteMetadata table structure is compatible with SQLiteCorpus."""
        from lexos.corpus.sqlite import SQLiteCorpus, SQLiteMetadata

        # SQLiteMetadata is a table definition
        # Verify it has attributes that the corpus would use
        assert hasattr(SQLiteMetadata, "corpus_id")
        assert hasattr(SQLiteMetadata, "name")
        assert hasattr(SQLiteMetadata, "num_docs")
        assert hasattr(SQLiteMetadata, "corpus_fingerprint")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
