"""database.py.

Simple database integration using pure SQLAlchemy with SQLModel compatibility.

This is a compatibility layer for SQLModel 0.0.24 that works around primary key issues.

Last Updated: November 20, 2025
Last Tested: November 20, 2025
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
from uuid import UUID, uuid4

from spacy.tokens import Doc

# Pydantic imports removed to avoid validation issues with LexosModelCache
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text

from lexos.corpus.record import Record
from lexos.corpus.utils import LexosModelCache
from lexos.exceptions import LexosException

# Create base for SQLAlchemy tables
Base = declarative_base()


class SQLiteRecord(Base):
    """SQLAlchemy table for record storage."""

    __tablename__ = "records"

    # Primary identification
    id = Column(String, primary_key=True)
    name = Column(String)

    # Content storage
    content_text = Column(Text, nullable=False)
    content_doc_bytes = Column(LargeBinary)

    # Status and metadata
    is_active = Column(Boolean, default=True)
    is_parsed = Column(Boolean, default=False)
    model = Column(String)

    # Content statistics (denormalized for query performance)
    num_tokens = Column(Integer, default=0)
    num_terms = Column(Integer, default=0)
    vocab_density = Column(Float, default=0.0)

    # Serialized metadata as JSON string
    metadata_json = Column(Text, default="{}")
    extensions_list = Column(Text, default="[]")

    # Data integrity and versioning
    data_source = Column(String)
    content_hash = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)


class SQLiteMetadata(Base):
    """SQLAlchemy table for corpus metadata."""

    __tablename__ = "corpus_metadata"

    # Corpus identification
    corpus_id = Column(String, primary_key=True)
    name = Column(String)

    # Aggregate statistics
    num_docs = Column(Integer, default=0)
    num_active_docs = Column(Integer, default=0)
    num_tokens = Column(Integer, default=0)
    num_terms = Column(Integer, default=0)

    # Configuration
    corpus_dir = Column(String, nullable=False)

    # Serialized metadata
    metadata_json = Column(Text, default="{}")
    analysis_results_json = Column(Text, default="{}")

    # Versioning and integrity
    corpus_fingerprint = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)


class SQLiteBackend:
    """Database interface for corpus operations using SQLite with full-text search."""

    def __del__(self):
        """Destructor to ensure database connections are closed."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup

    def __init__(self, database_path: Union[str, Path] = ":memory:", **kwargs: Any):
        """Initialize the corpus database.

        Args:
            database_path: Path to SQLite database file, or ":memory:" for in-memory database
            **kwargs: Additional keyword arguments for SQLAlchemy engine creation
        """
        self.database_path = str(database_path)
        self.engine = create_engine(f"sqlite:///{self.database_path}", **kwargs)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._initialize_database()

    def _db_record_to_record(
        self,
        db_record: SQLiteRecord,
        include_doc: bool = True,
        model_cache: Optional[LexosModelCache] = None,
    ) -> Record:
        """Convert a SQLiteRecord back to a Record object."""
        # Deserialize metadata
        metadata = json.loads(db_record.metadata_json)
        extensions = json.loads(db_record.extensions_list)

        # Handle content deserialization
        if include_doc and db_record.content_doc_bytes and db_record.is_parsed:
            # Deserialize spaCy Doc
            content = self._deserialize_doc_content(
                db_record.content_doc_bytes, db_record.model, model_cache
            )
        else:
            # Use text content
            content = db_record.content_text

        # Create Record object
        record = Record(
            id=db_record.id,
            name=db_record.name,
            is_active=db_record.is_active,
            content=content,
            model=db_record.model,
            extensions=extensions,
            data_source=db_record.data_source,
            meta=metadata,
        )

        return record

    def _deserialize_doc_content(
        self,
        doc_bytes: bytes,
        model: Optional[str] = None,
        model_cache: Optional[LexosModelCache] = None,
    ) -> Doc:
        """Deserialize spaCy Doc from bytes."""
        try:
            # Use Record's deserialization method
            temp_record = Record(id=str(uuid4()), content="")
            return temp_record._doc_from_bytes(doc_bytes, model, model_cache)
        except Exception as e:
            raise LexosException(f"Failed to deserialize spaCy Doc: {str(e)}")

    def _initialize_database(self):
        """Initialize database schema and enable full-text search."""
        # Create all tables
        Base.metadata.create_all(self.engine)

        # Enable FTS5 full-text search
        with self.SessionLocal() as session:
            # Create FTS5 virtual table for full-text search
            session.execute(
                text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
                    record_id,
                    name,
                    content_text,
                    metadata_text
                )
            """)
            )

            # Create triggers to keep FTS table synchronized
            session.execute(
                text("""
                CREATE TRIGGER IF NOT EXISTS records_fts_insert AFTER INSERT ON records
                BEGIN
                    INSERT INTO records_fts(record_id, name, content_text, metadata_text)
                    VALUES (new.id, new.name, new.content_text, new.metadata_json);
                END
            """)
            )

            session.execute(
                text("""
                CREATE TRIGGER IF NOT EXISTS records_fts_delete AFTER DELETE ON records
                BEGIN
                    DELETE FROM records_fts WHERE record_id = old.id;
                END
            """)
            )

            session.execute(
                text("""
                CREATE TRIGGER IF NOT EXISTS records_fts_update AFTER UPDATE ON records
                BEGIN
                    UPDATE records_fts
                    SET name = new.name,
                        content_text = new.content_text,
                        metadata_text = new.metadata_json
                    WHERE record_id = new.id;
                END
            """)
            )

            session.commit()

    def _record_to_db_record(self, record: Record) -> SQLiteRecord:
        """Convert a Record object to SQLiteRecord for database storage."""
        # Extract text content
        if record.is_parsed and isinstance(record.content, Doc):
            content_text = record.content.text
            # Serialize spaCy Doc if parsed
            content_doc_bytes = record._doc_to_bytes()
        else:
            content_text = str(record.content) if record.content else ""
            content_doc_bytes = None

        # Calculate content hash
        content_hash = hashlib.sha256(content_text.encode()).hexdigest()

        # Calculate statistics
        num_tokens = record.num_tokens() if record.is_parsed else 0
        num_terms = record.num_terms() if record.is_parsed else 0
        vocab_density = record.vocab_density() if record.is_parsed else 0.0

        # Serialize metadata
        metadata_json = json.dumps(record.meta, default=str)
        extensions_list = json.dumps(record.extensions)

        timestamp = datetime.now().isoformat()

        db_record = SQLiteRecord()
        db_record.id = str(record.id)
        db_record.name = record.name
        db_record.content_text = content_text
        db_record.content_doc_bytes = content_doc_bytes
        db_record.is_active = record.is_active
        db_record.is_parsed = record.is_parsed
        db_record.model = record.model
        db_record.num_tokens = num_tokens
        db_record.num_terms = num_terms
        db_record.vocab_density = vocab_density
        db_record.metadata_json = metadata_json
        db_record.extensions_list = extensions_list
        db_record.data_source = record.data_source
        db_record.content_hash = content_hash
        db_record.created_at = timestamp
        db_record.updated_at = timestamp

        return db_record

    def add_record(self, record: Record) -> None:
        """Add a Record to the database."""
        with self.SessionLocal() as session:
            # Check if record already exists
            existing = (
                session.query(SQLiteRecord)
                .filter(SQLiteRecord.id == str(record.id))
                .first()
            )
            if existing:
                raise LexosException(
                    f"Record with ID {record.id} already exists in database"
                )

            # Convert Record to SQLiteRecord
            db_record = self._record_to_db_record(record)

            session.add(db_record)
            session.commit()

    def close(self):
        """Close the database connection and clean up resources."""
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()

    def delete_record(self, record_id: str) -> bool:
        """Delete a record from the database."""
        with self.SessionLocal() as session:
            record = (
                session.query(SQLiteRecord).filter(SQLiteRecord.id == record_id).first()
            )
            if record:
                session.delete(record)
                session.commit()
                return True
            return False

    def filter_records(
        self,
        is_active: Optional[bool] = None,
        is_parsed: Optional[bool] = None,
        model: Optional[str] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[Record]:
        """Filter records by various criteria."""
        with self.SessionLocal() as session:
            query = session.query(SQLiteRecord)

            if is_active is not None:
                query = query.filter(SQLiteRecord.is_active == is_active)
            if is_parsed is not None:
                query = query.filter(SQLiteRecord.is_parsed == is_parsed)
            if model is not None:
                query = query.filter(SQLiteRecord.model == model)
            if min_tokens is not None:
                query = query.filter(SQLiteRecord.num_tokens >= min_tokens)
            if max_tokens is not None:
                query = query.filter(SQLiteRecord.num_tokens <= max_tokens)

            if limit is not None:
                query = query.limit(limit)

            results = query.all()

            return [
                self._db_record_to_record(db_record, include_doc=False)
                for db_record in results
            ]

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate corpus statistics from the database."""
        with self.SessionLocal() as session:
            # Basic counts
            total_records = session.execute(
                text("SELECT COUNT(*) FROM records")
            ).scalar()
            active_records = session.execute(
                text("SELECT COUNT(*) FROM records WHERE is_active = 1")
            ).scalar()
            parsed_records = session.execute(
                text("SELECT COUNT(*) FROM records WHERE is_parsed = 1")
            ).scalar()

            # Token statistics
            total_tokens = (
                session.execute(
                    text("SELECT SUM(num_tokens) FROM records WHERE is_active = 1")
                ).scalar()
                or 0
            )
            total_terms = (
                session.execute(
                    text("SELECT SUM(num_terms) FROM records WHERE is_active = 1")
                ).scalar()
                or 0
            )

            # Vocabulary density statistics
            avg_vocab_density = (
                session.execute(
                    text(
                        "SELECT AVG(vocab_density) FROM records WHERE is_active = 1 AND num_tokens > 0"
                    )
                ).scalar()
                or 0
            )

            return {
                "total_records": total_records,
                "active_records": active_records,
                "parsed_records": parsed_records,
                "total_tokens": total_tokens,
                "total_terms": total_terms,
                "average_vocab_density": avg_vocab_density,
            }

    # Note: `get_stats()` is the canonical method name. Older code that used
    # `get_corpus_stats()` should call `get_stats()` instead. This wrapper was
    # removed to keep the sqlite submodule's API consistent with the
    # `Corpus` public API. If you need backward compatibility across the
    # deprecated database modules, see `src/lexos/database/database_simple.py`.

    def get_record(
        self,
        record_id: str,
        include_doc: bool = True,
        model_cache: Optional[LexosModelCache] = None,
    ) -> Optional[Record]:
        """Retrieve a Record from the database."""
        with self.SessionLocal() as session:
            db_record = (
                session.query(SQLiteRecord).filter(SQLiteRecord.id == record_id).first()
            )
            if not db_record:
                return None

            return self._db_record_to_record(db_record, include_doc, model_cache)

    def search_records(
        self,
        query: str,
        limit: int = 100,
        include_inactive: bool = False,
        model_filter: Optional[str] = None,
    ) -> list[Record]:
        """Perform full-text search on records."""
        with self.SessionLocal() as session:
            # Build FTS query - Fix: Use proper FTS5 syntax and avoid duplicate joins
            fts_query = text("""
                SELECT DISTINCT r.* FROM records r
                WHERE r.id IN (
                    SELECT record_id FROM records_fts
                    WHERE records_fts MATCH :query
                )
                AND (:include_inactive OR r.is_active = 1)
                AND (:model_filter IS NULL OR r.model = :model_filter)
                ORDER BY r.created_at DESC
                LIMIT :limit
            """)

            result = session.execute(
                fts_query,
                {
                    "query": query,
                    "include_inactive": include_inactive,
                    "model_filter": model_filter,
                    "limit": limit,
                },
            )

            records = []
            for row in result:
                # Convert row to SQLiteRecord manually
                db_record = SQLiteRecord()
                for i, col in enumerate(SQLiteRecord.__table__.columns):
                    setattr(db_record, col.name, row[i])

                record = self._db_record_to_record(db_record, include_doc=False)
                records.append(record)

            return records

    def update_record(self, record: Record) -> None:
        """Update an existing record in the database."""
        with self.SessionLocal() as session:
            existing = (
                session.query(SQLiteRecord)
                .filter(SQLiteRecord.id == str(record.id))
                .first()
            )
            if not existing:
                raise LexosException(
                    f"Record with ID {record.id} not found in database"
                )

            # Update the existing record
            updated_record = self._record_to_db_record(record)
            for key, value in updated_record.__dict__.items():
                if (
                    key != "_sa_instance_state" and key != "id"
                ):  # Skip SQLAlchemy metadata and primary key
                    setattr(existing, key, value)

            session.commit()
