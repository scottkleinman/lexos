"""corpus.py.

Last updated: August 15, 2026
Last tested: August 15, 2026

This code is designed to work by default with UUID4 for the ID field, which is a universally unique identifier. UUID7 is a better choice but does not yet have full support in the Python standard library and Pydantic. Once that takes place, it can be easily changed in the Record model. Alternaively, the ID can be set to an incrementing integer with `id_type="integer"`.

To reproduce the web app's Statistics module, call `stats = Corpus.get_token_stats()` to get a `CorpusStats` object. Its method produce the web app's calculations and output. By default, the `get_token_stats()` method retrieves stats for the entire corpus, but you can pass parameters to filter active records or settings accepted by the vectorizer. You can also pass an arbitrary list of tuples containing the record ID, name, and tokens to retrieve statistics for any list of pre-tokenised records.


# TODO:
- Consider adding method to normalize extensions across entire corpus (PM suggestion: detect all extensions in corpus docs and set values to None if not present in a given Doc)
- Complete communication architecture implementation once peer modules (kmeans, topwords, kwic, text classification) are available
- Implement result validation and versioning for external module results
- Add cross-module result correlation capabilities
"""

import shutil
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import srsly
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    validate_call,
)
from spacy.tokens import Doc
from wasabi import msg

from lexos.corpus import Record
from lexos.corpus.corpus_stats import CorpusStats
from lexos.corpus.utils import LexosModelCache, RecordsDict
from lexos.exceptions import LexosException


class Corpus(BaseModel):
    """A collection of Record objects."""

    corpus_dir: str = Field(
        "corpus", description="The path to the directory where the corpus is stored."
    )
    corpus_metadata_file: str = Field(
        "corpus_metadata.json",
        description="The name of the corpus metadata file.",
    )
    name: str = Field(None, description="The name of the corpus.")
    records: RecordsDict = Field({}, description="Dictionary of records in the corpus.")
    names: dict[str, list[str]] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(
        {},
        description="Metadata dictionary for arbitrary metadata relating to the corpus.",
    )
    analysis_results: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Storage for results from external analysis modules (kmeans, topwords, kwic, etc.)",
    )
    model_cache: LexosModelCache = Field(
        LexosModelCache(),
        description="A cache for spaCy models used in the Corpus.",
        exclude=True,
    )
    num_active_docs: int = Field(
        0, description="Number of active records in the corpus."
    )
    num_docs: int = Field(0, description="Total number of records in the corpus.")
    num_terms: int = Field(0, description="Total number of unique terms in the corpus.")
    num_tokens: int = Field(0, description="Total number of tokens in the corpus.")
    terms: set = Field(set(), description="Set of unique terms in the corpus.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialise the Corpus with a data directory and a metadata file."""
        # Allow Path input for corpus_dir while preserving the model's string field.
        if "corpus_dir" in data and isinstance(data["corpus_dir"], Path):
            data["corpus_dir"] = str(data["corpus_dir"])
        super().__init__(**data)
        corpus_dir = Path(self.corpus_dir)
        Path(corpus_dir / "data").mkdir(parents=True, exist_ok=True)

        # Load existing metadata if it exists, otherwise create new
        metadata_file = corpus_dir / self.corpus_metadata_file
        if metadata_file.exists():
            # Load existing metadata to preserve record info
            existing_metadata = srsly.read_json(metadata_file)
            # Preserve the 'meta' dict which contains record metadata
            if "meta" in existing_metadata and existing_metadata["meta"]:
                self.meta = existing_metadata["meta"]

        # NOTE: We use model_dump() on the Corpus model here. The Corpus
        # computed fields (e.g., `terms` is a set) are safe to serialize
        # and do not raise the `LexosException`. We explicitly convert
        # `terms` to a list to make it JSON-serializable. If future
        # computed fields are added to Corpus that rely on external state
        # or can raise, this call should be revised to exclude those fields
        # (e.g., model_dump(exclude=[...])).
        data = self.model_dump()
        data["terms"] = list(data["terms"])
        srsly.write_json(metadata_file, data)
        msg.good("Corpus created.")

    def __iter__(self) -> Iterable[Record]:
        """Make the corpus iterable.

        Returns:
            Iterator[Record]: An iterator over the Record objects in the corpus.
        """
        return iter(self.records.values())

    def __repr__(self):
        """Return a string representation of the Corpus."""
        fields = {field: getattr(self, field) for field in self.model_fields_set}
        field_list = [f"{k}={v}" for k, v in fields.items()]
        rep = f"Corpus({', '.join(sorted(field_list))})"
        return rep

    @property
    def active_terms(self) -> set:
        """Return the set of active terms in the Corpus.

        Returns:
            set: A set of active term strings found in active parsed records.
        """
        active_terms = set()
        for record in self.records.values():
            if record.is_parsed and record.is_active:
                active_terms.update(record.terms.keys())
        return active_terms

    @property
    def meta_df(self) -> pd.DataFrame:
        """Return a DataFrame of the Corpus metadata."""
        if not self.meta:
            raise LexosException("No metadata available in the Corpus.")
        df = pd.DataFrame([self.meta])
        df.fillna("", inplace=True)
        return df

    @property
    def num_active_tokens(self) -> int:
        """Return the number of active tokens in the Corpus.

        Returns:
            int: The total number of tokens in active parsed records.
        """
        if len(self.active_terms) == 0:
            return 0
        return sum(
            record.num_tokens()
            for record in self.records.values()
            if record.is_active and record.is_parsed
        )

    @property
    def num_active_terms(self) -> int:
        """Return the number of active terms in the Corpus."""
        if len(self.active_terms) == 0:
            return 0
        return len(self.active_terms)

    def _add_to_corpus(self, record: Record, cache: Optional[bool] = False) -> None:
        """Add a record to the Corpus.

        Args:
            record (Record): A Record doc.
            cache (Optional[bool]): Whether to cache the record. Defaults to False.

        Returns:
            None
        """
        # Update corpus records table
        # We intentionally exclude computed fields here when dumping a
        # Record for meta storage because those computed properties (e.g.,
        # `terms`, `text`, `tokens`) may attempt to evaluate state-dependent
        # computed values that can raise for unparsed records. By explicitly
        # excluding them and then annotating `num_tokens`/`num_terms` using
        # guarded access below, we avoid calling computed fields on records
        # that are not parsed.
        meta = record.model_dump(
            exclude=["content", "terms", "text", "tokens"], mode="json"
        )
        # Ensure ID is always string for JSON serialization (redundant with mode="json" but kept for clarity)
        if "id" in meta:
            meta["id"] = str(meta["id"])
        num_tokens = record.num_tokens() if record.is_parsed else 0
        num_terms = record.num_terms() if record.is_parsed else 0
        meta["num_tokens"] = num_tokens
        meta["num_terms"] = num_terms
        # Use string ID as key to avoid UUID serialization issues
        self.meta[str(record.id)] = meta

        # Save the record to disk -- currently, this is always done
        corpus_dir = Path(self.corpus_dir)
        filename = f"{record.id}.bin"
        filepath = corpus_dir / "data" / filename
        record.meta["filename"] = str(filename)
        record.meta["filepath"] = str(filepath)
        record.to_disk(record.meta["filepath"])

        # Update the Corpus records dictionary
        record_id_str = str(record.id)
        self.records[record_id_str] = record

        # Update the Corpus names
        if record.name not in self.names:
            self.names[record.name] = []
        self.names[record.name].append(str(record.id))  # Explicitly convert to string

        # Update the Corpus statistics
        self._update_corpus_state()

    def _ensure_unique_name(self, name: str = None) -> str:
        """Ensure that no names are duplicated in the Corpus.

        Args:
            name (str): The record name.

        Returns:
            A string.
        """
        if not name:
            return f"untitled_{uuid.uuid1()}"
        if name in self.names:
            return f"{name}_{uuid.uuid1()}"
        return name

    def _generate_unique_id(self, type: str = "uuid4") -> str:
        """Generate a unique ID for the record.

        Args:
            type (str): The type of ID to generate. Can be "integer" or "uuid4". Defaults to "uuid4".

        Returns:
            str: A unique ID for the record.
        """
        if type == "integer":
            # Generate an integer ID from existing numeric keys.
            integer_ids = [
                int(key) for key in self.records.keys() if str(key).isdigit()
            ]
            return max(integer_ids, default=0) + 1
        elif type == "uuid4":
            # Generate initial UUID
            new_id = str(uuid.uuid4())

            # Keep generating new UUIDs until one is not in the records dic
            while new_id in self.records:
                new_id = str(uuid.uuid4())
            return new_id
        else:
            raise LexosException(
                f"Invalid ID type '{type}'. Must be 'integer' or 'uuid4'."
            )

    def _get_by_name(self, name: str) -> list[str]:
        """Get all record IDs from the Corpus by name.

        Args:
            name (str): The name of the record(s) to get.

        Returns:
            list[str]: A list of record IDs with the given name.
        """
        if name not in self.names:
            raise LexosException(
                f"Record with name {name} does not exist in the Corpus."
            )
        return self.names[name]

    def _update_corpus_state(self):
        """Update the Corpus state after adding or removing records.

        Note:
            This method recalculates the number of records, active records,
            terms, tokens, and unique terms in the entire Corpus.
        """
        self.num_docs = len(self.records)
        self.num_active_docs = sum(
            1 for r in self.records.values() if r and r.is_active
        )
        self.num_terms = sum(
            r.num_terms() for r in self.records.values() if r and r.is_parsed
        )
        self.num_tokens = sum(
            r.num_tokens() for r in self.records.values() if r and r.is_parsed
        )
        # We call model_dump() on Corpus to create a JSON of the corpus
        # metadata. This excludes Record-specific computed fields and the
        # `records` mapping so we only write top-level corpus metadata.
        # In particular, Record-computed fields are excluded so we don't
        # force evaluation across records which could trigger exceptions
        # for unparsed records.
        corpus_data = self.model_dump(
            exclude=["content", "terms", "text", "tokens", "records"]
        )
        # Convert any remaining UUIDs to strings
        for key, value in corpus_data.items():
            if hasattr(value, "hex"):  # UUID objects have .hex attribute
                corpus_data[key] = str(value)

        srsly.write_json(
            Path(self.corpus_dir) / self.corpus_metadata_file,
            corpus_data,
        )

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert non-JSON-serializable types to strings in metadata.

        Args:
            metadata: Original metadata dictionary

        Returns:
            Sanitized metadata dictionary with JSON-serializable values
        """
        from datetime import date, datetime
        from pathlib import Path
        from uuid import UUID

        def sanitize_value(value: Any) -> Any:
            """Sanitize a single value for JSON serialization.

            Args:
                value: The value to sanitize

            Returns:
                The sanitized value, which is JSON-serializable
            """
            if isinstance(value, dict):
                return self._sanitize_metadata(value)
            if isinstance(value, list):
                return [sanitize_value(item) for item in value]
            if isinstance(value, (UUID, Path)):
                return str(value)
            if isinstance(value, (datetime, date)):
                return value.isoformat()
            return value

        return {key: sanitize_value(value) for key, value in metadata.items()}

    def _normalize_content_items(
        self, content: Doc | Record | str | list[Doc | Record | str]
    ) -> list[Doc | Record | str]:
        """Normalize add content into a list of record items."""
        if isinstance(content, (Doc, Record, str)):
            return [content]
        return list(content)

    def _create_record(
        self,
        item: Doc | Record | str,
        record_id: str,
        name: Optional[str],
        is_active: Optional[bool],
        model: Optional[str],
        extensions: Optional[list[str]],
        metadata: Optional[dict[str, Any]],
    ) -> Record:
        """Create a Record from content or return an existing Record."""
        if isinstance(item, Record):
            if item.id and str(item.id) in self.records:
                raise LexosException(
                    f"Record with ID {item.id} already exists in the Corpus."
                )
            return item

        record_kwargs = dict(
            id=record_id,
            name=name,
            is_active=is_active,
            content=item,
            model=model,
            data_source=None,
        )
        if extensions is not None:
            record_kwargs["extensions"] = extensions
        if metadata is not None:
            record_kwargs["meta"] = metadata
        return Record(**record_kwargs)

    def _normalize_ids(
        self,
        id: Optional[str | list[str]] = None,
        name: Optional[str | list[str]] = None,
    ) -> list[str]:
        """Normalize ID and name inputs into a list of record IDs."""
        if not id and not name:
            raise LexosException("Must provide either an ID or a name.")

        if isinstance(id, str):
            ids = [id]
        elif isinstance(id, list):
            ids = id
        else:
            ids = []

        if name and not id:
            if isinstance(name, str):
                name = [name]
            ids = []
            for n in name:
                ids.extend(self._get_by_name(n))

        return ids

    def _load_record_by_id(self, id: str) -> Record:
        """Load a record from memory or disk by ID."""
        if id in self.records:
            return self.records[id]
        record = self.records[id]
        return record._from_disk(
            record.meta["filepath"], record.model, self.model_cache
        )

    def _remove_record_by_id(self, id: str) -> None:
        """Remove a single record from the corpus."""
        entry = self.records.pop(id)
        if entry.name not in self.names:
            raise LexosException(
                f"Record with name {entry.name} does not exist in the Corpus."
            )
        self.names[entry.name].remove(str(entry.id))
        if not self.names[entry.name]:
            self.names.pop(entry.name)

    def _filter_stats_records(self, active_only: bool) -> list[Record]:
        """Filter records for statistics computation.

        Args:
            active_only (bool): Whether to include only active records.

        Returns:
            list[Record]: The filtered list of records.
        """
        if active_only:
            return [record for record in self.records.values() if record.is_active]
        return list(self.records.values())

    def _build_stats_token_list(
        self, records: list[Record], type: str
    ) -> list[tuple[str, str, list[str]]]:
        """Build a list of tokens for statistics computation.

        Args:
            records (list[Record]): The list of records to process.
            type (str): The type of statistics to compute. Can be "tokens" or "characters".

        Returns:
            list[tuple[str, str, list[str]]]: The list of tokens for each record.
        """
        if type == "tokens":
            return [
                (str(record.id), record.name, self._get_token_strings(record))
                for record in records
            ]

        if type == "characters":
            return [
                (
                    str(record.id),
                    record.name,
                    list(record.content.text)
                    if record.is_parsed
                    else list(record.content),
                )
                for record in records
            ]

        raise LexosException(f"Unsupported stats type: {type}")

    def _get_token_strings(self, record: Record) -> list[str]:
        """Get the token strings from a Record.

        Args:
            record (Record): The record to extract token strings from.

        Returns:
            list[str]: The list of token strings.
        """
        if record.is_parsed:
            return [token.text for token in record.content]
        return record.content.split()

    @validate_call(config=model_config)
    def add(
        self,
        content: Doc | Record | str | list[Doc | Record | str],
        name: Optional[str] = None,
        is_active: Optional[bool] = True,
        model: Optional[str] = None,
        extensions: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        id_type: Optional[str] = "uuid4",
        cache: Optional[bool] = False,
    ):
        """Add a record to the Corpus.

        Args:
            content (Doc | Record | str | list[Doc | Record | str]): A text string, Record, or a spaCy Doc, or a list of any of these.
            name (str): A name for the record.
            is_active (bool): Whether or not the record is active.
            model (str): The name of the language model used to parse the record (optional).
            extensions (list[str]): A list of extension names to add to the record.
            metadata (dict[str, Any]): A dict containing any metadata.
            id_type (str): The type of ID to generate. Can be "integer" or "uuid4". Defaults to "uuid4".
            cache (bool): Whether or not to cache the record.
        """
        if metadata is not None:
            metadata = self._sanitize_metadata(metadata)

        items = self._normalize_content_items(content)

        for item in items:
            new_id = self._generate_unique_id(type=id_type)
            record = self._create_record(
                item=item,
                record_id=new_id,
                name=name,
                is_active=is_active,
                model=model,
                extensions=extensions,
                metadata=metadata,
            )

            if metadata:
                record.meta.update(metadata)

            self._add_to_corpus(record, cache=cache)

    def _add_to_corpus_without_state_update(
        self, record: Record, cache: Optional[bool] = False
    ) -> None:
        """Add a record to the Corpus without updating corpus state.

        This is an internal method used for batch operations where state
        updates are deferred until all records are added.

        Args:
            record (Record): A Record doc.
            cache (Optional[bool]): Whether to cache the record. Defaults to False.

        Returns:
            None
        """
        # Update corpus records table
        meta = record.model_dump(
            exclude=["content", "terms", "text", "tokens"], mode="json"
        )
        # Ensure ID is always string for JSON serialization
        if "id" in meta:
            meta["id"] = str(meta["id"])
        num_tokens = record.num_tokens() if record.is_parsed else 0
        num_terms = record.num_terms() if record.is_parsed else 0
        meta["num_tokens"] = num_tokens
        meta["num_terms"] = num_terms
        # Use string ID as key
        self.meta[str(record.id)] = meta

        # Save the record to disk
        corpus_dir = Path(self.corpus_dir)
        filename = f"{record.id}.bin"
        filepath = corpus_dir / "data" / filename
        record.meta["filename"] = str(filename)
        record.meta["filepath"] = str(filepath)
        record.to_disk(record.meta["filepath"])

        # Update the Corpus records dictionary
        record_id_str = str(record.id)
        self.records[record_id_str] = record

        # Update the Corpus names
        if record.name not in self.names:
            self.names[record.name] = []
        self.names[record.name].append(str(record.id))

        # Note: _update_corpus_state() is NOT called here

    def add_from_files(
        self,
        paths: Path | str | list[Path | str],
        max_workers: Optional[int] = None,
        worker_strategy: str = "auto",
        batch_size: int = 100,
        show_progress: bool = True,
        name_template: Optional[str] = None,
        is_active: bool = True,
        model: Optional[str] = None,
        extensions: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        id_type: str = "uuid4",
    ) -> None:
        """Load files directly into corpus using parallel I/O.

        This method streams files into the corpus without holding all
        content in memory, making it suitable for very large datasets.
        Files are loaded in parallel using the ParallelLoader with all
        its optimization features (smart file ordering, auto-tuning, etc.).

        State updates are deferred until all files are loaded for optimal
        performance.

        Args:
            paths (Path | str | list[Path | str]): File paths or directories to load.
            max_workers (Optional[int]): Maximum number of worker threads.
                If None, auto-calculated based on worker_strategy.
            worker_strategy (str): Worker allocation strategy. Options:
                - "auto": Analyzes file types and chooses optimal strategy (default)
                - "io_bound": More workers for I/O-intensive operations
                - "cpu_bound": Fewer workers for CPU-intensive operations
                - "balanced": Middle ground between I/O and CPU
            batch_size (int): Number of files to process in each batch. Default 100.
            show_progress (bool): Whether to show progress bar. Default True.
            name_template (Optional[str]): Template for generating record names.
                Can include {filename}, {stem}, {index}. If None, uses filename stem.
            is_active (bool): Whether records should be marked as active. Default True.
            model (Optional[str]): Name of language model used to parse records.
            extensions (Optional[list[str]]): List of extension names to add to records.
            metadata (Optional[dict[str, Any]]): Metadata to add to all records.
            id_type (str): Type of ID to generate ("integer" or "uuid4"). Default "uuid4".

        Example:
            ```python
            corpus = Corpus("my_corpus")
            # Load all text files from a directory
            corpus.add_from_files("path/to/texts/")
            # With custom naming
            corpus.add_from_files(
                ["file1.txt", "file2.txt"],
                name_template="{stem}_{index}",
                metadata={"source": "collection_a"}
            )
            ```
        """
        from lexos.io.parallel_loader import ParallelLoader

        # Sanitize metadata if provided
        if metadata is not None:
            metadata = self._sanitize_metadata(metadata)

        # Create ParallelLoader with specified settings
        loader = ParallelLoader(
            max_workers=max_workers,
            worker_strategy=worker_strategy,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Track for error reporting
        loaded_count = 0
        error_count = 0
        errors = []

        # Stream files and add to corpus
        for index, (path, name, mime_type, text, error) in enumerate(
            loader.load_streaming(paths), start=1
        ):
            if error:
                error_count += 1
                errors.append((path, error))
                continue

            # Generate record name from template or use default
            if name_template:
                record_name = name_template.format(
                    filename=Path(path).name, stem=name, index=index
                )
            else:
                record_name = name

            # Generate unique ID
            record_id = self._generate_unique_id(type=id_type)

            # Create record kwargs
            record_kwargs = dict(
                id=record_id,
                name=record_name,
                is_active=is_active,
                content=text,
                model=model,
                data_source=str(path),
            )

            if extensions is not None:
                record_kwargs["extensions"] = extensions

            if metadata is not None:
                record_kwargs["meta"] = metadata.copy()

            # Create and add record without updating state
            record = Record(**record_kwargs)
            self._add_to_corpus_without_state_update(record)
            loaded_count += 1

        # Update corpus state once at the end
        self._update_corpus_state()

        # Report results
        from wasabi import msg

        msg.good(f"Loaded {loaded_count} files into corpus. Errors: {error_count}")

        if errors and error_count <= 10:  # Show first 10 errors
            msg.warn("Errors encountered:")
            for path, error in errors[:10]:
                msg.fail(f"  {path}: {error}")

    @validate_call(config=model_config)
    def filter_records(self, **metadata_filters: Any) -> list[Record]:
        """Return records matching metadata key-value pairs.

        Args:
            **metadata_filters (Any): Arbitrary metadata fields and their required values.

        Returns:
            List of Record objects matching all metadata criteria.
        """
        results = []
        for record in self.records.values():
            if not hasattr(record, "meta") or not isinstance(record.meta, dict):
                continue
            match = True
            for key, value in metadata_filters.items():
                if key not in record.meta or record.meta[key] != value:
                    match = False
                    break
            if match:
                results.append(record)
        return results

    @validate_call(config=model_config)
    def get(
        self,
        id: Optional[str | list[str]] = None,
        name: Optional[str | list[str]] = None,
    ) -> Record | list[Record]:
        """Get a record from the Corpus by ID.

        Tries to get the record from memory; otherwise loads it from file.

        Args:
            id (str | list[str]): A record id or list of ids from the Corpus records.
            name (str | list[str]): A record name or list of names from the Corpus records.

        Returns:
            Record | list[Record]: The record(s) with the given ID(s) or name(s).
        """
        ids = self._normalize_ids(id=id, name=name)
        result = [self._load_record_by_id(record_id) for record_id in ids]
        if len(result) == 1:
            return result[0]
        return result

    @validate_call(config=model_config)
    def get_stats(
        self,
        active_only: bool = True,
        type: str = "tokens",
        min_df: int | None = None,
        max_df: int | None = None,
        max_n_terms: int | None = None,
        token_list: list[tuple[str, str, list[str]]] = None,
    ) -> CorpusStats:
        """Get statistics for the Corpus.

        Args:
            active_only (bool): Whether to include only active records. Defaults to True.
            type (str): The type of statistics to compute. Can be "tokens" or "characters". Defaults to "tokens".
            min_df (int | None): Minimum document frequency for terms. Defaults to None.
            max_df (int | None): Maximum document frequency for terms. Defaults to None.
            max_n_terms (int | None): Maximum number of terms to include in the statistics. Defaults to None.
            token_list (list[tuple[str, str, list[str]]] | None): Optional pre-computed list of tokens for each record.

        Returns:
            CorpusStats: An object containing the computed statistics for the Corpus.
        """
        if token_list is not None:
            return CorpusStats(
                docs=token_list,
                min_df=min_df,
                max_df=max_df,
                max_n_terms=max_n_terms,
            )

        records = self._filter_stats_records(active_only)
        token_list = self._build_stats_token_list(records, type)

        return CorpusStats(
            docs=token_list,
            min_df=min_df,
            max_df=max_df,
            max_n_terms=max_n_terms,
        )

    @validate_call(config=model_config)
    def load(
        self,
        path: Path | str = None,
        corpus_dir: Optional[Path | str] = None,
        cache: Optional[bool] = False,
    ) -> None:
        """Load a Corpus from a zip archive or directory.

        Args:
            path (Path | str): The path of the zip archive or directory to load.
            corpus_dir (Optional[Path | str]): The directory where the Corpus is to be unzipped.
            cache (Optional[bool]): Whether to cache the records in the Corpus. Defaults to False.

        Returns:
            None
        """
        # Ensure that a corpus_dir exists, or create one if it doesn't
        if not corpus_dir:
            corpus_dir = Path(self.corpus_dir)
        else:
            corpus_dir = Path(corpus_dir)
        corpus_dir.mkdir(parents=True, exist_ok=True)

        # If the path is a file, try to unpack it as a zip archive
        if Path(path).is_file():
            try:
                shutil.unpack_archive(path, corpus_dir)
            except shutil.ReadError as e:
                raise LexosException(
                    f"Failed to unpack archive: {e}. Ensure the file is a valid zip archive."
                )

        # Open the metadata file and load the metadata
        metadata_path = corpus_dir / self.corpus_metadata_file
        metadata = srsly.read_json(metadata_path)
        for key, value in metadata.items():
            setattr(self, key, value)

        # If cache is set, load the records into the model cache
        if cache:
            for record in self.records.values():
                if isinstance(record, Record):
                    # Load the record from disk
                    record.from_disk(
                        corpus_dir / "data" / f"{record.id}.bin",
                        model=record.model,
                        model_cache=self.model_cache,
                    )
                else:
                    raise LexosException(
                        "Records in the Corpus must be of type Record."
                    )

    @validate_call(config=model_config)
    def save(self, path: Path | str = None) -> None:
        """Save the Corpus as a zip archive.

        Args:
            path (Path | str): The path to save the Corpus to.

        Returns:
            None
        """
        shutil.make_archive(path / f"{self.name}", "zip", self.corpus_dir)

    @validate_call(config=model_config)
    def remove(
        self,
        id: Optional[str | list[str]] = None,
        name: Optional[str | list[str]] = None,
    ) -> None:
        """Remove a record from the corpus by ID."""
        ids = self._normalize_ids(id=id, name=name)

        for record_id in ids:
            try:
                self._remove_record_by_id(record_id)
            except KeyError:
                raise LexosException(
                    f"Record with ID {record_id} does not exist in the Corpus."
                )

        self._update_corpus_state()

    @validate_call(config=model_config)
    def set(self, id: str, **props) -> None:
        """Set a property or properties of a record in the Corpus.

        Args:
            id (str): A record id.
            **props (dict): The dict containing any other properties to set.

        Returns:
            None
        """
        # Get the record by ID
        record = self.records[id]

        # Save the record's filepath, thenupdate the specified properties
        old_filepath = record.meta.get("filepath", None)
        record.set(**props)

        # If the filepath has changed, delete the old file
        if record.meta.get("filepath", None) != old_filepath:
            Path(old_filepath).unlink(missing_ok=True)

        # If the record has a filepath, ensure the file is in the data directory
        filepath = record.meta.get("filepath")
        if filepath and filepath not in str(Path(self.corpus_dir) / "data"):
            record.to_disk(filepath, extensions=record.extensions)

        # Update the record in the Corpus and update the corpus state
        self.records[id] = record
        self._update_corpus_state()

    @validate_call(config=model_config)
    def term_counts(
        self, n: Optional[int] = 10, most_common: Optional[bool] = True
    ) -> Counter:
        """Get a Counter with the most common Corpus term counts.

        Args:
            n (Optional[int]): The number of most common terms to return. Defaults to 10.
            most_common (Optional[bool]): If True, return the n most common terms; otherwise, return the n least common terms.

        Returns:
            A collections.Counter object containing the n most common term counts for all records in the Corpus.
        """
        # Count the terms in all records
        counter = Counter()
        for record in self.records.values():
            if record.is_parsed:
                counter.update(record.terms)

        # Optionally filter the results
        if most_common and n:
            return counter.most_common(n)
        elif not most_common and n:
            return counter.most_common()[: -n - 1 : -1]
        elif most_common is False and n is None:
            return counter.most_common()[::]
        else:
            return counter

    @validate_call(config=model_config)
    def to_df(
        self, exclude: list[str] = ["content", "terms", "tokens"]
    ) -> pd.DataFrame:
        """Return a table of the Corpus records.

        Args:
            exclude (list[str]): A list of fields to exclude from the dataframe. If you wish to exclude metadata fields with the same name as model fields, you can use the prefix "metadata_" to avoid conflicts.

        Returns:
            pd.DataFrame: A dataframe representing the records in the Corpus.
        """
        rows = []
        for record in self.records.values():
            if record is None:  # Skip None records
                continue

            # Get parsed status of the record to determine how to build the row
            parsed = getattr(record, "is_parsed", False)
            if parsed:
                row = record.model_dump(exclude=list(exclude))
            else:
                row = self._build_unparsed_row(record, exclude)

            # Patch the row with computed fields and metadata
            row = self._patch_df_row_for_record(row, record, exclude, parsed)
            row = self._add_metadata_to_row(row, exclude)

            rows.append(row)

        # Create a DataFrame from the rows
        if len(rows) == 0:
            # Return empty DataFrame with basic columns if no records
            return pd.DataFrame(columns=["id", "name", "is_active"])
        else:
            df = pd.DataFrame(rows)
            # Fill NaN with appropriate values based on column dtype
            fill_values = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_values[col] = 0
                elif pd.api.types.is_bool_dtype(df[col]):
                    fill_values[col] = False
                else:
                    fill_values[col] = ""

            return df.fillna(fill_values)  # Use assignment instead of inplace

    def _build_unparsed_row(self, record: Record, exclude: list[str]) -> dict[str, Any]:
        """Build a row for an unparsed record.

        Args:
            record (Record): The unparsed record.
            exclude (list[str]): A list of fields to exclude from the row.

        Returns:
            dict[str, Any]: A dictionary representing the row for the unparsed record.
        """
        base_fields = [
            "id",
            "name",
            "is_active",
            "content",
            "model",
            "extensions",
            "data_source",
            "meta",
        ]
        row = {}
        for f in base_fields:
            if f in exclude:
                continue
            value = getattr(record, f, None)
            if f == "content" and isinstance(value, Doc):
                value = value.text
            if f == "id" and value is not None:
                value = str(value)
            if f == "meta" and value is not None:
                value = record._sanitize_metadata(value)
            row[f] = value
        return row

    def _patch_df_row_for_record(
        self, row: dict[str, Any], record: Record, exclude: list[str], parsed: bool
    ) -> dict[str, Any]:
        """Patch the DataFrame row with computed fields from the record.

        Args:
            row (dict[str, Any]): The initial row dictionary.
            record (Record): The record to extract computed fields from.
            exclude (list[str]): A list of fields to exclude from the row.
            parsed (bool): Whether the record is parsed.

        Returns:
        dict[str, Any]: The patched row dictionary with computed fields.
        """
        computed = [
            ("terms", lambda r: list(r.terms), []),
            ("tokens", lambda r: r.tokens, []),
            ("num_terms", lambda r: r.num_terms(), 0),
            ("num_tokens", lambda r: r.num_tokens(), 0),
            ("text", lambda r: r.text, ""),
        ]
        for field, getter, default in computed:
            if field in exclude:
                continue
            row[field] = getter(record) if parsed else default
        return row

    def _add_metadata_to_row(
        self, row: dict[str, Any], exclude: list[str]
    ) -> dict[str, Any]:
        """Add metadata fields to the DataFrame row, avoiding conflicts with existing fields.

        Args:
            row (dict[str, Any]): The initial row dictionary.
            exclude (list[str]): A list of fields to exclude from the row.

        Returns:
            dict[str, Any]: The row dictionary with metadata fields added, avoiding conflicts.
        """
        metadata = row.pop("meta", {})
        for key, value in metadata.items():
            if key in exclude or f"metadata_{key}" in exclude:
                continue
            if key in row:
                key = f"metadata_{key}"
            row[key] = value
        return row

    # =============================================================================
    # COMMUNICATION ARCHITECTURE - Phase 1.5
    # =============================================================================

    @validate_call(config=model_config)
    def import_analysis_results(
        self,
        module_name: str,
        results_data: dict[str, Any],
        version: str = "1.0.0",
        overwrite: bool = False,
    ) -> None:
        """Import analysis results from external modules into corpus metadata.

        Args:
            module_name: Name of the external module (e.g., 'kmeans', 'topwords', 'kwic', 'text_classification')
            results_data: Dictionary containing the analysis results
            version: Version string for result versioning and compatibility
            overwrite: Whether to overwrite existing results for this module

        Note:
            This is a framework implementation. Full functionality requires
            peer modules to be implemented and their result schemas defined.

        Returns:
            None
        """
        # TODO: Add result schema validation once peer modules are available
        # TODO: Add proper versioning system for backward compatibility
        # TODO: Implement result correlation capabilities across modules

        if module_name in self.analysis_results and not overwrite:
            raise ValueError(
                f"Results for module '{module_name}' already exist. "
                f"Use overwrite=True to replace them."
            )

        # Basic result structure with metadata
        self.analysis_results[module_name] = {
            "version": version,
            "timestamp": pd.Timestamp.now().isoformat(),
            "corpus_state": {
                "num_docs": self.num_docs,
                "num_active_docs": self.num_active_docs,
                "corpus_fingerprint": self._generate_corpus_fingerprint(),
            },
            "results": results_data,
        }

        msg.good(f"Imported {module_name} analysis results (version {version})")

    @validate_call(config=model_config)
    def get_analysis_results(self, module_name: str = None) -> dict[str, Any]:
        """Retrieve analysis results from external modules.

        Args:
            module_name: Specific module name to retrieve, or None for all results

        Returns:
            Dictionary containing analysis results
        """
        if module_name:
            if module_name not in self.analysis_results:
                raise ValueError(f"No results found for module '{module_name}'")
            return self.analysis_results[module_name]

        return self.analysis_results

    @validate_call(config=model_config)
    def export_statistical_fingerprint(self) -> dict[str, Any]:
        """Export standardized statistical summary for external modules.

        Returns:
            Dictionary containing corpus statistical fingerprint for external module consumption

        Note:
            This provides the standardized API for external modules to consume corpus statistics.
        """
        # TODO: Expand fingerprint based on external module requirements
        # TODO: Add feature extraction optimized for different analysis types

        try:
            stats = self.get_stats(active_only=True)

            # Core statistical fingerprint
            fingerprint = {
                "corpus_metadata": {
                    "name": self.name,
                    "num_docs": self.num_docs,
                    "num_active_docs": self.num_active_docs,
                    "num_tokens": self.num_tokens,
                    "num_terms": self.num_terms,
                    "corpus_fingerprint": self._generate_corpus_fingerprint(),
                },
                "distribution_stats": stats.distribution_stats,
                "percentiles": stats.percentiles,
                "text_diversity": stats.text_diversity_stats,
                "basic_stats": {
                    "mean": stats.mean,
                    "std": stats.standard_deviation,
                    "iqr_values": stats.iqr_values,
                    "iqr_bounds": stats.iqr_bounds,
                },
                "document_features": stats.doc_stats_df.to_dict("records"),
                "term_frequencies": self.term_counts(
                    n=100, most_common=True
                ),  # Top 100 terms
            }

            return fingerprint

        except Exception as e:
            # Fallback fingerprint if CorpusStats fails
            return {
                "corpus_metadata": {
                    "name": self.name,
                    "num_docs": self.num_docs,
                    "num_active_docs": self.num_active_docs,
                    "num_tokens": self.num_tokens,
                    "num_terms": self.num_terms,
                    "corpus_fingerprint": self._generate_corpus_fingerprint(),
                },
                "error": f"Statistical analysis failed: {str(e)}",
                "basic_features": {
                    "document_ids": list(self.records.keys()),
                    "document_names": list(self.names.keys()),
                },
            }

    def _generate_corpus_fingerprint(self) -> str:
        """Generate a unique fingerprint for corpus state validation.

        Returns:
            SHA256 hash representing current corpus state
        """
        import hashlib

        # Create fingerprint from corpus state
        state_data = {
            "num_docs": self.num_docs,
            "num_active_docs": self.num_active_docs,
            "record_ids": sorted(self.records.keys()),
            "active_record_ids": sorted(
                [k for k, v in self.records.items() if v and v.is_active]
            ),
        }

        state_string = str(sorted(state_data.items()))
        return hashlib.sha256(state_string.encode()).hexdigest()[:16]  # First 16 chars

    @validate_call(config=model_config)
    def validate_analysis_compatibility(self, module_name: str) -> dict[str, Any]:
        """Validate if stored analysis results are compatible with current corpus state.

        Args:
            module_name: Name of the module to validate

        Returns:
            Dictionary containing validation results and recommendations
        """
        if module_name not in self.analysis_results:
            return {
                "compatible": False,
                "reason": f"No analysis results found for module '{module_name}'",
            }

        stored_results = self.analysis_results[module_name]
        stored_state = stored_results.get("corpus_state", {})
        current_fingerprint = self._generate_corpus_fingerprint()
        stored_fingerprint = stored_state.get("corpus_fingerprint", "")

        compatibility = {
            "compatible": stored_fingerprint == current_fingerprint,
            "current_fingerprint": current_fingerprint,
            "stored_fingerprint": stored_fingerprint,
            "stored_timestamp": stored_results.get("timestamp", "unknown"),
            "stored_version": stored_results.get("version", "unknown"),
        }

        if not compatibility["compatible"]:
            compatibility["reason"] = (
                "Corpus state has changed since analysis was performed"
            )
            compatibility["recommendation"] = (
                f"Re-run {module_name} analysis with current corpus state"
            )

            # Detailed state comparison
            compatibility["state_changes"] = {
                "num_docs": {
                    "stored": stored_state.get("num_docs", 0),
                    "current": self.num_docs,
                    "changed": stored_state.get("num_docs", 0) != self.num_docs,
                },
                "num_active_docs": {
                    "stored": stored_state.get("num_active_docs", 0),
                    "current": self.num_active_docs,
                    "changed": stored_state.get("num_active_docs", 0)
                    != self.num_active_docs,
                },
            }

        return compatibility

    # TODO: Add when peer modules are available
    # def correlate_analysis_results(self, module1: str, module2: str) -> dict:
    #     """Correlate results between different analysis modules."""
    #     pass

    # TODO: Add when peer module schemas are defined
    # def validate_result_schema(self, module_name: str, results_data: dict) -> bool:
    #     """Validate that results conform to expected schema for the module."""
    #     pass
