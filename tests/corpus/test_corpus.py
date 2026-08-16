"""test_corpus.py.

Coverage: 99%. Missing: 756, 986

Last Update: August 15, 2026
"""

import datetime
import shutil
import uuid
from collections import Counter
from pathlib import Path

import pytest
import spacy
import srsly

try:
    from lexos.corpus.corpus import Corpus
    from lexos.corpus.record import Record

    CORPUS_AVAILABLE = True
except Exception:
    CORPUS_AVAILABLE = False

from lexos.exceptions import LexosException

pytestmark = pytest.mark.skipif(
    not CORPUS_AVAILABLE, reason="corpus module unavailable"
)


@pytest.fixture
def corpus_dir(tmp_path):
    """Fixture to create a temporary corpus directory."""
    return str(tmp_path)


@pytest.fixture
def corpus(corpus_dir):
    """Fixture to create a Corpus instance for testing."""
    return Corpus(corpus_dir=corpus_dir, name="TestCorpus")


def test_add_single_record_and_state(corpus):
    """Test adding a single record to the corpus and verify state updates."""
    corpus.add("Hello world", name="doc1")
    assert corpus.num_docs == 1
    assert corpus.num_active_docs == 1
    assert len(corpus.records) == 1

    record = next(iter(corpus.records.values()))
    assert record.name == "doc1"
    assert record.content == "Hello world"
    assert record.is_parsed is False


def test_get_record_by_id_and_name(corpus):
    """Test retrieving records by ID and name from the corpus."""
    corpus.add("Alpha test", name="alpha")
    corpus.add("Beta test", name="beta")

    ids = list(corpus.records.keys())
    assert len(ids) == 2

    first = corpus.get(id=ids[0])
    assert first.name in {"alpha", "beta"}

    second = corpus.get(name=first.name)
    assert second.id == first.id


def test_metadata_is_sanitized(corpus):
    """Test that metadata is sanitized when adding a record to the corpus."""
    metadata = {
        "created": datetime.date(2026, 1, 1),
        "source": Path("/tmp/example.txt"),
        "ref": uuid.uuid4(),
    }
    corpus.add("Meta document", name="meta_doc", metadata=metadata)

    record = corpus.get(name="meta_doc")
    assert record.meta["created"] == "2026-01-01"
    assert record.meta["source"] == str(Path("/tmp/example.txt"))
    assert isinstance(record.meta["ref"], str)


def test_remove_by_id_and_name(corpus):
    """Test removing records by ID and name from the corpus."""
    corpus.add("One", name="one")
    corpus.add("Two", name="two")
    ids = list(corpus.records.keys())

    corpus.remove(id=ids[0])
    assert len(corpus.records) == 1

    corpus.remove(name="two")
    assert len(corpus.records) == 0
    assert corpus.names == {}
    assert corpus.num_docs == 0


def test_remove_invalid_id_raises(corpus):
    """Test that removing a record with an invalid ID raises an exception."""
    with pytest.raises(LexosException):
        corpus.remove(id="not-a-real-id")


def test_get_stats_and_to_df_single_record(corpus):
    """Test getting statistics and converting to DataFrame with a single record."""
    corpus.add("The quick brown fox", name="fox")

    stats = corpus.get_stats(active_only=False, type="tokens")
    assert hasattr(stats, "docs")
    assert len(stats.docs) >= 1

    df = corpus.to_df()
    assert df.shape[0] == 1
    assert "name" in df.columns
    assert df.loc[0, "name"] == "fox"
    assert bool(df.loc[0, "is_active"]) is True


def test_corpus_init_preserves_existing_metadata(tmp_path):
    """Test that initializing a Corpus with an existing metadata file preserves the metadata."""
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    initial = Corpus(corpus_dir=str(src_dir), name="Initial")
    metadata_path = src_dir / initial.corpus_metadata_file
    data = srsly.read_json(metadata_path)
    data["meta"] = {"project": "TestProject"}
    srsly.write_json(metadata_path, data)

    reloaded = Corpus(corpus_dir=src_dir, name="Reload")
    assert reloaded.meta.get("project") == "TestProject"


def test_repr_contains_name(corpus):
    """Test that the string representation of the Corpus includes its name."""
    rep = repr(corpus)
    assert "Corpus(" in rep
    assert "name=TestCorpus" in rep


def test_meta_df_raises_when_empty(corpus):
    """Test that accessing meta_df raises an exception when metadata is empty."""
    with pytest.raises(LexosException):
        _ = corpus.meta_df


def test_meta_df_returns_dataframe(corpus):
    """Test that meta_df returns a DataFrame with the correct metadata."""
    corpus.meta = {"project": "TestProject", "version": 1}
    df = corpus.meta_df
    assert df.loc[0, "project"] == "TestProject"
    assert df.loc[0, "version"] == 1


def test_active_terms_and_num_active_counts(corpus):
    """Test that active terms and counts are correctly computed for active and inactive records."""
    corpus.add("Active doc", name="active", is_active=True)
    corpus.add("Inactive doc", name="inactive", is_active=False)
    assert isinstance(corpus.active_terms, set)
    assert corpus.num_active_docs == 1
    assert corpus.num_active_terms == 0
    assert corpus.num_active_tokens == 0


def test_ensure_unique_name_behaviour(corpus):
    """Test that _ensure_unique_name generates unique names when duplicates exist."""
    corpus.names = {"foo": ["1"]}
    generated = corpus._ensure_unique_name(None)
    assert generated.startswith("untitled_")
    generated_duplicate = corpus._ensure_unique_name("foo")
    assert generated_duplicate.startswith("foo_")


def test_generate_unique_id_invalid_type_raises(corpus):
    """Test that _generate_unique_id raises an exception for an invalid type."""
    with pytest.raises(LexosException):
        corpus._generate_unique_id(type="invalid")


def test_iter_returns_records(corpus):
    """Test that iterating over the corpus yields the records in the correct order."""
    corpus.add("Iterate me", name="iter_doc")
    assert list(corpus) == list(corpus.records.values())


def test_active_terms_and_token_counts_for_parsed_record(corpus):
    """Test that active terms and token counts are correctly computed for a parsed record."""
    nlp = spacy.blank("en")
    record = Record(id="1", name="parsed_doc", content=nlp("hello world"))
    object.__setattr__(record, "is_parsed", True)
    object.__setattr__(record, "terms", Counter({"hello": 1, "world": 1}))
    object.__setattr__(record, "tokens", [token.text for token in record.content])
    record.is_active = True
    corpus.records[str(record.id)] = record

    assert corpus.active_terms == {"hello", "world"}
    assert corpus.num_active_terms == 2
    assert corpus.num_active_tokens == 2


def test_ensure_unique_name_duplicate_returns_new_name(corpus):
    """Test that _ensure_unique_name returns a new unique name when a duplicate exists."""
    corpus.names = {"foo": ["1"]}
    unique_name = corpus._ensure_unique_name("foo")
    assert unique_name != "foo"
    assert unique_name.startswith("foo_")


def test_ensure_unique_name_returns_same_name_when_unique(corpus):
    """Test that _ensure_unique_name returns the same name when it is already unique."""
    assert corpus._ensure_unique_name("unique_name") == "unique_name"


def test_generate_unique_id_uuid4_collision(corpus, monkeypatch):
    """Test that _generate_unique_id handles UUID4 collisions and generates a new unique ID."""
    import lexos.corpus.corpus as corpus_module

    existing = Record(
        id="00000000-0000-4000-8000-000000000123", name="dup", content="x"
    )
    corpus.records[str(existing.id)] = existing

    uuids = iter(
        [
            uuid.UUID("00000000-0000-4000-8000-000000000123"),
            uuid.UUID("00000000-0000-4000-8000-000000000456"),
        ]
    )
    monkeypatch.setattr(corpus_module.uuid, "uuid4", lambda: next(uuids))

    new_id = corpus._generate_unique_id(type="uuid4")
    assert new_id == "00000000-0000-4000-8000-000000000456"


def test_update_corpus_state_converts_uuid_like_values(corpus, tmp_path, monkeypatch):
    """Test that _update_corpus_state correctly converts UUID-like values to strings when saving metadata."""

    class FakeUUID:
        hex = "deadbeef"

        def __str__(self):
            return "deadbeef"

    corpus.corpus_dir = str(tmp_path)
    data = {
        "corpus_dir": str(tmp_path),
        "corpus_metadata_file": "corpus_metadata.json",
        "name": corpus.name,
        "records": {},
        "names": {},
        "meta": {},
        "analysis_results": {},
        "num_active_docs": 0,
        "num_docs": 0,
        "num_terms": 0,
        "num_tokens": 0,
        "terms": [],
        "fake_uuid": FakeUUID(),
    }
    monkeypatch.setattr(Corpus, "model_dump", lambda self, exclude: data)
    corpus._update_corpus_state()

    written = srsly.read_json(Path(corpus.corpus_dir) / corpus.corpus_metadata_file)
    assert written["fake_uuid"] == "deadbeef"


def test_create_record_accepts_extensions(corpus):
    """Test that _create_record correctly assigns extensions to the record."""
    record = corpus._create_record(
        item="content",
        record_id="1",
        name="record",
        is_active=True,
        model=None,
        extensions=["ext"],
        metadata=None,
    )
    assert record.extensions == ["ext"]


def test_normalize_ids_accepts_list(corpus):
    """Test that _normalize_ids correctly returns a list of IDs when provided with a list."""
    assert corpus._normalize_ids(id=["1", "2"]) == ["1", "2"]


def test_load_record_by_id_missing_raises_key_error(corpus):
    """Test that _load_record_by_id raises a KeyError when the record ID is missing."""
    with pytest.raises(KeyError):
        corpus._load_record_by_id("missing")


def test_load_record_by_id_uses_from_disk_when_contains_returns_false(
    corpus, monkeypatch
):
    """Test that _load_record_by_id uses the _from_disk method when the record is not already loaded."""

    class FakeRecord:
        def __init__(self):
            self.id = "1"
            self.meta = {"filepath": "fake.bin"}
            self.model = None

        def _from_disk(self, path, model, model_cache):
            return self

    fake_record = FakeRecord()

    class FakeRecords:
        def __contains__(self, key):
            return False

        def __getitem__(self, key):
            return fake_record

    object.__setattr__(corpus, "records", FakeRecords())

    loaded = corpus._load_record_by_id("1")
    assert loaded is fake_record


def test_get_token_strings_parsed(corpus):
    """Test that _get_token_strings returns the correct token strings for a parsed record."""
    nlp = spacy.blank("en")
    record = Record(id="1", name="parsed_doc", content=nlp("hello world"))
    object.__setattr__(record, "is_parsed", True)
    assert corpus._get_token_strings(record) == ["hello", "world"]


def test_filter_records_skips_entries_without_metadata(corpus):
    """Test that filter_records skips entries that do not have metadata."""
    corpus.add("Document A", name="docA", metadata={"group": "A"})
    corpus.records["bad"] = object()
    filtered = corpus.filter_records(group="A")
    assert len(filtered) == 1


def test_remove_record_by_id_raises_when_name_missing(corpus):
    """Test that _remove_record_by_id raises a LexosException when the record's name is missing."""
    record = Record(id="1", name="missing_name", content="hello")
    corpus.records[str(record.id)] = record
    with pytest.raises(LexosException):
        corpus._remove_record_by_id(str(record.id))


def test_load_uses_corpus_dir_when_corpus_dir_not_provided(tmp_path):
    """Test that load uses the corpus_dir attribute when corpus_dir is not provided."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    src = Corpus(corpus_dir=source_dir, name="LoadedCorpus")
    archive_path = tmp_path / "loaded"
    shutil.make_archive(str(archive_path), "zip", str(source_dir))

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    loaded = Corpus(corpus_dir=target_dir, name="placeholder")
    loaded.load(path=archive_path.with_suffix(".zip"))

    assert loaded.name == "LoadedCorpus"


def test_load_invalid_zip_raises(tmp_path):
    """Test that load raises a LexosException when an invalid zip file is provided."""
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("not a zip")
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    corpus = Corpus(corpus_dir=target_dir, name="placeholder")

    with pytest.raises(LexosException):
        corpus.load(path=invalid_file, corpus_dir=target_dir)


def test_load_cache_uses_record_from_disk_with_actual_record_objects(
    tmp_path, monkeypatch
):
    """Test that load with cache=True uses Record.from_disk when the records are actual Record objects."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    corpus = Corpus(corpus_dir=target_dir, name="placeholder")
    record = Record(id="1", name="cached", content="hello", is_active=True)

    monkeypatch.setattr(
        srsly,
        "read_json",
        lambda path: {
            "corpus_dir": str(target_dir),
            "corpus_metadata_file": "corpus_metadata.json",
            "name": "LoadedCorpus",
            "records": {"1": record},
            "names": {"cached": ["1"]},
            "meta": {},
            "analysis_results": {},
            "num_active_docs": 1,
            "num_docs": 1,
            "num_terms": 0,
            "num_tokens": 0,
            "terms": set(),
        },
    )
    monkeypatch.setattr(
        Record, "from_disk", lambda self, path, model, model_cache: None
    )

    corpus.load(path=tmp_path / "unused", corpus_dir=target_dir, cache=True)
    assert corpus.name == "LoadedCorpus"
    assert isinstance(corpus.records["1"], Record)


def test_load_cache_raises_when_record_is_not_a_record(tmp_path, monkeypatch):
    """Test that load with cache=True raises a LexosException when the records are not actual Record objects."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    corpus = Corpus(corpus_dir=target_dir, name="placeholder")

    monkeypatch.setattr(
        srsly,
        "read_json",
        lambda path: {
            "corpus_dir": str(target_dir),
            "corpus_metadata_file": "corpus_metadata.json",
            "name": "LoadedCorpus",
            "records": {"1": {"id": "1"}},
            "names": {"cached": ["1"]},
            "meta": {},
            "analysis_results": {},
            "num_active_docs": 1,
            "num_docs": 1,
            "num_terms": 0,
            "num_tokens": 0,
            "terms": set(),
        },
    )

    with pytest.raises(LexosException):
        corpus.load(path=tmp_path / "unused", corpus_dir=target_dir, cache=True)


def test_set_unlinks_old_filepath(tmp_path, corpus, monkeypatch):
    """Test that corpus.set updates the record's metadata and unlinks the old file path."""
    corpus.add("Hello", name="set_doc")
    record_id = next(iter(corpus.records.keys()))
    record = corpus.records[record_id]
    old_path = tmp_path / "old.bin"
    old_path.write_text("x")
    record.meta["filepath"] = str(old_path)
    new_path = tmp_path / "new.bin"
    monkeypatch.setattr(Record, "to_disk", lambda self, path, extensions=None: None)

    corpus.set(record_id, meta={"filepath": str(new_path)})
    assert not old_path.exists()


def test_term_counts_n_none_returns_empty_list(corpus):
    """Test that term_counts returns an empty list when n is None and most_common is False."""
    result = corpus.term_counts(n=None, most_common=False)
    assert result == []


def test_to_df_handles_none_records(corpus):
    """Test that to_df correctly handles records that are None."""
    record = Record(id="1", name="good", content="hello")
    corpus.records["bad"] = None
    corpus.records[str(record.id)] = record
    df = corpus.to_df(exclude=["content", "terms", "tokens"])
    assert df.loc[0, "name"] == "good"


def test_to_df_includes_parsed_record(corpus):
    """Test that to_df correctly includes a parsed record with terms and tokens."""
    nlp = spacy.blank("en")
    record = Record(id="1", name="parsed_doc", content=nlp("hello"), is_active=True)
    object.__setattr__(record, "is_parsed", True)
    object.__setattr__(record, "terms", Counter({"hello": 1}))
    object.__setattr__(record, "tokens", ["hello"])
    corpus.records[str(record.id)] = record
    df = corpus.to_df(exclude=["terms", "tokens"])
    assert df.loc[0, "name"] == "parsed_doc"
    assert bool(df.loc[0, "is_active"]) is True


def test_to_df_bool_fill_values(corpus):
    """Test that to_df correctly fills boolean values."""
    record = Record(id="1", name="parsed_doc", content="hello", is_active=True)
    corpus.records[str(record.id)] = record
    df = corpus.to_df(exclude=["terms", "tokens"])
    assert df["is_active"].dtype == bool


def test_to_df_returns_empty_dataframe_when_no_records(corpus):
    """Test that to_df returns an empty DataFrame when there are no records."""
    df = corpus.to_df(exclude=["content", "terms", "tokens"])
    assert df.empty
    assert list(df.columns) == ["id", "name", "is_active"]


def test_build_unparsed_row_converts_doc_content(corpus):
    """Test that _build_unparsed_row correctly converts a Doc object to string content."""
    nlp = spacy.blank("en")
    record = Record(id="1", name="doc_with_doc", content=nlp("hello"))
    row = corpus._build_unparsed_row(record, exclude=[])
    assert row["content"] == "hello"


def test_add_metadata_to_row_respects_exclude(corpus):
    """Test that _add_metadata_to_row correctly excludes specified metadata keys."""
    row = {"meta": {"name": "conflict", "foo": 1}}
    result = corpus._add_metadata_to_row(row, exclude=["name"])
    assert "name" not in result
    assert result["foo"] == 1


def test_get_analysis_results_with_and_without_module_name(corpus):
    """Test that get_analysis_results returns the correct results with and without a module name."""
    corpus.import_analysis_results("module", {"value": 1})
    assert corpus.get_analysis_results("module")["results"]["value"] == 1
    assert corpus.get_analysis_results()["module"]["results"]["value"] == 1
    with pytest.raises(ValueError):
        corpus.get_analysis_results("missing")


def test_export_statistical_fingerprint_success(corpus):
    """Test that export_statistical_fingerprint returns the correct structure when successful."""
    nlp = spacy.blank("en")
    record = Record(id="1", name="parsed_doc", content=nlp("a b"), is_active=True)
    object.__setattr__(record, "is_parsed", True)
    object.__setattr__(record, "terms", Counter({"a": 1, "b": 1}))
    object.__setattr__(record, "tokens", ["a", "b"])
    corpus.records["1"] = record
    corpus._update_corpus_state()

    fingerprint = corpus.export_statistical_fingerprint()
    assert fingerprint["corpus_metadata"]["name"] == corpus.name
    assert "document_features" in fingerprint


def test_export_statistical_fingerprint_fallback(corpus, monkeypatch):
    """Test that export_statistical_fingerprint returns an error structure when get_stats raises an exception."""
    monkeypatch.setattr(
        Corpus,
        "get_stats",
        lambda self, active_only=True: (_ for _ in ()).throw(Exception("boom")),
    )
    fallback = corpus.export_statistical_fingerprint()
    assert fallback["corpus_metadata"]["name"] == corpus.name
    assert "error" in fallback


def test_validate_analysis_compatibility_when_module_not_found(corpus):
    """Test that validate_analysis_compatibility returns False when the module is not found."""
    result = corpus.validate_analysis_compatibility("missing")
    assert result["compatible"] is False


def test_validate_analysis_compatibility_detects_incompatible_state(corpus):
    """Test that validate_analysis_compatibility detects an incompatible state."""
    corpus.import_analysis_results("module", {"value": 1})
    corpus.num_docs = 1
    result = corpus.validate_analysis_compatibility("module")
    assert result["compatible"] is False
    assert "state_changes" in result


def test_add_from_files_with_name_template(corpus, tmp_path, monkeypatch):
    """Test that add_from_files correctly applies the name template."""
    from lexos.io import parallel_loader

    class FakeLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load_streaming(self, paths):
            yield (tmp_path / "a.txt", "a", "text/plain", "A", None)

    monkeypatch.setattr(parallel_loader, "ParallelLoader", FakeLoader)

    corpus.add_from_files(
        [tmp_path / "a.txt"],
        show_progress=False,
        name_template="{stem}_{index}",
        extensions=["ext"],
    )
    assert len(corpus.records) == 1
    record = next(iter(corpus.records.values()))
    assert record.name.startswith("a_")
    assert record.extensions == ["ext"]


def test_term_counts_most_common_and_least_common(corpus, monkeypatch):
    """Test that term_counts raises a LexosException when the record is missing."""
    with pytest.raises(LexosException):
        corpus._get_by_name("missing")


def test_normalize_ids_raises_when_no_input(corpus):
    """Test that _normalize_ids raises a LexosException when no ID input is provided."""
    with pytest.raises(LexosException):
        corpus._normalize_ids()


def test_filter_stats_records_returns_both_active_and_all(corpus):
    """Test that _filter_stats_records returns the correct number of active and all records."""
    corpus.add("Active", name="a", is_active=True)
    corpus.add("Inactive", name="b", is_active=False)
    active = corpus._filter_stats_records(active_only=True)
    all_records = corpus._filter_stats_records(active_only=False)
    assert len(active) == 1
    assert len(all_records) == 2


def test_build_stats_token_list_characters(corpus):
    """Test that _build_stats_token_list correctly builds a token list for character type."""
    corpus.add("Hello", name="char_doc")
    records = list(corpus.records.values())
    token_list = corpus._build_stats_token_list(records, type="characters")
    assert token_list[0][2] == list("Hello")


def test_build_stats_token_list_invalid_type_raises(corpus):
    """Test that _build_stats_token_list raises a LexosException for an invalid type."""
    corpus.add("Hello", name="invalid_doc")
    with pytest.raises(LexosException):
        corpus._build_stats_token_list(list(corpus.records.values()), type="invalid")


def test_get_token_strings_unparsed(corpus):
    """Test that _get_token_strings returns the correct token strings for an unparsed record."""
    record = Record(id="1", name="r1", content="hello world")
    assert corpus._get_token_strings(record) == ["hello", "world"]


def test_add_list_content_adds_multiple_records(corpus):
    """Test that adding a list of content creates multiple records."""
    corpus.add(["one", "two"], name="batch")
    assert len(corpus.records) == 2


def test_add_record_object_duplicate_id_raises(corpus):
    """Test that adding a record with a duplicate ID raises a LexosException."""
    record = Record(id="1", name="dup", content="hello")
    corpus.add(record)
    with pytest.raises(LexosException):
        corpus.add(record)


def test_add_from_files_with_mock_loader(corpus, tmp_path, monkeypatch):
    """Test that add_from_files correctly uses a mock loader."""
    from lexos.io import parallel_loader

    class FakeLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load_streaming(self, paths):
            yield (tmp_path / "a.txt", "a", "text/plain", "A", None)
            yield (tmp_path / "b.txt", "b", "text/plain", None, "error")

    monkeypatch.setattr(parallel_loader, "ParallelLoader", FakeLoader)

    corpus.add_from_files(
        [tmp_path / "a.txt"], show_progress=False, metadata={"group": "A"}
    )
    assert len(corpus.records) == 1


def test_save_creates_zip(corpus, tmp_path):
    """Test that saving the corpus creates a zip archive."""
    corpus.add("Save me", name="save_doc")
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    corpus.save(out_dir)
    assert (out_dir / "TestCorpus.zip").exists()


def test_load_from_zip_archive(corpus, tmp_path):
    """Test that loading a corpus from a zip archive correctly restores the corpus state."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    src = Corpus(corpus_dir=source_dir, name="LoadedCorpus")
    archive_path = tmp_path / "loaded"
    shutil.make_archive(str(archive_path), "zip", str(source_dir))

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    corpus_to_load = Corpus(corpus_dir=target_dir, name="placeholder")
    corpus_to_load.load(path=archive_path.with_suffix(".zip"), corpus_dir=target_dir)
    assert corpus_to_load.name == "LoadedCorpus"


def test_set_updates_properties_and_state(corpus):
    """Test that corpus.set correctly updates record properties and corpus state."""
    corpus.add("Hello", name="set_doc")
    record_id = next(iter(corpus.records.keys()))
    corpus.set(record_id, is_active=False)
    assert corpus.records[record_id].is_active is False
    assert corpus.num_active_docs == 0


def test_term_counts_empty_and_none(corpus):
    """Test that term_counts returns an empty list when there are no records and n is None."""
    result = corpus.term_counts(n=None, most_common=False)
    assert result == []


def test_term_counts_n_none_most_common_true_returns_counter(corpus):
    """Test that term_counts returns a Counter when n is None and most_common is True."""
    record = Record(id="1", name="term_doc", content="hello")
    object.__setattr__(record, "is_parsed", True)
    object.__setattr__(record, "terms", Counter({"hello": 1}))
    corpus.records["1"] = record
    result = corpus.term_counts(n=None, most_common=True)
    assert isinstance(result, Counter)


def test_add_metadata_to_row_conflict(corpus):
    """Test that _add_metadata_to_row correctly handles metadata conflicts."""
    row = {
        "id": "1",
        "name": "doc",
        "is_active": True,
        "meta": {"name": "conflict", "x": 1},
    }
    result = corpus._add_metadata_to_row(row, exclude=[])
    assert result["metadata_name"] == "conflict"


def test_import_analysis_results_overwrite(corpus):
    """Test that import_analysis_results correctly handles overwriting existing results."""
    corpus.import_analysis_results("module", {"value": 1})
    with pytest.raises(ValueError):
        corpus.import_analysis_results("module", {"value": 2})
    corpus.import_analysis_results("module", {"value": 3}, overwrite=True)
    assert corpus.analysis_results["module"]["version"] == "1.0.0"


def test_add_integer_ids(corpus):
    """Test that adding records with integer IDs generates sequential integer IDs."""
    corpus.add("Integer ID doc", name="integer_doc", id_type="integer")
    assert len(corpus.records) == 1

    record_id = next(iter(corpus.records.keys()))
    assert record_id.isdigit()

    corpus.add("Second integer doc", name="integer_doc_2", id_type="integer")
    second_id = list(corpus.records.keys())[1]
    assert second_id.isdigit()
    assert int(second_id) == int(record_id) + 1


def test_filter_records_by_metadata(corpus):
    """Test that filter_records correctly filters records based on metadata."""
    corpus.add("Document A", name="docA", metadata={"group": "A"})
    corpus.add("Document B", name="docB", metadata={"group": "B"})

    filtered = corpus.filter_records(group="A")
    assert len(filtered) == 1
    assert filtered[0].name == "docA"


def test_filter_records_no_match_returns_empty(corpus):
    """Test that filter_records returns an empty list when no records match the filter criteria."""
    corpus.add("Document C", name="docC", metadata={"group": "C"})
    filtered = corpus.filter_records(group="Z")
    assert filtered == []


def test_sanitize_metadata_nested_values(corpus):
    """Test that _sanitize_metadata correctly sanitizes nested metadata values."""
    data = {
        "path": Path("/tmp/test.txt"),
        "timestamp": datetime.datetime(2026, 8, 15, 12, 0),
        "uuid": uuid.uuid4(),
        "items": [Path("/tmp/one"), datetime.date(2026, 8, 15), uuid.uuid4()],
        "nested": {"inner": Path("/tmp/inner.txt")},
    }
    sanitized = corpus._sanitize_metadata(data)
    assert isinstance(sanitized["path"], str)
    assert isinstance(sanitized["timestamp"], str)
    assert isinstance(sanitized["uuid"], str)
    assert isinstance(sanitized["items"], list)
    assert isinstance(sanitized["nested"]["inner"], str)


def test_normalize_content_items_handles_single_and_list(corpus):
    """Test that _normalize_content_items correctly normalizes single strings and lists of strings."""
    single = corpus._normalize_content_items("text")
    assert single == ["text"]
    multiple = corpus._normalize_content_items(["one", "two"])
    assert multiple == ["one", "two"]


def test_create_record_returns_existing_record(corpus):
    """Test that _create_record returns the existing record when it already exists in the corpus."""
    record = Record(id=uuid.uuid4(), name="existing_doc", content="hello")
    created = corpus._create_record(
        record,
        record_id=str(record.id),
        name="existing_doc",
        is_active=True,
        model=None,
        extensions=None,
        metadata=None,
    )
    assert created is record


def test_get_stats_with_token_list_override(corpus):
    """Test that get_stats correctly uses the provided token_list instead of building it from records."""
    token_list = [("1", "doc1", ["a", "b"])]
    stats = corpus.get_stats(token_list=token_list)
    assert len(stats.docs) == 1
    assert stats.docs[0][0] == "1"


def test_to_df_excludes_fields_and_handles_metadata(corpus):
    """Test that to_df correctly excludes specified fields and includes metadata."""
    corpus.add("Test content", name="doc_meta", metadata={"foo": "bar"})
    df = corpus.to_df(exclude=["content", "terms", "tokens"])
    assert "foo" in df.columns or "metadata_foo" in df.columns
    assert df.loc[0, "name"] == "doc_meta"


def test_remove_invalid_name_raises(corpus):
    """Test that removing a record with an invalid name raises a LexosException."""
    with pytest.raises(LexosException):
        corpus.remove(name="missing_name")


def test_load_updates_corpus_name_and_meta(tmp_path):
    """Test that loading a corpus from a zip archive updates the corpus name and metadata correctly."""
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    source = Corpus(corpus_dir=src_dir, name="SourceCorpus")
    source.add("Load test", name="load_doc", metadata={"project": "LoadProject"})

    archive_path = tmp_path / "archive"
    shutil.make_archive(str(archive_path), "zip", str(src_dir))

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    loaded = Corpus(corpus_dir=target_dir, name="placeholder")
    loaded.load(path=archive_path.with_suffix(".zip"), corpus_dir=target_dir)

    assert loaded.name == "SourceCorpus"
    assert any(
        isinstance(value, dict)
        and value.get("meta", {}).get("project") == "LoadProject"
        for value in loaded.meta.values()
    )


def test_add_from_files_with_name_template(corpus, tmp_path, monkeypatch):
    """Test that add_from_files correctly applies the name template and metadata."""
    from lexos.io import parallel_loader

    class FakeLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load_streaming(self, paths):
            yield (tmp_path / "a.txt", "a", "text/plain", "A", None)

    monkeypatch.setattr(parallel_loader, "ParallelLoader", FakeLoader)

    corpus.add_from_files(
        [tmp_path / "a.txt"],
        show_progress=False,
        name_template="{stem}_{index}",
        extensions=["ext"],
        metadata={"group": "A"},
    )
    assert len(corpus.records) == 1
    record = next(iter(corpus.records.values()))
    assert record.name.startswith("a_")
    assert record.extensions == ["ext"]
    assert record.meta["group"] == "A"


def test_term_counts_most_common_and_least_common(corpus, monkeypatch):
    """Test that term_counts returns the correct most common and least common terms."""
    record = Record(id="1", name="term_doc", content="hello")
    object.__setattr__(record, "is_parsed", True)
    object.__setattr__(record, "terms", Counter({"hello": 1, "world": 2}))
    corpus.records = {"1": record}
    result = corpus.term_counts(n=1, most_common=True)
    assert result[0][0] == "world"
    result_least = corpus.term_counts(n=1, most_common=False)
    assert result_least[0][0] == "hello"


def test_add_to_corpus_without_state_update(corpus, tmp_path):
    """Test that _add_to_corpus_without_state_update adds a record without updating the corpus state."""
    record_id = uuid.uuid4()
    record = Record(id=record_id, name="no_state", content="hello")
    corpus._add_to_corpus_without_state_update(record)
    assert corpus.records.get(str(record_id)) is record
    assert corpus.num_docs == 0


def test_build_unparsed_row_preserves_metadata(corpus):
    """Test that _build_unparsed_row preserves the record's metadata and content."""
    record = Record(id="2", name="dry_doc", content="hello", meta={"foo": "bar"})
    row = corpus._build_unparsed_row(record, exclude=[])
    assert row["meta"]["foo"] == "bar"
    assert row["content"] == "hello"
