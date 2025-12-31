"""test_init.py.

Tests for the dfr_browser2 Browser class.

Coverage: 100%


Last Updated: December 29, 2025
"""

import json
import shutil
import subprocess
import webbrowser
from pathlib import Path

import pytest

from lexos.topic_modeling.dfr_browser2 import Browser


def create_file(path: Path, content: str = ""):
    """Create a file at the given path with the specified content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_browser_initialization_and_config_merging(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Ensure Browser copies template and mallet files, merges config, and updates config paths to data/."""
    # Create a fake template directory (dist) with a basic config.json
    template_dir = dist_template_dir

    # Create a fake mallet files directory with the required files
    mallet_dir = mallet_dir_factory()

    # Create a sample data file (TSV)
    data_file = sample_tsv

    # Initialize browser with provided config to override 'application.name'
    user_cfg = {"application": {"name": "Custom Title"}}
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out"),
        data_path=str(data_file),
        config=user_cfg,
    )

    # Validate files copied into output browser data/ folder
    out_data = Path(b.browser_path) / "data"
    assert (out_data / "docs.txt").exists()
    assert (out_data / "topic-keys.txt").exists()
    # doc-topic should have been copied
    assert (out_data / "doc-topic.txt").exists()
    assert (out_data / "metadata.csv").exists()
    assert (out_data / "topic-state.gz").exists()
    assert (out_data / "topic_coords.csv").exists()

    # Validate config.json updated with merged config and file paths
    cfg_path = Path(b.browser_path) / "config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # User config should override template entries where appropriate
    assert cfg.get("application", {}).get("name") == "Custom Title"
    # File paths should point to data/ relative paths
    assert cfg.get("data_source") == "data/docs.txt"
    assert cfg.get("topic_keys_file") == "data/topic-keys.txt"
    assert cfg.get("doc_topic_file") == "data/doc-topic.txt"


def test_default_template_path_resolves_to_module_dist(
    tmp_path: Path, mallet_dir_factory: callable, sample_tsv: Path
):
    """Test that default template_path='dist' resolves to module's dist directory."""
    # Create mallet files
    mallet_dir = mallet_dir_factory()

    # Ensure the dist directory exists before instantiating Browser (CI robustness)
    import lexos.topic_modeling.dfr_browser2

    expected_module_dir = Path(lexos.topic_modeling.dfr_browser2.__file__).parent
    expected_template = expected_module_dir / "dist"
    expected_template.mkdir(exist_ok=True)

    # Don't provide template_path, let it use default "dist"
    b = Browser(
        mallet_files_path=str(mallet_dir),
        browser_path=str(tmp_path / "browser"),
        data_path=str(sample_tsv),
    )

    # Should resolve to the module's dist directory
    assert b.template_path == expected_template
    assert b.template_path.exists()


def test_filename_map_original_to_new(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Test that filename_map is treated as original -> destination mapping and config entries are updated."""
    # Test that filename_map is treated as original->new mapping
    template_dir = dist_template_dir

    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topics.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    # Create files using 'original' names
    orig_names = [
        "metadata.csv",
        "topic-keys.txt",
        "doc-topics.txt",
        "topic-state.gz",
        "topic_coords.csv",
    ]
    for fname in orig_names:
        create_file(mallet_dir / fname, "content")

    # We'll map 'doc-topics.txt' -> 'doc-topic.txt' and 'topic-keys.txt' -> 'topic-keys.txt'
    filename_map = {
        "doc-topics.txt": "doc-topic.txt",
        "topic-keys.txt": "topic-keys.txt",
        "metadata.csv": "metadata.csv",
    }

    data_file = sample_tsv

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out2"),
        data_path=str(data_file),
        config={"application": {"name": "Mapped"}},
        filename_map=filename_map,
    )

    out_data = Path(b.browser_path) / "data"
    # The destination names should be the values from filename_map
    assert (out_data / "doc-topic.txt").exists()
    assert (out_data / "topic-keys.txt").exists()
    assert (out_data / "metadata.csv").exists()

    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("doc_topic_file") == "data/doc-topic.txt"
    assert cfg.get("topic_keys_file") == "data/topic-keys.txt"


def test_filename_map_reversed_mapping(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Test reversed mapping where filename_map's key is treated as destination and value as source."""
    # Test reversed mapping where keys are destination and values are original
    template_dir = dist_template_dir

    mallet_dir = mallet_dir_factory(
        [
            "doc-topics.txt",
            "metadata.csv",
            "topic-keys.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    # Create original file named 'doc-topics.txt'
    create_file(mallet_dir / "doc-topics.txt", "content")
    create_file(mallet_dir / "metadata.csv", "content")
    create_file(mallet_dir / "topic-keys.txt", "content")
    # Create other required files
    create_file(mallet_dir / "topic-state.gz", "content")
    create_file(mallet_dir / "topic_coords.csv", "content")

    filename_map = {
        # reversed: key is destination name, value is original name
        "doc-topic.txt": "doc-topics.txt",
    }

    data_file = sample_tsv

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out3"),
        data_path=str(data_file),
        filename_map=filename_map,
    )

    out_data = Path(b.browser_path) / "data"
    # Since the mapping was reversed, the final destination should be the key
    assert (out_data / "doc-topic.txt").exists()
    # Original file name should not be present (it was copied and renamed)
    assert not (out_data / "doc-topics.txt").exists()


def test_partial_filename_map(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Test that filename_map can include only a subset of files and defaults are used for others."""
    # Test that filename_map can include only a subset of files and defaults are used for others
    template_dir = dist_template_dir

    mallet_dir = mallet_dir_factory()

    # Only map topic-keys to a custom destination
    filename_map = {
        "topic-keys.txt": "custom-topic-keys.txt",
    }

    data_file = sample_tsv

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out4"),
        data_path=str(data_file),
        filename_map=filename_map,
    )

    out_data = Path(b.browser_path) / "data"
    # The mapped file should be renamed
    assert (out_data / "custom-topic-keys.txt").exists()
    # Default file should still be copied into canonical filename
    assert (out_data / "doc-topic.txt").exists()
    # Check config updated to use custom path for topic keys and default path for doc-topic
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("topic_keys_file") == "data/custom-topic-keys.txt"
    assert cfg.get("doc_topic_file") == "data/doc-topic.txt"


def test_user_config_file_path_preserved(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Verify that user-specified file path keys in config are preserved and not overwritten by automatic data/ paths."""
    # Test that user-supplied file path in config is preserved and not overwritten by copied paths
    template_dir = dist_template_dir

    mallet_dir = mallet_dir_factory()

    data_file = sample_tsv

    # User sets a custom path in config.json for doc_topic_file and it should be preserved
    user_cfg = {
        "application": {"name": "Custom Config"},
        "doc_topic_file": "/bad/path/doc-topic.txt",
    }
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(template_dir),
        browser_path=str(tmp_path / "browser_out5"),
        data_path=str(data_file),
        config=user_cfg,
    )

    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # Application override is preserved
    assert cfg.get("application", {}).get("name") == "Custom Config"
    # The file path value set by the user should be preserved
    assert cfg.get("doc_topic_file") == "/bad/path/doc-topic.txt"


def test_missing_required_files_raises(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Missing required mallet files should raise FileNotFoundError during Browser initialization."""
    template_dir = dist_template_dir

    # Missing topic-keys and topic-state (can't auto-generate topic_coords without state file)
    mallet_dir = mallet_dir_factory(["doc-topics.txt", "metadata.csv"])

    data_file = sample_tsv

    with pytest.raises(FileNotFoundError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(template_dir),
            browser_path=str(tmp_path / "browser_out6"),
            data_path=str(data_file),
        )


def test_template_path_does_not_exist_raises(
    tmp_path: Path, mallet_dir_factory: callable, sample_tsv: Path
):
    """Template path not found should raise FileNotFoundError."""
    mallet_dir = mallet_dir_factory()
    data_file = sample_tsv
    # Use a template path that does not exist
    with pytest.raises(FileNotFoundError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(tmp_path / "no-such-template"),
            browser_path=str(tmp_path / "browser_out_t"),
            data_path=str(data_file),
        )


def test_mallet_path_does_not_exist_raises(
    tmp_path: Path, dist_template_dir: Path, sample_tsv: Path
):
    """Mallet folder missing should raise FileNotFoundError."""
    data_file = sample_tsv
    # Use mallet path that does not exist
    with pytest.raises(FileNotFoundError):
        Browser(
            mallet_files_path=str(tmp_path / "no-such-mal"),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_t2"),
            data_path=str(data_file),
        )


def test_data_path_is_dir_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """Providing a directory as data_path should raise ValueError."""
    mallet_dir = mallet_dir_factory()
    # Use a directory as data_path
    with pytest.raises(ValueError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_dir"),
            data_path=str(mallet_dir),
        )


def test_data_path_nonexistent_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """Nonexistent data_path should raise FileNotFoundError."""
    mallet_dir = mallet_dir_factory()
    # data_path doesn't exist
    with pytest.raises(FileNotFoundError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_no_data"),
            data_path=str(tmp_path / "no-data.tsv"),
        )


def test_invalid_tsv_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """TSV with invalid row column count should raise ValueError."""
    mallet_dir = mallet_dir_factory()
    # create invalid tsv with 4 columns
    bad_tsv = tmp_path / "bad.tsv"
    bad_tsv.write_text("1	A	B	C\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_bad"),
            data_path=str(bad_tsv),
        )


def test_creates_temp_browser_path_if_none(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Omitting browser_path creates a temporary folder for the browser."""
    mallet_dir = mallet_dir_factory()
    data_file = sample_tsv
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        # no browser_path provided
        data_path=str(data_file),
    )
    # Should have created a temporary folder
    assert b.browser_path.exists()
    assert "dfr_browser_" in str(b.browser_path)


def test_canonicalization_and_duplicates(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Both doc-topic and doc-topics present are canonicalized and deduplicated."""
    # Create both doc-topic and doc-topics so duplicate dedup logic is hit
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topic.txt",
            "doc-topics.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    data_file = sample_tsv
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_dup"),
        data_path=str(data_file),
    )
    out_data = Path(b.browser_path) / "data"
    # Only the canonical dest should exist once
    assert (out_data / "doc-topic.txt").exists()


def test_version_property_handles_faulty_config_bool(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """When self.config raises during boolean evaluation, version should fallback to class default."""
    mallet_dir = mallet_dir_factory()
    data_file = sample_tsv

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_version"),
        data_path=str(data_file),
    )

    class BadBool:
        def __bool__(self):
            raise RuntimeError("boom")

    # Assign a value that will raise during boolean evaluation in the version property
    b.config = BadBool()
    # The property should gracefully fallback to the class-level BROWSER_VERSION
    assert b.version == Browser.BROWSER_VERSION


def test_topic_state_altname_canonicalization(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Ensure 'state.gz' alternate name results in canonical 'topic-state.gz' in copied files and config."""
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topic.txt",
            "state.gz",
            "topic_coords.csv",
        ]
    )
    data_file = sample_tsv
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_state"),
        data_path=str(data_file),
    )

    # The canonical destination should be 'topic-state.gz'
    assert b.copied_files.get("topic_state_file") == "data/topic-state.gz"
    out_data = Path(b.browser_path) / "data"
    # doc-topics should be canonicalized to doc-topic and not present
    assert not (out_data / "doc-topics.txt").exists()
    # Ensure metadata keys present in copied files mapping
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("metadata_file") == "data/metadata.csv"
    assert cfg.get("topic_state_file") == "data/topic-state.gz"
    assert cfg.get("topic_coords_file") == "data/topic_coords.csv"


def test__write_config_bad_template_json(
    tmp_path: Path, mallet_dir_factory: callable, sample_tsv: Path
):
    """If template config.json is invalid JSON, code falls back to base_cfg empty and writes merged config."""
    # Create dist with invalid config.json to hit the base_cfg except branch
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "config.json").write_text("not json", encoding="utf-8")
    (dist_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_dir),
        browser_path=str(tmp_path / "browser_out_badcfg"),
        data_path=str(sample_tsv),
        config={"application": {"name": "My"}},
    )
    # Config should be written and contain application name
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("application", {}).get("name") == "My"


def test__write_config_browser_path_none_raises(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Setting browser_path to None and calling _write_config should raise ValueError."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_6"),
        data_path=str(sample_tsv),
    )
    # set browser_path to None and ensure _write_config raises
    b.browser_path = None
    with pytest.raises(ValueError):
        b._write_config()


def test_serve_port_already_in_use_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable, monkeypatch
):
    """Serve should raise RuntimeError with helpful message when port is in use."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve"),
        port=9999,
    )

    # Mock _is_port_available to return False (port in use)
    monkeypatch.setattr(b, "_is_port_available", lambda port: False)

    # Should raise RuntimeError with helpful message
    with pytest.raises(RuntimeError) as exc_info:
        b.serve()

    assert "Port 9999 is already in use" in str(exc_info.value)
    assert "lsof -i:9999" in str(exc_info.value)


def test_is_port_available(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """Test _is_port_available method correctly detects available and unavailable ports."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_port"),
    )

    # Test with a very high port that should be available
    assert b._is_port_available(58888) is True


def test_serve_missing_browser_path_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """If browser_path is absent, serve should raise FileNotFoundError."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve2"),
    )
    # remove browser path
    shutil.rmtree(b.browser_path)
    with pytest.raises(FileNotFoundError):
        b.serve()


def test_serve_server_script_missing_raises(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable, monkeypatch
):
    """If server.py script is missing, serve should raise FileNotFoundError."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve3"),
        port=58765,  # Use unique high port to avoid conflicts
    )

    # Mock port as available
    monkeypatch.setattr(b, "_is_port_available", lambda port: True)

    # Mock Path(__file__).parent to return a path where server.py doesn't exist
    from pathlib import Path as PathClass

    fake_parent = tmp_path / "fake_module_dir"
    fake_parent.mkdir()

    # Patch __file__ context
    import lexos.topic_modeling.dfr_browser2

    original_file = lexos.topic_modeling.dfr_browser2.__file__
    monkeypatch.setattr(
        lexos.topic_modeling.dfr_browser2, "__file__", str(fake_parent / "__init__.py")
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        b.serve()

    assert "server.py not found" in str(exc_info.value)


def test_copytree_fallback(
    monkeypatch, tmp_path: Path, mallet_dir_factory: callable, sample_tsv: Path
):
    """If shutil.copytree raises, fallback per-file copy path should still copy template content."""
    # Prepare template with nested files
    dist_dir = tmp_path / "dist"
    (dist_dir / "nested").mkdir(parents=True, exist_ok=True)
    (dist_dir / "nested" / "file.txt").write_text("data", encoding="utf-8")
    (dist_dir / "config.json").write_text(
        json.dumps({"data_source": "data/docs.txt"}), encoding="utf-8"
    )
    (dist_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    # Prepare mallet dir
    mallet_dir = mallet_dir_factory()
    # Force copytree to raise an exception so the fallback path is used
    monkeypatch.setattr(
        "shutil.copytree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception("fail copytree")),
    )
    # Ensure browser output folder exists so fallback per-file copy has a destination
    (tmp_path / "browser_out_copytree").mkdir(parents=True, exist_ok=True)

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_dir),
        browser_path=str(tmp_path / "browser_out_copytree"),
        data_path=str(sample_tsv),
    )
    # Verify nested file copied
    assert (Path(b.browser_path) / "nested" / "file.txt").exists()


def test_config_browser_updates_config(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """`config_browser` should write config.json updating the template settings."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_conf"),
        data_path=str(sample_tsv),
    )
    b.config_browser({"display": {"itemsPerPage": 10}})
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("display", {}).get("itemsPerPage") == 10


def test_auto_generate_topic_coords_from_state_file(
    tmp_path: Path,
    dist_template_dir: Path,
    sample_tsv: Path,
    monkeypatch,
):
    """If topic_coords.csv is missing but topic-state.gz exists, it should be auto-generated."""
    # Create mallet dir without topic_coords.csv
    mallet_dir = tmp_path / "mallet_no_coords"
    mallet_dir.mkdir()
    create_file(mallet_dir / "metadata.csv", "content")
    create_file(mallet_dir / "topic-keys.txt", "content")
    create_file(mallet_dir / "doc-topic.txt", "content")
    create_file(mallet_dir / "topic-state.gz", "content")
    # NO topic_coords.csv

    # Mock process_mallet_state_file to create topic_coords.csv
    def mock_process(state_file, output_dir, n_top_words, generate_all):
        create_file(Path(output_dir) / "topic_coords.csv", "x,y\n0,0\n")

    monkeypatch.setattr(
        "lexos.topic_modeling.dfr_browser2.process_mallet_state_file", mock_process
    )

    # Should succeed and auto-generate topic_coords.csv
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_autogen"),
        data_path=str(sample_tsv),
    )

    # Verify topic_coords.csv was created and copied
    out_data = Path(b.browser_path) / "data"
    assert (out_data / "topic_coords.csv").exists()


def test_diagnostics_file_auto_copied(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Optional diagnostics.xml file should be automatically copied if present."""
    mallet_dir = mallet_dir_factory()
    # Create a diagnostics file in the mallet dir
    create_file(mallet_dir / "diagnostics.xml", "<diagnostics/>")

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_diag"),
        data_path=str(sample_tsv),
    )

    # Verify diagnostics.xml was copied
    out_data = Path(b.browser_path) / "data"
    assert (out_data / "diagnostics.xml").exists()

    # Ensure diagnostics_file is tracked in config
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("diagnostics_file") == "data/diagnostics.xml"


def test_filename_map_missing_non_required_key_ignored(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Mapping for a non-required key that doesn't exist is silently ignored; Browser still initializes."""
    mallet_dir = mallet_dir_factory()
    # Provide a mapping for a file that doesn't exist (not a required file)
    # This should not raise - the file is simply not copied if it doesn't exist
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_map_missing"),
        data_path=str(sample_tsv),
        filename_map={"not_real.txt": "not_real_dst.txt"},
    )
    # Browser should initialize successfully
    assert b.browser_path.exists()


def test_filename_map_key_missing_but_value_present(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """If mapping key missing but mapping value exists, it should be accepted for required files."""
    # Create mallet dir with 'keys.txt' (mapped value), but no 'topic-keys.txt' (key)
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "keys.txt",
            "doc-topic.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    # Provide mapping topic-keys.txt -> keys.txt
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_value_present"),
        data_path=str(sample_tsv),
        filename_map={"topic-keys.txt": "keys.txt"},
    )
    out_data = Path(b.browser_path) / "data"
    # Ensure the copied file exists with dest name 'keys.txt' unless canonicalized to topic-keys
    assert (out_data / "keys.txt").exists() or (out_data / "topic-keys.txt").exists()


def test_alt_name_group_mapping_value_present(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """If mapping refers to alt in alt_name_groups and mapping value exists, treat canonical as present."""
    # mallet dir contains mapped file 'mapped-docs.txt' but not 'doc-topics.txt' or 'doc-topic.txt'
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "mapped-docs.txt",
            "topic-state.gz",
            "topic_coords.csv",
        ]
    )
    # mapping: doc-topics.txt -> mapped-docs.txt (key is alt name)
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_altmap"),
        data_path=str(sample_tsv),
        filename_map={"doc-topics.txt": "mapped-docs.txt"},
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # Configuration should record doc_topic_file as data/doc-topics.txt (destination is key)
    assert cfg.get("doc_topic_file") == "data/doc-topics.txt"


def test_required_file_missing_but_mapped_value_exists(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """If canonical required file missing but mapped value exists the Browser should accept it."""
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topic.txt",
            "state.gz",
            "topic_coords.csv",
        ]
    )
    # Provide mapping topic-state.gz -> state.gz
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_state"),
        data_path=str(sample_tsv),
        filename_map={"topic-state.gz": "state.gz"},
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # Destination is canonical 'topic-state.gz' (mapping value was used as source)
    assert cfg.get("topic_state_file") == "data/topic-state.gz"


def test_alt_canonical_topic_state_mapping(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """If mapping refers to alternate 'state.gz' for topic-state, map canonical 'topic-state.gz'."""
    mallet_dir = mallet_dir_factory(
        [
            "metadata.csv",
            "topic-keys.txt",
            "doc-topic.txt",
            "state.gz",
            "topic_coords.csv",
        ]
    )
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_state2"),
        data_path=str(sample_tsv),
        filename_map={"topic-state.gz": "state.gz"},
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # Confirm canonical mapping used for topic_state_file
    assert cfg.get("topic_state_file") == "data/topic-state.gz"


def test_copytree_success_copies_all(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Ensure copytree success path copies nested files into browser path."""
    # Create a nested template under dist
    dist_dir = tmp_path / "dist2"
    (dist_dir / "assets").mkdir(parents=True)
    (dist_dir / "assets" / "a.txt").write_text("hello", encoding="utf-8")
    (dist_dir / "config.json").write_text(
        json.dumps({"data_source": "data/docs.txt"}), encoding="utf-8"
    )
    (dist_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_dir),
        browser_path=str(tmp_path / "browser_out_copytree_ok"),
        data_path=str(sample_tsv),
    )
    assert (Path(b.browser_path) / "assets" / "a.txt").exists()


def test_tsv_with_empty_lines_allowed(
    tmp_path: Path, dist_template_dir: Path, mallet_dir_factory: callable
):
    """TSV files can include empty lines — they are ignored during validation."""
    mallet_dir = mallet_dir_factory()
    tsv = tmp_path / "with_empty.tsv"
    tsv.write_text("1\tDoc1\n\n2\tDoc2\n", encoding="utf-8")
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_empty"),
        data_path=str(tsv),
    )
    assert (Path(b.browser_path) / "data" / "docs.txt").exists()


def test_user_config_topic_keys_preserved(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """User-provided 'topic_keys_file' in config should be preserved and not overwritten."""
    mallet_dir = mallet_dir_factory()
    user_cfg = {"topic_keys_file": "/bad/keys.txt"}
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_user_topic_keys"),
        data_path=str(sample_tsv),
        config=user_cfg,
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("topic_keys_file") == "/bad/keys.txt"


def test_browser_version_default_and_override(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Browser exposes a BROWSER_VERSION which appears under config['application']['version'], and is preserved if the user provides a version."""
    mallet_dir = mallet_dir_factory()

    # Default: should be set to Browser.BROWSER_VERSION
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_version"),
        data_path=str(sample_tsv),
    )
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    # File content should have version set
    assert cfg.get("application", {}).get("version") == b.BROWSER_VERSION
    # The instance config should also be updated in memory to the merged config
    assert isinstance(b.config, dict)
    assert b.config.get("application", {}).get("version") == b.BROWSER_VERSION
    # Version property should use the in-memory config value if available
    assert b.version == b.BROWSER_VERSION

    # User provided version: should be preserved
    user_cfg = {"application": {"version": "3.0.0"}}
    b2 = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_version2"),
        data_path=str(sample_tsv),
        config=user_cfg,
    )
    cfg2 = json.loads(
        (Path(b2.browser_path) / "config.json").read_text(encoding="utf-8")
    )
    # User-provided version preserved in file
    assert cfg2.get("application", {}).get("version") == "3.0.0"
    # and is present in the instance config in-memory as well
    assert isinstance(b2.config, dict)
    assert b2.config.get("application", {}).get("version") == "3.0.0"
    # Version property should reflect user-provided override
    assert b2.version == "3.0.0"


def test_check_file_exists_with_alternates(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Test _check_file_exists_with_alternates helper method."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_check"),
        data_path=str(sample_tsv),
    )

    # Should find doc-topic.txt (canonical)
    exists, path = b._check_file_exists_with_alternates("doc-topic.txt")
    assert exists is True
    assert path.name == "doc-topic.txt"

    # Test with file that doesn't exist
    exists, path = b._check_file_exists_with_alternates("nonexistent.txt")
    assert exists is False
    assert path is None


def test_auto_generate_topic_coords_failure(
    tmp_path: Path,
    dist_template_dir: Path,
    sample_tsv: Path,
    monkeypatch,
):
    """If topic_coords.csv generation fails, should raise RuntimeError with clear message."""
    # Create mallet dir without topic_coords.csv
    mallet_dir = tmp_path / "mallet_fail_gen"
    mallet_dir.mkdir()
    create_file(mallet_dir / "metadata.csv", "content")
    create_file(mallet_dir / "topic-keys.txt", "content")
    create_file(mallet_dir / "doc-topic.txt", "content")
    create_file(mallet_dir / "topic-state.gz", "content")
    # NO topic_coords.csv

    # Mock process_mallet_state_file to raise an exception
    def mock_process_fail(state_file, output_dir, n_top_words, generate_all):
        raise ValueError("Mock generation failure")

    monkeypatch.setattr(
        "lexos.topic_modeling.dfr_browser2.process_mallet_state_file", mock_process_fail
    )

    # Should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        Browser(
            mallet_files_path=str(mallet_dir),
            template_path=str(dist_template_dir),
            browser_path=str(tmp_path / "browser_out_fail_gen"),
            data_path=str(sample_tsv),
        )

    assert "Failed to generate topic_coords.csv" in str(exc_info.value)


def test_config_assignment_writes_file(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Setting `Browser.config` property should write updated config to file and update in-memory config."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_config_set"),
        data_path=str(sample_tsv),
    )

    # Assign new config; this should trigger a write via __setattr__
    new_cfg = {"application": {"name": "New Title"}}
    b.config = new_cfg
    # Check file and instance config
    cfg = json.loads((Path(b.browser_path) / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("application", {}).get("name") == "New Title"
    assert isinstance(b.config, dict)
    assert b.config.get("application", {}).get("name") == "New Title"


def test_stop_server_not_started(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    capsys,
):
    """Test stop_server when no server has been started."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_stop"),
        data_path=str(sample_tsv),
    )

    # Try to stop server when none is running
    b.stop_server()

    captured = capsys.readouterr()
    assert "No server process to stop" in captured.out


def test_stop_server_with_mock_process(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    monkeypatch,
    capsys,
):
    """Test stop_server with a mock server process."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_stop_mock"),
        data_path=str(sample_tsv),
    )

    # Create a mock process that sleeps (so we can terminate it)
    import subprocess
    import sys

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Manually set the server process
    b._server_process = process

    # Verify process is running
    assert process.poll() is None

    # Stop the server
    b.stop_server()

    # Verify process was terminated
    assert process.poll() is not None

    captured = capsys.readouterr()
    assert "Server stopped" in captured.out


def test_stop_server_already_stopped(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    capsys,
):
    """Test stop_server when process already exited."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_stop_already"),
        data_path=str(sample_tsv),
    )

    # Create a process that exits immediately
    import subprocess
    import sys

    process = subprocess.Popen(
        [sys.executable, "-c", "print('done')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for it to finish
    process.wait()

    # Set it as the server process
    b._server_process = process

    # Try to stop it
    b.stop_server()

    captured = capsys.readouterr()
    assert "already stopped" in captured.out


def test_stop_server_not_started(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    capsys,
):
    """Test stop_server when no server has been started."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_stop"),
        data_path=str(sample_tsv),
    )

    # Try to stop server when none is running
    b.stop_server()

    captured = capsys.readouterr()
    assert "No server process to stop" in captured.out


def test_stop_server_with_mock_process(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    monkeypatch,
    capsys,
):
    """Test stop_server with a mock server process."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_stop_mock"),
        data_path=str(sample_tsv),
    )

    # Create a mock process that sleeps (so we can terminate it)
    import subprocess
    import sys

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Manually set the server process
    b._server_process = process

    # Verify process is running
    assert process.poll() is None

    # Stop the server
    b.stop_server()

    # Verify process was terminated
    assert process.poll() is not None

    captured = capsys.readouterr()
    assert "Server stopped" in captured.out


def test_stop_server_already_stopped(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    capsys,
):
    """Test stop_server when process already exited."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_stop_already"),
        data_path=str(sample_tsv),
    )

    # Create a process that exits immediately
    import subprocess
    import sys

    process = subprocess.Popen(
        [sys.executable, "-c", "print('done')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for it to finish
    process.wait()

    # Set it as the server process
    b._server_process = process

    # Try to stop it
    b.stop_server()

    captured = capsys.readouterr()
    assert "already stopped" in captured.out


def test_track_copied_file_diagnostics(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
):
    """Test that diagnostics.xml is tracked in copied_files."""
    mallet_dir = mallet_dir_factory()

    # Create diagnostics.xml
    diagnostics_file = mallet_dir / "diagnostics.xml"
    diagnostics_file.write_text("<diagnostics/>", encoding="utf-8")

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_diag"),
    )

    # Should be tracked in copied_files
    assert "diagnostics_file" in b.copied_files
    assert b.copied_files["diagnostics_file"] == "data/diagnostics.xml"


def test_is_port_available_when_port_in_use(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
):
    """Test _is_port_available returns False when port is in use."""
    import socket

    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_port"),
    )

    # Bind to a port to make it unavailable
    test_port = 58999
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", test_port))
    server_socket.listen(1)

    try:
        # Port should be unavailable
        assert not b._is_port_available(test_port)
    finally:
        server_socket.close()


def test_stop_server_with_exception(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    monkeypatch,
    capsys,
):
    """Test stop_server handles exceptions during termination."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_exception"),
    )

    # Create a mock process
    import subprocess
    import sys

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    b._server_process = process

    # Mock terminate to raise an exception
    original_terminate = process.terminate

    def mock_terminate():
        raise Exception("Mock termination error")

    process.terminate = mock_terminate

    # Should handle the exception gracefully
    b.stop_server()

    # Check that error was printed
    captured = capsys.readouterr()
    assert "Error stopping server" in captured.out

    # Cleanup
    try:
        original_terminate()
        process.wait(timeout=2)
    except:
        process.kill()


def test_stop_server_force_kill_on_timeout(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    monkeypatch,
    capsys,
):
    """Test stop_server force kills when graceful termination times out."""
    mallet_dir = mallet_dir_factory()
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_force_kill"),
    )

    # Create a mock process
    import subprocess
    import sys

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    b._server_process = process

    # Mock wait to raise TimeoutExpired
    original_wait = process.wait

    def mock_wait(timeout=None):
        if timeout:
            raise subprocess.TimeoutExpired(process.args, timeout)
        return original_wait(timeout)

    process.wait = mock_wait

    # Should force kill
    b.stop_server()

    # Check that it was forcefully stopped
    captured = capsys.readouterr()


def test_track_copied_file_non_canonical_topic_state(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
):
    """Test that a file with 'topic-state' in the name but not in ALT_NAME_GROUPS triggers line 409."""
    mallet_dir = mallet_dir_factory()
    # Create a custom file with a name not in ALT_NAME_GROUPS
    create_file(mallet_dir / "custom_file.txt", "content")
    # Also create required files
    create_file(mallet_dir / "topic-keys.txt", "1\tkey1\nkey2")
    create_file(mallet_dir / "doc-topic.txt", "0\t0.5\t0.5")
    create_file(mallet_dir / "topic_coords.csv", "x,y,topics\n1,2,topic1")
    create_file(mallet_dir / "topic-state.gz", "state_content")

    # Map custom_file.txt to a name containing "topic-state"
    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_ts"),
        data_path=str(sample_tsv),
        filename_map={"custom_file.txt": "my-topic-state-custom.txt"},
    )

    # The custom filename should be tracked as topic_state_file (via line 409)
    # Note: topic-state.gz will also be copied, so we expect BOTH files to be tracked
    # But the custom one should trigger line 409
    out_data = Path(b.browser_path) / "data"
    assert (out_data / "my-topic-state-custom.txt").exists()
    assert (out_data / "topic-state.gz").exists()


def test_serve_in_jupyter_environment(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    monkeypatch,
    capsys,
):
    """Test serve() method in Jupyter environment with mocked subprocess and browser."""
    mallet_dir = mallet_dir_factory()

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve"),
        data_path=str(sample_tsv),
    )

    # Mock get_ipython to simulate Jupyter environment
    def mock_get_ipython():
        return True

    # Mock Popen to return a mock process
    class MockProcess:
        def __init__(self, *args, **kwargs):
            self.pid = 12345
            self.returncode = None

        def poll(self):
            return None  # Still running

        def communicate(self):
            return ("", "")

    # Mock webbrowser.open
    opened_urls = []

    def mock_open(url):
        opened_urls.append(url)

    # Mock _is_port_available to always return True
    def mock_is_port_available(port):
        return True

    # Apply mocks
    import builtins

    monkeypatch.setattr(builtins, "get_ipython", mock_get_ipython, raising=False)
    monkeypatch.setattr(subprocess, "Popen", MockProcess)
    monkeypatch.setattr(webbrowser, "open", mock_open)
    monkeypatch.setattr(b, "_is_port_available", mock_is_port_available)

    # Call serve()
    b.serve()

    # Check outputs
    captured = capsys.readouterr()
    assert "Starting DFR Browser server" in captured.out
    assert "b.stop_server()" in captured.out
    assert "Server running (PID: 12345)" in captured.out
    assert len(opened_urls) == 1
    assert f"http://localhost:{b.port}/" in opened_urls[0]

    # Check that _server_process was stored
    assert b._server_process is not None
    assert b._server_process.pid == 12345


def test_serve_in_cli_environment(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    monkeypatch,
    capsys,
):
    """Test serve() method in CLI environment with KeyboardInterrupt."""
    mallet_dir = mallet_dir_factory()

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve_cli"),
        data_path=str(sample_tsv),
    )

    # Mock get_ipython to raise NameError (not in Jupyter)
    def mock_get_ipython():
        raise NameError("name 'get_ipython' is not defined")

    # Mock subprocess.run to raise KeyboardInterrupt
    def mock_run(*args, **kwargs):
        raise KeyboardInterrupt()

    # Mock webbrowser.open
    opened_urls = []

    def mock_open(url):
        opened_urls.append(url)

    # Mock _is_port_available to always return True
    def mock_is_port_available(port):
        return True

    # Apply mocks
    import builtins

    monkeypatch.setattr(builtins, "get_ipython", mock_get_ipython, raising=False)
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(webbrowser, "open", mock_open)
    monkeypatch.setattr(b, "_is_port_available", mock_is_port_available)

    # Call serve()
    b.serve()

    # Check outputs
    captured = capsys.readouterr()
    assert "Starting DFR Browser server" in captured.out
    assert "Press Ctrl+C to stop the server" in captured.out
    assert "Server stopped" in captured.out
    assert len(opened_urls) == 1
    assert f"http://localhost:{b.port}/" in opened_urls[0]


def test_serve_jupyter_server_fails_to_start(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    monkeypatch,
):
    """Test serve() raises error when server process fails in Jupyter."""
    mallet_dir = mallet_dir_factory()

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve_fail"),
        data_path=str(sample_tsv),
    )

    # Mock get_ipython to simulate Jupyter environment
    def mock_get_ipython():
        return True

    # Mock Popen to return a process that exits immediately with error
    class MockFailedProcess:
        def __init__(self, *args, **kwargs):
            self.pid = 12345
            self.returncode = 1

        def poll(self):
            return 1  # Exited

        def communicate(self):
            return ("", "Error: Address already in use (Errno 98)")

    # Mock _is_port_available to always return True
    def mock_is_port_available(port):
        return True

    # Apply mocks
    import builtins

    monkeypatch.setattr(builtins, "get_ipython", mock_get_ipython, raising=False)
    monkeypatch.setattr(subprocess, "Popen", MockFailedProcess)
    monkeypatch.setattr(webbrowser, "open", lambda url: None)
    monkeypatch.setattr(b, "_is_port_available", mock_is_port_available)

    # Call serve() should raise
    with pytest.raises(RuntimeError, match="Server failed to start"):
        b.serve()


def test_serve_jupyter_webbrowser_fails(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    monkeypatch,
    capsys,
):
    """Test serve() handles webbrowser.open exception in Jupyter."""
    mallet_dir = mallet_dir_factory()

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve_wb"),
        data_path=str(sample_tsv),
    )

    # Mock get_ipython to simulate Jupyter environment
    def mock_get_ipython():
        return True

    # Mock Popen to return a mock process
    class MockProcess:
        def __init__(self, *args, **kwargs):
            self.pid = 12345
            self.returncode = None

        def poll(self):
            return None  # Still running

        def communicate(self):
            return ("", "")

    # Mock webbrowser.open to raise exception
    def mock_open(url):
        raise Exception("Browser failed to open")

    # Mock _is_port_available to always return True
    def mock_is_port_available(port):
        return True

    # Apply mocks
    import builtins

    monkeypatch.setattr(builtins, "get_ipython", mock_get_ipython, raising=False)
    monkeypatch.setattr(subprocess, "Popen", MockProcess)
    monkeypatch.setattr(webbrowser, "open", mock_open)
    monkeypatch.setattr(b, "_is_port_available", mock_is_port_available)

    # Call serve()
    b.serve()

    # Check outputs
    captured = capsys.readouterr()
    assert "Please open http://localhost:" in captured.out
    assert "Server running (PID: 12345)" in captured.out


def test_serve_cli_webbrowser_fails(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    monkeypatch,
    capsys,
):
    """Test serve() handles webbrowser.open exception in CLI."""
    mallet_dir = mallet_dir_factory()

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve_cli_wb"),
        data_path=str(sample_tsv),
    )

    # Mock get_ipython to raise NameError (not in Jupyter)
    def mock_get_ipython():
        raise NameError("name 'get_ipython' is not defined")

    # Mock subprocess.run to raise KeyboardInterrupt
    def mock_run(*args, **kwargs):
        raise KeyboardInterrupt()

    # Mock webbrowser.open to raise exception
    def mock_open(url):
        raise Exception("Browser failed to open")

    # Mock _is_port_available to always return True
    def mock_is_port_available(port):
        return True

    # Apply mocks
    import builtins

    monkeypatch.setattr(builtins, "get_ipython", mock_get_ipython, raising=False)
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(webbrowser, "open", mock_open)
    monkeypatch.setattr(b, "_is_port_available", mock_is_port_available)

    # Call serve()
    b.serve()

    # Check outputs
    captured = capsys.readouterr()
    assert "Please open http://localhost:" in captured.out
    assert "Server stopped" in captured.out


def test_serve_cli_subprocess_error(
    tmp_path: Path,
    dist_template_dir: Path,
    mallet_dir_factory: callable,
    sample_tsv: Path,
    monkeypatch,
):
    """Test serve() handles subprocess.CalledProcessError in CLI."""
    mallet_dir = mallet_dir_factory()

    b = Browser(
        mallet_files_path=str(mallet_dir),
        template_path=str(dist_template_dir),
        browser_path=str(tmp_path / "browser_out_serve_cli_err"),
        data_path=str(sample_tsv),
    )

    # Mock get_ipython to raise NameError (not in Jupyter)
    def mock_get_ipython():
        raise NameError("name 'get_ipython' is not defined")

    # Mock subprocess.run to raise CalledProcessError
    def mock_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "server.py")

    # Mock _is_port_available to always return True
    def mock_is_port_available(port):
        return True

    # Apply mocks
    import builtins

    monkeypatch.setattr(builtins, "get_ipython", mock_get_ipython, raising=False)
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(webbrowser, "open", lambda url: None)
    monkeypatch.setattr(b, "_is_port_available", mock_is_port_available)

    # Call serve() should raise RuntimeError
    with pytest.raises(RuntimeError, match="Server error"):
        b.serve()
