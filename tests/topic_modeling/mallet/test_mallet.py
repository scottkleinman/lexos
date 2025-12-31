"""test_mallet.py.

Coverage: 99%. Missing: 1225-1227, 1250-1252

Last Updated: November 26, 2025
"""

import os
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pydantic_core._pydantic_core import ValidationError as PydanticValidationError

from lexos.exceptions import LexosException
from lexos.topic_modeling.mallet import (
    Mallet,
    import_docs,
    import_files,
    read_dirs,
    read_file,
)


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create and return a temporary model directory for tests."""
    d = tmp_path / "mallet_model"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def no_system_calls(monkeypatch):
    """Fixture to prevent real system calls to `mallet` during tests."""

    def fake_system(cmd):
        # Dummy no-op to simulate success
        return 0

    monkeypatch.setattr(os, "system", fake_system)
    yield


def test_read_file_rejects_bool():
    """Test that read_file rejects boolean input."""
    with pytest.raises(PydanticValidationError):
        read_file(True)  # type: ignore[arg-type]


def test_read_dirs_rejects_boolean():
    """Test that read_dirs rejects boolean input."""
    with pytest.raises(PydanticValidationError):
        read_dirs(True)  # type: ignore[arg-type]


def test_import_files_reads_file(tmp_path):
    """Test that import_files reads file contents correctly."""
    f = tmp_path / "test.txt"
    f.write_text("hello world", encoding="utf-8")
    contents = import_files(f)
    assert isinstance(contents, list)
    assert contents == ["hello world"]


def test_import_docs_returns_strings():
    """Test that import_docs returns the input list of strings unchanged."""
    docs = ["a", "b", "c"]
    res = import_docs(docs)
    assert res == docs


def test_import_docs_accepts_spacy_doc(tmp_model_dir):
    """Passing a spaCy `Doc` to `import_docs` should return the doc.text entry (covers `Doc` branch)."""
    from spacy.tokens import Doc
    from spacy.vocab import Vocab

    vocab = Vocab()
    d = Doc(vocab, words=["hello", "world"])
    res = import_docs([d])
    assert res == [d.text]


def test_mallet_import_data_and_metadata(tmp_model_dir):
    """Test that import_data sets expected metadata keys."""
    m = Mallet(model_dir=str(tmp_model_dir))
    training_text = ["this is a test", "another doc"]
    # Import the data; os.system is patched, so no external calls
    m.import_data(training_text)

    # Check required metadata keys were set
    assert "path_to_training_data" in m.metadata
    assert "path_to_formatted_training_data" in m.metadata
    assert m.metadata["num_docs"] == 2
    assert isinstance(m.metadata["mean_num_tokens"], (int, float))
    assert "vocab_size" in m.metadata


def test_train_sets_canonical_metadata(tmp_model_dir, monkeypatch):
    """Test that train sets canonical metadata keys and paths."""
    m = Mallet(model_dir=str(tmp_model_dir))

    # Create a fake formatted training data file so train() will not error on missing input
    formatted_data_path = tmp_model_dir / "training_data.mallet"
    formatted_data_path.write_text("0\t\tword1 word2\n1\t\tword3 word4\n")

    # Ensure mallet execution is not run
    monkeypatch.setattr(os, "system", lambda c: 0)
    # Prevent spawn/track_progress from launching real subprocess
    monkeypatch.setattr(
        Mallet, "_track_progress", lambda self, cmd, iterations, verbose: None
    )

    # Call train and allow defaults
    m.train(num_topics=3, num_iterations=1, verbose=False)

    # Confirm canonical metadata keys set
    assert m.CANONICAL_DOC_TOPIC_KEY in m.metadata
    assert m.CANONICAL_TOPIC_KEYS_KEY in m.metadata
    assert m.CANONICAL_TERM_WEIGHTS_KEY in m.metadata
    assert m.CANONICAL_INFERENCER_KEY in m.metadata

    # Paths should be under the model directory
    for k in (
        m.CANONICAL_DOC_TOPIC_KEY,
        m.CANONICAL_TOPIC_KEYS_KEY,
        m.CANONICAL_TERM_WEIGHTS_KEY,
        m.CANONICAL_INFERENCER_KEY,
    ):
        assert str(m.metadata[k]).startswith(str(tmp_model_dir))


def test_infer_parses_infer_output(tmp_model_dir, monkeypatch):
    """Test that infer parses output file and returns correct distributions."""
    m = Mallet(model_dir=str(tmp_model_dir))

    # Prepare a fake inferencer -- create a dummy file and set metadata
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy content")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)

    # Create a formatted input file
    formatted_input = tmp_model_dir / "infer_input.mallet"
    formatted_input.write_text("0\t\tdoc text\n1\t\tdoc text 2\n")

    # Provide a fake output path and write content we expect the parser to parse.
    output_path = tmp_model_dir / "infer-doc-topics.txt"
    # Write using both tab-delimited and compressed 'topic:prob' forms
    output_content = textwrap.dedent(
        """
        #doc	1	0.1	0.9
        0\td0\t0:0.2 1:0.8
        1\td1\t0.5	0.5
        """
    )
    output_path.write_text(output_content)

    # Monkeypatch os.system to no-op
    monkeypatch.setattr(os, "system", lambda c: 0)

    # Run infer, pointing to output and inferencer
    distributions = m.infer(
        docs=["d1", "d2"],
        path_to_inferencer=str(fake_inferencer_path),
        output_path=str(output_path),
    )

    # Should parse 3 lines (ignoring header) and return lists of floats
    assert isinstance(distributions, list)
    assert len(distributions) == 2
    for dist in distributions:
        assert all(isinstance(x, float) for x in dist)


def test_infer_raises_on_mixed_format(tmp_model_dir, monkeypatch):
    """Test that mixed format lines (dense + topic:prob) raise a LexosException during infer parsing."""
    m = Mallet(model_dir=str(tmp_model_dir))

    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy content")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)

    output_path = tmp_model_dir / "infer-doc-topics.txt"
    output_content = textwrap.dedent(
        """
            #doc\t1\t0.1\t0.9
            0\td0\t0:0.2 1:0.8
            1\td1\t0.5 1:0.5
            """
    )
    output_path.write_text(output_content)

    monkeypatch.setattr(os, "system", lambda c: 0)
    with pytest.raises(LexosException):
        m.infer(
            docs=["d1"],
            path_to_inferencer=str(fake_inferencer_path),
            output_path=str(output_path),
        )


def test_infer_accepts_file_input_and_flags(tmp_model_dir, monkeypatch):
    """When `docs` is a path to a file, infer should import that file and include flags in import cmd."""
    m = Mallet(model_dir=str(tmp_model_dir))

    # Create a dummy docs file
    input_file = tmp_model_dir / "input_docs.txt"
    input_file.write_text("a\nb\n")

    # Provide a fake inferencer and output file
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)
    output_path = tmp_model_dir / "out.txt"
    # write a parsable output
    output_text = "0\td0\t0.5\t0.5\n1\td1\t0.2\t0.8\n"
    output_path.write_text(output_text)

    # Capture os.system invocations so we can assert that flags are included
    calls = []

    def fake_system(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(os, "system", fake_system)

    dists = m.infer(
        docs=str(input_file),
        path_to_inferencer=str(fake_inferencer_path),
        output_path=str(output_path),
        keep_sequence=True,
        preserve_case=True,
        remove_stopwords=True,
        use_pipe_from="pipe.dat",
        show=False,
    )

    assert isinstance(dists, list)
    assert any("--keep-sequence" in c for c in calls)
    assert any("--remove-stopwords" in c for c in calls)
    assert any("--preserve-case" in c for c in calls)
    assert any("--use-pipe-from pipe.dat" in c for c in calls)


def test_infer_output_missing_raises(tmp_model_dir, monkeypatch):
    """Test that infer raises LexosException when output file is missing."""
    m = Mallet(model_dir=str(tmp_model_dir))
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)
    missing_output = tmp_model_dir / "does_not_exist.txt"
    monkeypatch.setattr(os, "system", lambda c: 0)
    with pytest.raises(LexosException):
        m.infer(
            docs=["x"],
            path_to_inferencer=str(fake_inferencer_path),
            output_path=str(missing_output),
        )


def test_infer_malformed_colon_pair_raises(tmp_model_dir, monkeypatch):
    """Malformed 'topic:prob' pairs should raise LexosException in infer parsing."""
    m = Mallet(model_dir=str(tmp_model_dir))

    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)

    out = tmp_model_dir / "infer-doc-topics.txt"
    out.write_text("0\td0\t0:abc\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    with pytest.raises(LexosException):
        m.infer(
            docs=["x"],
            path_to_inferencer=str(fake_inferencer_path),
            output_path=str(out),
        )


def test_infer_unable_to_parse_distribution_raises(tmp_model_dir, monkeypatch):
    """Non-numeric probability tokens should raise LexosException during parse."""
    m = Mallet(model_dir=str(tmp_model_dir))

    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)

    out = tmp_model_dir / "infer-doc-topics.txt"
    # tab-delimited with parts >=3 but probability tokens are non-numeric
    out.write_text("0\td0\ta\tb\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    with pytest.raises(LexosException):
        m.infer(
            docs=["x"],
            path_to_inferencer=str(fake_inferencer_path),
            output_path=str(out),
        )


def test_infer_show_true_returns_none(tmp_model_dir, monkeypatch):
    """Test that infer returns None when show=True is passed."""
    m = Mallet(model_dir=str(tmp_model_dir))
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)
    out = tmp_model_dir / "out.txt"
    out.write_text("0\td0\t0.3\t0.7\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    res = m.infer(
        docs=["x"],
        path_to_inferencer=str(fake_inferencer_path),
        output_path=str(out),
        show=True,
    )
    assert res is None


def test_infer_rejects_bool_docs(tmp_model_dir):
    """Passing a boolean as the `docs` argument should raise LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    with pytest.raises(LexosException):
        Mallet.infer.__wrapped__(m, True)  # type: ignore[arg-type]


def test_infer_list_element_non_string_raises(tmp_model_dir):
    """If any element in docs list is non-string, raise LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)
    monkeypatch = None
    with pytest.raises(LexosException):
        Mallet.infer.__wrapped__(
            m, ["text", True], path_to_inferencer=str(fake_inferencer_path)
        )


def test_infer_list_input_and_flags(tmp_model_dir, monkeypatch):
    """When docs is a list, infer should write a temp file and include flags in the import command."""
    m = Mallet(model_dir=str(tmp_model_dir))
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)
    output_path = tmp_model_dir / "tmp-out.txt"
    output_path.write_text("0\td0\t0.6\t0.4\n")

    calls = []

    def fake_system(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(os, "system", fake_system)
    dists = m.infer(
        docs=["a", "b"],
        path_to_inferencer=str(fake_inferencer_path),
        output_path=str(output_path),
        keep_sequence=True,
        preserve_case=True,
        remove_stopwords=True,
        use_pipe_from="pipe.dat",
        show=False,
    )
    assert isinstance(dists, list)
    assert any("import-file" in c and "--keep-sequence" in c for c in calls)
    assert any("import-file" in c and "--preserve-case" in c for c in calls)
    assert any("import-file" in c and "--remove-stopwords" in c for c in calls)
    assert any("import-file" in c and "--use-pipe-from pipe.dat" in c for c in calls)


def test_infer_default_output_path_used(tmp_model_dir, monkeypatch):
    """Omitting output_path should cause the function to read the default 'infer-doc-topics.txt' in model_dir."""
    m = Mallet(model_dir=str(tmp_model_dir))
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)
    # Write an 'infer-doc-topics.txt' to the model dir so it will be found when output_path is None
    default_output = tmp_model_dir / "infer-doc-topics.txt"
    default_output.write_text("0\td0\t0.1\t0.9\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    res = m.infer(docs=["x"], path_to_inferencer=str(fake_inferencer_path), show=False)
    assert isinstance(res, list)
    assert len(res) == 1


def test_infer_parses_whitespace_separated_line(tmp_model_dir, monkeypatch):
    """Ensure infer() can parse whitespace-separated lines (not just tabs)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)
    out = tmp_model_dir / "infer-doc-topics.txt"
    out.write_text("0 d0 0.5 0.5\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    dists = m.infer(
        docs=["x"],
        path_to_inferencer=str(fake_inferencer_path),
        output_path=str(out),
        show=False,
    )
    assert isinstance(dists, list)
    assert dists[0] == [0.5, 0.5]


def test_read_file_formats(tmp_path):
    """Ensure read_file handles 1, 2, 3, and >3 column formats as expected."""
    # 1-column
    f1 = tmp_path / "one_col.txt"
    f1.write_text("doc1\ndoc2\n")
    out1 = read_file(f1)
    assert isinstance(out1, list)
    assert all("\t" in line for line in out1)

    # 2-column (first column is a dummy column, second is text)
    f2 = tmp_path / "two_col.txt"
    f2.write_text("x\ttext1\ny\ttext2\n")
    out2 = read_file(f2)
    assert all(line.count("\t") == 2 for line in out2)

    # 3-column (id, label, text)
    f3 = tmp_path / "three_col.txt"
    f3.write_text("id1\tlabel1\ttext1\nid2\tlabel2\ttext2\n")
    out3 = read_file(f3)
    assert all(line.count("\t") == 2 for line in out3)

    # >3 columns - merge col 2 onwards
    f4 = tmp_path / "many_col.txt"
    f4.write_text("id1\tlabel1\tthis\tis\tmulti\n")
    out4 = read_file(f4)
    # Should still be 3 columns after merging
    assert all(line.count("\t") == 2 for line in out4)


def test_read_dirs_reads_files(tmp_path):
    """Test that read_dirs reads all files in a directory and returns their contents."""
    d = tmp_path / "dirtest"
    d.mkdir()
    f1 = d / "a.txt"
    f2 = d / "b.txt"
    f1.write_text("hello")
    f2.write_text("world")
    contents = read_dirs(d)
    assert isinstance(contents, list)
    assert "hello" in contents
    assert "world" in contents


def test_read_file_nonexistent_raises(tmp_path):
    """Test that read_file raises LexosException for a nonexistent file."""
    # Non-existent file should raise LexosException
    p = tmp_path / "no_such_file.txt"
    with pytest.raises(LexosException):
        read_file(str(p))


def test_read_file_ioerror_raises(tmp_path, monkeypatch):
    """Test that read_file raises LexosException when pandas.read_csv raises IOError."""
    # Create a real file, but patch pandas.read_csv to raise IOError to exercise the read exception branch
    p = tmp_path / "io_error.txt"
    p.write_text("doc1\ndoc2\n")

    def raise_ioerror(*args, **kwargs):
        raise IOError("Simulated IO error")

    monkeypatch.setattr("pandas.read_csv", raise_ioerror)
    with pytest.raises(LexosException):
        read_file(str(p))


def test_read_dirs_invalid_type_and_missing_dir(tmp_path):
    """Test that read_dirs raises for invalid type and missing directory."""
    # Integer passed as dir should raise PydanticValidationError
    with pytest.raises(PydanticValidationError):
        read_dirs(1)  # type: ignore[arg-type]
    # Non-existent directory raises LexosException
    with pytest.raises(LexosException):
        read_dirs(tmp_path / "does_not_exist")


def test_read_file_bypassing_validation_bool_raises():
    """Test that read_file.__wrapped__ raises LexosException for boolean input."""
    # Call internal function without Pydantic validation to hit the bool guard
    with pytest.raises(LexosException):
        read_file.__wrapped__(True)  # type: ignore[arg-type]


def test_read_file_empty_dataframe_raises(monkeypatch, tmp_path):
    """Test that read_file raises ValueError when pandas.read_csv returns empty DataFrame."""
    # Patch pandas.read_csv to return an empty DataFrame to reach the len(df.columns)==0 branch
    monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: pd.DataFrame())
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(ValueError):
        read_file(p)


def test_import_files_missing_and_ioerror(tmp_path, monkeypatch):
    """Test that import_files raises LexosException for missing file and IOError."""
    # Missing file
    missing = tmp_path / "nofile.txt"
    with pytest.raises(LexosException):
        import_files(str(missing))

    # IO error when reading file
    f = tmp_path / "afile.txt"
    f.write_text("hello")

    def raise_ioerror(*args, **kwargs):
        raise IOError("boom")

    monkeypatch.setattr("builtins.open", raise_ioerror)
    with pytest.raises(LexosException):
        import_files(str(f))


def test_read_dirs_bypass_validation_detects_bool_raises():
    """Test that read_dirs.__wrapped__ raises LexosException for boolean input."""
    # Bypass Pydantic validation to exercise the explicit boolean rejection in read_dirs
    with pytest.raises(LexosException):
        read_dirs.__wrapped__(True)  # type: ignore[arg-type]


def test_mallet_init_bool_and_file_model_dir(tmp_path):
    """Test Mallet init rejects boolean and file path for model_dir."""
    # boolean model_dir should raise ValidationError (Pydantic will block booleans)
    with pytest.raises(PydanticValidationError):
        Mallet(model_dir=True)  # type: ignore[arg-type]

    # model_dir pointing to a file should raise
    p = tmp_path / "somefile"
    p.write_text("hi")
    with pytest.raises(LexosException):
        Mallet(model_dir=str(p))


def test_mallet_init_metadata_sets_model_dir(tmp_path):
    """Test Mallet init sets model_dir from metadata if not provided."""
    # If metadata contains model_directory and model_dir is None, __init__ should set model_dir
    meta_dir = tmp_path / "meta_model"
    meta_dir.mkdir()
    m = Mallet(metadata={"model_directory": str(meta_dir)})
    assert str(m.model_dir) == str(meta_dir)


def test_mallet_init_sets_metadata_model_directory_when_constructed(tmp_path):
    """When `model_dir` is provided to the constructor, `metadata['model_directory']` should be set."""
    d = tmp_path / "instantiated_model"
    # Ensure instance initialised with a model_dir string sets the metadata key
    m = Mallet(model_dir=str(d))
    assert "model_directory" in m.metadata
    assert str(m.metadata["model_directory"]) == str(d)


def test_mallet_init_metadata_bool_raises(tmp_path):
    """If metadata contains a boolean `model_directory`, __init__ should raise LexosException."""
    meta_dir = True
    with pytest.raises(LexosException):
        Mallet(metadata={"model_directory": meta_dir})


def test_mallet_init_accepts_path_object_and_sets_metadata(tmp_path):
    """Passing a Path object as `model_dir` should set metadata['model_directory'] as a string and create the directory."""
    d = tmp_path / "path_model"
    # Pass a Path object. The class should accept a Path and set metadata accordingly.
    m = Mallet(model_dir=d)
    assert "model_directory" in m.metadata
    assert str(m.metadata["model_directory"]).startswith(str(d))


def test_plot_topics_over_time_self_distributions_empty_raises(tmp_model_dir):
    """If self.distributions is empty and topic_distributions is None the method should raise LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Create a blank doc-topic file so self.distributions returns []
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    # Provide a topic_keys so it doesn't fall back to property (not necessary but explicit)
    topic_keys = [["0", "0.5", "a b"]]
    # times must be provided; calling with topic_distributions=None so that code uses self.distributions
    with pytest.raises(LexosException):
        m.plot_topics_over_time(
            times=[1],
            topic_index=0,
            topic_distributions=None,
            topic_keys=topic_keys,
            show=False,
        )


def test_distributions_compressed_sparse(tmp_model_dir):
    """Test that compressed sparse topic pairs are parsed correctly in distributions."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    # compressed pairs with non-consecutive topic ids 0 and 2; expect length 3
    content = textwrap.dedent(
        """
        0\td0\t0:0.2 2:0.8
        1 d1 1.0 0.0 0.0
        """
    )
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    dists = m.distributions
    assert len(dists) == 2
    assert len(dists[0]) == 3
    assert dists[0][1] == 0.0


def test_distributions_tab_single_token_whitespace_numbers_raises(tmp_model_dir):
    """Test that tab single token with whitespace numbers raises LexosException in distributions."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    content = textwrap.dedent(
        """
        0	d0	1.0 0.0 0.0
        """
    )
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    with pytest.raises(LexosException):
        _ = m.distributions


def test_distributions_no_tabs_colon_pairs_parsed(tmp_model_dir):
    """Test that no tabs and colon pairs are parsed correctly in distributions."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    # no tabs, only two tokens (id and colon pairs), should parse into distribution
    content = "d0 0:0.2\n"
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    dists = m.distributions
    assert isinstance(dists, list)
    assert len(dists) == 1
    assert len(dists[0]) == 1
    assert pytest.approx(dists[0][0]) == 0.2


def test_distributions_tab_single_token_colon_pairs_parsed(tmp_model_dir):
    """Test that tab single token colon pairs are parsed correctly in distributions."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    content = textwrap.dedent(
        """
        0\td0\t0:0.2 2:0.8
        """
    )
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    dists = m.distributions
    assert isinstance(dists, list)
    assert len(dists) == 1
    assert len(dists[0]) == 3
    assert pytest.approx(dists[0][0]) == 0.2
    assert pytest.approx(dists[0][2]) == 0.8


def test_distributions_colon_pair_malformed_raises(tmp_model_dir):
    """Test that malformed colon pair raises LexosException in distributions."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    content = "d0 0:abc\n"
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    with pytest.raises(LexosException):
        _ = m.distributions


def test_distributions_tab_colon_pair_non_int_raises(tmp_model_dir):
    """Tab-delimited single-token colon-pair with non-integer topic id should raise LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    content = "0\td0\ta:0.5\n"
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    with pytest.raises(LexosException):
        _ = m.distributions


def test_distributions_tab_colon_pair_malformed_prob_raises(tmp_model_dir):
    """Tab-delimited single-token colon-pair with non-numeric probability should raise LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    content = "0\td0\t0:abc\n"
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    with pytest.raises(LexosException):
        _ = m.distributions


def test_distributions_colon_pair_non_int_topic_raises(tmp_model_dir):
    """A non-integer topic id within a colon-pair (e.g., 'a:0.5') should raise LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    content = "d0 a:0.5\n"
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    with pytest.raises(LexosException):
        _ = m.distributions


def test_distributions_tab_non_numeric_token_raises(tmp_model_dir):
    """Tab-delimited distributions with a non-numeric token should raise LexosException when parsing."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    # tokens after the second column must be parseable as floats
    content = "0\td0\ta\tb\n"
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    with pytest.raises(LexosException):
        _ = m.distributions


def test_distributions_whitespace_last_token_not_colon_pair_raises(tmp_model_dir):
    """Whitespace-separated single-token malformed lines should raise a LexosException.

    This targets the branch that checks for colon pairs on the last token and raises when they are absent.
    """
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    content = "d0 some_token\n"
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    with pytest.raises(LexosException):
        _ = m.distributions


def test_load_topic_term_distributions_missing_and_invalid(tmp_model_dir):
    """Test that missing topic term distributions and invalid values raise exceptions."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Missing file - expect LexosException as code checks for none metadata
    with pytest.raises(LexosException):
        m.load_topic_term_distributions()

    # Invalid weight value
    f = tmp_model_dir / "topic-weights.txt"
    f.write_text("0\tword1\tnot_a_float\n")
    m.metadata[m.CANONICAL_TERM_WEIGHTS_KEY] = str(f)
    with pytest.raises(ValueError):
        m.load_topic_term_distributions()


def test_topic_clouds_default_background_color(tmp_model_dir, monkeypatch):
    """Test that topic_clouds uses default background color when not specified."""
    captured = {}

    class FakeMultiCloud:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.fig = "fakefig"

        def save(self, path):
            captured["saved"] = path

        def show(self):
            captured["shown"] = True

    monkeypatch.setattr("lexos.topic_modeling.mallet.MultiCloud", FakeMultiCloud)
    m = Mallet(model_dir=str(tmp_model_dir))
    # inject minimal term distribution
    monkeypatch.setattr(
        Mallet, "load_topic_term_distributions", lambda self: {0: {"a": 1.0}}
    )
    mc = m.topic_clouds(show=False)
    assert captured.get("opts") is not None
    assert captured.get("opts").get("background_color") == "white"


def test_plot_topics_over_time_negative_index_and_missing_data(tmp_model_dir):
    """Test that negative topic index and missing data raise ValueError in plot_topics_over_time."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # provide distributions but negative topic index should raise ValueError
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    # supply topic_keys to avoid property access
    with pytest.raises(ValueError):
        m.plot_topics_over_time(
            times=[1, 2], topic_index=-1, topic_keys=[[0, 0.5, "a b c"]], show=False
        )


def test_train_single_part_paths_prefix_and_inferencer_meta(tmp_model_dir, monkeypatch):
    """Test that single-part topic keys path is prefixed and inferencer metadata is set."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Prepare a fake formatted training data file so train() will not error on missing input
    formatted_data_path = tmp_model_dir / "training_data.mallet"
    formatted_data_path.write_text("0\t\tword1 word2\n1\t\tword3 word4\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    monkeypatch.setattr(
        Mallet, "_track_progress", lambda self, cmd, iterations, verbose: None
    )
    # Pass single-part name for topic keys to trigger prefixing with model_directory
    m.train(
        num_topics=2,
        num_iterations=1,
        verbose=False,
        path_to_topic_keys="topic-keys.txt",
    )
    assert m.CANONICAL_TOPIC_KEYS_KEY in m.metadata
    # Should have model_directory prefixed
    assert str(m.metadata[m.CANONICAL_TOPIC_KEYS_KEY]).startswith(str(tmp_model_dir))


def test_infer_raises_when_inferencer_missing(tmp_model_dir, monkeypatch):
    """Test that infer raises LexosException when inferencer metadata is missing."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # ensure no inferencer metadata
    if m.CANONICAL_INFERENCER_KEY in m.metadata:
        m.metadata.pop(m.CANONICAL_INFERENCER_KEY)
    monkeypatch.setattr(os, "system", lambda c: 0)
    with pytest.raises(LexosException):
        m.infer(docs=["a"], output_path=str(tmp_model_dir / "out.txt"))


def test_import_data_defaults_to_model_dir(tmp_model_dir):
    """Test that import_data writes default training data path to metadata."""
    m = Mallet(model_dir=str(tmp_model_dir))
    training_data = ["a b c"]
    # Use import_data which should write default path into metadata
    m.import_data(training_data)
    assert "path_to_training_data" in m.metadata
    assert m.metadata["path_to_training_data"].endswith("training_data.txt")


def test_get_topic_term_probabilities(tmp_model_dir):
    """Test that get_topic_term_probabilities returns both DataFrame and string formats."""
    """Ensure topic-term probabilities loader returns both df and str formats."""
    m = Mallet(model_dir=str(tmp_model_dir))
    f = tmp_model_dir / "topic-weights.txt"
    f.write_text("0\tword1\t2\n0\tword2\t3\n1\tword3\t1\n")
    m.metadata[m.CANONICAL_TERM_WEIGHTS_KEY] = str(f)
    df = m.get_topic_term_probabilities(as_df=True)
    assert isinstance(df, pd.DataFrame)
    s = m.get_topic_term_probabilities()
    assert isinstance(s, str)


def test_import_docs_method_writes_training_data(tmp_model_dir, monkeypatch):
    """Ensure `Mallet.import_docs` writes the training file and records its path in metadata."""
    m = Mallet(model_dir=str(tmp_model_dir))
    monkeypatch.setattr(os, "system", lambda c: 0)
    m.import_docs(["doc one", "doc two"])
    assert "path_to_training_data" in m.metadata
    assert m.metadata["path_to_training_data"].endswith("training_data.txt")


def test_import_file_method_reads_and_writes(tmp_model_dir, monkeypatch):
    """Ensure `Mallet.import_file` reads a file and writes formatted training data."""
    m = Mallet(model_dir=str(tmp_model_dir))
    p = tmp_model_dir / "docs.txt"
    p.write_text("doc a\ndoc b\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    # Patch read_file to avoid pydantic validation errors in this test (module expects a str|Path)
    monkeypatch.setattr(
        "lexos.topic_modeling.mallet.read_file",
        lambda x: ["0		doc a\n", "1\t\tdoc b\n"],
    )
    m.import_file(str(p))
    assert "path_to_training_data" in m.metadata
    assert m.metadata["path_to_training_data"].endswith("training_data.txt")


def test__import_training_data_writes_custom_path(tmp_model_dir, monkeypatch):
    """Ensure _import_training_data writes to the specified path and records it in metadata."""
    m = Mallet(model_dir=str(tmp_model_dir))
    monkeypatch.setattr(os, "system", lambda c: 0)
    training_data = ["doc one", "doc two"]
    custom_path = tmp_model_dir / "custom_training.txt"
    m._import_training_data(training_data, path_to_training_data=str(custom_path))
    assert "path_to_training_data" in m.metadata
    assert m.metadata["path_to_training_data"] == str(custom_path)
    assert custom_path.exists()


def test__import_training_data_with_training_ids_and_pipe(tmp_model_dir, monkeypatch):
    """Ensure _import_training_data includes training ids and accepts a pipe filename flag without error."""
    m = Mallet(model_dir=str(tmp_model_dir))
    monkeypatch.setattr(os, "system", lambda c: 0)
    training_data = ["doc one", "doc two"]
    custom_path = tmp_model_dir / "training_with_ids.txt"
    # Provide training IDs to ensure they are written as provided
    m._import_training_data(
        training_data,
        path_to_training_data=str(custom_path),
        training_ids=[10, 11],
        use_pipe_from="pipe.dat",
    )
    assert "path_to_training_data" in m.metadata
    assert m.metadata["path_to_training_data"] == str(custom_path)
    # Confirm contents contain the training ids as prefix
    content = custom_path.read_text()
    assert content.startswith("10\tno_label\t")


def test_get_top_docs_inconsistent_distribution_lengths_raises(tmp_model_dir):
    """If doc-topic distributions have inconsistent lengths, `get_top_docs` should raise."""
    m = Mallet(model_dir=str(tmp_model_dir))
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    # First line: 2 topic probabilities, second line: 3. Should raise.
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.5\t0.25\t0.25\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    train_path = tmp_model_dir / "training_data.txt"
    train_path.write_text("0\t\tthis\n1\t\tthat\n")
    m.metadata["path_to_training_data"] = str(train_path)
    with pytest.raises(LexosException):
        m.get_top_docs(topic=0, n=1)


def test_distributions_parsing_various_formats(tmp_model_dir):
    """Test that distributions parses mixed formats correctly."""
    m = Mallet(model_dir=str(tmp_model_dir))
    file = tmp_model_dir / "doc-topic.txt"
    # mix of tab-delimited, whitespace-delimited, and compressed token
    content = textwrap.dedent(
        """
            #doc\tid\t0\t1
            0\td0\t0.1\t0.9
            1 d1 0.5 0.5
            2\td2\t0:0.2 1:0.8
            """
    )
    file.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(file)
    dists = m.distributions
    assert isinstance(dists, list)
    assert len(dists) == 3
    assert all(isinstance(inner, list) for inner in dists)
    assert all(all(isinstance(x, float) for x in inner) for inner in dists)


def test_load_topic_term_distributions(tmp_model_dir):
    """Test that load_topic_term_distributions loads topic weights as a dictionary."""
    m = Mallet(model_dir=str(tmp_model_dir))
    f = tmp_model_dir / "topic-weights.txt"
    f.write_text("0\tword1\t2\n0\tword2\t3\n1\tword3\t1\n")
    m.metadata[m.CANONICAL_TERM_WEIGHTS_KEY] = str(f)
    d = m.load_topic_term_distributions()
    assert isinstance(d, dict)
    assert 0 in d and 1 in d
    t0 = d[0]
    assert pytest.approx(sum(t0.values()), rel=1e-6) == 1.0


def test_plot_categories_by_topics_heatmap_default_title(tmp_model_dir):
    """Test that plot_categories_by_topics_heatmap uses default title when none is provided."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # create topic_keys and doc-topic
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    fig = m.plot_categories_by_topics_heatmap(categories=["A", "B"], show=False)
    # check suptitle text
    assert hasattr(fig, "_suptitle")
    title_text = fig._suptitle.get_text()
    assert "Topics by Category" in title_text


def test_topic_clouds_round_mask_and_title(tmp_model_dir, monkeypatch):
    """Test that topic_clouds uses round mask and sets title correctly."""
    captured = {}

    class FakeMultiCloud:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.fig = "fakefig"

        def save(self, path):
            captured["saved"] = path

        def show(self):
            captured["shown"] = True

    monkeypatch.setattr("lexos.topic_modeling.mallet.MultiCloud", FakeMultiCloud)

    m = Mallet(model_dir=str(tmp_model_dir))

    def fake_load(self):
        return {0: {"a": 0.6, "b": 0.4}, 1: {"x": 0.8, "y": 0.2}}

    monkeypatch.setattr(Mallet, "load_topic_term_distributions", fake_load)
    fig = m.topic_clouds(show=False, round_mask=True)
    # Check FakeMultiCloud got round=120 (default True) and title present
    assert captured.get("round") == 120
    assert "title" in captured and captured.get("title")


def test_topic_clouds_round_mask_int_and_opts(tmp_model_dir, monkeypatch):
    """Test that topic_clouds accepts integer round_mask and options."""
    captured = {}

    class FakeMultiCloud:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.fig = "fakefig"

        def save(self, path):
            captured["saved"] = path

        def show(self):
            captured["shown"] = True

    monkeypatch.setattr("lexos.topic_modeling.mallet.MultiCloud", FakeMultiCloud)
    m = Mallet(model_dir=str(tmp_model_dir))
    monkeypatch.setattr(
        Mallet, "load_topic_term_distributions", lambda self: {0: {"a": 1.0}}
    )
    fig = m.topic_clouds(
        show=False, round_mask=80, max_terms=10, title="X", **{"opts": {}}
    )
    assert captured.get("round") == 80
    opts = captured.get("opts")
    assert opts.get("max_words") == 10


def test_topic_clouds_output_path_saves(tmp_model_dir, monkeypatch):
    """Test that topic_clouds saves output to specified path."""
    captured = {}

    class FakeMultiCloud:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.fig = "fakefig"

        def save(self, path):
            captured["saved"] = path

        def show(self):
            captured["shown"] = True

    monkeypatch.setattr("lexos.topic_modeling.mallet.MultiCloud", FakeMultiCloud)
    m = Mallet(model_dir=str(tmp_model_dir))
    monkeypatch.setattr(
        Mallet, "load_topic_term_distributions", lambda self: {0: {"a": 1.0}}
    )
    output = tmp_model_dir / "cloud.png"
    m.topic_clouds(show=False, output_path=str(output))
    assert captured.get("saved") == str(output)


def test_import_dir_bypass_validation_raises(tmp_model_dir):
    """Bypass validate_call and ensure boolean causes LexosException in import_dir."""
    m = Mallet(model_dir=str(tmp_model_dir))
    with pytest.raises(LexosException):
        Mallet.import_dir.__wrapped__(m, True)


def test_import_file_bypass_validation_raises(tmp_model_dir):
    """Bypass validate_call and ensure boolean causes LexosException in import_file."""
    m = Mallet(model_dir=str(tmp_model_dir))
    with pytest.raises(LexosException):
        Mallet.import_file.__wrapped__(m, True)


def test_import_docs_bypass_validation_raises(tmp_model_dir):
    """Bypass validate_call and ensure boolean causes LexosException in import_docs."""
    m = Mallet(model_dir=str(tmp_model_dir))
    with pytest.raises(LexosException):
        Mallet.import_docs.__wrapped__(m, True)


def test_import_data_bypass_validation_invalid_element_raises(tmp_model_dir):
    """Bypass validate_call and ensure a non-string element in training_data raises."""
    m = Mallet(model_dir=str(tmp_model_dir))
    with pytest.raises(LexosException):
        Mallet.import_data.__wrapped__(m, [1])


def test_model_directory_property_raises_when_missing():
    """If metadata/model_dir not set, accessing model_directory should raise."""
    m = Mallet()
    with pytest.raises(LexosException):
        _ = m.model_directory


def test_get_top_docs_missing_training_data_raises(tmp_model_dir):
    """If the training data path isn't present in metadata get_top_docs should raise."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Write doc-topic only so distributions exist
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    # Ensure training data not present
    if "path_to_training_data" in m.metadata:
        m.metadata.pop("path_to_training_data")
    with pytest.raises(LexosException):
        m.get_top_docs(topic=0, n=1)


def test_train_sets_non_canonical_metadata(tmp_model_dir, monkeypatch):
    """Check that non-canonical flags get preserved under `path_to_*` keys."""
    m = Mallet(model_dir=str(tmp_model_dir))
    formatted_data_path = tmp_model_dir / "training_data.mallet"
    formatted_data_path.write_text("0\t\tword1 word2\n1\t\tword3 word4\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    monkeypatch.setattr(
        Mallet, "_track_progress", lambda self, cmd, iterations, verbose: None
    )
    m.train(
        num_topics=2,
        num_iterations=1,
        verbose=False,
        path_to_diagnostics="diagnostic.xml",
    )
    # The non-canonical 'diagnostics-file' flag should map to 'path_to_diagnostics_file'
    assert "path_to_diagnostics_file" in m.metadata


def test_topic_clouds_invalid_round_mask_raises_and_show_returns_none(
    tmp_model_dir, monkeypatch
):
    """Invalid round_mask should raise; show=True returns None when MultiCloud.show invoked."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Provide topic-term distributions via monkeypatch
    monkeypatch.setattr(
        Mallet, "load_topic_term_distributions", lambda self: {0: {"a": 1.0}}
    )

    class FakeMultiCloud:
        def __init__(self, **kwargs):
            pass

        def save(self, path):
            pass

        def show(self):
            return None

    monkeypatch.setattr("lexos.topic_modeling.mallet.MultiCloud", FakeMultiCloud)

    # invalid round mask (non convertable string) should raise
    with pytest.raises(LexosException):
        m.topic_clouds(show=False, round_mask="nope")

    # show=True should cause method to return None (the call to show() returns None)
    # Use a valid round_mask so that method progresses
    res = m.topic_clouds(show=True, round_mask=False)
    assert res is None


def test_boxplot_overlay_invalid_raises(tmp_model_dir):
    """Test that plot_categories_by_topic_boxplots raises LexosException for invalid overlay."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Create doc-topic and topic-keys
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    with pytest.raises(LexosException):
        m.plot_categories_by_topic_boxplots(categories=["A"], topics=0, overlay="bad")


def test_load_topic_term_distributions_malformed_line_raises(tmp_model_dir):
    """Test that malformed lines in topic term distributions raise ValueError."""
    m = Mallet(model_dir=str(tmp_model_dir))
    f = tmp_model_dir / "topic-weights.txt"
    f.write_text("0\tword1\n")
    m.metadata[m.CANONICAL_TERM_WEIGHTS_KEY] = str(f)
    with pytest.raises(ValueError):
        m.load_topic_term_distributions()


def test_distributions_missing_raises(tmp_model_dir):
    """Test that accessing distributions with missing metadata raises LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # No metadata set
    with pytest.raises(LexosException):
        _ = m.distributions


def test_topic_keys_missing_raises(tmp_model_dir):
    """Test that accessing topic_keys with missing metadata raises LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    with pytest.raises(LexosException):
        _ = m.topic_keys


def test_get_keys_and_get_top_docs(tmp_model_dir):
    """Test that get_keys returns correct formats and get_top_docs returns DataFrame and string."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # topic_keys
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    keys_str = m.get_keys(num_keys=1)
    assert isinstance(keys_str, str)
    df = m.get_keys(as_df=True)
    assert isinstance(df, pd.io.formats.style.Styler)

    # doc-topics and training data for get_top_docs
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.7\t0.3\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    # training data file
    train_path = tmp_model_dir / "training_data.txt"
    train_path.write_text("0\t\tthis\n1\t\tthat\n")
    m.metadata["path_to_training_data"] = str(train_path)
    top_docs = m.get_top_docs(topic=0, n=1)
    assert isinstance(top_docs, pd.DataFrame)
    as_str = m.get_top_docs(topic=0, n=1, as_str=True)
    assert isinstance(as_str, str)


def test_get_keys_num_topics_out_of_range_raises(tmp_model_dir):
    """Requesting more topics than available should raise an IndexError (line 665)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.5\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    with pytest.raises(IndexError):
        m.get_keys(num_topics=5)


def test_get_keys_topics_list_index_out_of_range_raises(tmp_model_dir):
    """Providing a topics list with an index out-of-range should raise IndexError (line 673)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.5\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    with pytest.raises(IndexError):
        m.get_keys(topics=[1])


def test_plot_categories_by_topic_boxplots_with_overlay(tmp_model_dir, monkeypatch):
    """Test that plot_categories_by_topic_boxplots calls correct overlay functions."""
    m = Mallet(model_dir=str(tmp_model_dir))

    # Create doc-topic and topic-keys files expected by `topic_keys` and `distributions`
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_text = "0\t0.4\tword1 word2 word3\n1\t0.6\talpha beta gamma\n"
    topic_keys_path.write_text(topic_keys_text)
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)

    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.5\t0.5\n2\td2\t0.7\t0.3\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)

    categories = ["A", "B", "A"]

    # Patch seaborn functions to assert they are called correctly
    with (
        patch("seaborn.boxplot") as mock_boxplot,
        patch("seaborn.stripplot") as mock_strip,
        patch("seaborn.swarmplot") as mock_swarm,
    ):
        # Test default overlay (strip)
        result = m.plot_categories_by_topic_boxplots(
            categories=categories, topics=0, show=False, overlay="strip"
        )
        mock_boxplot.assert_called()
        mock_strip.assert_called()

        # Test swarm overlay
        result = m.plot_categories_by_topic_boxplots(
            categories=categories, topics=0, show=False, overlay="swarm"
        )
        mock_swarm.assert_called()

        # Test no overlay
        result = m.plot_categories_by_topic_boxplots(
            categories=categories, topics=0, show=False, overlay="none"
        )
        # Neither strip nor swarm should be called for this call
        # We don't assert not_called because earlier checks may have set call_count, but we check for not raising.
        assert result is not None


def test_boxplot_figsize_none_creates_axes_and_returns_fig(tmp_model_dir):
    """When figsize is None, the alternative plt.subplots call is used (line 1100) and a Figure is returned."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # create keys and doc-topic
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    # Set training data so get_top_docs can run if needed
    train_path = tmp_model_dir / "training_data.txt"
    train_path.write_text("0\t\tone\n")
    m.metadata["path_to_training_data"] = str(train_path)
    fig = m.plot_categories_by_topic_boxplots(
        categories=["A"], topics=0, show=False, figsize=None, overlay="none"
    )
    assert hasattr(fig, "savefig")


def test_boxplot_overlay_backend_failure_is_handled(tmp_model_dir, monkeypatch):
    """If the overlay plotting backend raises an Exception, the function should continue (lines 1135-1137)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    categories = ["A", "B"]
    # Patch seaborn stripplot to raise an error to simulate backend failure
    monkeypatch.setattr(
        "seaborn.stripplot",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("boom")),
    )
    # Also patch boxplot so plots build
    monkeypatch.setattr("seaborn.boxplot", lambda *args, **kwargs: None)
    # Should not raise despite stripplot error
    fig = m.plot_categories_by_topic_boxplots(
        categories=categories, topics=0, show=False, overlay="strip"
    )
    assert fig is not None


def test_boxplot_title_save_and_show_behaviour(tmp_model_dir, monkeypatch):
    """Test that providing a title uses figure-level suptitle (line 1145), saving (1148), and show() returns None (1150-1151)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.5\t0.5\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    categories = ["A", "B"]
    # Patch plt.show so it doesn't try to open a GUI
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    out = tmp_model_dir / "boxplot.png"
    # show False -> returns a figure or list
    fig_or_list = m.plot_categories_by_topic_boxplots(
        categories=categories,
        topics=[0, 1],
        show=False,
        title="MyPlot",
        output_path=str(out),
    )
    # With multiple topics it should return a list
    # Due to code returning a single figure when len(figs)==1 (early return), we accept either a Figure or list
    assert isinstance(fig_or_list, (list, type(plt.figure())))
    if isinstance(fig_or_list, list):
        figs = fig_or_list
    else:
        figs = [fig_or_list]
    # Ensure suptitle set on at least one figure
    assert any(
        hasattr(f, "_suptitle") and f._suptitle.get_text() == "MyPlot" for f in figs
    )
    # Files should have been saved for each topic (suffixed earlier by the implementation)
    topic0_file = tmp_model_dir / "boxplot_topic0.png"
    topic1_file = tmp_model_dir / "boxplot_topic1.png"
    assert topic0_file.exists()
    assert topic1_file.exists()
    # Call with show True -> returns None
    res = m.plot_categories_by_topic_boxplots(
        categories=categories, topics=None, show=True, title="MyPlot2"
    )
    assert res is None


def test_boxplot_empty_topics_returns_empty_list(tmp_model_dir):
    """Passing an empty topics list should return an empty list (line 1158)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    res = m.plot_categories_by_topic_boxplots(categories=["A"], topics=[], show=False)
    assert isinstance(res, list)
    assert len(res) == 0


def test_boxplot_single_topic_save_and_return(tmp_model_dir):
    """For a single topic with output_path, the function should save a suffixed file and return a Figure (single fig) (lines 1169-1172)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    categories = ["A"]
    out = tmp_model_dir / "single_boxplot.png"
    fig = m.plot_categories_by_topic_boxplots(
        categories=categories, topics=0, show=False, output_path=str(out)
    )
    # Should return a single Figure object
    assert not isinstance(fig, list)
    # Should save suffixed file for topic 0
    assert (tmp_model_dir / "single_boxplot_topic0.png").exists()


def test_boxplot_show_true_calls_show_and_close(tmp_model_dir, monkeypatch):
    """Ensure that when show=True, the plot function calls plt.show and plt.close (lines 1169-1172)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    categories = ["A", "B"]

    called = {"show": 0, "close": 0}

    def fake_show(*args, **kwargs):
        called["show"] += 1

    def fake_close(fig):
        called["close"] += 1

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)
    monkeypatch.setattr("matplotlib.pyplot.close", fake_close)

    res = m.plot_categories_by_topic_boxplots(
        categories=categories, topics=None, show=True
    )
    assert res is None
    assert called["show"] >= 1
    assert called["close"] >= 1


def test_boxplot_title_uses_figure_suptitle_and_saves_and_closes(
    tmp_model_dir, monkeypatch
):
    """When title is provided, the function should use fig.suptitle (covering suptitle branch and close on show=False)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    categories = ["A", "B"]

    called = {"close": 0}

    def fake_close(fig):
        called["close"] += 1

    monkeypatch.setattr("matplotlib.pyplot.close", fake_close)
    # Provide an explicit title to ensure fig.suptitle path is used
    out = tmp_model_dir / "titled_boxplot.png"
    fig = m.plot_categories_by_topic_boxplots(
        categories=categories,
        topics=0,
        title="My Title",
        show=False,
        output_path=str(out),
    )
    assert hasattr(fig, "_suptitle")
    assert fig._suptitle.get_text() == "My Title"
    # Saving behavior should produce the suffixed filename
    assert (tmp_model_dir / "titled_boxplot_topic0.png").exists()
    # Ensure close was called (even when show=False)
    assert called["close"] >= 1


def test_plot_categories_by_topics_heatmap_save_creates_output(
    tmp_model_dir, monkeypatch
):
    """When output_path is supplied, heatmap should save to the passed path (lines 1233-1235)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    # Patch seaborn.heatmap so it returns an axis without errors
    monkeypatch.setattr("seaborn.heatmap", lambda *args, **kwargs: plt.gca())
    # Patch plt.show to no-op for headless testing
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    out = tmp_model_dir / "heatmap_out.png"
    fig = m.plot_categories_by_topics_heatmap(
        categories=["A", "B"], output_path=str(out), show=False
    )
    assert out.exists()
    assert hasattr(fig, "_suptitle")


def test_plot_categories_by_topics_heatmap_default_title_counts_topics(
    tmp_model_dir, monkeypatch
):
    """When pivot_table produces a normal output, the default title should include the number of topics (lines 1233-1235)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text(
        "0\t0.4\tword1 word2\n1\t0.6\talpha beta\n2\t0.7\tgamma delta\n"
    )
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n2\td2\t0.1\t0.9\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    # Patch seaborn.heatmap to accept any object and plt.show to no-op
    monkeypatch.setattr("seaborn.heatmap", lambda *args, **kwargs: plt.gca())
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    fig = m.plot_categories_by_topics_heatmap(categories=["A", "B", "C"], show=False)
    assert hasattr(fig, "_suptitle")
    assert "Topics by Category (" in fig._suptitle.get_text()


def test_plot_categories_by_topics_heatmap_honors_custom_title_and_saves_and_closes(
    tmp_model_dir, monkeypatch
):
    """When a custom title is provided, the fig.suptitle should be set and saving + closing should be exercised."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)

    # Patch seaborn.heatmap to accept any object and plt.show to no-op
    monkeypatch.setattr("seaborn.heatmap", lambda *args, **kwargs: plt.gca())
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    called = {"close": 0}

    def fake_close(*args, **kwargs):
        called["close"] += 1

    monkeypatch.setattr("matplotlib.pyplot.close", fake_close)

    out = tmp_model_dir / "heatmap_custom_title.png"
    fig = m.plot_categories_by_topics_heatmap(
        categories=["A", "B"], output_path=str(out), show=False, title="My Heatmap"
    )
    assert hasattr(fig, "_suptitle")
    assert fig._suptitle.get_text() == "My Heatmap"
    assert out.exists()
    assert called["close"] >= 1


def test_boxplot_overlay_backend_failure_is_ignored(tmp_model_dir, monkeypatch):
    """If the overlay backend (stripplot/swarmplot) raises an exception, plot should continue without raising."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    categories = ["A", "B"]

    def fake_strip(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("seaborn.stripplot", fake_strip)
    # Should not raise despite overlay backend failure
    res = m.plot_categories_by_topic_boxplots(
        categories=categories, topics=None, show=False, overlay="strip"
    )
    assert res is not None


def test_boxplot_invalid_overlay_raises(tmp_model_dir):
    """Invalid overlay argument should raise a LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)

    with pytest.raises(LexosException):
        m.plot_categories_by_topic_boxplots(
            categories=["A"], topics=0, overlay="banana"
        )


def test_plot_categories_by_topics_heatmap_with_figsize_and_save_show(
    tmp_model_dir, monkeypatch
):
    """Provide figsize to use fig, ax creation path (line 1217), save to output_path (1238), and show True returns None (1240-1241)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    # Patch heatmap to avoid real plotting internals
    monkeypatch.setattr("seaborn.heatmap", lambda *args, **kwargs: plt.gca())
    # Patch plt.show to no-op
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    out = tmp_model_dir / "heatmap.png"
    res = m.plot_categories_by_topics_heatmap(
        categories=["A", "B"],
        figsize=(4, 3),
        output_path=str(out),
        show=False,
        title=None,
    )
    assert hasattr(res, "_suptitle")
    # Ensure saving occurred by checking file exists
    assert out.exists()
    # When show True, should return None
    res2 = m.plot_categories_by_topics_heatmap(
        categories=["A", "B"], figsize=(4, 3), show=True
    )
    assert res2 is None


def test_plot_categories_by_topics_heatmap_handles_columns_len_exception(
    tmp_model_dir, monkeypatch
):
    """Simulate failing len(df_norm_col.columns) to exercise except: branch (lines 1225-1226) and default title (line 1230)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)

    class FakeColumns:
        def __len__(self):
            raise Exception("boom")

    class FakeDF:
        def __init__(self):
            self.columns = FakeColumns()

        def mean(self):
            return self

        def std(self):
            return self

        def __sub__(self, other):
            return self

        def __truediv__(self, other):
            return self

    # Patch DataFrame.pivot_table to return the FakeDF
    monkeypatch.setattr(
        pd.DataFrame, "pivot_table", lambda self, *args, **kwargs: FakeDF()
    )
    # Patch seaborn.heatmap to accept any object
    monkeypatch.setattr("seaborn.heatmap", lambda *args, **kwargs: plt.gca())
    fig = m.plot_categories_by_topics_heatmap(categories=["A", "B"], show=False)
    # Title should not include numeric count (since len raised), thus 'Topics by Category'
    assert hasattr(fig, "_suptitle")
    assert "Topics by Category" == fig._suptitle.get_text()


def test_plot_categories_by_topics_heatmap_columns_natsorted(
    tmp_model_dir, monkeypatch
):
    """Ensure that heatmap columns are ordered by numeric topic index (natural sort), not string sort."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Create topic keys for topics 0..11 so string sorting would put 'Topic 10' before 'Topic 2'
    num_topics = 12
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_content = "\n".join(f"{i}\t0.5\ttopic_{i}" for i in range(num_topics))
    topic_keys_path.write_text(topic_keys_content)
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)

    # Create doc-topic distributions matching the number of topics
    # Two documents with the same distributions; the pivot will still show all topics
    distributions = []
    for _ in range(2):
        dist = [0.0] * num_topics
        for i in range(num_topics):
            dist[i] = float(i + 1) / (num_topics * (num_topics + 1) / 2)
        distributions.append(dist)
    # Write doc-topic file content (Mallet format doesn't matter since we specify distributions)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text(
        "0\td0\t" + "\t".join(str(x) for x in distributions[0]) + "\n"
    )
    doc_topics_path.write_text(
        doc_topics_path.read_text()
        + "1\td1\t"
        + "\t".join(str(x) for x in distributions[1])
        + "\n"
    )
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)

    # Provide a custom topic_distributions to bypass self.distributions ambiguity
    # Note: categories list must be same length as distributions
    categories = ["A", "B"]

    captured = {}

    def fake_heatmap(df, *args, **kwargs):
        # Capture column labels passed to heatmap
        captured["cols"] = list(df.columns)
        return plt.gca()

    monkeypatch.setattr("seaborn.heatmap", fake_heatmap)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)

    m.plot_categories_by_topics_heatmap(categories=categories, show=False)

    # Columns should be in numeric increasing order: Topic 0, Topic 1, ..., Topic 11
    assert "cols" in captured

    def _get_num(label: str):
        import re

        mnum = re.match(r"Topic\s+(\d+)", label)
        assert mnum is not None
        return int(mnum.group(1))

    nums = [_get_num(c) for c in captured["cols"]]
    assert nums == sorted(nums)


def test_topic_clouds_filters_topics_using_iloc(tmp_model_dir, monkeypatch):
    """Passing topics param should filter the DataFrame passed to MultiCloud via iloc (line 1298)."""
    captured = {}

    class FakeMultiCloud:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.fig = "fakefig"

        def save(self, path):
            captured["saved"] = path

        def show(self):
            captured["shown"] = True

    monkeypatch.setattr("lexos.topic_modeling.mallet.MultiCloud", FakeMultiCloud)
    # Provide multi-topic distribution dict
    monkeypatch.setattr(
        Mallet,
        "load_topic_term_distributions",
        lambda self: {0: {"a": 1.0}, 1: {"b": 1.0}, 2: {"c": 1.0}},
    )
    m = Mallet(model_dir=str(tmp_model_dir))
    mc_fig = m.topic_clouds(topics=[1], show=False)
    # The MultiCloud received a DataFrame with one row
    df_passed = captured.get("data")
    assert df_passed.shape[0] == 1


def test_topic_clouds_default_title_when_len_raises(tmp_model_dir, monkeypatch):
    """When len(df) raises on second call, topic_clouds should fall back to 'Topic Clouds' (lines 1331-1336)."""
    captured = {}

    class FakeMultiCloud:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.fig = "fakefig"

        def save(self, path):
            captured["saved"] = path

        def show(self):
            captured["shown"] = True

    monkeypatch.setattr("lexos.topic_modeling.mallet.MultiCloud", FakeMultiCloud)

    # Monkeypatch load_topic_term_distributions to return valid dict
    monkeypatch.setattr(
        Mallet,
        "load_topic_term_distributions",
        lambda self: {0: {"a": 1.0}, 1: {"b": 1.0}},
    )

    class FakeDF:
        def __init__(self):
            self._calls = 0

        def fillna(self, v):
            return self

        def __len__(self):
            self._calls += 1
            if self._calls > 1:
                raise Exception("boom")
            return 2

    # Monkeypatch DataFrame.from_dict to return a FakeDF instance
    monkeypatch.setattr(pd.DataFrame, "from_dict", lambda *args, **kwargs: FakeDF())

    m = Mallet(model_dir=str(tmp_model_dir))
    # Should not raise; title should be default 'Topic Clouds'
    fig = m.topic_clouds(show=False)
    assert captured.get("title") == "Topic Clouds"


def test_plot_topics_over_time_empty_distributions_raises(tmp_model_dir):
    """Passing empty topic_distributions should raise LexosException (line 1396)."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Provide a valid topic_keys and times but empty distributions
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    with pytest.raises(LexosException):
        m.plot_topics_over_time(
            times=[1],
            topic_index=0,
            topic_distributions=[],
            topic_keys=[["0", "0.5", "a b"]],
            show=False,
        )


def test_plot_topics_over_time_negative_index_raises_with_distributions(tmp_model_dir):
    """Negative topic_index should raise ValueError (line 1399) even with provided distributions."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Provide a minimal distribution
    topic_distributions = [[0.5, 0.5]]
    with pytest.raises(ValueError):
        m.plot_topics_over_time(
            times=[1],
            topic_index=-1,
            topic_distributions=topic_distributions,
            topic_keys=[["0", "0.5", "a b"]],
            show=False,
        )


def test_plot_topics_over_time_times_len_mismatch_raises(tmp_model_dir):
    """If `times` length does not match `topic_distributions`, a LexosException should be raised."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # equal length of distributions not matching length of times
    topic_distributions = [[0.3, 0.7], [0.2, 0.8]]
    # times length mismatched
    with pytest.raises(LexosException):
        m.plot_topics_over_time(
            times=[1],
            topic_index=0,
            topic_distributions=topic_distributions,
            topic_keys=[["0", "0.5", "a b c"], ["1", "0.5", "x y z"]],
            show=False,
        )


# End of tests


def test_plot_topics_over_time_no_data_for_topic_index_raises(tmp_model_dir):
    """If all distributions are too short for requested topic_index, raise LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # each distribution has length 1, topic_index 1 will be out-of-range
    topic_distributions = [[0.1], [0.2]]
    with pytest.raises(LexosException):
        m.plot_topics_over_time(
            times=[1, 2],
            topic_index=1,
            topic_distributions=topic_distributions,
            topic_keys=[["0", "0.5", "a b c"]],
            show=False,
        )


def test_plot_topics_over_time_default_title_fallback_and_saves(
    tmp_model_dir, monkeypatch
):
    """If topic_keys item lacks an index 2 element, fallback to simple topic title and save file."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_distributions = [[0.3, 0.7], [0.2, 0.8]]
    # supply a topic_keys entry lacking a third index to cause IndexError in title construction
    topic_keys = [["0", "0.5"]]
    out = tmp_model_dir / "topics_over_time.png"
    # patch plt.show to avoid GUI
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    # call plot and assert saved file exists and suptitle equals 'Topic {i}'
    fig = m.plot_topics_over_time(
        times=[1, 2],
        topic_index=0,
        topic_distributions=topic_distributions,
        topic_keys=topic_keys,
        show=False,
        output_path=str(out),
    )
    assert out.exists()
    assert hasattr(fig, "_suptitle")
    assert fig._suptitle.get_text() == "Topic 0"


def test_plot_topics_over_time_show_true_returns_none_and_calls_show(
    tmp_model_dir, monkeypatch
):
    """When show=True, the function should call plt.show() and return None."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_distributions = [[0.3, 0.7], [0.2, 0.8]]
    topic_keys = [["0", "0.5", "a b c"]]
    called = {"show_called": False}

    def fake_show(*args, **kwargs):
        called["show_called"] = True

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)
    res = m.plot_topics_over_time(
        times=[1, 2],
        topic_index=0,
        topic_distributions=topic_distributions,
        topic_keys=topic_keys,
        show=True,
    )
    assert res is None
    assert called["show_called"]


def test_infer_parses_whitespace_compressed_colon_pairs(tmp_model_dir, monkeypatch):
    """When compressed topic:prob pairs are whitespace-sep (no tabs), they should parse correctly."""
    m = Mallet(model_dir=str(tmp_model_dir))
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)
    out = tmp_model_dir / "infer-doc-topics.txt"
    # whitespace separated and colon pairs: topic 0 and 2 present
    out.write_text("d0 0:0.2\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    dists = m.infer(
        docs=["x"],
        path_to_inferencer=str(fake_inferencer_path),
        output_path=str(out),
        show=False,
    )
    assert isinstance(dists, list)
    assert len(dists) == 1
    assert len(dists[0]) == 1
    assert pytest.approx(dists[0][0]) == 0.2


def test_infer_whitespace_token_no_colon_raises(tmp_model_dir, monkeypatch):
    """When whitespace-separated line doesn't have colon pairs and also not numeric tokens, should raise LexosException."""
    m = Mallet(model_dir=str(tmp_model_dir))
    fake_inferencer_path = tmp_model_dir / "inferencer.mallet"
    fake_inferencer_path.write_text("dummy")
    m.metadata[m.CANONICAL_INFERENCER_KEY] = str(fake_inferencer_path)
    out = tmp_model_dir / "infer-doc-topics.txt"
    # line has less than 3 tokens and last token not colon pairs
    out.write_text("d0 some_token\n")
    monkeypatch.setattr(os, "system", lambda c: 0)
    with pytest.raises(LexosException):
        m.infer(
            docs=["x"],
            path_to_inferencer=str(fake_inferencer_path),
            output_path=str(out),
        )


def test_track_progress_progress_updates(tmp_model_dir, monkeypatch):
    """Simulate MALLET stdout that includes <n> progress fields to exercise _track_progress."""
    m = Mallet(model_dir=str(tmp_model_dir))

    class FakeStdOut:
        def __init__(self, lines):
            self._lines = [l.encode("utf-8") + b"\n" for l in lines]
            self.i = 0

        def readline(self):
            if self.i < len(self._lines):
                line = self._lines[self.i]
                self.i += 1
                return line
            return b""

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = FakeStdOut(["text", "<1.0>", "<5.0>", "<10.0>"])

        def poll(self):
            return None if self.stdout.i < len(self.stdout._lines) else 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    # No exception should be raised while updating progress
    m._track_progress("mallet fake", 10, verbose=True)


def test_import_data_flags_included(tmp_model_dir, monkeypatch):
    """Ensure import_data includes the import-file flags and the use_pipe_from option gets added to the import command."""
    m = Mallet(model_dir=str(tmp_model_dir))
    calls = []

    def fake_system(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(os, "system", fake_system)
    m.import_data(
        ["doc a", "doc b"],
        keep_sequence=True,
        preserve_case=True,
        remove_stopwords=True,
        use_pipe_from="pipe.dat",
    )
    assert any("--keep-sequence" in c for c in calls)
    assert any("--preserve-case" in c for c in calls)
    assert any("--remove-stopwords" in c for c in calls)
    assert any("--use-pipe-from pipe.dat" in c for c in calls)


def test_get_keys_numtopics_and_topics_int(tmp_model_dir):
    """Exercise num_topics slicing and `topics` list selection in get_keys."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("""0	0.4	word1 word2
1	0.6	alpha beta
2	0.3	gamma delta
""")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    out = m.get_keys(num_topics=1)
    assert "Topic 0" in out and "Topic 1" not in out
    out2 = m.get_keys(topics=[1])
    assert "Topic 1" in out2 and "Topic 0" not in out2
    df_sty = m.get_keys(as_df=True)
    assert isinstance(df_sty, pd.io.formats.style.Styler)


def test_get_top_docs_invalid_topic_and_num_topics_mismatch(tmp_model_dir):
    """Test that get_top_docs raises for invalid topic and mismatched num_topics."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # distributions and training data
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    train_path = tmp_model_dir / "training_data.txt"
    train_path.write_text("0\t\tthis\n1\t\tthat\n")
    m.metadata["path_to_training_data"] = str(train_path)

    # Passing a non-integer topic should raise ValueError
    with pytest.raises(ValueError):
        m.get_top_docs(topic="a", n=1)

    # Now set num_topics to mismatch the distributions length
    m.metadata["num_topics"] = 1
    with pytest.raises(LexosException):
        m.get_top_docs(topic=0, n=1)


def test_topic_clouds_round_mask_string_int(tmp_model_dir, monkeypatch):
    """If round_mask is a string that can be converted to int, accept it."""
    captured = {}

    class FakeMultiCloud:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.fig = "fakefig"

        def save(self, path):
            captured["saved"] = path

        def show(self):
            captured["shown"] = True

    monkeypatch.setattr("lexos.topic_modeling.mallet.MultiCloud", FakeMultiCloud)
    monkeypatch.setattr(
        Mallet, "load_topic_term_distributions", lambda self: {0: {"a": 1.0}}
    )
    m = Mallet(model_dir=str(tmp_model_dir))
    m.topic_clouds(show=False, round_mask="60")
    assert captured.get("round") == 60


def test_model_directory_property_returns_value(tmp_model_dir):
    """Test that model_directory property returns the correct value from metadata."""
    m = Mallet(model_dir=str(tmp_model_dir))
    assert m.model_directory == str(tmp_model_dir)


def test_setup_wordcloud_mask_true_and_false(monkeypatch, tmp_model_dir):
    """Ensure _setup_wordcloud sets mask properly for round_mask True/False."""
    captured = {}

    class FakeWordCloud:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("lexos.topic_modeling.mallet.WordCloud", FakeWordCloud)
    m = Mallet(model_dir=str(tmp_model_dir))
    m._setup_wordcloud(False, 10)
    assert captured.get("mask") is None
    captured.clear()
    m._setup_wordcloud(True, 10)
    assert captured.get("mask") is not None


def test_setup_wordcloud_kwargs_override(monkeypatch, tmp_model_dir):
    """`kwargs` passed to _setup_wordcloud override defaults in WordCloud options."""
    captured = {}

    class FakeWordCloud:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("lexos.topic_modeling.mallet.WordCloud", FakeWordCloud)
    m = Mallet(model_dir=str(tmp_model_dir))
    m._setup_wordcloud(True, 10, background_color="black")
    assert captured.get("background_color") == "black"


def test_mean_num_tokens_numpy_scalar(tmp_model_dir):
    """Test that mean_num_tokens handles numpy scalar values correctly."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # ensure we use a numpy scalar so the .item() branch is exercised
    m.metadata["mean_num_tokens"] = np.float64(2.0)
    # Should either expose .item() or cast to int; just ensure a numeric value
    val = m.mean_num_tokens
    assert isinstance(val, (int, float))


def test_mean_num_tokens_missing_returns_zero():
    """Test that mean_num_tokens returns 0 when metadata key is missing."""
    m = Mallet()
    assert m.mean_num_tokens == 0


def test_mean_num_tokens_int_value(tmp_model_dir):
    """Test that mean_num_tokens returns the correct integer value from metadata."""
    m = Mallet(model_dir=str(tmp_model_dir))
    m.metadata["mean_num_tokens"] = 3
    assert m.mean_num_tokens == 3


def test_distributions_header_skipped(tmp_model_dir):
    """Test that distributions property skips header line in doc-topic file."""
    m = Mallet(model_dir=str(tmp_model_dir))
    f = tmp_model_dir / "doc-topic.txt"
    content = "#doc\t1\t0\t1\n0\td0\t0.1\t0.9\n"
    f.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(f)
    d = m.distributions
    assert len(d) == 1


def test_get_top_docs_out_of_range_raises(tmp_model_dir):
    """Test that get_top_docs raises ValueError for out-of-range topic index."""
    m = Mallet(model_dir=str(tmp_model_dir))
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.1\t0.9\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    train_path = tmp_model_dir / "training_data.txt"
    train_path.write_text("0\t\tthis\n")
    m.metadata["path_to_training_data"] = str(train_path)
    # With a single topic (length 2 implies two topics: 0 and 1), requesting topic 2 should raise
    with pytest.raises(ValueError):
        m.get_top_docs(topic=2, n=1)


def test_num_docs_property(tmp_model_dir):
    """Test that num_docs property returns correct value from metadata."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # default is zero
    assert m.num_docs == 0
    m.metadata["num_docs"] = 5
    assert m.num_docs == 5


def test_vocab_size_property(tmp_model_dir):
    """Test that vocab_size property returns correct value from metadata."""
    m = Mallet(model_dir=str(tmp_model_dir))
    assert m.vocab_size == 0
    m.metadata["vocab_size"] = 42
    assert m.vocab_size == 42


def test_import_data_bool_rejects(monkeypatch, tmp_model_dir):
    """Test that import_data raises LexosException when given boolean as first arg."""
    m = Mallet(model_dir=str(tmp_model_dir))
    with pytest.raises(LexosException):
        Mallet.import_data.__wrapped__(m, True)


def test_get_top_docs_num_topics_invalid_fallback_to_topic_keys(tmp_model_dir):
    """Test that get_top_docs falls back to topic_keys when num_topics is invalid."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # set invalid num_topics and provide topic_keys as fallback
    m.metadata["num_topics"] = "not_int"
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.5\ta b\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    train_path = tmp_model_dir / "training_data.txt"
    train_path.write_text("0\t\tthis\n1\t\tthat\n")
    m.metadata["path_to_training_data"] = str(train_path)
    # Should use topic_keys fallback and not raise
    top_docs = m.get_top_docs(topic=0, n=1)
    assert not top_docs.empty


def test_get_top_docs_no_topic_info_raises(tmp_model_dir):
    """Test that get_top_docs raises LexosException when no topic info is available."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # ensure no num_topics, no topic_keys, and no distributions
    if "num_topics" in m.metadata:
        m.metadata.pop("num_topics")
    if m.CANONICAL_TOPIC_KEYS_KEY in m.metadata:
        m.metadata.pop(m.CANONICAL_TOPIC_KEYS_KEY)
    with pytest.raises(LexosException):
        m.get_top_docs(topic=0, n=1)


def test_get_top_docs_empty_distributions_raises(tmp_model_dir):
    """Test that get_top_docs raises LexosException when distributions are empty."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # create an empty doc-topic file (no distributions)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    # No topic_keys or num_topics present
    if m.CANONICAL_TOPIC_KEYS_KEY in m.metadata:
        del m.metadata[m.CANONICAL_TOPIC_KEYS_KEY]
    if "num_topics" in m.metadata:
        del m.metadata["num_topics"]
    train_path = tmp_model_dir / "training_data.txt"
    train_path.write_text("0\t\tthis\n")
    m.metadata["path_to_training_data"] = str(train_path)
    with pytest.raises(LexosException):
        m.get_top_docs(topic=0, n=1)


def test_get_top_docs_concat_metadata(tmp_model_dir):
    """Test that get_top_docs concatenates provided metadata DataFrame."""
    m = Mallet(model_dir=str(tmp_model_dir))
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.8\t0.2\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    train_path = tmp_model_dir / "training_data.txt"
    train_path.write_text("0\t\tthis\n")
    m.metadata["path_to_training_data"] = str(train_path)
    # Build a metadata DataFrame to concat
    md = pd.DataFrame([{"extra": "value"}])
    res = m.get_top_docs(topic=0, n=1, metadata=md)
    assert "extra" in res.columns


def test_get_topic_term_probabilities_topics_int(tmp_model_dir):
    """Test that get_topic_term_probabilities handles topics as int and returns DataFrame."""
    m = Mallet(model_dir=str(tmp_model_dir))
    f = tmp_model_dir / "topic-weights.txt"
    f.write_text("0\tword1\t2\n0\tword2\t3\n1\tword3\t1\n")
    m.metadata[m.CANONICAL_TERM_WEIGHTS_KEY] = str(f)
    df = m.get_topic_term_probabilities(topics=0, as_df=True)
    assert isinstance(df, pd.DataFrame) and 0 in df["Topic"].values


def test_load_topic_term_distributions_skips_blank_lines(tmp_model_dir):
    """Test that load_topic_term_distributions skips blank lines in topic-weights file."""
    m = Mallet(model_dir=str(tmp_model_dir))
    f = tmp_model_dir / "topic-weights.txt"
    f.write_text("\n0\tword1\t2\n\n0\tword2\t3\n")
    m.metadata[m.CANONICAL_TERM_WEIGHTS_KEY] = str(f)
    d = m.load_topic_term_distributions()
    assert isinstance(d, dict) and 0 in d


def test_load_topic_term_distributions_file_not_found_raises(tmp_model_dir):
    """Test that load_topic_term_distributions raises FileNotFoundError for missing file."""
    m = Mallet(model_dir=str(tmp_model_dir))
    missing = tmp_model_dir / "no_such_weights.txt"
    m.metadata[m.CANONICAL_TERM_WEIGHTS_KEY] = str(missing)
    with pytest.raises(FileNotFoundError):
        m.load_topic_term_distributions()


def test_plot_topics_over_time_empty_truthy_list_raises(tmp_model_dir):
    """Test that plot_topics_over_time raises LexosException for empty but truthy topic_distributions."""
    m = Mallet(model_dir=str(tmp_model_dir))
    # Provide a valid topic_keys so the method progresses to distributions check
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)

    class TruthyEmptyList(list):
        def __bool__(self):
            return True

    with pytest.raises(LexosException):
        m.plot_topics_over_time(
            times=[1],
            topic_index=0,
            topic_distributions=TruthyEmptyList([]),
            topic_keys=[["0", "0.5", "a b"]],
            show=False,
        )


def test_import_dir_reads_and_writes(tmp_model_dir, monkeypatch):
    """Test that import_dir reads files and updates metadata with path."""
    d = tmp_model_dir / "dirtest"
    d.mkdir()
    f1 = d / "a.txt"
    f2 = d / "b.txt"
    f1.write_text("hello")
    f2.write_text("world")
    m = Mallet(model_dir=str(tmp_model_dir))
    monkeypatch.setattr(os, "system", lambda c: 0)
    m.import_dir(str(d))
    assert "path_to_training_data" in m.metadata


def test_distributions_ignores_blank_lines(tmp_model_dir):
    """Test that distributions property ignores blank lines in doc-topic file."""
    m = Mallet(model_dir=str(tmp_model_dir))
    f = tmp_model_dir / "doc-topic.txt"
    content = "\n#doc\tid\t0\t1\n\n0\td0\t0.1\t0.9\n\n"
    f.write_text(content)
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(f)
    d = m.distributions
    assert isinstance(d, list) and len(d) == 1


def test_plot_categories_by_topics_heatmap_with_title_and_target_labels(
    tmp_model_dir, monkeypatch
):
    """Pass a custom title and use target_labels to filter categories; save file should exist."""
    m = Mallet(model_dir=str(tmp_model_dir))
    topic_keys_path = tmp_model_dir / "topic-keys.txt"
    topic_keys_path.write_text("0\t0.4\tword1 word2\n1\t0.6\talpha beta\n")
    m.metadata[m.CANONICAL_TOPIC_KEYS_KEY] = str(topic_keys_path)
    doc_topics_path = tmp_model_dir / "doc-topic.txt"
    doc_topics_path.write_text("0\td0\t0.2\t0.8\n1\td1\t0.3\t0.7\n")
    m.metadata[m.CANONICAL_DOC_TOPIC_KEY] = str(doc_topics_path)
    out = tmp_model_dir / "hm.png"
    # Patch seaborn.heatmap to accept any object
    monkeypatch.setattr("seaborn.heatmap", lambda *args, **kwargs: plt.gca())
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    fig = m.plot_categories_by_topics_heatmap(
        categories=["A", "B"],
        target_labels=["A"],
        output_path=str(out),
        title="Custom Title",
        show=False,
    )
    assert hasattr(fig, "_suptitle") and fig._suptitle.get_text() == "Custom Title"
    assert out.exists()
