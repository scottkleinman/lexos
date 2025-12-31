"""Tests for the dfr_browser2 prepare_data.py module.

Coverage: 99%. Missing:  428, 436

Last Updated: December 24, 2025
"""

import csv
import gzip
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lexos.topic_modeling.dfr_browser2.prepare_data import (
    compute_mds,
    get_top_words_and_weights,
    jensen_shannon,
    jsd_matrix,
    normalize_doc_topic_proportions,
    process_mallet_state_file,
    sparse_doc_topic_matrix,
    topic_word_matrix_from_topic_words,
    write_basic_metadata_csv,
    write_doc_topic_counts_csv,
    write_doc_topic_txt,
    write_doc_topics_zip,
    write_topic_coords_csv,
    write_topic_keys_txt,
    write_topic_words_json,
)


@pytest.fixture
def sample_topic_words() -> list[dict]:
    """Return sample topic words data structure."""
    return [
        {"words": ["word1", "word2", "word3"], "weights": [10, 5, 3]},
        {"words": ["word4", "word5", "word6"], "weights": [8, 6, 2]},
        {"words": ["word7", "word8", "word9"], "weights": [12, 4, 1]},
    ]


@pytest.fixture
def sample_doc_topic_counts() -> list[dict]:
    """Return sample document-topic counts."""
    return [
        {0: 10, 1: 5, 2: 2},  # Doc 0
        {0: 3, 1: 12, 2: 1},  # Doc 1
        {0: 1, 1: 2, 2: 15},  # Doc 2
    ]


@pytest.fixture
def sample_state_file(tmp_path: Path) -> Path:
    """Create a sample MALLET state file for testing."""
    state_file = tmp_path / "topic-state.gz"

    content = """#doc source pos typeindex type topic
#alpha : 0.5 0.5
#beta : 0.01
0 file0 0 0 word1 0
0 file0 1 1 word2 0
0 file0 2 2 word3 1
1 file1 0 0 word1 1
1 file1 1 3 word4 1
1 file1 2 1 word2 0
2 file2 0 2 word3 0
2 file2 1 3 word4 1
2 file2 2 4 word5 1
"""

    with gzip.open(state_file, "wt") as f:
        f.write(content)

    return state_file


def test_jensen_shannon():
    """Test Jensen-Shannon divergence calculation."""
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])

    # Identical distributions should have JS divergence of 0
    result = jensen_shannon(p, q)
    assert result == pytest.approx(0.0, abs=1e-10)

    # Different distributions should have positive divergence
    p2 = np.array([0.9, 0.1])
    q2 = np.array([0.1, 0.9])
    result2 = jensen_shannon(p2, q2)
    assert result2 > 0


def test_topic_word_matrix_from_topic_words(sample_topic_words: list[dict]):
    """Test topic-word matrix generation."""
    vocab = [
        "word1",
        "word2",
        "word3",
        "word4",
        "word5",
        "word6",
        "word7",
        "word8",
        "word9",
    ]

    mat = topic_word_matrix_from_topic_words(sample_topic_words, vocab, top_n=3)

    # Check shape
    assert mat.shape == (3, 9)

    # Check that rows sum to 1 (normalized)
    for i in range(mat.shape[0]):
        assert mat[i].sum() == pytest.approx(1.0, abs=1e-10)

    # Check that topic 0 has highest weight for word1 (index 0)
    assert mat[0, 0] > mat[0, 1]


def test_topic_word_matrix_empty_topic():
    """Test topic-word matrix with no words in vocab."""
    topic_words = [
        {"words": ["unknown1", "unknown2"], "weights": [10, 5]},
    ]
    vocab = ["word1", "word2", "word3"]  # None of the topic words are in vocab

    mat = topic_word_matrix_from_topic_words(topic_words, vocab, top_n=3)

    # Should have zero matrix (no words matched)
    assert mat.shape == (1, 3)
    assert mat.sum() == 0.0


def test_jsd_matrix():
    """Test JSD distance matrix calculation."""
    # Create a simple topic-word matrix
    mat = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.3, 0.5],
            [0.33, 0.33, 0.34],
        ]
    )

    dist = jsd_matrix(mat)

    # Check shape
    assert dist.shape == (3, 3)

    # Diagonal should be 0 (topic compared to itself)
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-10)
    assert dist[1, 1] == pytest.approx(0.0, abs=1e-10)

    # Matrix should be symmetric
    assert dist[0, 1] == pytest.approx(dist[1, 0])


def test_compute_mds():
    """Test MDS coordinate computation."""
    # Create a simple distance matrix
    dist = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.7],
            [1.0, 0.7, 0.0],
        ]
    )

    coords = compute_mds(dist, n_components=2)

    # Check shape
    assert coords.shape == (3, 2)

    # Check that coordinates are not all the same
    assert not np.allclose(coords[0], coords[1])


def test_get_top_words_and_weights():
    """Test extraction of top words and their weights."""
    vocab = {0: "apple", 1: "banana", 2: "cherry", 3: "date"}
    topic_word_counts = [10, 5, 15, 3]

    result = get_top_words_and_weights(topic_word_counts, vocab, n=3)

    assert "words" in result
    assert "weights" in result
    assert len(result["words"]) == 3
    assert len(result["weights"]) == 3

    # Top word should be "cherry" (index 2, count 15)
    assert result["words"][0] == "cherry"
    assert result["weights"][0] == 15


def test_normalize_doc_topic_proportions(sample_doc_topic_counts: list[dict]):
    """Test normalization of document-topic counts to proportions."""
    num_topics = 3

    proportions = normalize_doc_topic_proportions(sample_doc_topic_counts, num_topics)

    # Check that we have proportions for all documents
    assert len(proportions) == 3

    # Check that each document has proportions for all topics
    for doc_props in proportions:
        assert len(doc_props) == num_topics

    # Check that proportions sum to 1 for each document
    for doc_props in proportions:
        assert sum(doc_props) == pytest.approx(1.0)

    # Check specific values for doc 0: {0: 10, 1: 5, 2: 2} -> total 17
    assert proportions[0][0] == pytest.approx(10 / 17)
    assert proportions[0][1] == pytest.approx(5 / 17)
    assert proportions[0][2] == pytest.approx(2 / 17)


def test_normalize_doc_topic_proportions_empty_doc():
    """Test normalization with a document that has no tokens."""
    doc_topic_counts = [
        {0: 10, 1: 5},
        {},  # Empty document
        {0: 5, 1: 10},
    ]
    num_topics = 2

    proportions = normalize_doc_topic_proportions(doc_topic_counts, num_topics)

    # Empty document should have all zeros
    assert proportions[1] == [0.0, 0.0]


def test_sparse_doc_topic_matrix():
    """Test conversion of dense doc-topic matrix to sparse representation."""
    # Dense matrix: topics x docs
    dense_matrix = [
        [5, 0, 3],  # Topic 0
        [0, 10, 0],  # Topic 1
        [2, 1, 0],  # Topic 2
    ]

    sparse = sparse_doc_topic_matrix(dense_matrix)

    assert "i" in sparse
    assert "p" in sparse
    assert "x" in sparse

    # Check that non-zero values are captured
    assert 5 in sparse["x"]
    assert 10 in sparse["x"]
    assert 3 in sparse["x"]
    assert 2 in sparse["x"]
    assert 1 in sparse["x"]


def test_write_topic_keys_txt(tmp_path: Path, sample_topic_words: list[dict]):
    """Test writing topic-keys.txt file."""
    output_dir = str(tmp_path)

    write_topic_keys_txt(sample_topic_words, output_dir)

    output_file = tmp_path / "topic-keys.txt"
    assert output_file.exists()

    content = output_file.read_text()
    lines = content.strip().split("\n")

    # Should have 3 topics
    assert len(lines) == 3

    # Check format: topic_number \t weight \t word1 word2 ...
    parts = lines[0].split("\t")
    assert parts[0] == "0"
    assert parts[1] == "1.0"
    assert "word1" in parts[2]


def test_write_doc_topic_txt(tmp_path: Path):
    """Test writing doc-topic.txt file."""
    doc_proportions = [
        [0.5, 0.3, 0.2],
        [0.1, 0.6, 0.3],
    ]

    write_doc_topic_txt(doc_proportions, str(tmp_path))

    output_file = tmp_path / "doc-topic.txt"
    assert output_file.exists()

    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 2

    # Check format: docNum \t docName \t prop1 \t prop2 ...
    parts = lines[0].split("\t")
    assert parts[0] == "0"
    assert parts[1] == "doc1"
    assert float(parts[2]) == pytest.approx(0.5)


def test_write_basic_metadata_csv(tmp_path: Path):
    """Test writing basic metadata.csv file."""
    num_docs = 5

    write_basic_metadata_csv(num_docs, str(tmp_path))

    output_file = tmp_path / "metadata.csv"
    assert output_file.exists()

    with open(output_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 5
    assert rows[0]["docNum"] == "0"
    assert rows[0]["docName"] == "doc1"


def test_write_basic_metadata_csv_skip_existing(tmp_path: Path):
    """Test that metadata.csv generation is skipped if file exists."""
    # Create existing metadata.csv
    existing_file = tmp_path / "metadata.csv"
    existing_file.write_text("custom,content\n", encoding="utf-8")

    write_basic_metadata_csv(10, str(tmp_path))

    # Should not overwrite
    content = existing_file.read_text()
    assert content == "custom,content\n"


def test_write_topic_words_json(tmp_path: Path, sample_topic_words: list[dict]):
    """Test writing tw.json file."""
    alpha = [0.5, 0.5, 0.5]

    write_topic_words_json(alpha, sample_topic_words, str(tmp_path))

    output_file = tmp_path / "tw.json"
    assert output_file.exists()

    with open(output_file, "r") as f:
        data = json.load(f)

    assert "alpha" in data
    assert "tw" in data
    assert data["alpha"] == alpha
    assert len(data["tw"]) == 3


def test_write_doc_topics_zip(tmp_path: Path):
    """Test writing dt.zip file."""
    sparse_matrix = {"i": [0, 1, 2], "p": [0, 1, 2, 3], "x": [5, 10, 3]}

    write_doc_topics_zip(sparse_matrix, str(tmp_path))

    output_file = tmp_path / "dt.zip"
    assert output_file.exists()

    # Extract and verify content
    with zipfile.ZipFile(output_file, "r") as zf:
        assert "dt.json" in zf.namelist()
        content = zf.read("dt.json")
        data = json.loads(content)

    assert data["i"] == [0, 1, 2]
    assert data["x"] == [5, 10, 3]


def test_write_topic_coords_csv(tmp_path: Path, sample_topic_words: list[dict]):
    """Test writing topic_coords.csv file."""
    write_topic_coords_csv(sample_topic_words, str(tmp_path), top_n=3)

    output_file = tmp_path / "topic_coords.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)

    assert "topic" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns
    assert len(df) == 3


def test_process_mallet_state_file_basic(sample_state_file: Path, tmp_path: Path):
    """Test basic processing of MALLET state file."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    process_mallet_state_file(
        state_file=str(sample_state_file),
        output_dir=str(output_dir),
        n_top_words=10,
        generate_all=False,
    )

    # Check that core files were created
    assert (output_dir / "topic-keys.txt").exists()
    assert (output_dir / "doc-topic.txt").exists()
    assert (output_dir / "topic_coords.csv").exists()
    assert (output_dir / "metadata.csv").exists()


def test_process_mallet_state_file_generate_all(
    sample_state_file: Path, tmp_path: Path
):
    """Test processing with generate_all=True creates additional files."""
    output_dir = tmp_path / "output_all"
    output_dir.mkdir()

    process_mallet_state_file(
        state_file=str(sample_state_file),
        output_dir=str(output_dir),
        n_top_words=10,
        generate_all=True,
    )

    # Check that all files were created
    assert (output_dir / "topic-keys.txt").exists()
    assert (output_dir / "doc-topic.txt").exists()
    assert (output_dir / "topic_coords.csv").exists()
    assert (output_dir / "metadata.csv").exists()
    assert (output_dir / "doc-topic-counts.csv").exists()
    assert (output_dir / "tw.json").exists()
    assert (output_dir / "dt.zip").exists()


def test_process_mallet_state_file_topic_keys_content(
    sample_state_file: Path, tmp_path: Path
):
    """Test that topic-keys.txt has correct number of topics."""
    output_dir = tmp_path / "output_keys"
    output_dir.mkdir()

    process_mallet_state_file(
        state_file=str(sample_state_file),
        output_dir=str(output_dir),
        n_top_words=10,
        generate_all=False,
    )

    # Read topic-keys.txt
    content = (output_dir / "topic-keys.txt").read_text()
    lines = content.strip().split("\n")

    # Should have 2 topics (0 and 1 from the state file)
    assert len(lines) == 2


def test_process_mallet_state_file_doc_topic_content(
    sample_state_file: Path, tmp_path: Path
):
    """Test that doc-topic.txt has correct number of documents and topics."""
    output_dir = tmp_path / "output_doc"
    output_dir.mkdir()

    process_mallet_state_file(
        state_file=str(sample_state_file),
        output_dir=str(output_dir),
        n_top_words=10,
        generate_all=False,
    )

    # Read doc-topic.txt
    content = (output_dir / "doc-topic.txt").read_text()
    lines = content.strip().split("\n")

    # Should have 3 documents (0, 1, 2 from the state file)
    assert len(lines) == 3

    # Each line should have proportions for 2 topics
    for line in lines:
        parts = line.split("\t")
        # Format: docNum, docName, prop0, prop1
        assert len(parts) == 4  # docNum + docName + 2 proportions


def test_process_mallet_state_file_validation(sample_state_file: Path, tmp_path: Path):
    """Test that validation catches mismatches between topics and proportions."""
    output_dir = tmp_path / "output_validate"
    output_dir.mkdir()

    # This should not raise since our fix ensures all docs have all topics
    process_mallet_state_file(
        state_file=str(sample_state_file),
        output_dir=str(output_dir),
        n_top_words=10,
        generate_all=False,
    )

    # Verify by checking that all documents have the same number of topic proportions
    content = (output_dir / "doc-topic.txt").read_text()
    lines = content.strip().split("\n")

    num_topics_per_doc = [
        len(line.split("\t")) - 2 for line in lines
    ]  # Subtract docNum and docName

    # All documents should have the same number of topics
    assert len(set(num_topics_per_doc)) == 1


def test_write_doc_topic_counts_csv(tmp_path: Path):
    """Test writing doc-topic-counts.csv file."""
    doc_topic_counts = [
        {0: 10, 1: 5, 2: 2},
        {0: 3, 1: 12, 2: 1},
        {0: 1, 1: 2, 2: 15},
    ]
    num_topics = 3

    write_doc_topic_counts_csv(doc_topic_counts, num_topics, str(tmp_path))

    output_file = tmp_path / "doc-topic-counts.csv"
    assert output_file.exists()

    with open(output_file, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Check header
    assert rows[0] == ["docNum", "topic0", "topic1", "topic2"]

    # Check first document
    assert rows[1] == ["0", "10", "5", "2"]


def test_jsd_matrix_comprehensive():
    """Test JSD matrix with more comprehensive data."""
    # Create diverse topic-word distributions
    mat = np.array(
        [
            [0.7, 0.2, 0.1],  # Topic heavily weighted towards first word
            [0.1, 0.2, 0.7],  # Topic heavily weighted towards last word
            [0.33, 0.34, 0.33],  # Uniform distribution
        ]
    )

    dist = jsd_matrix(mat)

    # Check all properties
    assert dist.shape == (3, 3)

    # Diagonal should be exactly 0
    for i in range(3):
        assert dist[i, i] == pytest.approx(0.0, abs=1e-10)

    # Matrix should be symmetric
    for i in range(3):
        for j in range(3):
            assert dist[i, j] == pytest.approx(dist[j, i])

    # Topics 0 and 1 should be most different
    assert dist[0, 1] > dist[0, 2]
    assert dist[0, 1] > dist[1, 2]


def test_sparse_doc_topic_matrix_comprehensive():
    """Test sparse matrix conversion with various patterns."""
    # Test with some zero columns and mixed non-zero values
    dense_matrix = [
        [10, 0, 0, 5],  # Topic 0: sparse
        [0, 20, 15, 0],  # Topic 1: sparse, different docs
        [3, 3, 3, 3],  # Topic 2: dense
    ]

    sparse = sparse_doc_topic_matrix(dense_matrix)

    # Verify structure
    assert "i" in sparse
    assert "p" in sparse
    assert "x" in sparse

    # p should have num_topics + 1 entries
    assert len(sparse["p"]) == 4  # 3 topics + 1

    # Count non-zeros
    assert len(sparse["x"]) == 8  # 10, 5, 20, 15, 3, 3, 3, 3

    # Verify values are correct
    assert 10 in sparse["x"]
    assert 20 in sparse["x"]
    assert 15 in sparse["x"]


def test_process_mallet_state_file_with_large_data(tmp_path: Path):
    """Test processing with more realistic data size."""
    state_file = tmp_path / "large-state.gz"

    # Create a larger state file with more documents and topics
    content = """#doc source pos typeindex type topic
#alpha : 0.5 0.5 0.5
#beta : 0.01
"""
    # Add multiple documents with various topic assignments
    doc_id = 0
    for doc_num in range(10):  # 10 documents
        for token_num in range(20):  # 20 tokens per document
            word_idx = token_num % 5  # 5 unique words
            topic = (doc_num + token_num) % 3  # Distribute across 3 topics
            content += f"{doc_num} file{doc_num} {token_num} {word_idx} word{word_idx} {topic}\n"

    with gzip.open(state_file, "wt") as f:
        f.write(content)

    output_dir = tmp_path / "large_output"
    output_dir.mkdir()

    process_mallet_state_file(
        state_file=str(state_file),
        output_dir=str(output_dir),
        n_top_words=5,
        generate_all=True,
    )

    # Verify all files were created
    assert (output_dir / "topic-keys.txt").exists()
    assert (output_dir / "doc-topic.txt").exists()
    assert (output_dir / "topic_coords.csv").exists()
    assert (output_dir / "metadata.csv").exists()
    assert (output_dir / "doc-topic-counts.csv").exists()
    assert (output_dir / "tw.json").exists()
    assert (output_dir / "dt.zip").exists()

    # Verify content of topic-keys.txt
    content = (output_dir / "topic-keys.txt").read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 3  # 3 topics

    # Verify content of doc-topic.txt
    content = (output_dir / "doc-topic.txt").read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 10  # 10 documents


def test_process_mallet_state_file_error_handling(tmp_path: Path):
    """Test error handling for malformed state files."""
    state_file = tmp_path / "bad-state.gz"

    # Create a state file with incomplete lines
    content = """#doc source pos typeindex type topic
#alpha : 0.5 0.5
#beta : 0.01
0 file0 0 0 word1 0
incomplete line here
1 file1 0 1 word2 1
"""

    with gzip.open(state_file, "wt") as f:
        f.write(content)

    output_dir = tmp_path / "bad_output"
    output_dir.mkdir()

    # Should process successfully, skipping bad lines
    process_mallet_state_file(
        state_file=str(state_file),
        output_dir=str(output_dir),
        n_top_words=10,
        generate_all=False,
    )

    # Should still create files
    assert (output_dir / "topic-keys.txt").exists()
    assert (output_dir / "doc-topic.txt").exists()


def test_compute_mds_with_warnings():
    """Test MDS computation with distance matrices that trigger warnings."""
    # Create a distance matrix with very small distances (near-identical topics)
    dist = np.array(
        [
            [0.0, 0.001, 0.001],
            [0.001, 0.0, 0.001],
            [0.001, 0.001, 0.0],
        ]
    )

    # Should complete without raising warnings
    coords = compute_mds(dist, n_components=2)

    assert coords.shape == (3, 2)
    assert not np.isnan(coords).any()


def test_normalize_doc_topic_proportions_all_topics():
    """Test that normalization includes all topics, not just those present."""
    # Document only has topics 0 and 2, but model has 4 topics
    doc_topic_counts = [
        {0: 10, 2: 5},  # Missing topics 1 and 3
    ]
    num_topics = 4

    proportions = normalize_doc_topic_proportions(doc_topic_counts, num_topics)

    # Should have proportions for all 4 topics
    assert len(proportions[0]) == 4

    # Topics 1 and 3 should be 0
    assert proportions[0][1] == 0.0
    assert proportions[0][3] == 0.0

    # Topics 0 and 2 should sum to 1
    assert proportions[0][0] + proportions[0][2] == pytest.approx(1.0)


def test_process_mallet_state_file_with_100k_tokens(tmp_path: Path, capsys):
    """Test processing with 100,000+ tokens to trigger progress messages."""
    state_file = tmp_path / "huge-state.gz"

    # Create a state file with >100,000 tokens to trigger progress print
    content = """#doc source pos typeindex type topic
#alpha : 0.5 0.5
#beta : 0.01
"""
    # Generate 100,001 tokens to trigger the progress message
    for i in range(100001):
        doc_num = i % 100  # 100 documents
        word_idx = i % 10  # 10 unique words
        topic = i % 2  # 2 topics
        content += f"{doc_num} file{doc_num} {i} {word_idx} word{word_idx} {topic}\n"

    with gzip.open(state_file, "wt") as f:
        f.write(content)

    output_dir = tmp_path / "huge_output"
    output_dir.mkdir()

    process_mallet_state_file(
        state_file=str(state_file),
        output_dir=str(output_dir),
        n_top_words=5,
        generate_all=False,
    )

    # Check that progress message was printed
    captured = capsys.readouterr()
    assert "Processed 100,000 tokens" in captured.out


def test_cli_basic_usage(tmp_path: Path, sample_state_file: Path):
    """Test command-line interface with basic usage."""
    import subprocess
    import sys

    output_dir = tmp_path / "cli_output"
    output_dir.mkdir()

    # Run the script as a command-line tool
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lexos.topic_modeling.dfr_browser2.prepare_data",
            str(sample_state_file),
            "-o",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    # Check successful execution
    assert result.returncode == 0
    assert (output_dir / "topic-keys.txt").exists()
    assert (output_dir / "doc-topic.txt").exists()


def test_cli_with_all_flag(tmp_path: Path, sample_state_file: Path):
    """Test command-line interface with --all flag."""
    import subprocess
    import sys

    output_dir = tmp_path / "cli_all_output"
    output_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lexos.topic_modeling.dfr_browser2.prepare_data",
            str(sample_state_file),
            "-o",
            str(output_dir),
            "--all",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    # Check that all files were created
    assert (output_dir / "topic-keys.txt").exists()
    assert (output_dir / "tw.json").exists()
    assert (output_dir / "dt.zip").exists()


def test_cli_with_top_words(tmp_path: Path, sample_state_file: Path):
    """Test command-line interface with --top-words option."""
    import subprocess
    import sys

    output_dir = tmp_path / "cli_topwords_output"
    output_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lexos.topic_modeling.dfr_browser2.prepare_data",
            str(sample_state_file),
            "-o",
            str(output_dir),
            "--top-words",
            "50",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "topic-keys.txt").exists()


def test_cli_missing_statefile(tmp_path: Path):
    """Test command-line interface with missing state file."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lexos.topic_modeling.dfr_browser2.prepare_data",
            "nonexistent-file.gz",
        ],
        capture_output=True,
        text=True,
    )

    # Should exit with error code 1
    assert result.returncode == 1
    assert "Error: State file not found" in result.stdout
