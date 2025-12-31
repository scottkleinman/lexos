"""prepare_data.py.

Comprehensive MALLET topic-state file processor for dfr-browser.

Processes a MALLET topic-state.gz file and generates all necessary files for dfr-browser:

- topic-keys.txt (topic words for browser)
- doc-topic.txt (normalized proportions for browser)
- topic_coords.csv (2D topic coordinates for browser)
- tw.json (topic-words JSON for advanced features and dfr-browser compatibility)
- dt.zip (sparse doc-topic matrix for advanced features and dfr-browser compatibility)
- metadata.csv (basic document metadata if not exists)


To use as a module, call

```python
from prepare_data import process_mallet_state_file

process_mallet_state_file(statefile, output_dir, top_words, generate_all=False)
```

Last Updated: December 24, 2025
Last Tested: December 24, 2025
"""

import argparse
import csv
import gzip
import json
import os
import warnings
import zipfile as zf
from collections import defaultdict

# For topic coordinate generation
import numpy as np
import pandas as pd
from sklearn.manifold import MDS


# --- Topic coordinate generation (from scale_topics.py) ---
def topic_word_matrix_from_topic_words(
    topic_words: list[dict], vocab: list[str], top_n: int = 15
) -> np.ndarray:
    """Create topic-word matrix from topic_words data structure.

    Args:
        topic_words (list[dict]): List of dicts with 'words' key for each topic
        vocab (list[str]): List of all unique words across topics
        top_n (int): Number of top words to consider per topic

    Returns:
        np.ndarray: Topic-word matrix of shape (num_topics, vocab_size)
    """
    mat = np.zeros((len(topic_words), len(vocab)))
    for i, topic in enumerate(topic_words):
        words = topic["words"][:top_n]
        for rank, word in enumerate(words):
            if word in vocab:
                mat[i, vocab.index(word)] = 1.0 / (rank + 1)
        if mat[i].sum() > 0:
            mat[i] /= mat[i].sum()
    return mat


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        p (np.ndarray): First probability distribution (1D np.ndarray)
        q (np.ndarray): Second probability distribution (1D np.ndarray)

    Returns:
        float: Jensen-Shannon divergence
    """
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def jsd_matrix(mat: np.ndarray) -> np.ndarray:
    """Compute Jensen-Shannon distance matrix for topic-word distributions.

    Args:
        mat (np.ndarray): Topic-word matrix (num_topics x vocab_size)

    Returns:
        np.ndarray: Jensen-Shannon distance matrix (num_topics x num_topics)
    """
    n = mat.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = jensen_shannon(mat[i], mat[j])
    return dist


def compute_mds(dist: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute MDS coordinates from distance matrix.

    Args:
        dist (np.ndarray): Distance matrix (num_topics x num_topics)
        n_components (int): Number of MDS components (default: 2)

    Returns:
        np.ndarray: MDS coordinates (num_topics x n_components)

    # NOTE:
    When all topics are identical or very similar, the Jensen-Shannon distance matrix will have very small or zero distances. Division by zero/near-zero triggers a RuntimeWarning in sklearn's MDS implementation. This may be caused by not enough documents to create distinct topics, by the model creating redundant/similar topics, by poor model quality, or by homogeneous data in which documents are too similar. This function will calculate coordinates anyway, so we suppress the warning.
    """
    # Suppress the specific RuntimeWarning from sklearn
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=42,
            n_init=4,
            max_iter=300,
        )
        coords = mds.fit_transform(dist)

    return coords


def write_topic_coords_csv(
    topic_words: list[dict], output_dir: str, top_n: int = 15
) -> None:
    """Generate and write topic_coords.csv for dfr-browser.

    Args:
        topic_words (list[dict]): List of topic words data structures
        output_dir (str): Directory to write the CSV file
        top_n (int): Number of top words to consider per topic
    """
    # Build vocab from all top words
    vocab = sorted({w for topic in topic_words for w in topic["words"][:top_n]})
    mat = topic_word_matrix_from_topic_words(topic_words, vocab, top_n=top_n)
    dist = jsd_matrix(mat)
    coords = compute_mds(dist)
    df = pd.DataFrame(
        {"topic": list(range(len(topic_words))), "x": coords[:, 0], "y": coords[:, 1]}
    )
    filepath = os.path.join(output_dir, "topic_coords.csv")
    df.to_csv(filepath, index=False)
    print(f"Wrote topic_coords.csv with {len(topic_words)} topics")


def get_top_words_and_weights(
    topic_word_counts: list[int], vocab: dict, n: int
) -> dict[str, list[str | float]]:
    """Get the top n words and their weights for a topic.

    Args:
        topic_word_counts (list[int]): List of word counts for the topic
        vocab (dict): Mapping of word indices to word strings
        n (int): Number of top words to retrieve

    Returns:
        dict: Dictionary with 'words' and 'weights' lists
    """
    word_indices = list(range(len(topic_word_counts)))
    word_indices.sort(key=lambda i: -topic_word_counts[i])
    return {
        "words": [vocab[i] for i in word_indices[:n]],
        "weights": [topic_word_counts[i] for i in word_indices[:n]],
    }


def sparse_doc_topic_matrix(doc_topic_matrix: list[list[int]]) -> dict:
    """Convert a dense doc-topic matrix to a sparse representation.

    Args:
        doc_topic_matrix (list[list[int]]): Dense doc-topic matrix (num_topics x num_docs)

    Returns:
        dict: Sparse matrix representation with 'i', 'p', 'x' keys
    """
    num_docs = len(doc_topic_matrix[0])
    indices = []
    indptr = [0]
    data = []
    nnz = 0  # Number of non-zeros
    for topic_counts in doc_topic_matrix:
        for doc_idx in range(num_docs):
            count = topic_counts[doc_idx]
            if count != 0:
                indices.append(doc_idx)
                data.append(count)
                nnz += 1
        indptr.append(nnz)
    return {"i": indices, "p": indptr, "x": data}


def normalize_doc_topic_proportions(
    doc_topic_counts: list[dict], num_topics: int
) -> list[list[float]]:
    """Convert raw counts to normalized proportions for each document.

    Args:
        doc_topic_counts (list[dict]): List of dicts with topic counts per document
        num_topics (int): Total number of topics in the model

    Returns:
        list[list[float]]: List of normalized topic proportions per document
    """
    proportions = []
    for doc_counts in doc_topic_counts:
        total = sum(doc_counts.values())
        if total == 0:
            # If document has no tokens, create zeros for all topics
            proportions.append([0.0] * num_topics)
        else:
            # Create normalized proportions for all topics (not just those present in this doc)
            doc_proportions = []
            for topic in range(num_topics):
                prop = doc_counts.get(topic, 0) / total
                doc_proportions.append(prop)
            proportions.append(doc_proportions)
    return proportions


def write_topic_keys_txt(topic_words: list[dict], output_dir: str) -> None:
    """Write topic-keys.txt file compatible with dfr-browser.

    Args:
        topic_words (list[dict]): List of topic words data structures
        output_dir (str): Directory to write the topic-keys.txt file
    """
    filepath = os.path.join(output_dir, "topic-keys.txt")
    with open(filepath, "w") as f:
        for i, topic in enumerate(topic_words):
            words = topic["words"][:15]  # Top 15 words for browser
            # Format: topic_number weight word1 word2 word3...
            f.write(f"{i}\t1.0\t{' '.join(words)}\n")
    print(f"Wrote topic-keys.txt with {len(topic_words)} topics")


def write_doc_topic_txt(doc_proportions: list[list[float]], output_dir: str) -> None:
    """Write doc-topic.txt file compatible with dfr-browser.

    Args:
        doc_proportions (list[list[float]]): List of normalized topic proportions per document
        output_dir (str): Directory to write the doc-topic.txt file
    """
    filepath = os.path.join(output_dir, "doc-topic.txt")
    with open(filepath, "w") as f:
        for doc_idx, proportions in enumerate(doc_proportions):
            # Format: docNum docName proportion1 proportion2 ...
            prop_str = "\t".join(f"{p:.10f}" for p in proportions)
            f.write(f"{doc_idx}\tdoc{doc_idx + 1}\t{prop_str}\n")
    print(f"Wrote doc-topic.txt with {len(doc_proportions)} documents")


def write_doc_topic_counts_csv(
    doc_topic_counts: list[dict], num_topics: int, output_dir: str
) -> None:
    """Write doc-topic-counts.csv with raw counts.

    Args:
        doc_topic_counts (list[dict]): List of dicts with topic counts per document
        num_topics (int): Total number of topics
        output_dir (str): Directory to write the CSV file
    """
    filepath = os.path.join(output_dir, "doc-topic-counts.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: docNum, topic0, topic1, ...
        writer.writerow(["docNum"] + [f"topic{i}" for i in range(num_topics)])
        for doc_idx, topic_counts in enumerate(doc_topic_counts):
            row = [doc_idx] + [topic_counts.get(i, 0) for i in range(num_topics)]
            writer.writerow(row)


def write_basic_metadata_csv(num_docs: int, output_dir: str) -> None:
    """Write basic metadata.csv if it doesn't exist.

    Args:
        num_docs (int): Number of documents
        output_dir (str): Directory to write the metadata.csv file
    """
    filepath = os.path.join(output_dir, "metadata.csv")
    if os.path.exists(filepath):
        print("metadata.csv already exists, skipping generation")
        return

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["docNum", "docName", "title", "author", "year"])
        for i in range(num_docs):
            writer.writerow([i, f"doc{i + 1}", f"Document {i + 1}", "Unknown", "2024"])
    print(f"Generated basic metadata.csv with {num_docs} documents")


def write_topic_words_json(
    alpha: list, topic_words: list[dict], output_dir: str
) -> None:
    """Write tw.json for advanced features.

    Args:
        alpha (list): List of alpha parameters for topics
        topic_words (list[dict]): List of topic words data structures
        output_dir (str): Directory to write the tw.json file
    """
    filepath = os.path.join(output_dir, "tw.json")
    output = {"alpha": alpha, "tw": topic_words}
    with open(filepath, "w") as f:
        json.dump(output, f)
    print("Wrote tw.json with topic-words data")


def write_doc_topics_zip(sparse_matrix: dict, output_dir: str) -> None:
    """Write dt.zip for advanced features.

    Args:
        sparse_matrix (dict): Sparse doc-topic matrix representation
        output_dir (str): Directory to write the dt.zip file
    """
    filepath = os.path.join(output_dir, "dt.zip")
    with zf.ZipFile(filepath, "w") as zipf:
        zipf.writestr("dt.json", json.dumps(sparse_matrix))
    print("Wrote dt.zip with sparse doc-topics matrix")


def process_mallet_state_file(
    state_file: str,
    output_dir: str = ".",
    n_top_words: int = 30,
    generate_all: bool = False,
) -> None:
    """Process MALLET topic-state file and generate dfr-browser files.

    Args:
        state_file (str): Path to MALLET topic-state.gz file
        output_dir (str): Directory to write output files
        n_top_words (int): Number of top words per topic to save
        generate_all (bool): Whether to generate additional files beyond core requirements
    """
    print(f"Processing MALLET state file: {state_file}")
    print(f"Output directory: {output_dir}")

    # Initialize data structures
    doc_topic_counts = []  # list of dicts: doc_idx -> {topic: count}
    topic_word_counts = defaultdict(
        lambda: defaultdict(int)
    )  # topic -> word_idx -> count
    # doc_lengths = defaultdict(int)  # doc_idx -> token_count
    vocab = dict()  # word_idx -> word_string

    last_doc_idx = 0
    current_doc_counts = defaultdict(int)
    max_topic = 0
    line_count = 0

    # Process the state file
    with gzip.open(state_file, "rt") as f:
        # Skip header and read alpha parameters
        f.readline()  # Skip #doc source pos typeindex type topic
        alpha_line = f.readline().strip().split(" ")[2:]
        alpha = list(map(float, alpha_line))
        beta_line = f.readline().strip().split(" ")[2]
        print(f"Found alpha parameters: {len(alpha)} topics")
        print(f"Beta value: {beta_line}")

        # Process each token line
        for line in f:
            line_count += 1
            if line_count % 100000 == 0:
                print(f"Processed {line_count:,} tokens...")

            parts = line.strip().split()
            if len(parts) < 6:
                continue

            doc_idx, source, pos, type_index, word, topic = parts
            doc_idx = int(doc_idx)
            type_index = int(type_index)
            topic = int(topic)
            max_topic = max(max_topic, topic)

            # Handle document transition
            if last_doc_idx != doc_idx:
                if current_doc_counts:  # Save previous document
                    doc_topic_counts.append(current_doc_counts)
                current_doc_counts = defaultdict(int)

            # Update counts
            current_doc_counts[topic] += 1
            topic_word_counts[topic][type_index] += 1
            # doc_lengths[doc_idx] += 1

            # Update vocabulary
            if type_index not in vocab:
                vocab[type_index] = word

            last_doc_idx = doc_idx

        # Add the last document
        if current_doc_counts:
            doc_topic_counts.append(current_doc_counts)

    num_topics = max_topic + 1
    num_docs = len(doc_topic_counts)
    print(
        f"Processed {line_count:,} tokens from {num_docs} documents with {num_topics} topics"
    )

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate topic words data
    topic_words = []
    for t in range(num_topics):
        word_counts = [topic_word_counts[t][i] for i in range(len(vocab))]
        topic_words.append(get_top_words_and_weights(word_counts, vocab, n_top_words))

    # Generate normalized proportions for dfr-browser
    doc_proportions = normalize_doc_topic_proportions(doc_topic_counts, num_topics)
    # Validation: Ensure all documents have the correct number of topics
    for doc_idx, proportions in enumerate(doc_proportions):
        if len(proportions) != num_topics:
            raise ValueError(
                f"Document {doc_idx} has {len(proportions)} topic proportions, "
                f"but expected {num_topics} topics. "
                f"This indicates a mismatch between doc-topic data and topic-keys data."
            )

    # Validation: Ensure topic_words matches num_topics
    if len(topic_words) != num_topics:
        raise ValueError(
            f"Generated {len(topic_words)} topic-word entries, "
            f"but expected {num_topics} topics from state file."
        )
    # Write core dfr-browser files (always generated)
    write_topic_keys_txt(topic_words, output_dir)
    write_doc_topic_txt(doc_proportions, output_dir)
    write_topic_coords_csv(topic_words, output_dir, top_n=15)
    write_basic_metadata_csv(num_docs, output_dir)

    # Write additional files if requested with --all flag
    if generate_all:
        write_doc_topic_counts_csv(doc_topic_counts, num_topics, output_dir)
        write_topic_words_json(alpha, topic_words, output_dir)

        # Generate sparse matrix for dt.zip
        dense_doc_topic = [
            [doc_topic_counts[d].get(t, 0) for d in range(num_docs)]
            for t in range(num_topics)
        ]
        sparse_matrix = sparse_doc_topic_matrix(dense_doc_topic)
        write_doc_topics_zip(sparse_matrix, output_dir)

    print("\n✅ All files generated successfully!")
    print(f"Your dfr-browser data is ready in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process MALLET topic-state file for dfr-browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s topic-state.gz                   # Generate core files in current directory
  %(prog)s topic-state.gz -o sample_data    # Generate core files in sample_data/ directory
  %(prog)s topic-state.gz --all             # Generate all files including advanced features
  %(prog)s topic-state.gz --top-words 50    # Include 50 top words per topic (default: 30)

Generated files:
  Core files (always created):
    - topic-keys.txt      (topic words for dfr-browser)
    - doc-topic.txt       (document-topic proportions)
    - topic_coords.csv    (2D topic coordinates for browser)
    - metadata.csv        (basic metadata if missing)

  Additional files (with --all flag):
    - doc-topic-counts.csv (raw topic counts per document)
    - tw.json             (topic-words JSON for advanced features)
    - dt.zip              (sparse doc-topic matrix)
        """,
    )

    parser.add_argument(
        "statefile", help="Path to MALLET topic-state file (gzipped or plain)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Output directory for generated files (default: current directory)",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=30,
        help="Number of top words per topic to save (default: 30)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all files including advanced features (default: core files only)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.statefile):
        print(f"Error: State file not found: {args.statefile}")
        exit(1)

    process_mallet_state_file(
        args.statefile, args.output_dir, args.top_words, generate_all=args.all
    )
