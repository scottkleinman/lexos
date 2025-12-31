# Corpus

## Overview

The Lexos Corpus module is used to manage, search, and analyze text collections. Whilst you can easily pass documents loaded from files or assigned in memory to any Lexos tool, the Corpus module provides useful ways of managing your documents, especially for larger collections. Think of a corpus as a smart filing cabinet for your texts. Each document in your corpus is wrapped in a `Record` object — a container that holds not just the text itself, but also metadata (like author or date) and optional linguistic analysis.

The corpus module allows you to serialize and de-serialize your records to disk, generate statistics about your documents, and activate and de-active records for analysis, and filter and search your documents.

---

### Records

Let's start with the foundation: the `Record`. A Record is simply a document with some metadata attached.

```python
from lexos.corpus import Record

record = Record(
    name="my_first_doc",
    content="This is the text of my document.",
    meta={"author": "Jane", "date": "2025-01-15"}
)
```

Under the hood, a Record can store plain text, a parsed spaCy Doc (for NLP analysis), or both. It also handles serialization — saving and loading from disk or database.

record = corpus.get(name=["saying_1", "fable_1"])

For more information, see [Working with Records](working_with_records.md).

---

### Creating a Corpus

A corpus is a collection of records. The simplest way to create a file-based corpus using the `Corpus` class.

```python
from lexos.corpus import Corpus

corpus = Corpus(corpus_dir="my_collection", name="My Texts")

# Add documents
corpus.add(
    content="The quick brown fox jumps over the lazy dog.",
    name="fable_1",
    metadata={"type": "fable", "year": 1900}
)

corpus.add(
    content="A journey of a thousand miles begins with a single step.",
    name="saying_1",
    metadata={"type": "saying", "author": "Lao Tzu"}
)

print(f"Corpus has {corpus.num_docs} documents")
```

Your documents are now stored as `Record` objects in the `my_collection` directory. Each record is saved as a file, and metadata is tracked in a central index. When you create a corpus, a file called `corpus_metadata.json` is created in the `my_collection` directory. This file contains metadata about your corpus which can be used as a form of "ground truth" for the corpus.

---

#### Loading Files from Disk

For loading multiple files into your corpus efficiently, use the `add_from_files()` method. This method provides memory-efficient streaming of files with parallel processing. For 100 files, `add_from_files()` is typically **1.4x faster** than loading files individually. Performance gains increase with larger file counts (1000+ files). It works with plain text files as well as PDFs, DOCX, and ZIP archives.

```python
from lexos.corpus import Corpus

corpus = Corpus(corpus_dir="my_collection", name="My Texts")

# Load all text files from a directory
corpus.add_from_files(
    paths="path/to/text/files",
    max_workers=4,
    show_progress=True,
    name_template="doc_{index:03d}",
    metadata={"collection": "my_texts", "year": 2025}
)

print(f"Loaded {corpus.num_docs} documents")
```

**Key Parameters:**

- **paths** (str | Path): Path to directory or file(s) to load
- **max_workers** (int): Number of parallel workers (default: 4)
- **worker_strategy** (str): Worker optimization strategy - `"auto"`, `"io_bound"`, `"cpu_bound"`, or `"balanced"` (default: `"auto"`)
- **batch_size** (int): Files per batch for progress tracking (default: 50)
- **show_progress** (bool): Display progress bars (default: True)
- **name_template** (str): Template for naming records with placeholders:
  - `{filename}`: Full filename with extension
  - `{stem}`: Filename without extension
  - `{index}`: Sequential number (use `:03d` for zero-padding)
- **is_active** (bool): Set records as active/inactive (default: True)
- **model** (str): Language model for NLP parsing (optional)
- **extensions** (list[str]): File extensions to include (default: common text formats)
- **metadata** (dict): Metadata to add to all loaded records
- **id_type** (str): ID generation method - `"uuid4"` or `"integer"` (default: `"uuid4"`)

**Example with Custom Naming:**

```python
# Load files with custom naming based on filename
corpus.add_from_files(
    paths="documents",
    name_template="{stem}",  # Use filename without extension
    metadata={"source": "project_alpha"}
)

# Or use sequential numbering
corpus.add_from_files(
    paths="documents",
    name_template="doc_{index:03d}",  # doc_001, doc_002, etc.
)
```

---

#### Accessing Your Documents

You can access corpus records with the `get()` method by passing the record's `id` or `name`.

```python
# Get a specific record by id
record = corpus.get(id="1")

# Get multiple records by name
records = corpus.get(name=["fable_1", "saying_1"])  # returns a list

# Get all records
all_records = list(corpus.records.values())
```

---

#### Displaying a Corpus

You can generate a pandas DataFrame of your corpus for easy inspection as a table.

```python
df = corpus.to_df()
print(df)
```

---

#### Filtering and Querying

You can filter records by metadata:

```python
# Find all fables
fables = corpus.filter_records(type="fable")
```

---

### The SQLite Backend

For larger projects—hundreds or thousands of documents—use the SQLite backend. This stores everything in a fast, searchable database:

```python
from lexos.corpus.sqlite import SQLiteCorpus

corpus = SQLiteCorpus(
    corpus_dir="my_collection",
    sqlite_path="corpus.db",
    name="My Texts",
    use_sqlite=True
)

# Add documents just like before
corpus.add(
    content="The digital revolution transformed society.",
    name="article_1",
    metadata={"source": "tech_journal", "topic": "technology"}
)

# Sync file-based records to the database
corpus.sync()
```

!!! Note
    The `sqlite_path` is the path to the location where you want to save the database. SQLiteCorpus maintains both file storage (for backup) and database storage (for fast queries). Syncing keeps them in sync. If you wish to use only a database without separate file storage, set `sqlite_only=True`.

#### Full-Text Search

The SQLite backend supports powerful full-text search:

```python
# Search for keywords
results = corpus.search("digital OR technology")

for record in results:
    print(f"{record.name}: {record.preview}")
```

#### Advanced Filtering

Filter by multiple criteria:

```python
# Find long documents that have been parsed with NLP
long_parsed = corpus.filter_records(
    is_parsed=True,
    min_tokens=500
)

# Find inactive records (useful for archiving)
archived = corpus.filter_records(is_active=False)
```

---

### Corpus Statistics

You can generate statistics to help you understand your data by calling the `get_stats()` method.

```python
stats = corpus.get_stats()
```

This gives you insights like:

- **Total records and active records** – How many documents do you have?
- **Token counts** – Total words and average words per document
- **Vocabulary size** – How many unique words?
- **Lexical diversity** – Measures like type-token ratio (TTR) that show vocabulary richness
- **Hapax legomena** – Words appearing only once (often interesting!)
- **Document length distribution** – Min, max, average document length

**Example output:**

```python
{
    'total_records': 100,
    'active_records': 98,
    'total_tokens': 15234,
    'avg_tokens_per_record': 152,
    'vocabulary_size': 3421,
    'hapax_ratio': 0.45,
    'lexical_diversity': {
        'ttr': 0.22,
        'rttr': 4.5
    },
    'doc_length': {
        'min': 10,
        'max': 2341,
    all_records = list(corpus.records.values())
        'std': 205
    }
}
```

Under the hood, the `get_stats()` method creates an instance of the `CorpusStats` class. For further details, see the separate sections below.

---

#### Generating Corpus Statistics with `CorpusStats`

The `corpus_stats` module provides comprehensive statistical analysis of your corpus. It calculates everything from basic counts to advanced lexical diversity measures, helping you understand the structure and quality of your text collection.

You do not technically need to create a corpus to use it. All you neeed is a list of tuples where each tuple contains:

- **ID**: A unique identifier for the document
- **Label**: A human-readable name
- **Tokens**: A list of tokens (words or other units)

You can then pass this tuple to the `CorpusStats` class, as shown below:

```python
from lexos.corpus.corpus_stats import CorpusStats

# Prepare your documents
docs = [
    ("doc1", "First Document", ["the", "quick", "brown", "fox"]),
    ("doc2", "Second Document", ["the", "lazy", "dog", "sleeps"]),
    ("doc3", "Third Document", ["a", "quick", "dog", "runs"])
]

# Create the statistics object
stats = CorpusStats(docs=docs)
```

!!! Note
    CorpusStats creates a Lexos Document-Term Matrix (DTM) internally. You can configure the DTM vectorizer with parameters like `min_df`, `max_df`, and `max_n_terms` to filter rare or common terms.

  ```python
  # Filter terms appearing in at least 2 documents
  stats = CorpusStats(docs=docs, min_df=2)
  ```

  See the Lexos `dtm` module documentation for further details.

#### Document Statistics Table

The `doc_stats_df` property gives you a pandas DataFrame with statistics for each document:

```python
df = stats.doc_stats_df
print(df)
```

**Example output:**

| Documents       | hapax_legomena | total_tokens | total_terms | vocabulary_density | hapax_dislegomena |
|-----------------|----------------|--------------|-------------|--------------------|-------------------|
| First Document  | 3              | 4            | 4           | 100.00             | 0                 |
| Second Document | 3              | 4            | 4           | 100.00             | 0                 |
| Third Document  | 2              | 4            | 4           | 100.00             | 0                 |

- **hapax_legomena**: Words appearing exactly once in this document
- **total_tokens**: Total word count
- **total_terms**: Number of unique words
- **vocabulary_density**: (unique words / total words) × 100
- **hapax_dislegomena**: Words appearing exactly twice

#### Basic Corpus Statistics

Get the mean and standard deviation across all documents:

```python
mean_tokens = stats.mean
std_dev = stats.standard_deviation

print(f"Average document length: {mean_tokens:.1f} tokens")
print(f"Standard deviation: {std_dev:.1f}")
```

#### Finding Outliers

Identify documents that are unusually short or long using two methods:

**1. Interquartile Range (IQR) Method:**

```python
# Get IQR-based outliers
outliers = stats.get_iqr_outliers()
for doc_id, doc_name in outliers:
    print(f"Outlier: {doc_name} (ID: {doc_id})")

# Get the IQR bounds
lower, upper = stats.iqr_bounds
print(f"Normal range: {lower:.0f} to {upper:.0f} tokens")
```

**2. Standard Deviation Method:**

```python
# Documents more than 2 standard deviations from mean
outliers = stats.get_std_outliers()
```

#### Distribution Analysis

Understand how document lengths are distributed:

```python
dist_stats = stats.distribution_stats

print(f"Skewness: {dist_stats['skewness']:.2f}")
print(f"Kurtosis: {dist_stats['kurtosis']:.2f}")
print(f"Is normally distributed: {dist_stats['is_normal']}")
print(f"Coefficient of variation: {dist_stats['coefficient_of_variation']:.2f}")
```

**Interpreting the metrics:**

- **Skewness**: Negative means more short docs, positive means more long docs, ~0 is balanced
- **Kurtosis**: Positive means more extreme values, negative means flatter distribution
- **Is normal**: Whether the distribution resembles a bell curve (Shapiro-Wilk test, p > 0.05)
- **Coefficient of variation**: Relative variability (std/mean); lower means more consistent document lengths

#### Percentile Analysis

Get detailed percentile breakdowns:

```python
percentiles = stats.percentiles

print(f"5th percentile: {percentiles['percentile_5']:.0f} tokens")
print(f"Median: {percentiles['percentile_50']:.0f} tokens")
print(f"95th percentile: {percentiles['percentile_95']:.0f} tokens")
print(f"Range: {percentiles['range']:.0f} tokens")
```

This tells you, for example, that 95% of documents are shorter than the 95th percentile value.

#### Text Diversity Statistics

Analyze vocabulary richness across your corpus:

```python
diversity = stats.text_diversity_stats

print(f"Mean Type-Token Ratio: {diversity['mean_ttr']:.3f}")
print(f"Corpus-level TTR: {diversity['corpus_ttr']:.3f}")
print(f"Hapax legomena ratio: {diversity['corpus_hapax_ratio']:.3f}")
print(f"Total hapax words: {diversity['total_hapax']}")
```

**Key metrics:**

- **Type-Token Ratio (TTR)**: Unique words / total words (higher = more diverse vocabulary)
- **Hapax ratio**: Proportion of words appearing once (indicates vocabulary growth potential)
- **Dislegomena ratio**: Proportion of words appearing twice

#### Advanced Lexical Diversity

Beyond simple TTR, get more sophisticated diversity measures:

```python
adv_diversity = stats.advanced_lexical_diversity

print(f"Mean CTTR: {adv_diversity['mean_cttr']:.3f}")
print(f"Mean RTTR: {adv_diversity['mean_rttr']:.3f}")
print(f"Diversity coefficient of variation: {adv_diversity['diversity_coefficient_variation']:.3f}")
```

- **CTTR (Corrected TTR)**: Types / √(2 × tokens) - less sensitive to text length
- **RTTR (Root TTR)**: Types / √tokens - another length-adjusted measure
- **Log TTR**: log(types) / log(tokens) - logarithmic scaling
- **Diversity CV**: How much diversity varies across documents

!!! Note
    These measures correct for the fact that longer texts naturally have lower TTR, making them more suitable for comparing texts of different lengths.

#### Zipf's Law Analysis

Test whether your corpus follows Zipf's law (a power law distribution where word frequency is inversely proportional to rank):

```python
zipf = stats.zipf_analysis

print(f"Zipf slope: {zipf['zipf_slope']:.3f}")  # Should be close to -1 for ideal Zipf
print(f"R-squared: {zipf['r_squared']:.3f}")
print(f"Follows Zipf's law: {zipf['follows_zipf']}")
print(f"Goodness of fit: {zipf['zipf_goodness_of_fit']}")
```

- A slope near -1 and high R² indicates the corpus follows natural language patterns
- "Excellent" or "good" fit suggests the corpus is representative of natural text
- Poor fit might indicate specialized vocabulary, small corpus, or unusual text types

#### Corpus Quality Metrics

Assess whether your corpus is balanced and sufficiently sampled:

```python
quality = stats.corpus_quality_metrics

# Document length balance
balance = quality['document_length_balance']
print(f"Length balance: {balance['classification']}")
print(f"Coefficient of variation: {balance['coefficient_variation']:.3f}")

# Vocabulary coverage
coverage = quality['corpus_coverage']
print(f"Total unique terms: {coverage['unique_terms']}")
print(f"Coverage ratio: {coverage['coverage_ratio']:.4f}")

# Sampling adequacy
richness = quality['vocabulary_richness']
print(f"Sampling adequacy: {richness['sampling_adequacy']}")
print(f"Hapax ratio: {richness['hapax_ratio']:.3f}")
```

**Interpreting quality metrics:**

**Balance classifications:**

- `very_balanced`: CV < 0.2 - documents are very similar in length
- `balanced`: CV < 0.4 - reasonable consistency
- `moderately_unbalanced`: CV < 0.6 - some variation
- `highly_unbalanced`: CV ≥ 0.6 - wide variation in document lengths

**Sampling adequacy:**

- `excellent`: < 10% hapax words - vocabulary is well covered
- `good`: 10-30% hapax - adequate sampling
- `adequate`: 30-50% hapax - borderline; more data may help
- `insufficient`: ≥ 50% hapax - too many rare words; need more documents

#### Comparing Groups

Compare statistics between two subsets of your corpus:

```python
# Compare documents from two different authors
results = stats.compare_groups(
    group1_labels=["doc1", "doc3"],
    group2_labels=["doc2"],
    metric="total_tokens",
    test_type="mann_whitney"  # or "t_test" or "welch_t"
)

print(f"Group 1 mean: {results['group1_mean']:.1f}")
print(f"Group 2 mean: {results['group2_mean']:.1f}")
print(f"P-value: {results['p_value']:.4f}")
print(f"Effect size: {results['effect_size']:.3f} ({results['effect_size_interpretation']})")
print(f"Significant: {results['is_significant']}")
```

**Test types:**

- **mann_whitney**: Non-parametric test (doesn't assume normal distribution)
- **t_test**: Assumes equal variances and normal distribution
- **welch_t**: Doesn't assume equal variances

!!! Note
    Effect sizes tell you the magnitude of difference. A "large" effect size means the difference is substantial, even if groups overlap; "small" means the difference is detectable but modest.

#### Bootstrap Confidence Intervals

Estimate confidence intervals for any metric using resampling:

```python
ci = stats.bootstrap_confidence_interval(
    metric="total_tokens",
    confidence_level=0.95,
    n_bootstrap=1000
)

print(f"Original mean: {ci['original_mean']:.1f}")
print(f"95% CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")
print(f"Margin of error: ±{ci['margin_of_error']:.1f}")
```

This tells you the range where the true mean likely falls, accounting for sampling variability.

#### Visualization

Create boxplots to visualize document length distribution:

**Seaborn (static):**

```python
stats.plot(column="total_tokens", type="seaborn_boxplot", title="Document Lengths")
```

**Plotly (interactive):**

```python
stats.plot(column="total_tokens", type="plotly_boxplot", title="Document Lengths")
```

You can plot any column from `doc_stats_df`, such as:

- `total_tokens`
- `total_terms`
- `vocabulary_density`
- `hapax_legomena`
