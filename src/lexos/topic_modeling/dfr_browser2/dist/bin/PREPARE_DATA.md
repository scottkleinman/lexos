# Preparing Data for DFR Browser 2

This guide explains how to use the `prepare_data.py` script to convert MALLET topic model output into the format required by DFR Browser 2.

## Overview

The `prepare_data.py` script processes a MALLET topic-state file and generates all necessary files for visualizing your topic model in DFR Browser 2:

### Core Files (Always Generated)

- **`topic-keys.txt`** - Topic words for the browser interface
- **`doc-topic.txt`** - Normalized document-topic proportions
- **`topic_coords.csv`** - 2D coordinates for topic visualization
- **`metadata.csv`** - Basic document metadata (if not already present)

### Additional Files (Optional, with `--all` flag)

- **`doc-topic-counts.csv`** - Raw topic counts per document
- **`tw.json`** - Topic-words JSON for advanced features
- **`dt.zip`** - Sparse document-topic matrix

## Requirements

### Required Dependencies

```bash
pip install numpy pandas scikit-learn
```

### MALLET Output Required

You need a MALLET topic-state file, typically named `topic-state.gz`. This file is generated when you run MALLET with the `--output-state` option:

```bash
mallet train-topics \
  --input corpus.mallet \
  --num-topics 50 \
  --output-state topic-state.gz \
  --output-doc-topics doc-topics.txt \
  --output-topic-keys topic-keys.txt
```

## Basic Usage

### Command Line

Navigate to the `src/bin` directory and run:

```bash
python prepare_data.py topic-state.gz
```

This will generate the core files in the current directory.

### Specify Output Directory

```bash
python prepare_data.py topic-state.gz -o ../../dist/data/myproject
```

This generates files in the specified output directory.

### Generate All Files

```bash
python prepare_data.py topic-state.gz -o ../../dist/data/myproject --all
```

This generates both core and additional files.

### Customize Top Words

```bash
python prepare_data.py topic-state.gz --top-words 50
```

This saves the top 50 words per topic (default is 30).

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `statefile` | Path to MALLET topic-state file | (required) |
| `-o`, `--output-dir` | Output directory for generated files | `.` (current directory) |
| `--top-words` | Number of top words per topic to save | `30` |
| `--all` | Generate all files including advanced features | `False` |

## Using as a Python Module

You can import and use the script in your own Python code:

```python
from prepare_data import process_mallet_state_file

# Basic usage
process_mallet_state_file(
    state_file="topic-state.gz",
    output_dir="output",
    n_top_words=30,
    generate_all=False
)

# Generate all files with custom settings
process_mallet_state_file(
    state_file="../../data/myproject/topic-state.gz",
    output_dir="../../dist/data/myproject",
    n_top_words=50,
    generate_all=True
)
```

## Understanding MALLET State File Format

The topic-state file contains a complete record of the topic model's final state, with one line per token:

```txt
#doc source pos typeindex type topic
#alpha : 2.5 2.5 2.5 ... (one value per topic)
#beta : 0.01
0 file:/doc1.txt 0 156 word 12
0 file:/doc1.txt 1 789 another 3
0 file:/doc1.txt 2 234 topic 12
...
```

**Columns:**

- **doc**: Document index (0-based)
- **source**: Document filename/identifier
- **pos**: Token position in document
- **typeindex**: Word type index (vocabulary ID)
- **type**: The actual word
- **topic**: Assigned topic number

The script reads this file and aggregates the data to create the output files.

## Output File Formats

### 1. topic-keys.txt

Format compatible with dfr-browser:

```txt
0   1.0 word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15
1   1.0 term1 term2 term3 term4 term5 term6 term7 term8 term9 term10 term11 term12 term13 term14 term15
2   1.0 text1 text2 text3 text4 text5 text6 text7 text8 text9 text10 text11 text12 text13 text14 text15
```

**Columns:**

- Topic number
- Weight (always 1.0 for compatibility)
- Top 15 words for the topic (space-separated)

### 2. doc-topic.txt

Format with normalized proportions:

```txt
0   doc1   0.0234567890 0.1234567890 0.0034567890 ...
1   doc2   0.0567890123 0.0234567890 0.1234567890 ...
2   doc3   0.0890123456 0.0567890123 0.0234567890 ...
```

**Columns:**

- Document number
- Document name
- Topic proportions (one per topic, sum to 1.0 per document)

### 3. topic_coords.csv

2D coordinates for topic visualization:

```csv
topic,x,y
0,-12.345,8.901
1,5.678,-3.456
2,10.234,15.678
```

**Columns:**

- **topic**: Topic number
- **x**: X-coordinate for visualization
- **y**: Y-coordinate for visualization

These coordinates are computed using:

1. Jensen-Shannon divergence between topic distributions
2. Multidimensional Scaling (MDS) to reduce to 2D

### 4. metadata.csv

Basic document metadata (generated only if file doesn't exist):

```csv
docNum,docName,title,author,year
0,doc1,Document 1,Unknown,2024
1,doc2,Document 2,Unknown,2024
2,doc3,Document 3,Unknown,2024
```

**Note:** Replace this with your actual metadata for better browsing experience.

### 5. doc-topic-counts.csv (with `--all`)

Raw topic counts per document:

```csv
docNum,topic0,topic1,topic2,topic3,...
0,12,5,23,8,...
1,8,15,3,19,...
2,25,2,11,6,...
```

Useful for analyzing raw model output and custom analysis.

### 6. tw.json (with `--all`)

Topic-words data in JSON format:

```json
{
  "alpha": [2.5, 2.5, 2.5, ...],
  "tw": [
    {
      "words": ["word1", "word2", "word3", ...],
      "weights": [123, 98, 87, ...]
    },
    {
      "words": ["term1", "term2", "term3", ...],
      "weights": [156, 112, 95, ...]
    }
  ]
}
```

Contains alpha hyperparameters and topic-word distributions.

### 7. dt.zip (with `--all`)

Sparse document-topic matrix in compressed format:

```json
{
  "i": [0, 1, 2, 5, 7, ...],    // Document indices
  "p": [0, 3, 7, 12, ...],       // Index pointers
  "x": [12, 5, 23, 8, 15, ...]   // Count values
}
```

Efficient storage for large corpora with many topics.

## Topic Coordinate Generation

### How It Works

The script generates 2D coordinates for visualizing topic relationships:

1. **Build topic-word matrix**: Creates a matrix of top words per topic with inverse-rank weighting
2. **Compute Jensen-Shannon distances**: Measures divergence between topic distributions
3. **Apply MDS**: Reduces the distance matrix to 2D coordinates
4. **Output coordinates**: Saves to `topic_coords.csv`

### Jensen-Shannon Divergence

Measures the similarity between two probability distributions. Lower values indicate more similar topics.

```python
JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
where M = 0.5 * (P + Q)
```

### Multidimensional Scaling (MDS)

Projects high-dimensional distances into 2D space while preserving relative distances as much as possible.

### RuntimeWarning Note

The script suppresses the following warning during coordinate generation:

```bash
RuntimeWarning: invalid value encountered in scalar divide
```

**This is normal** and occurs when topics are very similar (small or zero distances in the Jensen-Shannon matrix). The warning is automatically suppressed and coordinates are still generated correctly.

**Common causes:**

- Small corpus with limited topic variation
- Too many topics for the corpus size
- Homogeneous documents (all very similar)

**When to be concerned:**

- All topics cluster at a single point
- Coordinates contain NaN or infinite values
- Topic model quality is consistently poor

If you suspect a problem, you can comment out the code to suppress the warning to confirm whether these issues in your data may be responsible.

## Workflow: From MALLET to DFR Browser 2

### Step 1: Train Your Topic Model with MALLET

```bash
# Import documents
mallet import-dir \
  --input corpus/ \
  --output corpus.mallet \
  --keep-sequence \
  --remove-stopwords

# Train model
mallet train-topics \
  --input corpus.mallet \
  --num-topics 50 \
  --output-state topic-state.gz \
  --output-doc-topics doc-topics.txt \
  --output-topic-keys topic-keys.txt \
  --optimize-interval 10 \
  --num-iterations 1000
```

### Step 2: Prepare Metadata (Optional)

Create a `metadata.csv` file with your actual document information:

```csv
docNum,docName,title,author,year,journal,doi
0,doc1,Understanding Topic Models,Smith J.,2023,AI Review,10.1234/air.2023
1,doc2,Advanced NLP Techniques,Jones M.,2024,Comp. Ling.,10.5678/cl.2024
```

### Step 3: Process MALLET Output

```bash
cd src/bin

# Generate core files
python prepare_data.py \
  path_to_mallet-output/topic-state.gz \
  -o ../data

# Or generate all files
python prepare_data.py \
  path_to_mallet-output/topic-state.gz \
  -o ../data \
  --all \
  --top-words 50
```

### Step 4: Configure DFR Browser 2

Update `dist/config.json` to point to your data:

```json
{
  "data_dir": "data",
  "metadata_file": "data/metadata.csv",
  "topic_scaled_file": "data/topic_coords.csv",
  "info": {
    "title": "My Topic Model",
    "meta_info": "50 topics from 1000 documents"
  }
}
```

### Step 5: View in Browser

Serve DFR Browser 2, and open it in your web browser:

```bash
cd ..
python serve.py
# Navigate to http://localhost:8000
```

## Advanced Usage

### Custom Topic Coordinate Generation

You can customize the coordinate generation by modifying the script:

```python
from prepare_data import write_topic_coords_csv

# Use more top words for coordinate calculation
write_topic_coords_csv(topic_words, output_dir, top_n=25)
```

### Batch Processing Multiple Models

Process multiple MALLET outputs:

```bash
#!/bin/bash
for state_file in mallet-outputs/*/topic-state.gz; do
    project=$(dirname "$state_file" | xargs basename)
    python prepare_data.py \
        "$state_file" \
        -o "../data/" \
        --all
done
```

### Incremental Updates

If you retrain your model with the same documents:

```bash
# Keep your existing metadata.csv
# Only regenerate topic files
python prepare_data.py new-topic-state.gz -o existing-output-dir --all
```

The script will not affect your existing `metadata.csv` file.

## Performance

### Processing Speed

- **Small corpus** (< 1,000 docs): < 1 second
- **Medium corpus** (1,000 - 10,000 docs): 1-10 seconds
- **Large corpus** (10,000 - 100,000 docs): 10-60 seconds
- **Very large corpus** (> 100,000 docs): 1-5 minutes

Progress is reported every 100,000 tokens.

### Memory Usage

The script loads the entire topic-state file into memory. For very large corpora:

- **Estimate**: ~100-200 MB per million tokens
- **Recommendation**: For corpora with > 10 million tokens, ensure sufficient RAM (4GB+)

### Optimization Tips

1. **Use gzip compression**: MALLET's `.gz` files are automatically handled
2. **Limit top words**: Use `--top-words 30` instead of higher values
3. **Skip advanced files**: Only use `--all` when needed

## Troubleshooting

### Common Issues

**Problem:** "FileNotFoundError: [Errno 2] No such file or directory"

- **Solution:** Check that the topic-state file path is correct
- Verify: `ls -lh topic-state.gz`

**Problem:** "UnicodeDecodeError: 'utf-8' codec can't decode byte"

- **Solution:** Your state file may be corrupted or not a valid MALLET output
- Verify: `gzip -t topic-state.gz` (should report OK)

**Problem:** "IndexError: list index out of range"

- **Solution:** State file format may be incorrect or incomplete
- Check: Open the file and verify it has the expected header lines

**Problem:** RuntimeWarning about invalid value in scalar divide

- **Solution:** This is normal when topics are very similar (see "RuntimeWarning Note" section)
- **Action:** The script handles this automatically; coordinates are still generated

**Problem:** metadata.csv not being created

- **Solution:** File already exists in the output directory
- **Action:** Delete or rename existing metadata.csv to regenerate

**Problem:** Topic coordinates all at the same point

- **Solution:** Topics are too similar; model may need retraining
- **Try:**
  - Increase number of MALLET iterations
  - Adjust number of topics
  - Improve document preprocessing (stopwords, etc.)

## Examples

### Example 1: Quick Start

Process a small topic model:

```bash
python prepare_data.py topic-state.gz
```

### Example 2: Custom Output Directory

Generate files for a specific project:

```bash
python prepare_data.py \
  path_to_topic-state.gz \
  -o ../../data/model_state
```

### Example 3: All Files with More Top Words

Generate all files with 50 top words per topic:

```bash
python prepare_data.py \
  topic-state.gz \
  -o ../data \
  --all \
  --top-words 50
```

### Example 4: Python Module Usage

```python
from prepare_data import process_mallet_state_file

# Process multiple models
models = [
    ("psychology-model/topic-state.gz", "data/psychology"),
    ("history-model/topic-state.gz", "data/history"),
    ("literature-model/topic-state.gz", "data/literature")
]

for state_file, output_dir in models:
    print(f"Processing {state_file}...")
    process_mallet_state_file(
        state_file=state_file,
        output_dir=output_dir,
        n_top_words=50,
        generate_all=True
    )
    print(f"✓ Completed {output_dir}\n")
```

### Example 5: Batch Processing Script

Create a shell script to process multiple models:

```bash
#!/bin/bash
# process_all_models.sh

MODELS_DIR="mallet-outputs"
OUTPUT_BASE="../data"

for state in "$MODELS_DIR"/*/topic-state.gz; do
    project=$(basename "$(dirname "$state")")
    echo "Processing $project..."

    python src/bin/prepare_data.py \
        "$state" \
        -o "$OUTPUT_BASE/$project" \
        --all \
        --top-words 50

    echo "✓ Completed $project"
    echo ""
done

echo "All models processed!"
```

Run with:

```bash
chmod +x process_all_models.sh
./process_all_models.sh
```
