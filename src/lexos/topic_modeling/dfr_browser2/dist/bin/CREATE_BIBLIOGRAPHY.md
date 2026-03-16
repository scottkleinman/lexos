# Creating Bibliography Files

This guide explains how to use the `create_bibliography.py` script to convert metadata CSV files or existing CSL JSON files into Citation Style Language (CSL) JSON format with formatted citations for the DFR Browser 2 application.

## Overview

The `create_bibliography.py` script can work with two types of input:

### From CSV Files

1. Converts each entry to CSL JSON format
2. Formats citations using professional citation styles (APA, MLA, Chicago, etc.)
3. Outputs a `bibliography.json` file ready for use in DFR Browser 2

### From CSL JSON Files

1. Reads existing CSL JSON bibliography files
2. Adds or updates formatted citations for each entry
3. Preserves existing formatted citations (won't overwrite)
4. Outputs an enhanced bibliography file with citation strings

Note: Many bibliographical tools like Zotero will output bibliographies in CSL JSON format, and other formats can be crosswalked into CSL JSON format. See [citationstyles.org](https://citationstyles.org/developers/) for a list of resources.

## Requirements

### Required Dependencies

- pandas
- python-dateutil
- nameparser

### Optional Dependencies

For citation formatting with professional styles:

- citeproc-py

**Note:** Without `citeproc-py`, the script will still create valid CSL JSON but will use simple fallback citation formatting.

## Basic Usage

### Command Line - CSV Files

Navigate to the `src/bin` directory and run:

```bash
python create_bibliography.py
```

This will:

- Read `metadata.csv` from the same directory
- Create `bibliography.json` with Chicago Author-Date style citations
- Save the output in the same directory

### Command Line - CSL JSON Files

To add formatted citations to an existing CSL JSON file:

```bash
python create_bibliography.py --input bibliography.json --output bibliography_formatted.json
```

This will:

- Read `bibliography.json` from the same directory
- Add formatted citations to entries that don't have them
- Preserve any existing formatted citations
- Save the enhanced output to `bibliography_formatted.json`

### With Custom Options

**CSV Input:**

```bash
python create_bibliography.py --input ../data/metadata.csv \
                               --output ../data/bibliography.json \
                               --style apa \
                               --debug
```

**JSON Input:**

```bash
python create_bibliography.py --input my_bibliography.json \
                               --output my_bibliography_formatted.json \
                               --style mla \
                               --debug
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Input file: metadata CSV or CSL JSON | `metadata.csv` |
| `--output` | Output JSON file path | `bibliography.json` |
| `--style` | Citation style to use | `chicago-author-date` |
| `--debug` | Enable verbose debug output | `False` |

**Note:** The script automatically detects the input format based on the file extension (`.json` for CSL JSON, otherwise CSV).

## Citation Styles

### Available Styles

The script supports any CSL style. Common shortcuts include:

| Shortcut | Full Style Name | Use Case |
|----------|----------------|----------|
| `chicago` | chicago-author-date | History, humanities (author-date) |
| `chicago-note` | chicago-note-bibliography | History, humanities (footnotes) |
| `apa` | apa | Psychology, social sciences |
| `mla` | modern-language-association | Literature, humanities |
| `harvard` | harvard-cite-them-right | General academic (UK) |

### Specifying a Style

```bash
# Using a shortcut
python create_bibliography.py --style apa

# Using full style name
python create_bibliography.py --style modern-language-association
```

## Using as a Python Module

You can import and use the script in your own Python code:

```python
from create_bibliography import create_bibliography

# From CSV
create_bibliography(
    input="metadata.csv",
    output="bibliography.json",
    style="chicago-author-date",
    debug=False
)

# From CSL JSON - add formatted citations
create_bibliography(
    input="bibliography.json",
    output="bibliography_formatted.json",
    style="apa",
    debug=False
)

# With custom paths
create_bibliography(
    input="../data/metadata.csv",
    output="../data/bibliography.json",
    style="apa",
    debug=True
)
```

## Input Formats

### CSV Format

### Required Columns

At minimum, your `metadata.csv` should have:

- `title` - Document title
- Either `author` or author information will be inferred

### Common Columns

The script recognizes and maps these common columns to CSL fields:

| CSV Column | CSL Field | Description |
|------------|-----------|-------------|
| `author` | `author` | Author name(s) |
| `title` | `title` | Document title |
| `year` | `issued` | Publication year |
| `date` | `issued` | Publication date |
| `journal` | `container-title` | Journal name |
| `volume` | `volume` | Volume number |
| `issue` | `issue` | Issue number |
| `pages` | `page` | Page range |
| `doi` | `DOI` | Digital Object Identifier |
| `url` | `URL` | Web address |
| `publisher` | `publisher` | Publisher name |
| `location` | `publisher-place` | Publication location |
| `abstract` | `abstract` | Abstract text |

### Example CSV Structure

```csv
author,title,year,journal,volume,issue,pages,doi
"Smith, John",Understanding Topic Models,2023,Journal of AI,15,3,45-67,10.1234/jai.2023
"Jones, Mary; Davis, Bob",Advanced NLP Techniques,2024,Computational Linguistics,28,1,12-34,10.5678/cl.2024
```

### Author Format

The script handles multiple author formats:

**Single author:**

```csv
"Smith, John"
"John Smith"
```

**Multiple authors (various separators):**

```csv
"Smith, John; Jones, Mary"
"Smith, John and Jones, Mary"
"Smith, John & Jones, Mary"
```

### Date/Year Format

The script is flexible with date formats:

```csv
2023
2023-05-15
May 15, 2023
Spring 2023
```

### CSL JSON Format

If you already have a CSL JSON file, the script can process it to add formatted citations.

#### Valid CSL JSON Structure

Your input JSON must be an array of objects:

```json
[
  {
    "type": "article-journal",
    "id": "smith2023",
    "title": "Understanding Topic Models",
    "author": [
      {
        "family": "Smith",
        "given": "John"
      }
    ],
    "issued": {
      "date-parts": [[2023]]
    },
    "container-title": "Journal of AI"
  }
]
```

#### Required Fields for CSL JSON Input

- Each entry should have a unique `id` (will be auto-generated if missing)
- Each entry should have a `type` (defaults to "article" if missing)
- All standard CSL fields are supported

#### Preserving Existing Citations

If an entry already has a `formatted-citation` field, the script will:

- **Skip it** and preserve the existing formatted citation
- **Not overwrite** with a new citation
- **Report** how many entries were skipped in the output

This allows you to:

- Manually edit specific citations and preserve them
- Re-run the script with a different style without affecting pre-formatted entries
- Mix auto-generated and manually crafted citations

## Output Format

### CSL JSON Structure

Each entry in `bibliography.json` follows CSL JSON format:

```json
{
  "type": "article-journal",
  "id": "understanding_topic_m_smith_2023",
  "title": "Understanding Topic Models",
  "author": [
    {
      "family": "Smith",
      "given": "John"
    }
  ],
  "issued": {
    "date-parts": [[2023]]
  },
  "container-title": "Journal of AI",
  "volume": "15",
  "issue": "3",
  "page": "45-67",
  "DOI": "10.1234/jai.2023",
  "formatted-citation": "Smith, John. 2023. \"Understanding Topic Models\". <em>Journal of AI</em> 15 (3): 45-67. https://doi.org/10.1234/jai.2023"
}
```

### Formatted Citations

The `formatted-citation` field contains the ready-to-display citation string in your chosen style. Note that HTML tags for italics are added for display in DFR Browser 2.

**Chicago Author-Date:**

```html
Smith, John. 2023. "Understanding Topic Models". <em>Journal of AI</em> 15 (3): 45-67.
```

**APA:**

```html
Smith, J. (2023). Understanding topic models. <em>Journal of AI</em>, 15(3), 45-67.
```

**MLA:**

```html
Smith, John. "Understanding Topic Models." <em>Journal of AI</em>, vol. 15, no. 3, 2023, pp. 45-67.
```

## Citation Generation Process

### Priority Order

The script generates citations using this priority:

1. **Pre-formatted citation** - If your CSV has a `formatted_citation` column, it uses that directly
2. **Citeproc-py formatting** - Automatted formatting using the specified citation style and the CSL-JSON fields
3. **Fallback formatting** - Simple author-year-title format if `citeproc-py` fails

### Fallback Citation Format

If both conditions (1) and (2) fail, the script creates a basic fallback citation:

**With author:**

```html
Smith, John. 2023. <em>Understanding Topic Models</em>.
```

**Without author:**

```html
<em>Understanding Topic Models</em>. 2023.
```

### When Fallback Occurs

Fallback formatting is used when:

- `citeproc-py` is not installed
- The citation style cannot be loaded
- Date parsing returns a `literal` value (e.g., "Spring 2023") instead of `date_parts`
- Author parsing fails completely
- An unexpected error occurs during formatting

## Troubleshooting

### Debug Mode

Enable debug output to see detailed processing information:

```bash
python create_bibliography.py --debug
```

Debug mode shows:

- Each entry being processed
- Field validation and mapping
- Citation formatting attempts and failures
- Sample output entry

### Common Issues

**Problem:** "No entries were converted"

- **Solution:** Check that your CSV file exists and has a `title` column

**Problem:** "Expected a JSON array" error

- **Solution:** CSL JSON input must be an array `[...]`, not an object `{...}`

**Problem:** Citations look plain (no formatting)

- **Solution:** Install citeproc-py: `pip install citeproc-py`

**Problem:** "Failed to format citation" warnings

- **Solution:** Check date and author formats in your CSV. The script will use fallback formatting.

**Problem:** Missing year in citations

- **Solution:** Ensure your CSV has a `year` or `date` column with valid dates

**Problem:** Authors not displaying correctly

- **Solution:** Use standard formats like "Last, First" or "First Last"

**Problem:** Some citations not being formatted (JSON input)

- **Solution:** Check debug output - entries with existing `formatted-citation` fields are preserved and skipped

### Field Validation

The script automatically:

- Drops invalid CSL fields
- Excludes common non-bibliographic fields (`docnum`, `filename`, etc.)
- Generates unique IDs for each entry
- Provides default values for required fields

Fields that are automatically excluded:

```python
EXCLUDED_FIELDS = {
    "docnum", "docname", "documentnumber", "documentname",
    "doc_num", "doc_name", "index", "idx", "row", "record",
    "entry", "item", "filename", "file_name", "path",
    "filepath", "file_path"
}
```

## Advanced Usage

### Custom Field Mapping

The script automatically maps common CSV column names to CSL fields. You can add custom mappings by modifying the `CSL_FIELD_MAPPING` dictionary in the script.

### Batch Processing

Process multiple metadata files:

```bash
#!/bin/bash
for dir in data/*/; do
    python create_bibliography.py \
        --input "${dir}metadata.csv" \
        --output "${dir}bibliography.json" \
        --style chicago
done
```

### Re-styling Existing Bibliographies

Change citation styles for existing CSL JSON files:

```bash
# Convert from Chicago to APA
python create_bibliography.py \
    --input bibliography_chicago.json \
    --output bibliography_apa.json \
    --style apa

# Convert from MLA to Harvard
python create_bibliography.py \
    --input bibliography_mla.json \
    --output bibliography_harvard.json \
    --style harvard
```

**Note:** This preserves entries that already have formatted citations. To force re-formatting, remove the `formatted-citation` field from entries first.

### Workflow: CSV → Multiple Citation Styles

Generate bibliographies in multiple styles from one CSV:

```python
from create_bibliography import create_bibliography

styles = ["chicago", "apa", "mla", "harvard"]

for style in styles:
    create_bibliography(
        input="metadata.csv",
        output=f"bibliography_{style}.json",
        style=style,
        debug=False
    )
    print(f"✓ Created bibliography with {style} style")
```

### Integration with DFR Browser Workflow

Typical workflow:

1. **Prepare metadata:**

   ```bash
   # Your metadata.csv should be in `yourproject/data`
   ```

2. **Generate bibliography:**

   ```bash
   cd yourproject/bin
   python create_bibliography.py \
       --input ../data/metadata.csv \
       --output ../data/bibliography.json \
       --style chicago
   ```

3. **Update config:**

    In `yourproject/config.json`, make sure you have the following entry:

   ```json
   "bibliography": {
    "path": "data/bibliography.json",
    "style": "chicago", // Or whatever style you used for your bibliography
    "locale": "en-US"
   }
   ```

## Performance

### Processing Speed

- **~1000 entries/second** without citation formatting
- **~100-500 entries/second** with citeproc-py formatting (depends on citation style complexity)

### Large Datasets

For datasets with >10,000 entries:

- Progress is reported every 1,000 entries
- Use `--debug` sparingly (only for troubleshooting samples)
- Consider processing in batches if memory is limited

## Output Statistics

After processing, the script reports different statistics based on input type:

### CSV Input

```bash
Read 5000 records from metadata.csv
Processed 5000 entries...
Successfully converted 4998 entries to CSL format
Warning: 2 entries failed conversion
Successfully formatted 4950 citations using style 'chicago-author-date'
Warning: 48 entries failed citation formatting
Saved 4998 CSL entries to bibliography.json
```

This tells you:

- **Total records** read from CSV
- **Successful conversions** to CSL format
- **Failed conversions** (entries that couldn't be processed at all)
- **Formatted citations** (entries with professional citation strings)
- **Failed formatting** (entries using fallback citations)

### JSON Input

```bash
Processing CSL JSON file: bibliography.json
Using citation style: apa
Loaded 1000 CSL entries from bibliography.json
Successfully processed 1000 entries
Found 150 entries with existing formatted citations
Successfully formatted 820 new citations using style 'apa'
Warning: 30 entries failed citation formatting
Saved 1000 CSL entries to bibliography_formatted.json
```

This tells you:

- **Total entries** loaded from JSON
- **Already formatted** (entries that were skipped/preserved)
- **New formatted citations** (entries that were processed)
- **Failed formatting** (entries that couldn't be formatted)

## Best Practices

### 1. Data Preparation

**For CSV files:**

- Ensure consistent author formatting across your CSV
- Use standard date formats (YYYY-MM-DD or YYYY)
- Include DOIs and URLs when available
- Validate your CSV before processing (check for encoding issues)

**For CSL JSON files:**

- Ensure JSON is valid and formatted as an array
- Verify all entries have unique `id` fields
- Check that date fields use proper CSL format (`date-parts` or `literal`)
- Backup your file before processing

### 2. Citation Quality

- Install `citeproc-py` for professional formatting
- Choose an appropriate citation style for your discipline
- Review a sample of formatted citations before deploying
- Use `--debug` to identify problematic entries

## Examples

### Example 1: Basic Conversion

```bash
# Convert with default settings (Chicago style)
python create_bibliography.py
```

### Example 2: APA Style for Psychology Journal

```bash
python create_bibliography.py \
    --input psychology_metadata.csv \
    --output psychology_bibliography.json \
    --style apa
```

### Example 3: MLA Style with Debug Output

```bash
python create_bibliography.py \
    --input literature_metadata.csv \
    --output literature_bibliography.json \
    --style mla \
    --debug
```

### Example 4: Adding Citations to Existing CSL JSON

```bash
python create_bibliography.py \
    --input my_bibliography.json \
    --output my_bibliography_with_citations.json \
    --style chicago
```

### Example 5: Re-styling an Existing Bibliography

```bash
# Convert existing Chicago bibliography to APA
python create_bibliography.py \
    --input bibliography_chicago.json \
    --output bibliography_apa.json \
    --style apa
```

### Example 6: Processing Multiple Projects

```python
from create_bibliography import create_bibliography

projects = [
    ("psychology", "apa"),
    ("history", "chicago"),
    ("literature", "mla")
]

for project, style in projects:
    create_bibliography(
        input=f"../data/{project}/metadata.csv",
        output=f"../data/{project}/bibliography.json",
        style=style,
        debug=False
    )
    print(f"✓ Processed {project}")
```

### Example 7: Round-Trip Processing

```python
# 1. Create CSL JSON from CSV
create_bibliography(
    input="metadata.csv",
    output="bibliography_raw.json",
    style="chicago",
    debug=False
)

# 2. Manually edit some citations in bibliography_raw.json

# 3. Re-process with different style (preserves manual edits)
create_bibliography(
    input="bibliography_raw.json",
    output="bibliography_final.json",
    style="apa",  # Only formats entries without existing citations
    debug=False
)
```

## Use Cases

### Use Case 1: Initial Bibliography Creation

**Scenario:** You have a CSV of metadata and need to create a bibliography.

```bash
python create_bibliography.py \
    --input metadata.csv \
    --output bibliography.json \
    --style chicago
```

### Use Case 2: Changing Citation Styles

**Scenario:** You want to offer your bibliography in multiple citation styles.

```bash
# Create APA version
python create_bibliography.py \
    --input bibliography.json \
    --output bibliography_apa.json \
    --style apa

# Create MLA version
python create_bibliography.py \
    --input bibliography.json \
    --output bibliography_mla.json \
    --style mla
```

### Use Case 3: Incremental Updates

**Scenario:** You've added new entries to your CSV and want to update the bibliography.

```bash
# Re-process the entire CSV
python create_bibliography.py \
    --input metadata_updated.csv \
    --output bibliography.json \
    --style chicago
```

### Use Case 4: Manual Citation Editing

**Scenario:** Most citations are fine, but a few need manual tweaking.

1. Generate bibliography normally
2. Manually edit specific `formatted-citation` fields in the JSON
3. Re-run script - manual edits will be preserved

```bash
# After manual edits, this won't overwrite your changes
python create_bibliography.py \
    --input bibliography.json \
    --output bibliography.json \
    --style chicago
```

## Support and Resources

### CSL Resources

- **CSL Specification:** [https://citationstyles.org/](https://citationstyles.org/)
- **Zotero Style Repository:** [https://www.zotero.org/styles](https://www.zotero.org/styles)
- **CSL Styles Repository:** [https://github.com/citation-style-language/styles](https://github.com/citation-style-language/styles)
- **CSL Style Search:** [https://editor.citationstyles.org/searchByName/](https://editor.citationstyles.org/searchByName/)

### Citeproc-py Documentation

- **GitHub:** [https://github.com/brechtm/citeproc-py](https://github.com/brechtm/citeproc-py)
- **PyPI:** [https://pypi.org/project/citeproc-py/](https://pypi.org/project/citeproc-py/)
