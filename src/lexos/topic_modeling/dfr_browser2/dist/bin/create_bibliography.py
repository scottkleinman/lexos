"""create_bibliography.py.

Reads metadata.csv or CSL JSON and converts it to a file in Citation Style Language (CSL) JSON format.
Each entry is formatted with citeproc-py and includes the formatted citation string.
Outputs to bibliography.json in the same directory.

Requirements:
  - pandas
  - python-dateutil
  - nameparser
  - citeproc-py

To use as a module:

```python
from create_bibliography import create_bibliography

# From CSV
create_bibliography(
    "metadata.csv", "bibliography.json", "chicago-author-date", debug=False
)

# From existing CSL JSON (adds formatted citations)
create_bibliography(
    "bibliography.json", "bibliography_formatted.json", "apa", debug=False
)
```
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from dateutil import parser as date_parser
from nameparser import HumanName

# Import citeproc-py components (optional dependency)
try:
    from citeproc import (
        Citation,
        CitationItem,
        CitationStylesBibliography,
        CitationStylesStyle,
    )
    from citeproc.source.json import CiteProcJSON

    CITEPROC_AVAILABLE = True
except ImportError:
    CITEPROC_AVAILABLE = False

# Valid CSL JSON fields (comprehensive list)
VALID_CSL_FIELDS = {
    # Core fields
    "type",
    "id",
    "title",
    "author",
    "editor",
    "translator",
    "container-title",
    "collection-title",
    "container-author",
    "container-editor",
    "publisher",
    "publisher-place",
    "page",
    "volume",
    "issue",
    "number",
    "number-of-pages",
    "number-of-volumes",
    "edition",
    "version",
    "section",
    "chapter-number",
    # Date fields
    "issued",
    "accessed",
    "submitted",
    "event-date",
    "original-date",
    # Identifier fields
    "ISBN",
    "ISSN",
    "DOI",
    "URL",
    "PMID",
    "PMCID",
    "call-number",
    # Additional metadata
    "abstract",
    "note",
    "annote",
    "keyword",
    "language",
    "original-title",
    "original-publisher",
    "original-publisher-place",
    "references",
    "source",
    "status",
    "genre",
    "medium",
    "archive",
    "archive-place",
    "archive_location",
    "authority",
    "citation-key",
    "citation-label",
    "collection-number",
    "container-title-short",
    "dimensions",
    "event",
    "event-place",
    "first-reference-note-number",
    "jurisdiction",
    "scale",
    "title-short",
    "year-suffix",
    # Legal and special fields
    "court",
    "document-name",
    "legislation-number",
    "legal-status",
    "history",
    "supplement",
    "locator",
    "reviewed-title",
    "reviewed-author",
    # Custom field for formatted citation
    "formatted-citation",
}

# CSL field mapping for common bibliographic fields
CSL_FIELD_MAPPING = {
    "title": "title",
    "journal": "container-title",
    "publication": "container-title",
    "publisher": "publisher",
    "volume": "volume",
    "issue": "issue",
    "number": "issue",
    "pages": "page",
    "page": "page",
    "doi": "DOI",
    "isbn": "ISBN",
    "issn": "ISSN",
    "url": "URL",
    "abstract": "abstract",
    "keywords": "keyword",
    "language": "language",
    "edition": "edition",
    "series": "collection-title",
    "location": "publisher-place",
    "city": "publisher-place",
}

# Fields that should be excluded (common CSV fields that are not CSL fields)
EXCLUDED_FIELDS = {
    "docnum",
    "docname",
    "documentnumber",
    "documentname",
    "doc_num",
    "doc_name",
    "index",
    "idx",
    "row",
    "record",
    "entry",
    "item",
    "filename",
    "file_name",
    "path",
    "filepath",
    "file_path",
}


def parse_authors(author_string: str) -> list:
    """Parse author string into CSL author format.

    Handles multiple authors separated by common delimiters.

    Args:
        author_string (str): String containing one or more author names

    Returns:
        list: List of author objects in CSL format
    """
    if not author_string or pd.isna(author_string):
        return []

    author_string = author_string.strip()

    # Detect the format and split accordingly
    authors = []

    # Common patterns for multiple authors
    # Priority order: semicolons, " and ", " & ", then newlines
    if ";" in author_string:
        # Semicolon is unambiguous - always separates authors
        authors = [a.strip() for a in author_string.split(";") if a.strip()]
    elif " and " in author_string.lower():
        # " and " separates authors
        authors = [
            a.strip()
            for a in re.split(r"\s+and\s+", author_string, flags=re.IGNORECASE)
            if a.strip()
        ]
    elif " & " in author_string:
        # Ampersand separates authors
        authors = [a.strip() for a in author_string.split(" & ") if a.strip()]
    elif "\n" in author_string:
        # Newlines separate authors
        authors = [a.strip() for a in author_string.split("\n") if a.strip()]
    else:
        # No clear multi-author separator found
        # Check if we have multiple commas (indicating "Last, First; Last, First" without semicolons)
        comma_count = author_string.count(",")

        if comma_count == 1:
            # Single comma - likely "Last, First" format for one author
            authors = [author_string]
        elif comma_count > 1:
            # Multiple commas - could be multiple "Last, First" authors separated by commas
            # This is ambiguous, but we'll try to detect the pattern
            # Look for ", " followed by a capital letter (likely a new last name)
            # Pattern: "Smith, John, Jones, Mary" should split after "John," not after "Smith,"

            # Try to split on comma followed by space and capital letter
            # But preserve "Last, First" structure
            potential_authors = re.split(r",\s+(?=[A-Z])", author_string)

            # Check if we have pairs that look like "Last First" after splitting
            if len(potential_authors) > 1 and all(
                "," not in a for a in potential_authors[1:]
            ):
                # If none of the subsequent splits have commas, they're likely first names
                # Recombine them as "Last, First" pairs
                authors = []
                for i in range(0, len(potential_authors), 2):
                    if i + 1 < len(potential_authors):
                        authors.append(
                            f"{potential_authors[i]}, {potential_authors[i + 1]}"
                        )
                    else:
                        authors.append(potential_authors[i])
            else:
                # Can't determine pattern, treat as single author
                authors = [author_string]
        else:
            # No commas - single author in "First Last" format
            authors = [author_string]

    # Parse each author name
    parsed_authors = []
    for author in authors:
        if not author:
            continue

        try:
            # Use HumanName parser which handles both "First Last" and "Last, First"
            name = HumanName(author.strip())

            author_obj = {}
            if name.last and name.last.strip():
                author_obj["family"] = name.last.strip()
            if name.first and name.first.strip():
                author_obj["given"] = name.first.strip()
            if name.middle and name.middle.strip():
                # Add middle name to given name
                if "given" in author_obj:
                    author_obj["given"] = f"{author_obj['given']} {name.middle.strip()}"
                else:
                    author_obj["given"] = name.middle.strip()
            if name.suffix and name.suffix.strip():
                author_obj["suffix"] = name.suffix.strip()

            # Only add if we have at least a family name
            if "family" in author_obj:
                parsed_authors.append(author_obj)
            else:
                # No family name found - use literal
                parsed_authors.append({"literal": author.strip()})
        except Exception:
            # Fallback to literal name
            parsed_authors.append({"literal": author.strip()})

    return parsed_authors


def parse_date(date_string: str) -> dict | None:
    """Parse date string into CSL date format.

    Args:
        date_string (str): Date string in various formats

    Returns:
        dict: CSL date object with date-parts
    """
    if not date_string or pd.isna(date_string):
        return None

    # If it's just a year
    if re.match(r"^\d{4}$", str(date_string).strip()):
        return {"date-parts": [[int(date_string)]]}

    try:
        # Try to parse as date
        parsed_date = date_parser.parse(str(date_string))
        return {"date-parts": [[parsed_date.year, parsed_date.month, parsed_date.day]]}
    except Exception:
        # If parsing fails, try to extract year
        year_match = re.search(r"\b(\d{4})\b", str(date_string))
        if year_match:
            return {"date-parts": [[int(year_match.group(1))]]}
        elif date_string is not None:
            return {"literal": date_string}
        else:
            return None


def normalize_field_name(field_name: str) -> str:
    """Normalize field names to lowercase and handle common variations.

    Args:
        field_name (str): Original field name

    Returns:
        str: Normalized field name
    """
    return field_name.lower().strip().replace(" ", "").replace("_", "").replace("-", "")


def validate_csl_entry(entry: dict, debug: bool = False):
    """Validate and clean a CSL entry to ensure it's properly formatted.

    Args:
        entry (dict): CSL entry to validate
        debug (bool): Enable debug output

    Returns:
        dict: Validated and cleaned CSL entry
    """
    if debug:
        print(f"  Validating entry with keys: {list(entry.keys())}")

    # Start with a clean entry containing only valid CSL fields
    cleaned_entry = {}

    # Required fields
    if "type" not in entry:
        cleaned_entry["type"] = "article-journal"  # Default type
    else:
        cleaned_entry["type"] = entry["type"]

    if "id" not in entry or not entry["id"]:
        # Generate a reliable ID
        title_part = (
            str(entry.get("title", "untitled"))[:20]
            .replace(" ", "_")
            .replace(",", "")
            .replace(".", "")
        )
        author_part = "unknown"
        if isinstance(entry.get("author"), list) and entry.get("author"):
            author_part = str(entry["author"][0].get("family", "unknown"))[:10].replace(
                " ", "_"
            )
        year_part = "nodate"
        if entry.get("issued"):
            year_part = str(entry["issued"].get("date-parts", [[0]])[0][0])
        cleaned_entry["id"] = f"{title_part}_{author_part}_{year_part}".lower()
    else:
        cleaned_entry["id"] = entry["id"]

    # Ensure ID is valid (no spaces, special characters that might cause issues)
    cleaned_entry["id"] = re.sub(r"[^\w\-_]", "", str(cleaned_entry["id"]))
    if not cleaned_entry["id"]:  # If ID becomes empty after cleaning
        cleaned_entry["id"] = f"item_{id(entry)}"  # Use memory address as fallback

    # Ensure title exists
    if "title" not in entry or not entry["title"]:
        cleaned_entry["title"] = "Untitled"
    else:
        cleaned_entry["title"] = str(entry["title"]).strip()

    # Copy only valid CSL fields from the original entry
    for key, value in entry.items():
        # Skip keys we've already processed
        if key in ["type", "id", "title"]:
            continue

        if key in VALID_CSL_FIELDS and value is not None and value != "":
            if isinstance(value, str):
                value = value.strip()
                if value:  # Only add non-empty strings
                    cleaned_entry[key] = value
            else:
                cleaned_entry[key] = value
        elif key not in EXCLUDED_FIELDS and debug:
            # Don't report common excluded fields, but report others for debugging
            print(f"    Dropping field '{key}' (not in valid CSL fields)")

    return cleaned_entry


def format_citation_with_citeproc(
    csl_entry: dict, style: str = "chicago-author-date", debug: bool = False
) -> str:
    """Format a single CSL entry as a citation using citeproc-py.

    Args:
        csl_entry (dict): CSL entry to format
        style (str): Citation style to use
        debug (bool): Enable debug output

    Returns:
        str: Formatted citation string or None if formatting fails
    """
    if not CITEPROC_AVAILABLE:
        if debug:
            print(
                f"Warning: citeproc-py not available, skipping citation formatting for {csl_entry.get('id', 'unknown')}"
            )
        return None

    # Try to use actual citeproc-py formatting first
    try:
        if debug:
            print(
                f"Attempting citeproc-py formatting for {csl_entry.get('id', 'unknown')}"
            )

        # Create a bibliography source with just this entry
        bib_source = CiteProcJSON([csl_entry])

        # Load citation style
        try:
            bib_style = CitationStylesStyle(style, validate=False)
            if debug:
                print(f"  ✓ Loaded style: {style}")
        except Exception as e:
            if debug:
                print(f"  ❌ Could not load style '{style}': {e}")
                print("  ↳ Trying chicago-author-date as fallback")
            bib_style = CitationStylesStyle("chicago-author-date", validate=False)

        # Create bibliography
        bibliography = CitationStylesBibliography(bib_style, bib_source)

        # Correct method: Create Citation object with CitationItem
        citation = Citation([CitationItem(csl_entry["id"])])

        # Register the citation
        bibliography.register(citation)

        # Generate bibliography
        bibliography_items = bibliography.bibliography()

        # Generate the formatted citation
        if bibliography_items and len(bibliography_items) > 0:
            citation_str = str(bibliography_items[0]).strip()
            # Remove any HTML tags and decode HTML entities
            citation_str = re.sub(r"<[^>]+>", "", citation_str)
            citation_str = (
                citation_str.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
            )
            # Fix double periods (common in APA style after initials)
            citation_str = re.sub(r"\.\.+", ".", citation_str)
            if debug:
                print(f"  ✓ Generated citation: {citation_str}")
            return citation_str
        else:
            if debug:
                print("  ❌ No bibliography items generated")
            raise Exception("No bibliography items generated")

    except Exception as citeproc_error:
        if debug:
            print(f"  ❌ citeproc-py formatting failed: {citeproc_error}")
            print("  ↳ Using fallback citation format")

    # Fallback to simple citation format
    if debug:
        print(f"Creating fallback citation for {csl_entry.get('id', 'unknown')}")

    try:
        # Create a simple citation format as fallback
        citation_parts = []

        # Add authors
        if "author" in csl_entry and csl_entry["author"]:
            authors = csl_entry["author"]
            if isinstance(authors, list) and len(authors) > 0:
                if len(authors) == 1:
                    author = authors[0]
                    if "family" in author:
                        # Format as "Family, Given" for single author
                        author_name = author["family"]
                        if "given" in author:
                            author_name = f"{author['family']}, {author['given']}"
                        citation_parts.append(author_name)
                elif len(authors) <= 3:
                    author_names = []
                    for author in authors:
                        if "family" in author:
                            # Format as "Family, Given" for multiple authors
                            author_name = author["family"]
                            if "given" in author:
                                author_name = f"{author['family']}, {author['given']}"
                            author_names.append(author_name)
                    citation_parts.append("; ".join(author_names))
                else:
                    if "family" in authors[0]:
                        # Format first author with given name, then "et al."
                        first_author = authors[0]["family"]
                        if "given" in authors[0]:
                            first_author = (
                                f"{authors[0]['family']}, {authors[0]['given']}"
                            )
                        citation_parts.append(f"{first_author} et al.")

        # Add year
        if "issued" in csl_entry and csl_entry["issued"]:
            if "date-parts" in csl_entry["issued"]:
                date_parts = csl_entry["issued"]["date-parts"]
                if date_parts and len(date_parts[0]) > 0:
                    citation_parts.append(f"({date_parts[0][0]})")
            elif "literal" in csl_entry["issued"]:
                citation_parts.append(f"({csl_entry['issued']['literal']})")

        # Add title
        if "title" in csl_entry:
            citation_parts.append(f'"{csl_entry["title"]}"')

        # Add container
        if "container-title" in csl_entry:
            citation_parts.append(f"{csl_entry['container-title']}")

        fallback_citation = ". ".join(citation_parts) + "."

        if debug:
            print(f"Created fallback citation: {fallback_citation}")

        return fallback_citation

    except Exception as e:
        if debug:
            print(
                f"Error creating fallback citation for {csl_entry.get('id', 'unknown')}: {e}"
            )
        return None


def convert_metadata_to_csl(
    metadata_file: str, style: str = "chicago-author-date", debug: bool = False
) -> list:
    """
    Convert metadata.csv to CSL JSON format with formatted citations.

    Args:
        metadata_file (str): Path to metadata.csv file
        style (str): Citation style for formatting
        debug (bool): Enable debug output

    Returns:
        list: List of CSL entries with formatted citations
    """
    try:
        # Read CSV file
        df = pd.read_csv(metadata_file)
        print(f"Read {len(df)} records from {metadata_file}")

        # Convert each row to CSL format
        csl_entries = []
        failed_conversions = 0
        failed_formatting = 0

        for index, row in df.iterrows():
            try:
                # Create CSL entry
                entry = {
                    "type": "article-journal",  # Default type for journal articles
                    "id": f"item_{index + 1}",
                }

                # Parse authors
                if "author" in row and pd.notna(row["author"]):
                    authors = parse_authors(row["author"])
                    if authors:
                        entry["author"] = authors

                # Add title
                if "title" in row and pd.notna(row["title"]):
                    entry["title"] = str(row["title"]).strip()
                else:
                    entry["title"] = f"Document {index + 1}"  # Fallback title

                # Parse date/year
                if "year" in row and pd.notna(row["year"]):
                    parsed_date = parse_date(row["year"])
                    if parsed_date:
                        entry["issued"] = parsed_date
                elif "date" in row and pd.notna(row["date"]):
                    parsed_date = parse_date(row["date"])
                    if parsed_date:
                        entry["issued"] = parsed_date

                # Map other fields
                for col in df.columns:
                    if col.lower() in [
                        "author",
                        "title",
                        "year",
                        "date",
                        "docnum",
                        "docname",
                    ]:
                        continue  # Already processed or excluded

                    if pd.notna(row[col]):
                        normalized_col = normalize_field_name(col)

                        # Check if it maps to a CSL field
                        csl_field = None
                        for csv_field, csl_name in CSL_FIELD_MAPPING.items():
                            if normalized_col == normalize_field_name(csv_field):
                                csl_field = csl_name
                                break

                        if csl_field:
                            entry[csl_field] = str(row[col]).strip()
                        elif col.lower() not in EXCLUDED_FIELDS:
                            # Keep as custom field (normalized)
                            field_name = col.lower().replace(" ", "-")
                            entry[field_name] = str(row[col]).strip()

                # Validate the CSL entry
                validated_entry = validate_csl_entry(entry, debug)

                # Check metadata for a formatted citation
                if "formatted_citation" in entry:
                    formatted_citation = entry["formatted_citation"]
                # Format citation with citeproc-py if not in metadata
                else:
                    formatted_citation = format_citation_with_citeproc(
                        validated_entry, style, debug
                    )
                if formatted_citation:
                    validated_entry["formatted-citation"] = formatted_citation
                else:
                    failed_formatting += 1
                    if validated_entry["author"] and validated_entry["author"] != "":
                        validated_entry["formatted-citation"] = (
                            f"{validated_entry['author']}. {validated_entry['year']}. <em>{validated_entry['title']}</em>."
                        )
                    else:
                        validated_entry["formatted-citation"] = (
                            f"<em>{validated_entry['title']}</em>. {validated_entry['year']}."
                        )
                    if debug:
                        print(
                            f"Failed to format citation for entry {index + 1}. Using fallback {validated_entry['formatted-citation']}."
                        )

                csl_entries.append(validated_entry)

                # Progress reporting
                if debug and (index + 1) % 100 == 0:
                    print(f"Processed {index + 1} entries...")
                elif (index + 1) % 1000 == 0:
                    print(f"Processed {index + 1} entries...")

            except Exception:
                failed_conversions += 1
                if debug:
                    print(f"Error processing row {index + 1}")
                continue

        print(f"Successfully converted {len(csl_entries)} entries to CSL format")
        if failed_conversions > 0:
            print(f"Warning: {failed_conversions} entries failed conversion")
        if failed_formatting > 0:
            print(f"Warning: {failed_formatting} entries failed citation formatting")

        return csl_entries

    except Exception as e:
        print(f"Error processing metadata file: {e}")
        return None


def process_csl_json(
    input_file: str,
    style: str = "chicago-author-date",
    debug: bool = False,
):
    """Process existing CSL JSON file and add formatted citations.

    Args:
        input_file (str): Path to input CSL JSON file
        style (str): Citation style for formatting
        debug (bool): Enable debug output

    Returns:
        list: List of CSL entries with formatted citations
    """
    try:
        # Load existing CSL JSON
        with open(input_file, "r", encoding="utf-8") as f:
            csl_entries = json.load(f)

        if not isinstance(csl_entries, list):
            print(f"Error: Expected a JSON array, got {type(csl_entries).__name__}")
            return None

        print(f"Loaded {len(csl_entries)} CSL entries from {input_file}")

        if debug:
            print(f"Processing with citation style: {style}")

        formatted_count = 0
        failed_formatting = 0
        already_formatted = 0

        # Process each entry
        for i, entry in enumerate(csl_entries, 1):
            if debug and i % 100 == 0:
                print(f"Processing entry {i}/{len(csl_entries)}...")

            # Check if entry already has a formatted citation
            if "formatted-citation" in entry:
                already_formatted += 1
                if debug:
                    print(f"Entry {i} already has formatted citation, skipping")
                continue

            # Ensure entry has required fields
            if "id" not in entry:
                # Generate an ID if missing
                entry["id"] = f"entry_{i}"

            if "type" not in entry:
                entry["type"] = "article"  # Default type

            # Format citation
            try:
                formatted_citation = format_citation_with_citeproc(entry, style, debug)
                if formatted_citation:
                    entry["formatted-citation"] = formatted_citation
                    formatted_count += 1
                else:
                    failed_formatting += 1
                    if debug:
                        print(
                            f"Failed to format citation for entry {i}: {entry.get('id', 'unknown')}"
                        )
            except Exception as e:
                failed_formatting += 1
                if debug:
                    print(f"Error formatting entry {i}: {e}")

        # Print summary
        print(f"Successfully processed {len(csl_entries)} entries")
        if already_formatted > 0:
            print(
                f"Found {already_formatted} entries with existing formatted citations"
            )
        if formatted_count > 0:
            print(
                f"Successfully formatted {formatted_count} new citations using style '{style}'"
            )
        if failed_formatting > 0:
            print(f"Warning: {failed_formatting} entries failed citation formatting")

        return csl_entries

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        return None
    except Exception as e:
        print(f"Error processing CSL JSON file: {e}")
        return None


def create_bibliography(
    input: str = "metadata.csv",
    output: str = "bibliography.json",
    style: str = "chicago-author-date",
    debug: bool = False,
):
    """Process the metadata.

    Args:
        input (str): Input metadata CSV or CSL JSON file
        output (str): Output JSON file
        style (str): Citation style for formatting
        debug (bool): Enable debug output for troubleshooting
    """
    # Map common style shortcuts to their full names
    style_mapping = {
        "chicago": "chicago-author-date",
        "chicago-note": "chicago-note-bibliography",
        "mla": "modern-language-association",
        "apa": "apa",
        "harvard": "harvard-cite-them-right",
    }

    # Apply style mapping if the user provided a shortcut
    original_style = style
    if style.lower() in style_mapping:
        style = style_mapping[style.lower()]
        if debug:
            print(f"Mapped style '{original_style}' to '{style}'")

    # Get the directory of this script
    script_dir = Path(__file__).parent

    # Construct paths relative to script directory
    input_file = script_dir / input
    output_file = script_dir / output

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        return

    # Check if citeproc-py is available
    if not CITEPROC_AVAILABLE:
        print("WARNING: citeproc-py is not installed!")
        print(
            "Citations will be converted to CSL JSON format but will NOT include formatted citation strings."
        )
        print(
            "To enable citation formatting, install citeproc-py with: pip install citeproc-py"
        )
        print("")

    # Detect input file format
    is_json_input = input_file.suffix.lower() == ".json"

    if is_json_input:
        print(f"Processing CSL JSON file: {input_file}")
        print(f"Using citation style: {style}")
        csl_entries = process_csl_json(str(input_file), style, debug)
    else:
        print(f"Converting {input_file} to CSL JSON format...")
        print(f"Using citation style: {style}")
        # Convert metadata to CSL
        csl_entries = convert_metadata_to_csl(str(input_file), style, debug)

    if not csl_entries:
        print("Error: No entries were converted. Exiting.")
        return

    # Save to JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(csl_entries, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(csl_entries)} CSL entries to {output_file}")

        # Print some statistics
        formatted_count = sum(
            1 for entry in csl_entries if "formatted-citation" in entry
        )

        if CITEPROC_AVAILABLE:
            print(
                f"Successfully formatted {formatted_count} citations using style '{style}'"
            )
            if formatted_count == 0:
                print(
                    "WARNING: No citations were successfully formatted. Check debug output for errors."
                )
        else:
            print("Citation formatting was skipped (citeproc-py not installed)")
            print(
                "CSL JSON entries were created successfully but without formatted citation strings."
            )

        if debug and csl_entries:
            print("\n=== Sample entry ===")
            sample = csl_entries[0]
            print(json.dumps(sample, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error saving to {output_file}: {e}")


def main():
    """Main function to handle command line arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert metadata CSV or CSL JSON to formatted bibliography with citations"
    )
    parser.add_argument(
        "--style",
        default="chicago-author-date",
        help="Citation style for formatting (default: chicago-author-date)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output for troubleshooting"
    )
    parser.add_argument(
        "--input",
        default="metadata.csv",
        help="Input file: metadata CSV or CSL JSON (default: metadata.csv)",
    )
    parser.add_argument(
        "--output",
        default="bibliography.json",
        help="Output JSON file (default: bibliography.json)",
    )

    args = parser.parse_args()
    create_bibliography(args.input, args.output, args.style, args.debug)


if __name__ == "__main__":
    main()
