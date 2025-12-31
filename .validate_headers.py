""".validate_headers.py.

Pre-commit hook to validate and normalize date formats and coverage information in file headers.

Usage:
    python validate_headers.py <filepath>
"""

import re
import sys
from datetime import datetime
from pathlib import Path


def parse_date(date_str: str) -> datetime | None:
    """Parse various date formats into a datetime object.

    Args:
        date_str: The date string to parse

    Returns:
        datetime object or None if parsing fails
    """
    # Try various date formats
    formats = [
        "%B %d, %Y",  # November 27, 2025
        "%Y-%m-%d",  # 2025-11-27
        "%m/%d/%Y",  # 11/27/2025
        "%d/%m/%Y",  # 27/11/2025
        "%b %d, %Y",  # Nov 27, 2025
        "%Y/%m/%d",  # 2025/11/27
        "%d %B %Y",  # 27 November 2025
        "%B %Y",  # November 2025 (day defaults to 1)
        "%Y-%m",  # 2025-11
        "%m-%d-%Y",  # 11-27-2025
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return None


def format_date(dt: datetime) -> str:
    """Format datetime object to 'Month Day, Year' format.

    Args:
        dt: datetime object

    Returns:
        Formatted date string (e.g., "November 27, 2025")
    """
    # Format with non-zero-padded day
    return dt.strftime("%B {day}, %Y").format(day=dt.day)


def validate_coverage_pattern(line: str) -> tuple[bool, str]:
    """Validate coverage pattern in test files.

    Args:
        line: The line containing coverage information

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Pattern: "Coverage: 100%" or "Coverage: X%. Missing: list"
    match_100 = re.search(r"Coverage:\s*100%", line)
    if match_100:
        return True, ""

    match_partial = re.search(r"Coverage:\s*(\d+)%\.\s*Missing:\s*(.+)", line)
    if match_partial:
        percentage = int(match_partial.group(1))
        missing_str = match_partial.group(2).strip()

        # Validate percentage is 0-100
        if not (0 <= percentage <= 100):
            return False, f"Coverage percentage {percentage} is not in range 0-100"

        # Validate missing is a comma-separated list of integers, ranges, or a single integer/range
        # Valid formats: "42", "42, 58", "201-205", "42, 201-205, 300"
        try:
            missing_parts = [part.strip() for part in missing_str.split(",")]
            for part in missing_parts:
                # Check if it's a range (e.g., "201-205")
                if "-" in part:
                    range_parts = part.split("-")
                    if len(range_parts) != 2:
                        raise ValueError(f"Invalid range format: {part}")
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    if start > end:
                        raise ValueError(f"Invalid range (start > end): {part}")
                else:
                    # It's a single integer
                    int(part)  # Will raise ValueError if not an integer
            return True, ""
        except ValueError as e:
            return (
                False,
                f"Missing line numbers must be integers or ranges (e.g., '42', '201-205', '42, 201-205'), got: {missing_str}. Error: {str(e)}",
            )

    return (
        False,
        "Coverage pattern must be 'Coverage: 100%' or 'Coverage: X%. Missing: line_numbers'",
    )


def process_file(filepath: str) -> dict[str, str]:
    """Process a file and validate/normalize its header.

    Args:
        filepath: Path to the file to process

    Returns:
        Dictionary with processing results and any modifications needed
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read first 50 lines
    with open(path, "r", encoding="utf-8") as f:
        lines = [next(f, None) for _ in range(50)]
        lines = [line for line in lines if line is not None]

    results = {
        "filepath": str(path),
        "is_test_file": False,
        "in_tests_folder": "tests" in path.parts,
        "errors": [],
        "warnings": [],
        "modifications": {},
    }

    # Check if it's a test file
    if results["in_tests_folder"] and path.name.startswith("test_"):
        results["is_test_file"] = True

    # Process lines
    found_last_updated = False
    found_last_tested = False
    found_coverage = False

    for i, line in enumerate(lines):
        # Check for "Last Updated: "
        if "Last Updated:" in line:
            found_last_updated = True
            match = re.search(r"Last Updated:\s*(.+?)(?:\n|$)", line)
            if match:
                date_str = match.group(1).strip()
                parsed_date = parse_date(date_str)

                if parsed_date:
                    formatted = format_date(parsed_date)
                    if date_str != formatted:
                        results["modifications"][f"line_{i + 1}_last_updated"] = {
                            "old": date_str,
                            "new": formatted,
                            "line_number": i + 1,
                        }
                else:
                    results["warnings"].append(
                        f"Line {i + 1}: Could not parse date '{date_str}' in 'Last Updated:'"
                    )

        # Check for "Last Tested: " (only for non-test files)
        if not results["in_tests_folder"] and "Last Tested:" in line:
            found_last_tested = True
            match = re.search(r"Last Tested:\s*(.+?)(?:\n|$)", line)
            if match:
                date_str = match.group(1).strip()
                parsed_date = parse_date(date_str)

                if parsed_date:
                    formatted = format_date(parsed_date)
                    if date_str != formatted:
                        results["modifications"][f"line_{i + 1}_last_tested"] = {
                            "old": date_str,
                            "new": formatted,
                            "line_number": i + 1,
                        }
                else:
                    results["warnings"].append(
                        f"Line {i + 1}: Could not parse date '{date_str}' in 'Last Tested:'"
                    )

        # Check for "Coverage: " (only for test files)
        if results["is_test_file"] and "Coverage:" in line:
            found_coverage = True
            is_valid, error_msg = validate_coverage_pattern(line)
            if not is_valid:
                results["errors"].append(f"Line {i + 1}: {error_msg}")

    # Check if Coverage is required but missing
    if results["is_test_file"] and not found_coverage:
        results["errors"].append(
            "Missing required 'Coverage:' pattern in test file header"
        )

    return results


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python validate_headers.py <filepath> [<filepath> ...]")
        sys.exit(1)

    all_passed = True

    # Process each file passed as argument
    for filepath in sys.argv[1:]:
        try:
            results = process_file(filepath)

            print(f"\n{'=' * 60}")
            print(f"Processing: {results['filepath']}")
            print(f"Test file: {results['is_test_file']}")
            print(f"In tests folder: {results['in_tests_folder']}")
            print("=" * 60)

            if results["errors"]:
                print("ERRORS:")
                for error in results["errors"]:
                    print(f"  ❌ {error}")
                print()
                all_passed = False

            if results["warnings"]:
                print("WARNINGS:")
                for warning in results["warnings"]:
                    print(f"  ⚠️  {warning}")
                print()

            if results["modifications"]:
                print("MODIFICATIONS NEEDED:")
                for key, mod in results["modifications"].items():
                    print(f"  Line {mod['line_number']}:")
                    print(f"    Old: {mod['old']}")
                    print(f"    New: {mod['new']}")
                print()
            else:
                print("✅ No modifications needed\n")

        except Exception as e:
            print(f"\nERROR processing {filepath}: {e}\n")
            all_passed = False

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
