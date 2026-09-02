"""Validate DOI consistency in README citation information.

This check ensures the DOI shown in markdown text matches the DOI in the link.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOI_RE = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
CONCEPT_DOI_RE = r"10\.\d{4,9}/zenodo\.\d+"
DOI_LINK_RE = re.compile(
    rf"\[(?:doi:)?(?P<text>{DOI_RE})\]\(https?://(?:dx\.)?doi\.org/(?P<link>{DOI_RE})\)",
    re.IGNORECASE,
)


def main() -> int:
    """Run DOI consistency checks on README markdown."""
    readme_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("README.md")
    if not readme_path.exists():
        print(f"ERROR: README file not found: {readme_path}")
        return 1

    content = readme_path.read_text(encoding="utf-8")
    marker = "## 📝 Citation Information"
    section = content.split(marker, maxsplit=1)[1] if marker in content else content
    matches = list(DOI_LINK_RE.finditer(section))

    if not matches:
        print("ERROR: No DOI markdown link found in README citation section.")
        return 1

    has_error = False
    for match in matches:
        text_doi = match.group("text").lower()
        link_doi = match.group("link").lower()

        if not re.fullmatch(CONCEPT_DOI_RE, text_doi, flags=re.IGNORECASE):
            has_error = True
            print(
                "ERROR: Citation DOI must be a Zenodo concept DOI in README: "
                f"found={text_doi}"
            )

        if text_doi != link_doi:
            has_error = True
            print(
                "ERROR: DOI text/link mismatch in README citation section: "
                f"text={text_doi} link={link_doi}"
            )

    if has_error:
        return 1

    print("README DOI check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
