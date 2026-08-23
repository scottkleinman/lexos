"""util.py.

Last Updated: August 23, 2026
Last Tested: August 23, 2026
"""


def sanitize_label_text(label: str) -> str:
    """Sanitize data-driven plot labels for whitespace and linebreak visibility."""
    if label is None:
        return ""
    if "\r\n" in label or "\r" in label or "\n" in label:
        label = label.replace("\r\n", "<linebreak>")
        label = label.replace("\r", "<linebreak>").replace("\n", "<linebreak>")
    if " " in label or "\t" in label:
        label = label.replace(" ", "<whitespace>").replace("\t", "<whitespace>")
    return label
