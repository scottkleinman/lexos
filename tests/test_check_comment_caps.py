"""Tests for tools.check_comment_caps."""

from importlib import util
from pathlib import Path


def load_check_comment_caps_module():
    """Load the comment-capitalization hook as a module."""
    module_path = (
        Path(__file__).resolve().parents[1] / "tools" / "check_comment_caps.py"
    )
    spec = util.spec_from_file_location("check_comment_caps", module_path)
    module = util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lowercase_continuation_line_is_allowed(tmp_path):
    """A lower-case second line in a full-line comment block should pass."""
    module = load_check_comment_caps_module()
    path = tmp_path / "sample.py"
    path.write_text(
        "# This will process the DTM and hit line 226 multiple times\n"
        "# including cases where count_value > 0 is False (zero counts)\n",
        encoding="utf-8",
    )

    assert module.check_python(path) == []


def test_isolated_lowercase_comment_still_fails(tmp_path):
    """A standalone lower-case full-line comment should still fail."""
    module = load_check_comment_caps_module()
    path = tmp_path / "sample.py"
    path.write_text(
        "# including cases where count_value > 0 is False\n", encoding="utf-8"
    )

    assert module.check_python(path) == [
        (1, "Comment should start with a capital letter")
    ]


def test_indented_comment_block_with_lowercase_continuation_is_allowed(tmp_path):
    """Indented comment-only blocks should allow lower-case wrapped continuation lines."""
    module = load_check_comment_caps_module()
    path = tmp_path / "sample.py"
    path.write_text(
        "def example():\n"
        "    # WARNING: This renders the code unusable if the data contains float counts\n"
        "    # such as topic model distributions. It doesn't seem necessary for any of\n"
        "    # our current use cases, so I'm commenting it out for now.\n"
        "    # Ensure counts are integers\n"
        "    # counts = Counter({k: int(v) for k, v in counts.items()})\n",
        encoding="utf-8",
    )

    assert module.check_python(path) == []
