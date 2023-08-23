"""test_io_smart.py.

Test file for the io.smart module.

Note: When downloading files from GitHub, you will sometimes get an error
reading "An existing connection was forcibly closed by the remote host."
This will cause a test failure. Generally waiting a minute or so and
re-running will fix the problem.
"""
import pytest

from lexos.io.smart import Loader

# Test data
test_dir = "../test_data"
raw_url = "https://raw.githubusercontent.com/scottkleinman/lexos/main/tests/test_data"

data = [
    (f"{test_dir}/txt/Austen_Pride_sm.txt", 1),
    (f"{raw_url}/txt/Austen_Pride_sm.txt", 1),
    ([f"{test_dir}/txt/Austen_Pride_sm.txt", f"{test_dir}/txt/Austen_Sense_sm.txt"], 2),
    ([f"{raw_url}/txt/Austen_Pride_sm.txt", f"{raw_url}/txt/Austen_Sense_sm.txt"], 2),
    (f"{test_dir}/docx/Austen_Pride_sm.docx", 1),
    (f"{raw_url}/docx/Austen_Pride_sm.docx", 1),
    (
        [
            f"{test_dir}/docx/Austen_Pride_sm.docx",
            f"{test_dir}/docx/Austen_Sense_sm.docx",
        ],
        2,
    ),
    (
        [
            f"{raw_url}/docx/Austen_Pride_sm.docx",
            f"{raw_url}/docx/Austen_Sense_sm.docx",
        ],
        2,
    ),
    (f"{test_dir}/pdf/Austen_Pride_sm.pdf", 1),
    (f"{raw_url}/pdf/Austen_Pride_sm.pdf", 1),
    ([f"{test_dir}/pdf/Austen_Pride_sm.pdf", f"{test_dir}/pdf/Austen_Sense_sm.pdf"], 2),
    ([f"{raw_url}/pdf/Austen_Pride_sm.pdf", f"{raw_url}/pdf/Austen_Sense_sm.pdf"], 2),
    (f"{test_dir}/txt/", 4),
    ("https://github.com/scottkleinman/lexos/tree/main/tests/test_data/txt", 4),
    (f"{test_dir}/zip/txt.zip", 2),
]

# Test functions
@pytest.mark.parametrize("input, expected", data)
def test_load(input, expected):
    """Test loader with txt, docx, pdf, and zip files, as well as directories."""
    loader = Loader()
    loader.load(input)
    assert len(loader.texts) == expected
