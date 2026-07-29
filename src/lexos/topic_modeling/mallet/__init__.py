"""__init__.py.

Last Updated: 2026-06-19
Last Tested: 2026-06-20
"""

from .mallet import (
    MALLET_BINARY_PATH,
    Mallet,
    import_docs,
    import_files,
    read_dirs,
    read_file,
)

__all__ = [
    "MALLET_BINARY_PATH",
    "Mallet",
    "import_docs",
    "import_files",
    "read_dirs",
    "read_file",
]
