"""__init__.py.

Last Updated: June 19, 2026
Last Tested: TBD
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
