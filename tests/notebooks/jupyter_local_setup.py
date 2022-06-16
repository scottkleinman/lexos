"""jupyter_local_setup.py.

Prepares a Jupyter notebook for local testing.
"""

# Python imports
from typing import Optional
import os
import sys


def main(args: Optional[list] = None):
    """Run main function.

    Args:
        args (list): The command line arguments.
    """
    if len(args) > 1:
        lexos_dir = args[1]
    else:
        lexos_dir = "lexos"
    set_env_paths(lexos_dir)


def set_env_paths(lexos_dir: str = "lexos"):
    """Set the environment variable PATH to include the lexos directory.

    Args:
        lexos_dir (str): The lexos directory.
    """
    sys.path.insert(0, os.path.abspath(lexos_dir))
    print(f"System path set to `{lexos_dir}`.")


if __name__ == "__main__":
    main(sys.argv)
