"""keyterms.py.

Last Updated: November 10, 2025
Last Tested: November 10, 2025
"""

from spacy.tokens import Doc

# Register a custom extension for keyterms if not already set
if not Doc.has_extension("keyterms"):
    Doc.set_extension("keyterms", default=None, force=True)
