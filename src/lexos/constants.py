"""constants.py.

Lexos constants.
"""

from natsort import natsort_keygen, ns

# Numbers
FILE_START = 1024
MIN_ENCODING_DETECT = 10000
MIN_NEWLINE_DETECT = 1000

# Mime types
TEXT_TYPES = {"text/plain", "text/html", "text/xml"}
PDF_TYPES = {"application/pdf"}
DOCX_TYPES = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
ZIP_TYPES = {"application/zip"}
INVALID_FILE_TYPES = {"application/vnd.python.pickle"}

# Cutter.ByteCutter
LEXOS_MILESTONE_FLAG = "<LEXOS_MILESTONE_FLAG>"

# DTM
SORTING_ALGORITHM = natsort_keygen(alg=ns.LOCALE)
# SORTING_ALGORITHM = ns.LOCALE
