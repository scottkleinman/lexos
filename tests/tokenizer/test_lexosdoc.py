"""test_lexosdoc.py."""

# Import a minimal text loader class, the functions for scrubber pipelines,
# and the scrubber function registry
from lexos.io.basic import Loader
from lexos.tokenizer import lexosdoc, make_doc

# Load a text
data = ["tests/test_data/txt/Austen_Pride.txt"]
loader = Loader()
loader.load(data)

# Make a spaCy doc
doc = make_doc(loader.texts[0][0:1000])

# Make a LexosDoc
lexos_doc = lexosdoc.LexosDoc(doc)
tokens = lexos_doc.get_tokens()
attrs = lexos_doc.get_token_attrs()
df = lexos_doc.to_dataframe()
