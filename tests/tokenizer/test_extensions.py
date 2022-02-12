"""test_extensions.py."""

# Import a minimal text loader class, the functions for scrubber pipelines,
# and the scrubber function registry
from lexos.io.basic import Loader
from lexos.tokenizer import extensions, make_doc

# Load a text
data = ["tests/test_data/Austen_Pride.txt"]
loader = Loader()
loader.load(data)

# Make a spaCy doc
doc = make_doc(loader.texts[0][0:50])
for token in doc[0:5]:
    print((token._.is_fruit, token._.get("is_fruit")))
