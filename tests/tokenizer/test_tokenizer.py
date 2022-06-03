"""test_tokenizer.py."""

# Import a minimal text loader class, the functions for scrubber pipelines,
# and the scrubber function registry
from lexos.io.basic import Loader
from lexos.tokenizer import make_doc, make_docs

# Load some texts
data = ["tests/test_data/txt/Austen_Pride.txt", "tests/test_data/txt/Austen_Sense.txt"]
loader = Loader()
loader.load(data)

# Make a spaCy doc
doc = make_doc(loader.texts[0])

# Get the tokens
tokens = [token.text for token in doc]
print("Preview of Tokens:")
print(tokens[0:50])

# Make multiple spaCy docs
docs = make_docs(loader.texts)

# Get the tokens
for i, doc in enumerate(docs):
    print(f"Preview of Tokens in Doc{i}:")
    tokens = [token.text for token in doc]
    print(tokens[0:25])
    print()
