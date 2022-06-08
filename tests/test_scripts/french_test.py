"""french_test.py.

This script is a sample script that shows how to use the lexos package.

To run it, first configure the path to your sample data and to the output file you wish to create.

Then run `poetry run python french_test.py`
"""
# Configuration
data = """Quand M. Bilbon Sacquet, de Cul-de-Sac, annonça qu'il donnerait à l'occasion de son undécante-unième anniversaire une réception d'une magnificence particulière, une grande excitation régna dans Hobbitebourg, et toute la ville en parla."""

# Test to make sure the French model is loaded:
try:
    import spacy

    nlp = spacy.load("fr_core_news_sm")
except BaseException:
    msg = """

    The French model is not installed. You will need to add the following to your pyproject.toml file:

    [tool.poetry.dependencies.fr_core_news_sm]
    url = "https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.2.0/fr_core_news_sm-3.2.0.tar.gz"

    Then run `poetry install`.
    """
    raise BaseException(msg)

# Lexos imports
print("Loading Lexos tools...")
from lexos import tokenizer
from lexos.io.smart import Loader

# Create the loader and load the data
print("Loading data...")
# loader = Loader()
# loader.load(data)
# text = loader.texts[0]

text = data

# Turn the text into a spaCy doc
print("Making spaCy doc...")
doc = tokenizer.make_doc(text, model="fr_core_news_sm")

# Print the tokens with some filtering
print("Token list without punctuation:\n")
print("Token", "Lemma", "POS", "Stopword\n")
for token in doc:
    if not token.is_punct and not token.is_space:
        print(token.text, token.lemma_, token.pos_, token.is_stop)

print("\nDone!")