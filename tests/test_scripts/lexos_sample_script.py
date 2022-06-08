"""lexos_sample_script.py.

This script is a sample script that shows how to use the lexos package.

To run it, first configure the path to your sample data and to the output file you wish to create.

Then run `poetry run python lexos_sample_script.py`
"""
# Configuration
data = "../test_data/txt/Austen_Pride.txt"
dendrogram_file = "Austen_Pride_dendrogram.png"

# Lexos imports
from lexos import tokenizer
from lexos.cluster.dendrogram import Dendrogram
from lexos.cutter import Ginsu, Machete
from lexos.dtm import DTM
from lexos.io.smart import Loader
from lexos.scrubber.pipeline import make_pipeline, pipe
from lexos.scrubber.registry import load_components, scrubber_components

# Create the loader and load the data
loader = Loader()
loader.load(data)
text = loader.texts[0]

#################
# You _could_ do some scrubbing, but it's not necessary

# Load a component from the registry
# lower_case = scrubber_components.get("lower_case")
# remove_digits = scrubber_components.get("digits")

# Or, if you want to do several at once...
# title_case, remove_digits = load_components(("lower_case", "digits"))

# Make the pipeline
# scrub = make_pipeline(lower_case, pipe(remove_digits))

# Scrub the text
# scrubbed_text = scrub(text)
#################

# Cutting the data into 10 segments
cutter = Machete()
segments = cutter.splitn(text, n=10, as_string=True)

# Turn the scrubbed texts into a spaCy docs
docs = tokenizer.make_docs(segments)

# Convert to the docs to lower case token lists with filtering
filtered_segments = []
for doc in docs:
    filtered_segments.append(
        [
            token.norm
            for token in doc
            if not token.is_punct and not token.is_digit and not token.is_space
        ]
    )

# Make a DTM
labels = [f"Austen{i+1}" for i, _ in enumerate(filtered_segments)]
dtm = DTM(filtered_segments, labels)

# Make a dendrogram from the DTM
dendrogram = Dendrogram(dtm, show=False)

# Save the dendrogram
dendrogram.savefig(dendrogram_file)
print(f"Saved {dendrogram_file}.")
