"""lexos_sample_script.py.

This script is a sample script that shows how to use the lexos package.

To run it, first configure the path to your sample data and to the output file you wish to create.

Then run `poetry run python lexos_sample_script.py`
"""
# Configuration
data = "../test_data/txt/Austen_Pride.txt"
dendrogram_file = "Austen_Pride_dendrogram.png"

# Lexos imports
print("Loading Lexos tools...")
import itertools
from lexos import tokenizer
from lexos.cluster.dendrogram import Dendrogram
from lexos.cutter import Ginsu, Machete
from lexos.dtm import DTM
from lexos.io.smart import Loader
from lexos.scrubber.pipeline import make_pipeline, pipe
from lexos.scrubber.registry import load_components, scrubber_components

# Create the loader and load the data
print("Loading data...")
loader = Loader()
loader.load(data)
text = loader.texts[0]

#################
# You _could_ do some scrubbing, but it's not necessary

# Load a component from the registry
# print("Scrubbing data...")
# remove_digits = scrubber_components.get("digits")

# Or, if you want to do several at once...
# title_case, remove_digits = load_components(("lower_case", "digits"))

# Make the pipeline
# scrub = make_pipeline(lower_case, pipe(remove_digits))

# Scrub the text
# scrubbed_text = scrub(text)
#################

# Cutting the data into 10 segments
print("Cutting data into 10 segments...")
cutter = Machete()
segments = cutter.splitn(text, n=10)
# Cutter returns [[segment1, segment2, ...], [segment1, segment2, ...], ...], where
# each segement list corresponds to one source text. Since we are only using one
# source text for this experiment, we can flatten the list.
flat_segments = list(itertools.chain(*segments))

# Turn the scrubbed texts into a spaCy docs
print("Making spaCy docs...")
docs = tokenizer.make_docs(flat_segments)

cutter = Ginsu()
all_doc_segments = cutter.splitn(docs[0], n=10)
print("Ginsu segments:")
segment_docs = []
for doc_segments in all_doc_segments[0:1]:
    segment_docs.append(tokenizer.make_docs(doc_segments))

print(f"Segment Docs: len({segment_docs})")
print(f"First Segment Doc: len({segment_docs[0]})")
# Convert to the docs to lower case token lists with filtering
# Important: This loop and subsequent code don't work if you don't flatten
# the segments list since it assumes that you are starting with only one text.
# print("Converting to lowercase, removing punctuation, digits, and whitespace...")
# filtered_segments = []
# for doc in docs:
#     filtered_segments.append(
#         [
#             token.norm
#             for token in doc
#             if not token.is_punct and not token.is_digit and not token.is_space
#         ]
#     )

# # Make a DTM
# print("Making DTM...")
# labels = [f"Austen{i+1}" for i, _ in enumerate(filtered_segments)]
# dtm = DTM(filtered_segments, labels)

# # Make a dendrogram from the DTM
# print("Clustering...")
# dendrogram = Dendrogram(dtm, show=False)

# # Save the dendrogram
# print("Saving dendrogram...")
# dendrogram.savefig(dendrogram_file)
# print(f"Saved {dendrogram_file}.")

print("Done!")