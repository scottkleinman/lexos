"""test_remove.py."""

# Import a minimal text loader class, the functions for scrubber pipelines,
# and the scrubber function registry
from lexos.io.smart import Loader
from lexos.scrubber.pipeline import make_pipeline, pipe
from lexos.scrubber.registry import load_components, scrubber_components

# Load a component from the registry
accents = scrubber_components.get("accents")

# Or, if you want to do several at once...
components = (
    "brackets",
    "digits",
    "tags",
    "new_lines",
    "pattern",
    "punctuation",
    "tabs",
)
brackets, digits, tags, new_lines, pattern, punctuation, tabs = load_components(
    components
)

# Test out the components
text = "This is a test. Testing 123"
scrubbed_text = punctuation(text)
print(scrubbed_text)
print()

# Now let's try a pipeline on a real text

# Load a text
data = "tests/test_data/txt/Austen_Pride.txt"
loader = Loader()
loader.load(data)
text = loader.texts[0]

# Make a pipeline (the `pipe()` method is required for passing arguments to functions
scrub = make_pipeline(
    accents,
    brackets,
    digits,
    tags,
    new_lines,
    pipe(pattern, pattern="est"),
    punctuation,
    tabs,
)

# Scrub the text using the pipeline
scrubbed_text = scrub(text)
print("Preview:")
print(scrubbed_text[0:50])
