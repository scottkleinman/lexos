"""test_normalize.py."""

# Import a minimal text loader class, the functions for scrubber pipelines,
# and the scrubber function registry
from lexos.io.basic import Loader
from lexos.scrubber.pipeline import make_pipeline, pipe
from lexos.scrubber.registry import load_components, scrubber_components

# Load a component from the registry
lower_case = scrubber_components.get("lower_case")

# Or, if you want to do several at once...
components = (
    "bullet_points",
    "hyphenated_words",
    "quotation_marks",
    "repeating_chars",
    "unicode",
    "whitespace",
)
(
    bullet_points,
    hyphenated_words,
    quotation_marks,
    repeating_chars,
    unicode,
    whitespace,
) = load_components(components)

# Test out the components
text = "This is a test. Testing 123"
scrubbed_text = lower_case(text)
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
    unicode,
    whitespace,
    quotation_marks,
    hyphenated_words,
    bullet_points,
    lower_case,
    pipe(repeating_chars, chars="??"),
    lower_case,
)

# Scrub the text using the pipeline
scrubbed_text = scrub(text)
print("Preview:")
print(scrubbed_text[0:50])
