"""test_replace.py."""

# Import a minimal text loader class, the functions for scrubber pipelines,
# and the scrubber function registry
from lexos.io.smart import Loader
from lexos.scrubber.pipeline import make_pipeline, pipe
from lexos.scrubber.registry import load_components, scrubber_components

# Load a component from the registry
currency_symbols = scrubber_components.get("currency_symbols")

# Or, if you want to do several at once...
components = (
    "re_digits",
    "emails",
    "emojis",
    "hashtags",
    "re_pattern",
    "phone_numbers",
    "re_punctuation",
    "special_characters",
    "tag_map",
    "urls",
    "user_handles",
)

(
    re_digits,
    emails,
    emojis,
    hashtags,
    re_pattern,
    phone_numbers,
    re_punctuation,
    special_characters,
    tag_map,
    urls,
    user_handles,
) = load_components(components)

# Test out the components
text = "<p>This is a test. Testing $123.</p>"
scrubbed_text = currency_symbols(text)
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
    re_digits,
    emails,
    emojis,
    hashtags,
    pipe(re_pattern, pattern={"est": "ZZZ"}),
    phone_numbers,
    re_punctuation,
    pipe(special_characters, is_html=True),
    pipe(tag_map, map={"p": {"action": "remove_tag", "attribute": ""}}),
    urls,
    user_handles,
)

# Scrub the text using the pipeline
scrubbed_text = scrub(text)
print("Preview:")
print(scrubbed_text[0:50])
