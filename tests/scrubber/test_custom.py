"""test_custom.py."""

# Import a minimal text loader class, the functions for scrubber pipelines,
# and the scrubber function registry
from lexos.io.basic import Loader
from lexos.scrubber.pipeline import make_pipeline, pipe
from lexos.scrubber.registry import load_components, scrubber_components


# Define a custom component and register it
def title_case(text: str) -> str:
    """Convert the text to title case.

    Args:
        text: The text to convert to title case.

    Returns:
        str: The text converted to title case.
    """
    return text.title()


scrubber_components.register("title_case", func=title_case)

# Load a component from the registry
title_case = scrubber_components.get("title_case")

# Test out the component
text = "This is a test. Testing 123"
scrubbed_text = title_case(text)
print(scrubbed_text)
print()

# Now let's try a pipeline on a real text

# Load a text
data = "tests/test_data/txt/Austen_Pride.txt"
loader = Loader()
loader.load(data)
text = loader.texts[0]

# Make a pipeline (the `pipe()` method is required for passing arguments to functions
lower_case = scrubber_components.get("lower_case")
scrub = make_pipeline(lower_case, title_case)

# Scrub the text using the pipeline
scrubbed_text = scrub(text)
print("Preview:")
print(scrubbed_text[0:50])