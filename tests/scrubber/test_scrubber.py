"""test_scrubber.py."""

# Import a minimal text loader class, the functions for scrubber pipelines,
# and the scrubber function registry
from lexos.io.basic import Loader
from lexos.scrubber.pipeline import make_pipeline
from lexos.scrubber.registry import scrubber_components
from lexos.scrubber.scrubber import Scrubber

# Load a text
data = "tests/test_data/txt/Austen_Pride.txt"
loader = Loader()
loader.load(data)
text = loader.texts[0]

lower_case = scrubber_components.get("lower_case")
scrub = make_pipeline(lower_case)
pipeline = lower_case

s = Scrubber()
s.add_pipeline(pipeline)
show_pipeline = s.get_pipeline()
texts = s.scrub(text)
for text in texts:
    print(text[0:50])
