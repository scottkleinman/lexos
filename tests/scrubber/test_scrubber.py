"""test_scrubber.py.

Coverage: 100%
Last Update: 16 May 2025
"""

from functools import partial

import catalogue
import pytest

from lexos.exceptions import LexosException
from lexos.scrubber.registry import scrubber_components
from lexos.scrubber.scrubber import Pipe, Scrubber, scrub


@pytest.fixture
def scrubber():
    """Scrubber fixture."""
    return Scrubber()


def test_pipe_class():
    """Test the Pipe class."""
    pipe = Pipe(name="digits", opts={"only": ["1"]})
    assert pipe.name == "digits"
    assert pipe.opts == {"only": ["1"]}
    assert pipe.factory == scrubber_components


def test_pipe_class_exemption():
    """Test the Pipe class and its __call__ method, including error handling."""
    # NameError branch: simulate factory.get raising NameError
    import catalogue

    class DummyFactory(catalogue.Registry):
        def get(self, name):
            raise NameError("dummy")

    pipe_bad = Pipe(name="not_a_real_component", factory=DummyFactory("dummy"))
    with pytest.raises(LexosException) as excinfo:
        pipe_bad("test")
    assert "dummy" in str(excinfo.value)

    # RegistryError branch: simulate factory.get raising catalogue.RegistryError
    class DummyFactory2(catalogue.Registry):
        def get(self, name):
            raise catalogue.RegistryError("dummy registry error")

    pipe_bad2 = Pipe(name="not_a_real_component", factory=DummyFactory2("dummy2"))
    with pytest.raises(LexosException) as excinfo2:
        pipe_bad2("test")
    assert "dummy registry error" in str(excinfo2.value)


def test_add_pipe_string(scrubber):
    """Test adding a named component to the scrubber pipeline."""
    scrubber.add_pipe("lower_case")
    assert len(scrubber._components) == 1
    assert scrubber._components[0].name == "lower_case"
    assert scrubber.pipes == ["lower_case"]


def test_add_pipe_partial(scrubber):
    """Test adding a partial containing a named component and kwargs to the scrubber pipeline."""
    from lexos.scrubber.remove import digits

    scrubber.add_pipe(partial(digits, only=["1"]))
    assert len(scrubber._components) == 1
    assert scrubber._components[0].name == "digits"
    assert scrubber._components[0].opts == {"only": ["1"]}
    assert scrubber.pipes == ["digits"]


def test_add_pipe_tuple(scrubber):
    """Test adding a tuple containing a named component and kwargs to the scrubber pipeline."""
    scrubber.add_pipe(("digits", {"only": ["1"]}))
    assert len(scrubber._components) == 1
    assert scrubber._components[0].name == "digits"
    assert scrubber._components[0].opts == {"only": ["1"]}
    assert scrubber.pipes == ["digits"]


def test_add_pipe_pipe(scrubber):
    """Test adding a Pipe object to the scrubber pipeline."""
    pipe = Pipe(name="digits", opts={"only": ["1"]})
    scrubber.add_pipe(pipe)
    assert len(scrubber._components) == 1
    assert isinstance(scrubber._components[0], Pipe)
    assert scrubber._components[0].name == "digits"
    assert scrubber._components[0].opts == {"only": ["1"]}
    assert scrubber.pipes == ["digits"]


def test_add_pipe_multiple_string(scrubber):
    """Test adding multiple named components to the scrubber pipeline."""
    scrubber.add_pipe(["lower_case", "digits"])
    assert len(scrubber._components) == 2
    assert scrubber._components[0].name == "lower_case"


def test_add_pipe_multiple_tuple(scrubber):
    """Test adding multiple tuples to the scrubber pipeline."""
    pipeline = [("lower_case", {}), ("digits", {"only": ["1"]})]
    scrubber.add_pipe(pipeline)
    assert len(scrubber._components) == 2
    assert scrubber._components[0].name == "lower_case"


def test_add_pipe_multiple_pipe(scrubber):
    """Test adding multiple Pipe objects to the scrubber pipeline."""
    pipeline = [
        Pipe(name="lower_case", opts={}),
        Pipe(name="digits", opts={"only": ["1"]}),
    ]
    scrubber.add_pipe(pipeline)
    assert len(scrubber._components) == 2
    assert scrubber._components[0].name == "lower_case"


def test_add_pipe_first(scrubber):
    """Test adding a named component to the start of the scrubber pipeline."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits", first=True)
    assert scrubber._components[0].name == "digits"


def test_add_pipe_last(scrubber):
    """Test adding a named component to the end of the scrubber pipeline."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits", last=True)
    assert scrubber._components[0].name == "lower_case"


def test_add_pipe_multiple_positions(scrubber):
    """Test adding a named component with multiple positions set."""
    scrubber.add_pipe("lower_case")
    with pytest.raises(LexosException, match="Only one of before"):
        scrubber.add_pipe("digits", first=True, last=True)


def test_add_pipe_before(scrubber):
    """Test adding a component before another."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits", before="lower_case")
    assert scrubber._components[0].name == "digits"


def test_add_pipe_after(scrubber):
    """Test adding a component after another."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits", after="lower_case")
    assert scrubber._components[0].name == "lower_case"


def test_add_pipe_before_after_invalid(scrubber):
    """Test adding a component before or after a non-existent component."""
    scrubber.add_pipe("lower_case")
    with pytest.raises(LexosException, match="The component name"):
        scrubber.add_pipe("digits", before="invalid_pipe")
    with pytest.raises(LexosException, match="The component name"):
        scrubber.add_pipe("digits", after="invalid_pipe")


def test_add_pipe_numeric_before(scrubber):
    """Test adding a component before or after a non-existent component."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits")
    scrubber.add_pipe("remove_whitespace", before=1)
    assert scrubber._components[1].name == "remove_whitespace"


def test_add_pipe_numeric_after(scrubber):
    """Test adding a component before or after a non-existent component."""
    scrubber.add_pipe("lower_case")
    scrubber.add_pipe("digits")
    scrubber.add_pipe("remove_whitespace", after=0)
    assert scrubber._components[1].name == "remove_whitespace"


def test_get_pipe_index_before_out_of_bounds(scrubber):
    """Test that _get_pipe_index raises ValueError for negative integer 'before'."""
    scrubber.add_pipe("lower_case")
    with pytest.raises(ValueError, match="Index -1 out of range."):
        scrubber._get_pipe_index(before=-1)
    with pytest.raises(ValueError, match="Index 2 out of range."):
        scrubber._get_pipe_index(before=2)


def test_add_pipe_invalid_component_type(scrubber):
    """Test that add_pipe raises LexosException for invalid component type."""

    class NotAComponent:
        pass

    with pytest.raises(
        LexosException,
        match="Components must be strings, tuples, functools.partial, or Pipe objects.",
    ):
        scrubber.add_pipe(NotAComponent())


def test_add_pipe_merge_opts(scrubber):
    """Test that add_pipe merges options for an existing component."""
    scrubber.add_pipe(("digits", {"a": 1, "b": 2}))
    scrubber.add_pipe(("digits", {"b": 3, "c": 4}))
    assert scrubber._components[1].opts == {"a": 1, "b": 3, "c": 4}


def test_pipe(scrubber):
    """Test scrubbing texts with the pipeline."""
    scrubber.add_pipe("lower_case")
    texts = ["Apple1", "Banana2"]
    result = list(scrubber.pipe(texts))
    assert result == ["apple1", "banana2"]


def test_pipe_disable(scrubber):
    """Test scrubbing texts with an iterable as the pipeline."""
    pipeline = ["lower_case", "digits"]
    scrubber.add_pipe(pipeline)
    texts = ["Apple1", "Banana2"]
    result = list(scrubber.pipe(texts, disable=["digits"]))
    assert result == ["apple1", "banana2"]


def test_pipe_cfg(scrubber):
    """Test scrubbing texts with the pipeline."""
    scrubber.add_pipe(("digits", {"only": ["1"]}))
    component_cfg = {"digits": {"only": ["2"]}}
    texts = ["Apple1", "Banana2"]
    result = list(scrubber.pipe(texts, component_cfg=component_cfg))
    assert result == ["Apple1", "Banana"]


def test_remove_pipe(scrubber):
    """Test removing a pipe from the scrubber."""
    pipe = Pipe(name="test_pipe")
    scrubber.add_pipe(pipe)
    scrubber.remove_pipe("test_pipe")
    assert len(scrubber._components) == 0


def test_reset(scrubber):
    """Test resetting the scrubber pipeline."""
    pipe = Pipe(name="test_pipe")
    scrubber.add_pipe(pipe)
    scrubber.reset()
    assert len(scrubber._components) == 0


def test_scrub(scrubber):
    """Test scrubbing a text with the pipeline."""
    scrubber.add_pipe(("digits", {"only": ["1"]}))
    result = scrubber.scrub("apple1")
    assert result == "apple"


def test_invalid_pipe_index(scrubber):
    """Test invalid pipe index raises LexosException."""
    with pytest.raises(LexosException):
        scrubber._get_pipe_index(before=0, after=1)


def test_invalid_pipe_name(scrubber):
    """Test invalid pipe name raises LexosException."""
    with pytest.raises(LexosException):
        scrubber._get_pipe_index(before="invalid_pipe")


def test_get_pipe_index_after_out_of_range(scrubber):
    """Test that _get_pipe_index raises ValueError for out-of-range 'after'."""
    scrubber.add_pipe("lower_case")
    with pytest.raises(ValueError, match="Index 1 out of range."):
        scrubber._get_pipe_index(after=1)
    with pytest.raises(ValueError, match="Index -1 out of range."):
        scrubber._get_pipe_index(after=-1)


def test_empty_pipeline(scrubber):
    """Test scrubbing with an empty pipeline returns the text."""
    assert scrubber.scrub("text") == "text"


def test_get_pipe_index_invalid_combination_hits_final_raise(scrubber):
    """Test that _get_pipe_index raises ValueError for truly invalid types."""

    class Dummy:
        pass

    with pytest.raises(
        ValueError, match="Invalid combination of before, after, first, last."
    ):
        scrubber._get_pipe_index(before=Dummy())


def test_scrub_with_callable():
    """Test scrubbing with a callable."""

    def mock_callable(text):
        return text.upper()

    result = scrub("test", [mock_callable])
    assert result == "TEST"


def test_scrub_with_partial():
    """Test scrubbing with a partial callable."""

    def mock_callable(text, suffix):
        return text + suffix

    mock_partial = partial(mock_callable, suffix="!")
    result = scrub("test", [mock_partial])
    assert result == "test!"


def test_scrub_with_tuple():
    """Test scrubbing with a tuple containing a callable and its kwargs."""

    def mock_callable(text, prefix):
        return prefix + text

    result = scrub("test", [(mock_callable, {"prefix": "Hello, "})])
    assert result == "Hello, test"


def test_scrub_with_string():
    """Test scrubbing with a named component."""
    result = scrub("TEST", ["lower_case"])
    assert result == "test"


def test_scrub_with_pipeline():
    """Test scrubbing with a pipeline of components."""
    from lexos.scrubber.normalize import lower_case
    from lexos.scrubber.remove import punctuation

    Pipeline = ["digits", lower_case, (punctuation, {"only": "!"})]
    result = scrub("TEST123.!", Pipeline)
    assert result == "test."


def test_scrub_with_invalid_component():
    """Test that scrub raises LexosException for an invalid component."""
    with pytest.raises(LexosException):
        scrub("test", ["nonexistent_component"])


def test_scrub_string_else_branch():
    """Test AttributeError exception."""

    class DummyFactory:
        def dummyFunction(input: str) -> str:
            return input

    with pytest.raises(LexosException):
        result = scrub("abc", ["dummy_component"], factory=DummyFactory())
