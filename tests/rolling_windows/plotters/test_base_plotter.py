"""test_base_plotter.py.

Coverage: 100%
Last Updated: February 9, 2025
"""

import pytest

from lexos.rolling_windows.plotters.base_plotter import BasePlotter


@pytest.fixture
def base_plotter():
    """Create a BasePlotter instance for testing."""
    return BasePlotter()


def test_metadata_basic(base_plotter):
    """Test basic metadata retrieval."""
    metadata = base_plotter.metadata
    assert base_plotter.id == "base_plotter"
    assert isinstance(metadata, dict)


def test_set_attrs_single(base_plotter):
    """Test setting a single attribute."""
    base_plotter._set_attrs(test_attr="test_value")
    assert hasattr(base_plotter, "test_attr")
    assert base_plotter.test_attr == "test_value"


def test_set_attrs_multiple(base_plotter):
    """Test setting multiple attributes."""
    attrs = {"attr1": "value1", "attr2": 123, "attr3": [1, 2, 3]}
    base_plotter._set_attrs(**attrs)
    for key, value in attrs.items():
        assert hasattr(base_plotter, key)
        assert getattr(base_plotter, key) == value


def test_set_attrs_none_value(base_plotter):
    """Test that None values are not set as attributes."""
    base_plotter._set_attrs(test_attr=None)
    assert not hasattr(base_plotter, "test_attr")


def test_set_attrs_empty_kwargs(base_plotter):
    """Test setting no attributes."""
    original_attrs = set(dir(base_plotter))
    base_plotter._set_attrs()
    after_attrs = set(dir(base_plotter))
    assert original_attrs == after_attrs


def test_metadata_with_attributes(base_plotter):
    """Test metadata after setting custom attributes."""
    base_plotter._set_attrs(custom_attr="test")
    metadata = base_plotter.metadata
    assert "custom_attr" in metadata
    assert metadata["custom_attr"] == "test"
