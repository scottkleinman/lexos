"""test_registry.py.

Coverage: 100%
Last Update: 2025-01-19.
"""

import catalogue
import pytest

# from lexos.scrubber.registry import scrubber_components
from lexos.scrubber.registry import get_component, get_components, scrubber_components


def test_load_existing_component():
    """Test loading an existing component."""
    component_name = "lower_case"
    component = get_component(component_name)
    assert callable(component)
    assert component("TEST") == "test"


def test_load_non_existing_component():
    """Test loading a non-existing component."""
    component_name = "non_existing"
    with pytest.raises(catalogue.RegistryError):
        get_component(component_name)


def test_get_components():
    """Test loading multiple components."""
    component_names = ("digits", "lower_case")
    components = get_components(component_names)
    for component in components:
        assert callable(component)
