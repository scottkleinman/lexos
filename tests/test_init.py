"""test_init.py.

Tests for the top-level `lexos` package initialization module.

Coverage: 100%
Last Update: August 9, 2026
"""

import importlib
import sys
import types

from rich.table import Table

import lexos


def test_package_metadata_and_info_mapping():
    """Metadata constants should match the exported __info__ mapping."""
    assert lexos.__title__ == "Lexos"
    assert lexos.__license__ == "MIT"
    assert lexos.__status__ == "pre-release"
    assert lexos.__version_info__ == (0, 2, 0, "beta")

    assert lexos.__info__["title"] == lexos.__title__
    assert lexos.__info__["version"] == lexos.__version__
    assert lexos.__info__["docs"] == lexos.__docs__
    assert lexos.__info__["repo"] == lexos.__repo__
    assert lexos.__info__["copyright"] == lexos.__copyright__
    assert lexos.__info__["license"] == lexos.__license__
    assert lexos.__info__["status"] == lexos.__status__


def test_public_exports_are_declared_in_lazy_imports():
    """All declared public exports should be backed by lazy import entries."""
    for name in lexos.__all__:
        assert name in lexos._LAZY_IMPORTS


def test_dir_includes_public_exports():
    """The module dir should include all public symbols from __all__."""
    module_dir = dir(lexos)
    for name in lexos.__all__:
        assert name in module_dir


def test_getattr_unknown_symbol_raises_attribute_error():
    """Unknown attributes should raise AttributeError with module context."""
    try:
        lexos.__getattr__("DefinitelyMissingSymbol")
        raise AssertionError("Expected AttributeError")
    except AttributeError as exc:
        assert "has no attribute" in str(exc)
        assert "DefinitelyMissingSymbol" in str(exc)


def test_getattr_lazy_import_and_caching(monkeypatch):
    """__getattr__ should lazy import once and cache the resolved value."""
    fake_module_name = "tests._fake_lexos_lazy_module"
    fake_attr = "FakeSymbol"

    class FakeSymbol:
        pass

    fake_module = types.ModuleType(fake_module_name)
    fake_module.FakeSymbol = FakeSymbol
    monkeypatch.setitem(sys.modules, fake_module_name, fake_module)

    monkeypatch.setitem(lexos._LAZY_IMPORTS, fake_attr, (fake_module_name, fake_attr))
    lexos.__dict__.pop(fake_attr, None)

    calls = []

    def fake_import_module(name):
        calls.append(name)
        return importlib.import_module(name)

    monkeypatch.setattr(lexos, "import_module", fake_import_module)

    value_first = lexos.__getattr__(fake_attr)
    value_second = getattr(lexos, fake_attr)

    assert value_first is FakeSymbol
    assert value_second is FakeSymbol
    assert calls == [fake_module_name]


def test_get_info_prints_table(monkeypatch):
    """get_info should print a rich Table via Console.print."""
    printed = []

    def fake_print(self, obj="", *args, **kwargs):
        printed.append(obj)

    monkeypatch.setattr("rich.console.Console.print", fake_print)

    lexos.get_info()

    assert any(isinstance(item, Table) for item in printed)
