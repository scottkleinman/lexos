"""test_init.py.

Tests for the public API exported by lexos.corpus.

Coverage: 100%

Last Update: 2025-11-15.
"""


class TestCorpusInitExports:
    """Test the explicit export surface in lexos.corpus."""

    def test_exports_match_phase_1_contract(self):
        """Phase 1 should expose a stable, explicit API."""
        import lexos.corpus

        assert lexos.corpus.__all__ == [
            "Corpus",
            "Record",
            "CorpusStats",
            "RecordsDict",
        ]

    def test_exported_symbols_exist(self):
        """Every symbol in __all__ should resolve to a non-None object."""
        import lexos.corpus

        for export_name in lexos.corpus.__all__:
            assert hasattr(lexos.corpus, export_name)
            assert getattr(lexos.corpus, export_name) is not None

    def test_legacy_symbol_not_exported(self):
        """LexosModelCache remains available from lexos.corpus.utils, not package root."""
        import lexos.corpus

        assert "LexosModelCache" not in lexos.corpus.__all__
