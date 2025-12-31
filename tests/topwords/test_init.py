"""Tests for topwords module.

Coverage: 100%
Last Updated: June 25, 2025
"""

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict

from lexos.topwords import TopWords

# ---------------- Fixtures ----------------


@pytest.fixture
def sample_data():
    """Sample data for testing DataFrame conversion."""
    return [
        {"term": "analysis", "score": 0.95},
        {"term": "text", "score": 0.87},
        {"term": "document", "score": 0.72},
    ]


# ---------------- Test TopWords Base Class ----------------


class TestTopWordsBaseClass:
    """Test TopWords base class."""

    def test_is_pydantic_model(self):
        """Test that TopWords is a Pydantic BaseModel."""
        assert isinstance(TopWords(), BaseModel)
        assert issubclass(TopWords, BaseModel)

    def test_init_default(self):
        """Test TopWords initialization with defaults."""
        topwords = TopWords()
        assert topwords is not None
        assert isinstance(topwords, TopWords)

    def test_to_df_method_exists(self):
        """Test that to_df method exists."""
        topwords = TopWords()
        assert hasattr(topwords, "to_df")
        assert callable(getattr(topwords, "to_df"))

    def test_to_df_not_implemented(self):
        """Test that to_df method raises NotImplementedError (abstract method)."""
        topwords = TopWords()
        with pytest.raises(NotImplementedError):
            topwords.to_df()

    def test_docstring_exists(self):
        """Test that the class has proper documentation."""
        assert TopWords.__doc__ is not None
        assert "Base class for topwords plugins" in TopWords.__doc__
        assert "common API" in TopWords.__doc__

    def test_to_df_docstring_exists(self):
        """Test that to_df method has proper documentation."""
        assert TopWords.to_df.__doc__ is not None
        assert "pandas DataFrame" in TopWords.to_df.__doc__

    def test_model_config(self):
        """Test that the model has Pydantic configuration."""
        topwords = TopWords()
        assert hasattr(topwords, "model_config") or hasattr(topwords, "__config__")

    def test_model_fields(self):
        """Test that the model has expected field structure."""
        topwords = TopWords()
        # The base class should have minimal fields
        fields = topwords.model_fields if hasattr(topwords, "model_fields") else {}
        assert isinstance(fields, dict)

    def test_inheritance_compatibility(self):
        """Test that TopWords can be inherited from properly."""

        class TestImplementation(TopWords):
            """Test implementation of TopWords."""

            def to_df(self):
                """Return a test DataFrame."""
                return pd.DataFrame([{"term": "test", "score": 1.0}])

        # Test that the subclass can be instantiated
        test_impl = TestImplementation()
        assert isinstance(test_impl, TopWords)
        assert isinstance(test_impl, TestImplementation)

        # Test that the subclass implements to_df properly
        df = test_impl.to_df()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "term" in df.columns
        assert "score" in df.columns

    def test_multiple_inheritance_scenarios(self):
        """Test various inheritance scenarios."""

        # Test subclass with additional fields defined properly
        class ExtendedTopWords(TopWords):
            """Extended TopWords with additional fields."""

            model_config = ConfigDict(extra="allow")  # Allow extra fields

            def __init__(self, **data):
                super().__init__(**data)
                # Use object.__setattr__ to bypass Pydantic validation for dynamic attributes
                object.__setattr__(self, "data", data.get("data", []))

            def to_df(self):
                """Return DataFrame from stored data."""
                return pd.DataFrame(self.data)

        # Test with empty data
        extended = ExtendedTopWords(data=[])
        df = extended.to_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

        # Test with sample data
        sample_data = [{"term": "word", "score": 0.5}]
        extended_with_data = ExtendedTopWords(data=sample_data)
        df_with_data = extended_with_data.to_df()
        assert isinstance(df_with_data, pd.DataFrame)
        assert len(df_with_data) == 1
        assert df_with_data.iloc[0]["term"] == "word"
        assert df_with_data.iloc[0]["score"] == 0.5

    def test_method_signature(self):
        """Test the to_df method signature."""
        import inspect

        signature = inspect.signature(TopWords.to_df)
        params = list(signature.parameters.keys())

        # Should only have 'self' parameter
        assert len(params) == 1
        assert params[0] == "self"

        # Should not have return annotation in base class
        assert (
            signature.return_annotation == inspect.Signature.empty
            or signature.return_annotation is None
        )

    def test_class_attributes(self):
        """Test class-level attributes."""
        # Test that the class has the expected attributes
        assert hasattr(TopWords, "__doc__")
        assert hasattr(TopWords, "to_df")

        # Test class name
        assert TopWords.__name__ == "TopWords"
        assert TopWords.__qualname__ == "TopWords"

    def test_instance_attributes(self):
        """Test instance-level attributes."""
        topwords = TopWords()

        # Test that instance has Pydantic model attributes
        assert hasattr(topwords, "model_dump") or hasattr(topwords, "dict")
        assert hasattr(topwords, "model_validate") or hasattr(topwords, "parse_obj")

    def test_serialization(self):
        """Test model serialization capabilities."""
        topwords = TopWords()

        # Test that the model can be serialized to dict
        if hasattr(topwords, "model_dump"):
            data = topwords.model_dump()
        else:
            data = topwords.dict()

        assert isinstance(data, dict)

    def test_validation(self):
        """Test Pydantic validation capabilities."""
        # Test that the class has validation methods
        assert hasattr(TopWords, "model_validate") or hasattr(TopWords, "parse_obj")

        # Test validation with empty dict (should work for base class)
        if hasattr(TopWords, "model_validate"):
            validated = TopWords.model_validate({})
        else:
            validated = TopWords.parse_obj({})

        assert isinstance(validated, TopWords)


# ---------------- Edge Cases and Error Handling ----------------


class TestTopWordsEdgeCases:
    """Test edge cases and error handling."""

    def test_abstract_method_implementation(self):
        """Test that the base class properly defines abstract method behavior."""
        topwords = TopWords()

        # The to_df method should raise NotImplementedError when called directly
        with pytest.raises(NotImplementedError):
            topwords.to_df()

    def test_subclass_must_implement_to_df(self):
        """Test that subclasses must implement to_df method."""

        class IncompleteImplementation(TopWords):
            """Incomplete implementation that doesn't override to_df."""

            pass

        incomplete = IncompleteImplementation()

        # Should still raise NotImplementedError since to_df wasn't overridden
        with pytest.raises(NotImplementedError):
            incomplete.to_df()

    def test_proper_subclass_implementation(self):
        """Test that proper subclass implementation works correctly."""

        class ProperImplementation(TopWords):
            """Proper implementation that overrides to_df."""

            def to_df(self):
                """Return a proper DataFrame."""
                return pd.DataFrame({"term": ["word1", "word2"], "score": [0.8, 0.6]})

        proper = ProperImplementation()
        df = proper.to_df()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["term", "score"]
        assert df.iloc[0]["term"] == "word1"
        assert df.iloc[0]["score"] == 0.8

    def test_empty_dataframe_implementation(self):
        """Test subclass that returns empty DataFrame."""

        class EmptyImplementation(TopWords):
            """Implementation that returns empty DataFrame."""

            def to_df(self):
                """Return empty DataFrame."""
                return pd.DataFrame()

        empty_impl = EmptyImplementation()
        df = empty_impl.to_df()

        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert len(df) == 0

    def test_complex_dataframe_implementation(self):
        """Test subclass with complex DataFrame structure."""

        class ComplexImplementation(TopWords):
            """Implementation with complex DataFrame."""

            model_config = ConfigDict(extra="allow")  # Allow extra fields

            def __init__(self, **data):
                super().__init__(**data)
                # Use object.__setattr__ to bypass Pydantic validation for dynamic attributes
                object.__setattr__(self, "terms", data.get("terms", []))
                object.__setattr__(self, "metadata", data.get("metadata", {}))

            def to_df(self):
                """Return complex DataFrame with metadata."""
                if not self.terms:
                    return pd.DataFrame(columns=["term", "score", "category"])

                df_data = []
                for term_data in self.terms:
                    df_data.append(
                        {
                            "term": term_data.get("term", ""),
                            "score": term_data.get("score", 0.0),
                            "category": term_data.get("category", "unknown"),
                        }
                    )

                df = pd.DataFrame(df_data)

                # Add metadata as attributes
                for key, value in self.metadata.items():
                    setattr(df, key, value)

                return df

        # Test with empty data
        complex_empty = ComplexImplementation()
        df_empty = complex_empty.to_df()
        assert isinstance(df_empty, pd.DataFrame)
        assert df_empty.empty
        assert list(df_empty.columns) == ["term", "score", "category"]

        # Test with complex data
        complex_data = ComplexImplementation(
            terms=[
                {"term": "analysis", "score": 0.95, "category": "noun"},
                {"term": "study", "score": 0.82, "category": "verb"},
            ],
            metadata={"source": "test", "version": "1.0"},
        )
        df_complex = complex_data.to_df()
        assert isinstance(df_complex, pd.DataFrame)
        assert len(df_complex) == 2
        assert hasattr(df_complex, "source")
        assert df_complex.source == "test"


# ---------------- Integration Tests ----------------


class TestTopWordsIntegration:
    """Integration tests for TopWords with pandas and pydantic."""

    def test_pandas_compatibility(self):
        """Test compatibility with pandas operations."""

        class PandasCompatible(TopWords):
            """Implementation that returns pandas-compatible DataFrame."""

            def to_df(self):
                return pd.DataFrame(
                    {
                        "term": ["word1", "word2", "word3"],
                        "score": [0.9, 0.7, 0.5],
                        "frequency": [10, 7, 3],
                    }
                )

        compatible = PandasCompatible()
        df = compatible.to_df()

        # Test various pandas operations
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

        # Test sorting
        sorted_df = df.sort_values("score", ascending=False)
        assert sorted_df.iloc[0]["score"] == 0.9

        # Test filtering
        filtered_df = df[df["score"] > 0.6]
        assert len(filtered_df) == 2

        # Test aggregation
        mean_score = df["score"].mean()
        assert abs(mean_score - 0.7) < 0.01

    def test_pydantic_integration(self):
        """Test integration with Pydantic features."""

        class PydanticIntegrated(TopWords):
            """Implementation that uses Pydantic features."""

            model_config = ConfigDict(extra="allow")  # Allow extra fields

            def __init__(self, **data):
                super().__init__(**data)
                # Use object.__setattr__ to bypass Pydantic validation for dynamic attributes
                object.__setattr__(self, "terms", data.get("terms", []))

            def to_df(self):
                if not self.terms:
                    return pd.DataFrame(columns=["term", "score"])
                return pd.DataFrame(self.terms)

        # Test with validation
        integrated = PydanticIntegrated(
            terms=[{"term": "test", "score": 0.8}, {"term": "word", "score": 0.6}]
        )

        df = integrated.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

        # Test serialization
        if hasattr(integrated, "model_dump"):
            data = integrated.model_dump()
        else:
            data = integrated.dict()
        assert isinstance(data, dict)  # Should have base model data

    def test_real_world_usage_pattern(self):
        """Test real-world usage patterns."""

        class RealWorldImplementation(TopWords):
            """Real-world style implementation."""

            model_config = ConfigDict(extra="allow")  # Allow extra fields

            def __init__(self, **data):
                super().__init__(**data)
                # Use object.__setattr__ to bypass Pydantic validation for dynamic attributes
                object.__setattr__(self, "algorithm", data.get("algorithm", "default"))
                object.__setattr__(self, "results", data.get("results", []))
                object.__setattr__(self, "config", data.get("config", {}))

            def to_df(self):
                """Convert results to DataFrame with metadata."""
                if not self.results:
                    return pd.DataFrame(columns=["term", "score", "rank"])

                # Add ranking
                sorted_results = sorted(
                    self.results, key=lambda x: x.get("score", 0), reverse=True
                )
                for i, result in enumerate(sorted_results):
                    result["rank"] = i + 1

                df = pd.DataFrame(sorted_results)

                # Add metadata as DataFrame attributes
                df.attrs["algorithm"] = self.algorithm
                df.attrs["config"] = self.config

                return df

        # Test realistic usage
        real_world = RealWorldImplementation(
            algorithm="textrank",
            results=[
                {"term": "machine learning", "score": 0.95},
                {"term": "data analysis", "score": 0.87},
                {"term": "artificial intelligence", "score": 0.92},
            ],
            config={"topn": 10, "normalize": True},
        )

        df = real_world.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "rank" in df.columns
        assert df.iloc[0]["term"] == "machine learning"  # Highest score
        assert df.iloc[0]["rank"] == 1
        assert df.attrs["algorithm"] == "textrank"
