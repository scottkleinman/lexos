"""test_llm_labeler.py.

Tests for TopicLabelerConfig and TopicLabelerClient.

Coverage: 97%. Missing: 183, 222, 243, 285, 301
Last Updated: 28 July, 2026
"""

from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

from lexos.topic_modeling.mallet.llm_labeler import (
    TopicLabelerClient,
    TopicLabelerConfig,
    label_mallet_topics,
)


def test_topic_labeler_config_requires_auth():
    """Test that TopicLabelerConfig requires either api_key or base_url."""
    # Should work with api_key
    config = TopicLabelerConfig(provider="openai", model="gpt-4", api_key="sk-test")
    assert config.api_key == "sk-test"

    # Should work with base_url
    config = TopicLabelerConfig(
        provider="local", model="llama3", base_url="http://localhost:11434/v1"
    )
    assert config.base_url == "http://localhost:11434/v1"

    # Should fail with neither
    with pytest.raises(ValidationError) as excinfo:
        TopicLabelerConfig(provider="openai", model="gpt-4")
    assert "Either 'api_key' or 'base_url' must be provided." in str(excinfo.value)


def test_topic_labeler_config_provider_lower():
    """Test that provider is case-insensitive and converted to lowercase."""
    config = TopicLabelerConfig(provider="OpenAI", model="gpt-4", api_key="sk-test")
    assert config.provider == "openai"


def test_generate_label_unsupported_provider():
    """Test that TopicLabelerClient raises ValueError for unsupported providers."""
    config = TopicLabelerConfig(provider="unsupported", model="xyz", api_key="test")
    client = TopicLabelerClient(config=config)
    with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
        client.generate_label(["word1", "word2"])


@patch("requests.post")
def test_call_openai_compatible(mock_post):
    """Test OpenAI compatible API call."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Historical Events"}}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    config = TopicLabelerConfig(
        provider="openai", model="gpt-4", api_key="sk-test", temperature=0.5
    )
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["war", "history", "battle"])

    assert label == "Historical Events"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "gpt-4"
    assert kwargs["json"]["temperature"] == 0.5
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"


@patch("requests.post")
@patch("time.sleep")
def test_call_openai_compatible_rate_limit(mock_sleep, mock_post):
    """Test OpenAI compatible API rate limit handling (429)."""
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429

    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.json.return_value = {
        "choices": [{"message": {"content": "Historical Events"}}]
    }

    # First call returns 429, second call returns 200
    mock_post.side_effect = [mock_response_429, mock_response_ok]

    config = TopicLabelerConfig(
        provider="openai", model="gpt-4", api_key="sk-test", max_retries=2
    )
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["war", "history"])

    assert label == "Historical Events"
    assert mock_post.call_count == 2
    mock_sleep.assert_called_with(2.0)


@patch("requests.post")
def test_call_openai_compatible_rate_limit_exhausted(mock_post):
    """Test OpenAI compatible API rate limit exhausted."""
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429
    mock_post.return_value = mock_response_429

    config = TopicLabelerConfig(
        provider="openai", model="gpt-4", api_key="sk-test", max_retries=2
    )
    client = TopicLabelerClient(config=config)
    with pytest.raises(
        Exception, match="OpenAI/Local API rate limit reached repeatedly"
    ):
        with patch("time.sleep"):
            client.generate_label(["war", "history"])


@patch("requests.post")
@patch("time.sleep")
def test_call_openai_compatible_network_retry(mock_sleep, mock_post):
    """Test OpenAI compatible API retry on network error."""
    mock_response_ok = MagicMock(status_code=200)
    mock_response_ok.json.return_value = {
        "choices": [{"message": {"content": "Success"}}]
    }
    mock_post.side_effect = [
        requests.exceptions.RequestException("Network Error"),
        mock_response_ok,
    ]

    config = TopicLabelerConfig(
        provider="openai", model="gpt-4", api_key="sk-test", max_retries=2
    )
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["word"])
    assert label == "Success"
    assert mock_post.call_count == 2


@patch("requests.post")
def test_call_gemini(mock_post):
    """Test Gemini API call."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Natural Science"}]}}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    config = TopicLabelerConfig(provider="gemini", model="gemini-1.5", api_key="key")
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["biology", "chemistry"])

    assert label == "Natural Science"
    assert "googleapis.com" in mock_post.call_args[0][0]
    assert "models/gemini-1.5" in mock_post.call_args[0][0]


@patch("requests.post")
@patch("time.sleep")
def test_call_gemini_rate_limit(mock_sleep, mock_post):
    """Test Gemini API rate limit handling (429)."""
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429

    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Success"}]}}]
    }

    mock_post.side_effect = [mock_response_429, mock_response_ok]

    config = TopicLabelerConfig(
        provider="gemini", model="gemini-1.5", api_key="key", max_retries=2
    )
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["biology"])

    assert label == "Success"
    assert mock_post.call_count == 2


@patch("requests.post")
def test_call_gemini_blocked(mock_post):
    """Test Gemini API blocked response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"promptFeedback": {"blockReason": "SAFETY"}}
    mock_post.return_value = mock_response

    config = TopicLabelerConfig(provider="gemini", model="gemini-1.5", api_key="key")
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["word"])
    assert label == "[Blocked: SAFETY]"


@patch("requests.post")
def test_call_gemini_no_content(mock_post):
    """Test Gemini API no content response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"candidates": [{"finishReason": "RECITATION"}]}
    mock_post.return_value = mock_response

    config = TopicLabelerConfig(provider="gemini", model="gemini-1.5", api_key="key")
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["word"])
    assert label == "[No content: RECITATION]"


@patch("requests.post")
def test_call_claude(mock_post):
    """Test Claude API call."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": [{"text": "Philosophy"}]}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    config = TopicLabelerConfig(provider="claude", model="claude-3", api_key="sk-ant")
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["plato", "aristotle"])

    assert label == "Philosophy"
    assert "anthropic.com" in mock_post.call_args[0][0]
    assert mock_post.call_args[1]["headers"]["x-api-key"] == "sk-ant"


@patch("requests.post")
@patch("time.sleep")
def test_call_claude_rate_limit(mock_sleep, mock_post):
    """Test Claude API rate limit handling (429)."""
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429

    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.json.return_value = {"content": [{"text": "Success"}]}

    mock_post.side_effect = [mock_response_429, mock_response_ok]

    config = TopicLabelerConfig(
        provider="claude", model="claude-3", api_key="sk-ant", max_retries=2
    )
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["word"])
    assert label == "Success"
    assert mock_post.call_count == 2


@patch("requests.post")
def test_call_claude_network_retry(mock_post):
    """Test Claude API retry on network error."""
    mock_response_ok = MagicMock(status_code=200)
    mock_response_ok.json.return_value = {"content": [{"text": "Success"}]}
    mock_post.side_effect = [
        requests.exceptions.RequestException("Network Error"),
        mock_response_ok,
    ]

    config = TopicLabelerConfig(
        provider="claude", model="claude-3", api_key="sk-ant", max_retries=2
    )
    client = TopicLabelerClient(config=config)
    with patch("time.sleep"):
        label = client.generate_label(["word"])
    assert label == "Success"
    assert mock_post.call_count == 2


def test_label_mallet_topics_specific_numbers(tmp_path):
    """Test label_mallet_topics with specific topic numbers (topic_nums)."""
    topic_keys = tmp_path / "topic_keys.txt"
    topic_keys.write_text("0\t0.5\tw1 w2\n1\t0.3\tw3 w4\n2\t0.2\tw5 w6\n")

    config = TopicLabelerConfig(provider="local", model="llama3", api_key="test")

    with patch.object(TopicLabelerClient, "generate_label") as mock_gen:
        mock_gen.return_value = "Label"
        # Test with single int
        labels = label_mallet_topics(str(topic_keys), config, topic_nums=1)
        assert labels == {1: "Label"}

        # Test with empty list (should label nothing)
        labels = label_mallet_topics(str(topic_keys), config, topic_nums=[])
        assert labels == {}


def test_label_mallet_topics(tmp_path):
    """Test the parsing and labeling of a Mallet topic keys file."""
    topic_keys = tmp_path / "topic_keys.txt"
    topic_keys.write_text(
        "0\t0.5\twar history battle\n1\t0.3\tbiology chemistry physics\n"
    )

    config = TopicLabelerConfig(provider="local", model="llama3", api_key="test")

    with patch.object(TopicLabelerClient, "generate_label") as mock_gen:
        mock_gen.side_effect = ["Historical Events", "Science"]
        labels = label_mallet_topics(str(topic_keys), config)

        assert labels == {0: "Historical Events", 1: "Science"}
        assert mock_gen.call_count == 2


@patch("requests.post")
def test_call_openai_compatible_reasoning(mock_post):
    """Test OpenAI compatible API call with reasoning."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "", "reasoning_content": "Reasoned Title"}}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    # Test with o1 model and include_reasoning
    config = TopicLabelerConfig(
        provider="openai", model="o1-mini", api_key="sk-test", include_reasoning=True
    )
    client = TopicLabelerClient(config=config)
    label = client.generate_label(["war", "history"])

    assert label == "Reasoned Title"
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["include_reasoning"] is True
    assert kwargs["json"]["reasoning_effort"] == "medium"


def test_generate_label_custom_prompt():
    """Test generate_label with a custom prompt."""
    config = TopicLabelerConfig(
        provider="local",
        model="llama3",
        api_key="test",
        prompt="Summarize this: ",
    )
    client = TopicLabelerClient(config=config)

    with patch.object(client, "_call_openai_compatible") as mock_call:
        client.generate_label(["word1", "word2"])
        mock_call.assert_called_once()
        sent_prompt = mock_call.call_args[0][0]
        assert sent_prompt.startswith("Summarize this: ")
        assert "Words: word1, word2" in sent_prompt


def test_label_mallet_topics_subset(tmp_path):
    """Test label_mallet_topics with a subset of topic numbers."""
    topic_keys = tmp_path / "topic_keys.txt"
    topic_keys.write_text("0\t0.5\tw1 w2\n1\t0.3\tw3 w4\n2\t0.2\tw5 w6\n")

    config = TopicLabelerConfig(provider="local", model="llama3", api_key="test")

    with patch.object(TopicLabelerClient, "generate_label") as mock_gen:
        mock_gen.return_value = "Label"
        # Label only topic 1
        labels = label_mallet_topics(str(topic_keys), config, topic_nums=[1])

        assert labels == {1: "Label"}
        assert mock_gen.call_count == 1


def test_label_mallet_topics_failure_graceful(tmp_path):
    """Test that label_mallet_topics handles API failures gracefully."""
    topic_keys = tmp_path / "topic_keys.txt"
    topic_keys.write_text("0\t0.5\tw1 w2\n")

    config = TopicLabelerConfig(provider="local", model="llama3", api_key="test")

    with patch.object(TopicLabelerClient, "generate_label") as mock_gen:
        mock_gen.side_effect = Exception("API Down")
        labels = label_mallet_topics(str(topic_keys), config)

        assert 0 in labels
        assert "Labelling failed: API Down" in labels[0]
