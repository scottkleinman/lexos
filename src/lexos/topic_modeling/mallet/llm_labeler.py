"""llm_labeler.py.

Client for labeling topics using various LLM providers.

Examples:
    - `config = TopicLabelerConfig(provider="claude", model="claude-3-5-sonnet-20240620", api_key="sk-ant-...", n_terms=15)`
    - `config = TopicLabelerConfig(provider="gemini", model="gemini-1.5-flash", api_key="AIzaSy...", n_terms=15)`
    - `config = TopicLabelerConfig(provider="local", model="llama3", base_url="http://localhost:11434/v1/chat/completions", n_terms=15)`


Last Updated: 28 July, 2026
Last Tested: 28 July, 2026
"""

import time
from typing import Optional

import requests
from pydantic import BaseModel, Field, model_validator
from tqdm.auto import tqdm

from lexos.util import ensure_list


class TopicLabelerConfig(BaseModel):
    """Configuration for the TopicLabelerClient."""

    provider: str = Field(
        ..., description="The LLM provider to use (e.g., 'openai', 'gemini', 'claude')."
    )
    model: str = Field(..., description="The specific model to use from the provider.")
    api_key: Optional[str] = Field(
        None, description="API key for the LLM provider, if required."
    )
    base_url: Optional[str] = Field(
        None,
        description="Base URL for the LLM provider's API, if different from the default.",
    )
    n_terms: Optional[int] = Field(
        15, description="Number of top terms to consider from the topic model cluster."
    )
    temperature: Optional[float] = Field(
        0.1, description="Default to low creativity for clean labeling."
    )
    max_tokens: Optional[int] = Field(
        50, description="Default to short, concise outputs."
    )
    documents_snippet: Optional[str] = Field(
        "", description="Optional snippet of contextual context from the documents."
    )
    prompt: Optional[str] = Field(
        None, description="Optional custom prompt to override the default prompt."
    )
    include_reasoning: Optional[bool] = Field(
        False,
        description="Whether to request reasoning (thinking) from models that support it.",
    )
    timeout: Optional[int] = Field(
        120, description="Timeout in seconds for API requests to the LLM provider."
    )
    max_retries: Optional[int] = Field(
        5,
        description="Maximum number of retries for API requests in case of rate limiting.",
    )

    @model_validator(mode="after")
    def check_api_auth(self) -> "TopicLabelerConfig":
        """Ensures either an api_key or a base_url is provided."""
        if not self.api_key and not self.base_url:
            raise ValueError("Either 'api_key' or 'base_url' must be provided.")
        return self

    def __init__(self, **data):
        """Initializes the TopicLabelerConfig with the specified provider in lower case."""
        super().__init__(**data)
        self.provider = self.provider.lower()


class TopicLabelerClient(BaseModel):
    """Client for labeling topics using various LLM providers."""

    config: TopicLabelerConfig = Field(
        ..., description="Configuration for the TopicLabelerClient."
    )

    def generate_label(self, top_words: list[str]) -> str:
        """Sends a structured prompt to the selected model provider.

        Args:
            top_words (list[str]): List of high-frequency words from a topic model cluster.

        Returns:
            str: The generated label for the topic.
        """
        if self.config.prompt is None:
            prompt = (
                f"Analyze the following high-frequency words from a topic model cluster:\n"
                f"Words: {', '.join(top_words)}\n"
                f"Contextual context snippet: {self.config.documents_snippet}\n\n"
                f"Provide 1 concise, clear title/label (3-5 words max) summarizing this topic. "
                f"Return ONLY the plain text label without any preamble or quotes."
            )
        else:
            prompt = self.config.prompt
            prompt += f"\nWords: {', '.join(top_words)}\nContextual context snippet: {self.config.documents_snippet}"

        if self.config.provider == "openai" or self.config.provider == "local":
            return self._call_openai_compatible(prompt)
        elif self.config.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.config.provider == "claude":
            return self._call_claude(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _call_openai_compatible(self, prompt: str) -> str:
        """Calls an OpenAI-compatible API endpoint with the given prompt.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated label from the model.
        """
        # Works out of the box for OpenAI, Ollama (localhost), LM Studio, and vLLM
        url = self.config.base_url or "https://openai.com"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,  # Supported universally by OpenAI/Ollama/vLLM
        }

        # Handle reasoning/thinking mode for models that support it
        # Note: Some providers use 'reasoning_effort', others use 'include_reasoning'
        if self.config.include_reasoning:
            payload["include_reasoning"] = True
            # For OpenAI o1/o3 models:
            if "o1" in self.config.model or "o3" in self.config.model:
                payload["reasoning_effort"] = "medium"

        delay = 2.0  # Base delay in seconds

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.config.timeout
                )

                # Catch standard HTTP rate limiting codes
                if response.status_code == 429:
                    if attempt == self.config.max_retries - 1:
                        raise Exception(
                            "OpenAI/Local API rate limit reached repeatedly. Halting."
                        )

                    print(f"\n[Rate Limited] Provider busy. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Double the wait time (2s -> 4s -> 8s...)
                    continue

                response.raise_for_status()
                res_json = response.json()
                content = res_json["choices"][0]["message"].get("content", "")

                # Handle specific models (like Gemma 4 QAT) that return empty content
                # but put the response in reasoning_content or other fields if max_tokens is reached
                if (
                    not content
                    and "reasoning_content" in res_json["choices"][0]["message"]
                ):
                    content = res_json["choices"][0]["message"]["reasoning_content"]

                return content.strip()

            except requests.exceptions.RequestException as e:
                # If it's a network glitch rather than a 429, retry up to max_retries
                if attempt == self.config.max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2

    def _call_gemini(self, prompt: str) -> str:
        """Handles Google Gemini API endpoints with the given prompt.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated label from the model.
        """
        # Google Gemini API endpoint format:
        # https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}
        model_name = self.config.model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={self.config.api_key}"

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }

        delay = 2.0  # Start with a 2-second delay if blocked

        for attempt in range(self.config.max_retries):
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.config.timeout
            )
            # Check if we hit the Free Tier RPM ceiling
            if response.status_code == 429:
                if attempt == self.config.max_retries - 1:
                    raise Exception(
                        "Gemini API rate limit exceeded repeatedly. Try again later."
                    )

                print(
                    f"\n[Rate Limited] Free tier cap hit. Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff (2s -> 4s -> 8s...)
                continue
            response.raise_for_status()
            break

        res_json = response.json()
        if "candidates" not in res_json or not res_json["candidates"]:
            # Check for block reasons or errors
            if (
                "promptFeedback" in res_json
                and "blockReason" in res_json["promptFeedback"]
            ):
                return f"[Blocked: {res_json['promptFeedback']['blockReason']}]"
            return "[No candidates returned]"

        candidate = res_json["candidates"][0]
        if "content" not in candidate or "parts" not in candidate["content"]:
            # Check for safety ratings or finish reasons
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            return f"[No content: {finish_reason}]"

        return candidate["content"]["parts"][0]["text"].strip()

    def _call_claude(self, prompt: str) -> str:
        """Handles Anthropic Claude API endpoints with the given prompt.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated label from the model.
        """
        url = "https://anthropic.com"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,  # Required parameter for Claude
        }
        delay = 2.0  # Base delay in seconds

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.config.timeout
                )

                # Catch Anthropic specific rate limits or tier blocks
                if response.status_code == 429:
                    if attempt == self.config.max_retries - 1:
                        raise Exception(
                            "Claude API rate limit reached repeatedly. Halting."
                        )

                    print(
                        f"\n[Rate Limited] Anthropic API cooling down. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                response.raise_for_status()
                return response.json()["content"][0]["text"].strip()

            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2


def label_mallet_topics(
    topic_keys_path: str,
    config: TopicLabelerConfig,
    topic_nums: Optional[int | list[int]] = None,
) -> dict[int, str]:
    """Parses Mallet's output topic keys file and assigns AI labels.

    Args:
        topic_keys_path: Path to the Mallet output file (usually `topic_keys.txt`)
        config: TopicLabelerConfig instance containing provider, model, api_key, and base_url
        topic_nums: Optional list of topic numbers to label. If None, all topics will be labeled.

    Returns:
        A dictionary mapping topic IDs to their generated labels.
    """
    client = TopicLabelerClient(config=config)

    topic_labels = {}
    topic_nums = ensure_list(topic_nums) if topic_nums is not None else None

    # Read the file lines first to count them for the progress bar
    with open(topic_keys_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if not topic_nums or len(topic_nums) == 0:
        total = len(lines)
    else:
        total = len(topic_nums)

    pbar = tqdm(lines, total=total, desc="Labeling Topics", unit="Topic", leave=True)
    for line in pbar:
        # Mallet topic keys format: [topic_id] [weight] [word1] [word2] ...
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            topic_id = int(parts[0])
            if topic_nums is not None and topic_id not in topic_nums:
                continue

            # Update postfix instead of description to keep a single line
            pbar.set_postfix({"Topic": topic_id})

            # Grab top n_terms keywords representing the cluster
            top_words = parts[2].split(" ")[: client.config.n_terms]

            # Fetch generated label from LLM
            try:
                label = client.generate_label(top_words=top_words)
                topic_labels[topic_id] = label.strip('"').strip("'").replace("\\", "")
            except Exception as e:
                # Fallback gracefully to old index notation if an API error occurs
                topic_labels[topic_id] = f"Topic {topic_id} (Labelling failed: {e})"

    return topic_labels
