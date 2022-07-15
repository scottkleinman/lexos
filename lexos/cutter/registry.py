"""registry.py."""
import re
from typing import Callable

import catalogue

# Create the registry
tokenizers = catalogue.create("lexos", "tokenizers")

# Whitespace tokenizer
def whitespace_tokenizer(text: str) -> list:
    """Tokenize on whitespace, keeping whitespace.

    Args:
        text: The text to tokenize.

    Returns:
        list: A list of pseudo-word tokens.
    """
    return re.findall(r"\S+\s*", text)


# Character tokenizer
def character_tokenizer(text: str) -> list:
    """Tokenize by single characters, keeping whitespace.

    Args:
        text: The text to tokenize.

    Returns:
        list: A list of character tokens.
    """
    return [char for char in text]


# Linebreak tokenizer
def linebreak_tokenizer(text: str) -> list:
    """Tokenize by linebreaks, keeping whitespace.

    Args:
        text: The text to tokenize.

    Returns:
        list: A list of line tokens.
    """
    return text.splitlines(keepends=True)


# Register default normalize components
tokenizers.register("whitespace", func=whitespace_tokenizer)
tokenizers.register("character", func=character_tokenizer)
tokenizers.register("linebreak", func=linebreak_tokenizer)


def load(s: str) -> Callable:
    """Load a single tokenizer from a string.

    Args:
        s: The name of the function.

    Returns:
        list: A tokenizer function.
    """
    return tokenizers.get(s)
