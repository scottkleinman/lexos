"""whitespace_counter.py.

Last Update: December 20, 2025
Last Tested: December 20, 2025

This module provides a whitespace tokenizer that captures line breaks and counts runs of spaces.
"""

import re
from typing import Iterable

import spacy
from pydantic import validate_call
from spacy.tokens import Doc, Token

from lexos.exceptions import LexosException
from lexos.tokenizer import Tokenizer

try:
    default_model = spacy.load("xx_sent_ud_sm")
except ImportError:
    raise LexosException(
        "The default model is not available. Please run `python -m spacy download xx_sent_ud_sm` from the command line."
    )


class WhitespaceCounter(Tokenizer):
    """Whitespace tokenizer that captures line breaks and counts runs of spaces."""

    def _get_token_widths(self, text: str) -> tuple[list[str], list[int]]:
        """Get the widths of tokens in a doc.

        Args:
            text (str): The input text.

        Returns:
            tuple[list[str], list[int]]: A tuple containing the tokens and widths.
        """
        # Pattern: words, line breaks, or runs of spaces
        pattern = re.compile(r"([^\s\n]+)|(\n)|([ ]{2,})|([ ])")
        tokens = []
        widths = []
        for match in pattern.finditer(text):
            word, newline, multi_space, single_space = match.groups()
            if word:
                tokens.append(word)
                widths.append(len(word))  # Use number of characters in word
            elif newline:
                tokens.append("\n")
                widths.append(1)  # Use 1 to indicate a line break
            elif multi_space:
                tokens.append(" ")
                widths.append(len(multi_space))
            elif single_space:
                tokens.append(" ")
                widths.append(1)
        return tokens, widths

    @validate_call
    def make_doc(
        self, text: str, max_length: int = None, disable: list[str] = []
    ) -> Doc:
        """Return a doc from a text.

        Args:
            text (str): The text to be parsed.
            max_length (int): The maximum length of the doc.
            disable (list[str]): A list of spaCy pipeline components to disable.

        Returns:
            Doc: A spaCy doc object.
        """
        # Override instance settings with keyword arguments
        if max_length:
            self.max_length = max_length
            self.nlp.max_length = max_length
        if disable:
            self.nlp.select_pipes(disable=disable)
        tokens, widths = self._get_token_widths(text)
        if not Token.has_extension("width"):
            Token.set_extension("width", default=0)
        doc = Doc(self.nlp.vocab, words=tokens)
        for token, count in zip(doc, widths):
            token._.width = count
        # Apply pipeline components manually, skipping those in 'disable'
        for name, proc in self.nlp.pipeline:
            if name not in disable:
                doc = proc(doc)
        return doc

    @validate_call
    def make_docs(
        self,
        texts: Iterable[str],
        max_length: int = None,
        disable: Iterable[str] = [],
        chunk_size: int = 1000,
    ) -> Iterable[Doc]:
        """Return a generator of docs from an iterable of texts, processing in chunks.

        Args:
            texts (Iterable[str]): The texts to process.
            max_length (int, optional): Maximum doc length.
            disable (Iterable[str], optional): Pipeline components to disable.
            chunk_size (int, optional): Number of docs to process per chunk.

        Yields:
            Doc: spaCy Doc objects.
        """
        if max_length:
            self.max_length = max_length
            self.nlp.max_length = max_length

        if not Token.has_extension("width"):
            Token.set_extension("width", default=0)
        enabled_pipes = [
            (name, proc) for name, proc in self.nlp.pipeline if name not in disable
        ]

        def chunker(iterable, size):
            chunk = []
            for item in iterable:
                chunk.append(item)
                if len(chunk) == size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

        for text_chunk in chunker(texts, chunk_size):
            docs = []
            for text in text_chunk:
                tokens, widths = self._get_token_widths(text)
                doc = Doc(self.nlp.vocab, words=tokens)
                for token, count in zip(doc, widths):
                    token._.width = count
                docs.append(doc)
            for _, proc in enabled_pipes:
                docs = [proc(doc) for doc in docs]
            yield from docs
