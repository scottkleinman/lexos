# Tokenizer

## Overview

This module is designed to provide tools for tokenizing texts using natural language processing (NLP) models. There are [four tokenizer classes](#tokenizers), two of which use spaCy NLP models to tokenize input texts. These modules must be installed locally in order for the tokenizer module to run correctly. By default, the `xx_sent_ud_sm` model is used. If another model is desired, those models must also be installed.

## Tokenizers

The tokenizer module includes 4 classes across 2 files.

- [lexos.tokenizer](tokenizer.md)
  - [`SliceTokenizer`](tokenizer.md#lexos.tokenizer.SliceTokenizer)
    - Simple slice tokenizer
    - Can be used to generate character tokens/ngrams
  - [`Tokenizer`](tokenizer.md/#lexos.tokenizer.Tokenizer)
    - Tokenizes texts using spaCy NLPs
    - Takes in raw text as input
    - Returns spaCy docs that contains the tokens of the given input text
    - Includes filtering of digits, punctuation, stopwords, etc.
    - Supports all spaCy NLP models
  - [`WhitespaceTokenizer`](tokenizer.md/#lexos.tokenizer.WhitespaceTokenizer)
    - Tokenizes on whitespace
- [lexos.tokenizer.ngrams](ngrams.md)
  - [`Ngrams`](ngrams.md/#lexos.tokenizer.ngrams.Ngrams)
    - Returns ngrams from a text, spaCy doc, or list of tokens
    - Includes filtering of digits, punctuation, stopwords, whitespace, etc.
    - User selected size of ngrams
