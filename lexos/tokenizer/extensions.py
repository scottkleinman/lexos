"""extensions.py.

This is a set of extensions to spaCy docs allowing custom attributes
and methods. Typically, they would be accessed with an underscore
prefix like `token._.is_fruit` or as `token._.get("is_fruit")`.
"""
from spacy.tokens import Token

# A custom attribute to return a boolean as to whether a token is a fruit
fruits = ["apple", "pear", "banana", "orange", "strawberry"]
is_fruit_getter = lambda token: token.text in fruits
Token.set_extension("is_fruit", getter=is_fruit_getter)
