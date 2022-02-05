# Extensions

This is a set of extensions to spaCy docs allowing custom attributes
and methods. Typically, they woudld be accessed with an underscore
prefix like `doc._.is_fruit` or `doc._.get("is_fruit")`.

Extensions are set with code like

```python
fruits = ["apple", "pear", "banana", "orange", "strawberry"]
is_fruit_getter = lambda token: token.text in fruits
Token.set_extension("is_fruit", getter=is_fruit_getter)
```

See the spaCy <a href="https://spacy.io/usage/processing-pipelines#custom-components-attributes" targe="_blank">custom attributes</a> documentation for full details.

### ::: lexos.tokenizer.extensions.is_fruit_getter
    rendering:
      show_root_heading: true
      heading_level: 3

!!! note

    This is really a proof of concept function. A better example can be added in the future.
