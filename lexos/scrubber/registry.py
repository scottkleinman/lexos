"""registry.py."""
import catalogue

from . import normalize, remove, replace

# Create the registry
scrubber_components = catalogue.create("lexos", "scrubber_components")

# Register default normalize components
scrubber_components.register("bullet_points", func=normalize.bullet_points)
scrubber_components.register("hyphenated_words", func=normalize.hyphenated_words)
scrubber_components.register("lower_case", func=normalize.lower_case)
scrubber_components.register("quotation_marks", func=normalize.quotation_marks)
scrubber_components.register("repeating_chars", func=normalize.repeating_chars)
scrubber_components.register("unicode", func=normalize.unicode)
scrubber_components.register("whitespace", func=normalize.whitespace)

# Register default remove components
scrubber_components.register("accents", func=remove.accents)
scrubber_components.register("brackets", func=remove.brackets)
scrubber_components.register("digits", func=remove.digits)
scrubber_components.register("new_lines", func=remove.new_lines)
scrubber_components.register("pattern", func=remove.pattern)
scrubber_components.register("punctuation", func=remove.punctuation)
scrubber_components.register("tabs", func=remove.tabs)
scrubber_components.register("tags", func=remove.tags)

# Register default replace components
scrubber_components.register("currency_symbols", func=replace.currency_symbols)
scrubber_components.register("re_digits", func=replace.digits)
scrubber_components.register("emails", func=replace.emails)
scrubber_components.register("emojis", func=replace.emojis)
scrubber_components.register("hashtags", func=replace.hashtags)
scrubber_components.register("re_pattern", func=replace.pattern)
scrubber_components.register("phone_numbers", func=replace.phone_numbers)
scrubber_components.register("re_punctuation", func=replace.punctuation)
scrubber_components.register("special_characters", func=replace.special_characters)
scrubber_components.register("tag_map", func=replace.tag_map)
scrubber_components.register("urls", func=replace.urls)
scrubber_components.register("user_handles", func=replace.user_handles)

def load_components(t: tuple):
    """Load components from a tuple.

    Args:
        t: A tuple containing string names of functions.
    """
    for item in t:
        yield scrubber_components.get(item)
