# Using Milestones

Milestones are specified locations in the text that designate structural or sectional divisions. A milestone can be either a designated unit *within* the text or a placemarker inserted between sections of text. The Lexos `milestones` module provides methods for identifying milestone locations by searching for patterns you designate. There three separate classes for identifying milestones in different ways: `StringMilestones`, `TokenMilestones`, and `SpanMilestones`. We will look at each of these in turn.

## `StringMilestones`

The `StringMilestones` class is used for extracting and storing milestones in strings or spaCy Doc objects. It uses regular expressions to find patterns and returns their locations and text.

Here is a basic example:

```python
# Import the StringMilestones class
from lexos.milestones.string_milestones import StringMilestones

# A sample doc
doc = "The quick brown fox jumps over the lazy dog."

# Create a String Milestones instance and search for the pattern "quick"
milestones = StringMilestones(doc=doc, patterns="quick")

# Print the start character, end character, and text of each milestone
for milestone in milestones:
    print(milestone.start, milestone.end, milestone.text)

# 4 9 quick
```

The `.spans` property returns a list of `StringSpan` objects that have `text`, `start`, and `end` attributes. As shown, above, you can iterate through the `Milestones` object direclt to return items from this list.

You can use regex patterns to match more complex milestones. For instance, in the example below, we match the string "Chapter" followed by one or more digits.

```python
text = "Chapter 1\nThis is the text of the first chapter.\nChapter 2\nThis is the text of the second chapter."

milestones = StringMilestones(doc=text, patterns="Chapter \\d+")

# Print the start character, end character, and text of each milestone
for milestone in milestones:
    print(milestone.start, milestone.end, milestone.text)

# 0 9 Chapter 1
# 49 58 Chapter 2
```

By default, searches are case-sensitive. To ignore case, set `case_sensitive=False`:

```python
text = "Chapter 1\nThis is the text of the first chapter.\nChapter 2\nThis is the text of the second chapter."

milestones = StringMilestones(doc=text, patterns="Chapter", case_sensitive=False)

# Print the start character, end character, and text of each milestone
for milestone in milestones:
    print(milestone.start, milestone.end, milestone.text)

# 0 7 Chapter
# 40 47 chapter
# 49 56 Chapter
# 90 97 chapter
```

You can pass a list of patterns:

```python
milestones = StringMilestones(doc=text, patterns=["This", "Chapter"], case_sensitive=False)

for milestone in milestones:
    print(milestone.start, milestone.end, milestone.text)

# 0 7 Chapter
#10 14 This
#40 47 chapter
#49 56 Chapter
#59 63 This
#90 97 chapter
```

### Updating Patterns and Settings

You can update the patterns or case sensitivity using the `.set()` method:

```python
milestones.set("The", case_sensitive=False)
for milestone in milestones:
    print(milestone.start, milestone.end, milestone.text)

# 18 21 the
# 30 33 the
# 67 70 the
# 79 82 the
```

This will erase any previous settings.

## `TokenMilestones`

The `TokenMilestones` class is used for extracting and storing milestones tokenized text, such as spaCy Doc objects. It differs from `StringMilestones` in that it matches against full tokens.

Let's start by importing the Lexos `Tokenizer` class to create a spaCy Doc object. We'll also import the `TokenMilestones` class.

```python
# Import Tokenizer and TokenMilestones
from lexos.tokenizer import Tokenizer
from lexos.milestones.token_milestones import TokenMilestones

text = "Chapter 1: Introduction. Chapter 2: Methods."
tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc(text)
```

Now, let's extract milestones for the word "Chapter".

```python
milestones = TokenMilestones(doc=doc)
matches = milestones.get_matches(patterns=["Chapter"])
print(matches)

# ["Chapter", "Chapter"]
```

This functions as a quick check to make sure that the pattern as worked. By default, `get_matches` detects tokens that exactly match the pattern (although you can set `case_sensitive` to False). Other ways of matching will be discussed below.

Unlike `StringMilestones` the output of `TokenMilestones` is stored *in the original Doc object*. To do this, you must pass the matches to the Doc object using the `set_milestones()` method.

```python
milestones.set_milestones(matches)
```

The effect of this method is to create two custom extensions to the doc's tokens: `milestone_iob` and `milestone_label`. The former indicates whether the token is inside ("I"), outside ("O"), or at the beginning ("B") of a milestone. The `milestone_label` provides the complete text of the milestone for any "B" token.

```python
milestones = TokenMilestones(doc=doc)
matches = milestones.get_matches(patterns="Chapter")
milestones.set_milestones(matches)
for token in doc:
    print(token.text, token._.milestone_iob, token._.milestone_label)

# Chapter B Chapter
# 1 O
# : O
# Introduction O
# . O
# Chapter B Chapter
# 2 O
# : O
# Methods O
# . O
```

!!! note
    The `milestone_iob` and `milestone_label` attributes are custom spaCy extensions, which need to be accessed with the `._.` prefix.

In some cases, the milestone text itself is content that you do not wish to include in your analysis. For instance, you may have placed a milestone marker like `<MILESTONE>` at points in your original text, and you wish to identify divisions *on either side* of that marker. The `set_milestones` method provides a `start` parameter which allows you to choose to set the "B" tag "before" or "after" the matched token(s). So, if you are matching "Chapter 2", `start=before` would place the "B" attribute on the token before "Chapter", and `start=after` (the more likely choice) would place the "B" tag on the token after "Chapter 2". If, for instance, you were cutting the document into segments, you might want each segment to begin with content, rather than a chapter heading. If course, this would leave the chapter heading (the milestone) at the end of the previous segment. If you wish to remove the milestone tokens from the document completely, you can set `remove=True`.

!!! note
    Note that the `remove` parameter creates a copy of the original Doc object with the milestone tokens removed. This means that the remaining tokens may not have the same indices in the copy as in the original.

### Setting the Search Mode

As noted above, by default `TokenMilestones` searches for token strings that match the search pattern(s) exactly (allowing for the case-sensitivity setting). You can match multiple tokens by setting `mode="phrase"` in `get_matches`.

```python
milestones = TokenMilestones(doc=doc)
matches = milestones.get_matches(patterns="Chapter 1", mode="phrase")
milestones.set_milestones(matches)
for token in doc:
    print(token.text, token._.milestone_iob, token._.milestone_label)

# Chapter B Chapter
# 1 I
# : O
# Introduction O
# . O
# Chapter O
# 2 O
# : O
# Methods O
# . O
```

You can also perform more nuanced matching using `mode="rule"`. This allows you to submit rules using spaCy's powerful `Matcher` class. Here is an example:

```python
pattern = [{"TEXT": "Chapter"}, {"IS_DIGIT": True}]
milestones = TokenMilestones(doc=doc)
matches = milestones.get_matches(patterns=[pattern], mode="rule")
milestones.set_milestones(matches)
for token in doc:
    print(token.text, token._.milestone_iob, token._.milestone_label)

# Chapter B Chapter
# 1 I
# : O
# Introduction O
# . O
# Chapter B Chapter
# 2 I
# : O
# Methods O
# . O
```

!!! note
    See spaCy's <a href="https://spacy.io/usage/rule-based-matching" target="_blank">Rule-Based Matching documentation</a> for a complete explanation of the syntax for formulating matching patterns. Note that your case-sensitivity setting may override any case handling in your search pattern(s).

### Removing and Resetting Milestones

You can remove previously set milestones with the `remove` method. For instance, the milestones set above can be removed as follows:

```python
milestones.remove(matches)
for token in doc:
    print(token.text, token._.milestone_iob, token._.milestone_label)

# Chapter O
# 1 O
# : O
# Introduction O
# . O
# Chapter O
# 2 O
# : O
# Methods O
# . O
```

If you wish to remove *all* milestones efficiently from a Doc object, simply call the `reset` method.

## `SpanMilestones`

Span milestones are used to group spans together for analysis or visualization. Span milestones differ from normal milestones in that milestones are "invisible" structural boundaries between spans or groups of spans (e.g. sentence or line breaks). Thus, instead of storing a list of patterns representing milestones, span milestones store the groups of spans themselves.

There are three subclasses that inherit from `SpanMilestones`: `LineMilestones`, `SentenceMilestones`, and `CustomMilestones`. The `LineMilestones` class is the easiest to understand. It splits the text on line breaks and generates a list of spaCy `Span` objects. These can be accessed through the `spans` of both the `Milestones` and the `Doc` objects:

```python
from lexos.milestones.span_milestones import LineMilestones

text = "Chapter 1: Introduction.\nChapter 2: Methods."
tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc(text)
milestones = LineMilestones(doc=doc)
milestones.set()

# Print the text of the spans in a list
print(milestones.spans)
print(doc.spans)

# [Chapter 1: Introduction., Chapter 2: Methods.]
# {'milestones': [Chapter 1: Introduction., Chapter 2: Methods.]}
```

You can iterate through the `milestones.spans` list directly, as shown below:

```python
for milestone in milestones:
    print(milestone.start, milestone.end, milestone.text)

# 0 5 Chapter 1: Introduction.
# 6 11 Chapter 2: Methods.
# 0 5 Chapter 1: Introduction.
# 6 11 Chapter 2: Methods.
```

There is also a `to_list()` method, which returns a list of dictionaries providing additional indexing information, should you need it.

```python
print(milestones.to_list())
[{'text': 'Chapter 1: Introduction.', 'characters': 'Chapter 1: Introduction', 'start_token': 0, 'end_token': 5, 'start_char': 0, 'end_char': 23}, {'text': 'Chapter 2: Methods.', 'characters': 'Chapter 2: Methods', 'start_token': 6, 'end_token': 11, 'start_char': 25, 'end_char': 43}]
```

!!! note
    By default, the pattern used to identify line breaks is "\n", but this can be customed with the `pattern` keyword when calling `set`. By default, all line breaks are not included in the milestone spans, but this can be disabled with `remove_linebreak= False`.

The `SentenceMilestones` class works in a similar way:

```python
from lexos.milestones.span_milestones import SentenceMilestones

text = "This is sentence 1. This is sentence 2."
tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc(text)
milestones = SentenceMilestones(doc=doc)
milestones.set()

# Print the text of the spans in a list
print(milestones.spans)
print(doc.spans)
print(list(doc.sents))

# [This is sentence 1., This is sentence 12]
# {'milestones': [This is sentence 1., This is sentence 2.]}
# [This is sentence 1., This is sentence 12]
```

Note that the `Doc` object already has a `sents` attribute that contains a generator sentence spans. This is generated automatically *if and only if* your language model has a sentence segmenter. If it does not, you cannot use the `SentenceMilestones` class and will need to rely on the custom approach discussed below. See the <a href="https://spacy.io/usage/linguistic-features#sbd" target="_blank">spaCy documentation</a> for further information on creating Doc objects with sentence segmentation in the pipeline.

The `CustomMilestones` class can be used to generate milestones based on arbitrary spans. A good way to demonstrate this is to reproduce the sentence segments shown above.

```python
from lexos.milestones.span_milestones import CustomMilestones

text = "This is sentence 1. This is sentence 2."
tokenizer = Tokenizer(model="en_core_web_sm")
doc = tokenizer.make_doc(text)
spans = [doc[0:5], doc[5:10]]
milestones = CustomMilestones(doc=doc)
milestones.set(spans)

# Print the text of the spans in a list
print(milestones.spans)

# [This is sentence 1., This is sentence 2.]
```

Here we have manually set our spans to include the first and last five tokens, which happen to coincide with sentence boundaries. But we could easily create spans separated in other ways.

!!! important
    Unlike the previous two classes, `CustomMilestones` requires you to pass a list of `Span` objects to the `set` method.

### Additional Settings and Methods

All three classes have additional `max_label_length` and `step` parameters. The `max_label_length` is the maximum number of characters in a token's `milestones_label` attribute (the default is 20). The `step` parameter takes an integer indicating the number of spans per item in the milestones list. For instance, if you wanted to have a milestone every tenth sentence, setting `step=10` would mean that every item in the `milestones.spans` list would consist of ten sentences. This parameter can similarly be used to group lines or custom spans.

All three classes have a `reset` method, which will remove all spans from both the `Milestones` and `Doc` objects.
