# Remove

The `remove` component of `Scrubber` contains a set of functions for removing strings and patterns from text.

### ::: lexos.scrubber.remove.accents
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.scrubber.remove.brackets
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.scrubber.remove.digits
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.scrubber.remove.new_lines
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.scrubber.remove.pattern
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.scrubber.remove.punctuation
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.scrubber.remove.tabs
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.scrubber.remove.tags
    rendering:
      show_root_heading: true
      heading_level: 3

!!! note
  Tag handling has been ported over from the Lexos web app, which uses `BeautifulSoup` and `lxml` to parse the tree. It will be good to watch the development of <a href="https://github.com/rushter/selectolax" target="_blank">selectolax</a>, which claims to be more efficient, at least for HTML. An implementation with spaCy is available in the <a href="https://github.com/pmbaumgartner/spacy-html-tokenizer" target="_blank">spacy-html-tokenizer</a>, though it may not be right for integration into Lexos since the output is a doc in which tokens are sentences.
