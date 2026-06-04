# Recipes

Recipes are sample configuration files variables needed to achieve specific results:

## `update_en_core_web_sm`

This recipe updates the built-in spaCy "en_core_web_sm" model. It retrains the tagger whilst keeping the other pipeline components "frozen" in the same state as the built-in model. Since the tagger relies on `tok2vec`, it replaces the default vectors with its own (thus the use of `replace_listeners`). When training, the `tagger`, `attribute_ruler`, `lemmatizer`, `parser`, and `ner` components are all included in the annotations. Note that for the English-language models, spaCy POS and morphological feature labels are determined by `attribute_ruler` mapping rules, rather than by the `morphologizer` pipeline component.

**Testing Notes:**

- This recipe has been used to produce a valid model that reproduces the results of "en_core_web_sm" with updated data.
- The original "en_core_web_sm" model identifies the term "eleventy" in the first sentence of _The Lord of the Rings_ as a noun. This labelling was corrected in the training data (a single sentence) to match the labelling for "seventy", and the updated model trained on this data now correctly labels "eleventy" as a number.
- This is a toy sample. A use case would be to create training data using "en_core_web_sm" on, say, a novel, correct the labels, and then re-train the model.

## `multilingual_tagger`

This recipe trains from scratch a multilingual model with only the tagger in the pipeline. It is thus a good starting point for a wide variety of languages.

## `tagger_parser_ud`

This recipe is simply a copy of the configuration file for the spaCy [`tagger_parser_ud` project](https://github.com/explosion/projects/tree/v3/pipelines/tagger_parser_ud). If users wish to replicate that the project from within the Lexos API, they can simply initialise their project with this config before copying their assets.
