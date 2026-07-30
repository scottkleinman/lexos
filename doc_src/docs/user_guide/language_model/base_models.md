# Choosing and Setting Up Base Models

When you call `LanguageModel()` with `base_model=...`, each pipeline component starts from existing weights rather than random initialisation. This is *fine-tuning* — it gives us a starting point that you then adjust and improve for your needs. This will typically produces better results than training from scratch, provided the source model is a reasonable starting point for your target. For example if you want a model for a specific dialect or historical stage of a langauge, you can get a base model for the langauge as a whole then fine-tune on your specific data.

This guide covers how to find or build a suitable source model for each situation.

---

## Understanding the `base_model` dict

The dict maps each component name to a source model. The source is either an installed spaCy model name or a local path to a trained model directory:

```python
base_model={
    "tok2vec":              "en_core_web_sm",          # installed model
    "tagger":               "en_core_web_sm",
    "morphologizer":        "path/to/ud_model/model-best",  # local path
    "trainable_lemmatizer": "path/to/ud_model/model-best",
    "parser":               "path/to/ud_model/model-best",
}
```

Components omitted from the dict are initialised from scratch. You can mix sourced and scratch components in any combination.

---

## The annotation scheme problem

Before choosing a source model, understand the annotation scheme issue.

spaCy's English models (`en_core_web_*`) were trained on **OntoNotes** data, which uses different dependency labels from **Universal Dependencies**:

| OntoNotes (spaCy English) | Universal Dependencies |
| --- | --- |
| `dobj` | `obj` |
| `prep` | `case` |
| `pobj` | `obl` |
| `nsubjpass` | `nsubj:pass` |

If your training data uses UD labels (as the Universal Dependencies treebanks do), sourcing the parser from `en_core_web_sm` gives a mismatched starting point — the model will need to unlearn its existing labels before learning yours. For the parser, **always source from a UD-trained model** when your data is UD-annotated.

For `tok2vec` and `tagger`, the mismatch is less critical: tok2vec weights are label-agnostic, and tagger Penn labels are close enough to UD POS that the representations transfer well.

---

## Option 1: Pre-trained spaCy models

spaCy provides models for 70+ languages. These are the easiest starting point and cover `tok2vec` and `tagger` for most languages.

**Find models:** [https://spacy.io/models](https://spacy.io/models)

**Install:**

```bash
python -m spacy download en_core_web_sm   # English, small
python -m spacy download de_core_news_sm  # German, small
```

**Component coverage in English spaCy models:**

| Component | In `en_core_web_*`? | Notes |
| --- | --- | --- |
| tok2vec | Yes | Good for all English use cases |
| tagger | Yes | Penn Treebank labels (close to UD POS) |
| morphologizer | No | Not present; use UD-trained model |
| trainable_lemmatizer | No | Uses rule-based lemmatizer instead |
| parser | Yes, but OntoNotes labels | Mismatch with UD data — use UD-trained model |

**For non-English languages — check UD coverage first.** Many non-English spaCy models include UD-trained parsers and morphologizers in addition to tok2vec and tagger. Before looking at Options 2 or 3, check your language's model page at [https://spacy.io/models](https://spacy.io/models) and look at the component list to see whats available and sources to see what type of data its trained on (UD or OntoNotes).

SpaCy models generally don't offer all 5 specifcally trainable lemmatizers. If the model only covers `tok2vec` and `tagger` (as the English models do), use it for those two and a UD-trained model (Options 2 or 3 below) for `morphologizer`, `trainable_lemmatizer`, and `parser`.

**Rule of thumb:** Any component omitted from the `base_model` dict is trained from scratch. You can mix and match — source the components with a good match and omit the rest.

---

## Option 2: Community UD-trained models

Before training your own base model, check whether one already exists:

- **spaCy Universe** — [https://spacy.io/universe](https://spacy.io/universe) — community-contributed models, many UD-trained
- **Hugging Face Hub** — search `spacy` + your language code
- **spaCy model releases** — some language teams publish UD-trained models alongside official releases

If you find a UD-trained model, use its path for all five components:

```python
base_model="path/to/downloaded/ud_model/model-best"
# or as a dict if you want different sources per component
```

---

## Option 3: Train your own base model from related UD data

Use this option when no pre-built UD model exists for your language (Options 1 and 2 both came up empty), but a related UD treebank is available — for example, a modern-language treebank you plan to use as a stepping stone before fine-tuning on your specific historical, domain-specific, or dialect data.

The two-step process:

1. **Train a base model** on the related UD treebank (this step — follow the main tutorial with `base_model` omitted)
2. **Fine-tune** on your specific UD data, sourcing all components from the `model-best` produced in step 1 (follow the main tutorial again, passing that path as `base_model`)

**For step 1, follow the [main tutorial](../../tutorials/language_model/tutorial.ipynb).** Pass the following:

- **`data`** — the related UD treebank. See [Getting Training Data](training_data.md) to find one.
- **`base_model`** — omit it entirely, or omit only the components with no suitable source. Any component not specified trains from scratch.

After training, the `model-best` directory is your base model. Pass its path as `base_model` when you run the main tutorial again on your specific data.

**Don't have specific UD data yet?** Use the trained base model to bootstrap your own domain-specific annotations: the model pre-annotates new text, you correct the output, and you iterate. See [Building a Specialized Model: Iterative Workflow](../../tutorials/language_model/advanced_workflow.ipynb) for the full process.

**Multilingual fallback:** if no spaCy model exists for your language at all, use `lang="xx"` (spaCy's multilingual code). Results will depend on the quantity and quality of your training data.
