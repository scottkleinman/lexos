# Getting Training Data

The module expects training data in **CONLL-U format** — the standard used by the [Universal Dependencies](https://universaldependencies.org/) project. This guide covers where to find existing CONLL-U data and how to create your own.

---

## What CONLL-U looks like

CONLL-U files are plain text: one token per line, 10 tab-separated fields, blank lines between sentences.

```text
# sent_id = ewt-train-1
# text = From the AP comes this story.
1   From    from    ADP    IN    _              3   case    _   _
2   the     the     DET    DT    Definite=Def   3   det     _   _
3   AP      AP      PROPN  NNP   Number=Sing    4   obl     _   _
4   comes   come    VERB   VBZ   ...            0   root    _   _
5   this    this    DET    DT    ...            6   det     _   _
6   story   story   NOUN   NN    Number=Sing    4   nsubj   _   _
7   .       .       PUNCT  .     _              4   punct   _   SpaceAfter=No
```

**The 10 fields:** ID · FORM (word) · LEMMA · UPOS (universal POS) · XPOS (language POS) · FEATS (morphological features) · HEAD (dependency head) · DEPREL (dependency relation) · DEPS · MISC

The module trains all five pipeline components from these fields:

- UPOS → morphologizer (also trains POS via tagger using XPOS)
- FEATS → morphologizer
- LEMMA → trainable_lemmatizer
- HEAD + DEPREL → parser

> **Note on OntoNotes:** OntoNotes is a separate annotation scheme used by spaCy's English models (`en_core_web_*`). OntoNotes data is proprietary (LDC licence required) and uses incompatible dependency labels (`dobj`/`prep`/`pobj` vs. UD's `obj`/`case`/`obl`). This tutorial does not cover OntoNotes-based training. Do not mix OntoNotes data with UD treebank data.

---

## Option 1: Download an existing UD treebank

The [Universal Dependencies project](https://universaldependencies.org/) distributes annotated treebanks for 100+ languages, all in CONLL-U format. This is the fastest path.

**Download:** [https://universaldependencies.org/#download](https://universaldependencies.org/#download)

Most treebanks are under Creative Commons or similar open licences. Each download includes pre-split train/dev/test files — you can skip `split_conllu()` and pass them directly to `copy_assets()`:

```python
model.copy_assets(
    train="path/to/en_ewt-ud-train.conllu",
    dev="path/to/en_ewt-ud-dev.conllu",
    test="path/to/en_ewt-ud-test.conllu",
)
```

**English treebanks:**

| Treebank | Train sentences | Domain |
| --- | --- | --- |
| EWT (English Web Treebank) | ~12,500 | Web text: emails, reviews, forums, blogs |
| GUM (Georgetown Multilayer) | ~9,000 | Diverse: academic, fiction, news, interviews |
| ParTUT | ~1,800 | Parliament proceedings, Wikipedia |
| LinES | ~3,000 | Fiction and technical writing |

Combining treebanks (e.g. EWT + GUM) increases training data and usually improves results.

**Finding treebanks for other languages:** Browse by language at [https://universaldependencies.org/](https://universaldependencies.org/). Most major European languages have multiple treebanks. For less-resourced languages, there may be only one small treebank or none at all.

---

## Option 2: Annotate your own data

If no treebank exists for your language, historical period, or domain, you will need to create your own by making your own annotations.

### Annotation tools

| Tool | Notes |
| --- | --- |
| [INCEpTION](https://inception-project.github.io/) | Full-featured; supports UD annotation natively; collaborative; exports CONLL-U; free and open source |
| [Arborator Grew](https://arboratorgrew.elizia.net/) | Designed specifically for UD dependency trees; web-based; exports CONLL-U |
| [Prodigy](https://prodi.gy/) | Commercial tool from Explosion (creators of spaCy); tight spaCy integration; can auto-annotate with your model, display predictions for correction, and export training data; designed for fast keyboard-driven correction; requires a licence — SpaCy offers many helpful tools like this worth checking out, though most aren't free-use |
| [brat](https://brat.nlplab.org/) | Simpler and general-purpose; good for getting started; requires some setup for UD |

For new UD annotation projects, **INCEpTION** or **Arborator Grew** are the recommended free tools — both support the UD annotation guidelines natively. **Prodigy** is especially useful if you are running the iterative bootstrap-and-correct workflow described in [Building a Specialized Model: Iterative Workflow](../../tutorials/language_model/advanced_workflow.ipynb) — it wraps the auto-annotate-and-correct loop into a single UI built for speed.

### UD annotation guidelines

Universal Dependencies maintains detailed guidelines at [https://universaldependencies.org/guidelines.html](https://universaldependencies.org/guidelines.html). Read the guidelines for your language before starting — annotation decisions made early are expensive to revise later.

For historical or non-standard varieties (Early Modern English, Middle French, etc.), the modern language guidelines are the starting point. Document any variant-specific decisions in an annotation manual so consistency is maintained across annotators.

### How many sentences do you need?

Annotation is slow — careful UD annotation takes roughly 30–60 minutes per 100 tokens (longer for complex sentences or unfamiliar genres).

**For fine-tuning on top of a base model:**

| Annotated sentences | Expected outcome |
| --- | --- |
| 100–300 | Visible improvement on in-domain POS and morphology; parser results noisy |
| 300–1,000 | Solid POS and morphology; parser converges; lemmatizer may still underperform rule-based |
| 1,000+ | Reliable improvements across all components including lemmatization |

**For training a base model from scratch** (no existing source model), add roughly an order of magnitude: expect marginal results below ~1,000 sentences, reasonable results from ~5,000.

---

## Option 3: Silver data (bootstrap and correct)

If hand annotation from scratch is too slow, use the model to accelerate the process:

1. Run your current model (or any related model) on unannotated text to produce automatic annotations
2. Correct the output in an annotation tool — correcting errors is significantly faster than annotating from scratch
3. Add the corrected sentences to your training set and retrain
4. Repeat — each round improves accuracy and speeds up correction

For full instructions on running this workflow — including how to export annotations from your model, how to correct CONLL-U data, and how to combine rounds — see [Building a Specialized Model: Iterative Workflow](../../tutorials/language_model/advanced_workflow.ipynb).

---

## Format requirements and tips

- Files must be **UTF-8 encoded**
- Sentences are separated by **blank lines** (not just between tokens)
- Sentence-level comment lines (`# sent_id`, `# text`) are recommended but not required by the module
- **Multi-word tokens** (e.g. CONLL-U line `1-2  gonna`) are handled by `merge_subtokens=True` in `convert_assets()` — the default
- The module's `split_conllu()` function expects a **single file** and handles splitting into train/dev/test. If you already have separate files, pass them directly to `copy_assets()` and skip `split_conllu()`
- If your data is in a format other than CONLL-U (e.g. Penn Treebank `.mrg`, CoNLL-2003 IOB), you will need to convert it first. spaCy's `spacy convert` command handles some formats; for others, custom conversion scripts are needed
