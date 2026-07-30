# Training Language Models

The `language_model` module Python wrapper around spaCy's training workflow for fine-tuning language models on custom corpora. The module is designed for any language or domain where a generic pre-built model is insufficient — historical varieties, specialised domains, low-resource languages, or any text where higher annotation accuracy matters.

The module connects directly to the Lexos tokenizer: once a model is trained and packaged, pass its path to `tokenizer.make_doc()` and it will be used for all linguistic feature assignment.

Users interact with the `LanguageModel` class and `split_conllu` utility — there is no need to edit config files or run command-line tools. Developers who want to add new training configurations do so by adding recipe `.cfg` files to the `recipes/` folder.

> **Note on terminology:** In spaCy, *pipeline* refers to a configuration of components and algorithms for assigning feature labels to text, while *model* refers to the statistical weights those algorithms produce. Since trained pipelines generate statistical models, these terms overlap — this README uses them interchangeably where the distinction is not important.

---

## Getting started

The fastest path to a trained model is the step-by-step tutorial notebook:

- **[Fine-tuning a Language Model for Your Texts](../../tutorials/language_model/tutorial.ipynb){target="_blank"}** — the main tutorial; covers the complete workflow from data preparation to a packaged model

Supplementary guides for specific setup tasks:

- **[Getting Training Data](training_data.md)** — where to find CONLL-U treebanks, how to annotate your own data, and the iterative bootstrap method
- **[Choosing and Setting Up Base Models](base_models.md)** — how to find or train a suitable source model for your language and annotation scheme
- **[Building a Specialized Model: Iterative Workflow](../../tutorials/language_model/advanced_workflow.ipynb)** — for users with no existing training data, or who want to improve a model they have already trained
- **[Transformer-based Training](../../tutorials/language_model/transformer_tutorial.ipynb)** — fine-tuning a pre-trained transformer backbone instead of the CNN for higher accuracy (requires a GPU); MacBERTh, a historical-English model, is the worked example
- **[Tuning Training Settings](training_settings.md)** — learning rates, early stopping, batch sizes, and data normalization for both CNN and transformer pipelines

---

## Architecture

The module has two layers:

**`LanguageModel` class** — manages the directory structure, generates or loads the spaCy config, and exposes the training lifecycle as simple method calls. It is a thin orchestration layer; it contains no ML logic.

**spaCy Python API** — does the actual work. Every method delegates to a spaCy function:

| Method | spaCy function |
| --- | --- |
| `convert_assets()` | `spacy.cli.convert` |
| `train()` | `spacy.training.loop.train` |
| `evaluate()` | `spacy.cli.evaluate` |
| `package()` | `spacy.cli.package` |

The module intentionally avoids Weasel (spaCy's CLI project system). Weasel is designed for command-line workflows; the Python API is better suited to a library.

### Directory structure

Every `LanguageModel` instance creates and manages this layout:

```text
model_dir/
├── config.cfg              spaCy training configuration (single source of truth)
├── assets/{lang}/          raw CONLL-U input files (archival copies)
├── corpus/{lang}/          converted .spacy binary files (training input)
├── training/{lang}/        trained model checkpoints
│   ├── model-best/         checkpoint with highest dev score
│   └── model-last/         checkpoint from the final training step
└── metrics/{lang}/         evaluation output ({lang}.json)
```

---

## The training pipeline

The default pipeline trains five Universal Dependencies components:

| Component | Predicts |
| --- | --- |
| `tok2vec` | Token vectors (shared encoder backbone — no direct output) |
| `tagger` | Penn Treebank POS tags (XPOS column in UD: `NN`, `VBD`, …) |
| `morphologizer` | UD morphological features (`Tense=Past`, `Number=Plur`, …) and universal POS |
| `trainable_lemmatizer` | Lemmas via learned edit-tree patterns |
| `parser` | UD dependency structure and labels (`nsubj`, `obj`, `case`, …) |

`tok2vec` is the shared backbone — all other components read their token representations from it via `Tok2VecListener`. During training, gradients from all five tasks flow back through tok2vec, making its representations richer than any single-task model.

**UD vs OntoNotes:** spaCy's built-in English models (`en_core_web_sm/md/lg`) use the OntoNotes annotation scheme with different dependency labels (`dobj`, `prep`, `pobj` instead of UD's `obj`, `case`, `obl`). This module uses Universal Dependencies throughout. See [Choosing and Setting Up Base Models](base_models.md) for how this affects source model selection.

---

## Configuration and recipes

spaCy's training is entirely controlled by a `config.cfg` file (Thinc's TOML-based format). `LanguageModel` generates this file automatically on instantiation and maintains an in-memory copy as `model.config` (a Thinc `Config` object, which is a dict subclass).

### Config generation modes

| `base_model` | `recipe` | What happens |
| --- | --- | --- |
| `None` | `None` | `spacy.cli.init_config()` generates a scratch-training config |
| `str` or `dict` | `None` | `_generate_finetune_config()` builds a component-sourcing config |
| any | provided | Recipe file is loaded as-is; `base_model` is ignored for config |

### Recipes

`recipes/` contains vetted `.cfg` files for specific training scenarios. Pass a recipe path via `recipe=` to use one instead of auto-generation:

| Recipe | Description |
| --- | --- |
| `default_ud.cfg` | Full five-component UD pipeline, trained from scratch |
| `finetune_ud.cfg` | Structural base used internally by `_generate_finetune_config()` for component-sourcing runs |
| `multilingual_tagger.cfg` | Tagger-only model for any language, trained from scratch; a good starting point for languages with limited resources |
| `transformer_ud.cfg` | Full UD pipeline backed by a pre-trained transformer (default: MacBERTh for historical English); requires the `transformers` extra and a GPU |

To add a new recipe: write a valid spaCy config and drop it in `recipes/`. Leave `[paths]` values as `null` — they are injected by `copy_assets()`.

### Modifying the config

After instantiation, the config can be modified as a nested dict:

```python
model.config["training"]["max_steps"] = 5000
model.config["training"]["optimizer"]["learn_rate"] = 0.0001
model.save_config()
```

Call `save_config()` to write changes to disk before training, since `train()` re-reads `config.cfg` from disk.

For which settings are worth changing — learning rates, early stopping, batch sizes — and sensible ranges for each pipeline type, see [Tuning Training Settings](training_settings.md).

---

## Pre-training validation

`train()` runs a preflight check via `validate()` before starting. This catches common problems — missing assets, empty corpus files, invalid config — before training begins.

```python
# validate() is called automatically by train(); call it manually to inspect:
model.validate()
model.train()

# Skip validation if you know the setup is correct:
model.train(skip_validation=True)
```

---

## Fine-tuning via component sourcing

Fine-tuning starts from an existing model's weights instead of random initialisation. In spaCy, this is done by replacing a component's `factory = "..."` definition with `source = "model_path_or_name"`. `_generate_finetune_config()` generates this config automatically from the `base_model` parameter.

### Three forms of `base_model`

**Scratch training** (default):

```python
model = LanguageModel("my_model")
```

**All components from one model** — convenience form:

```python
model = LanguageModel("my_model", base_model="path/to/ud_model/model-best")
```

> **Note:** This only works if the source model contains every component in your pipeline. For the standard full UD pipeline, use the per-component dict form below unless your source model was itself trained with this module (and therefore has all five components).

**Per-component source mapping** — full control:

```python
model = LanguageModel("my_model", base_model={
    "tok2vec":              "en_core_web_sm",
    "tagger":               "en_core_web_sm",
    "morphologizer":        "path/to/ud_model/model-best",
    "trainable_lemmatizer": "path/to/ud_model/model-best",
    "parser":               "path/to/ud_model/model-best",
})
```

Components omitted from the dict are initialised from scratch using the architecture defined in `default_ud.cfg`.

### The tok2vec constraint (resolved automatically)

Factory-defined components reference tok2vec's output width via the config variable `${components.tok2vec.model.encode.width}`. When tok2vec is sourced, that config path disappears and the variable would break — but the module resolves this automatically. If `tok2vec` is sourced, `_get_tok2vec_width()` reads the actual width from the source model's config, then `_patch_tok2vec_width()` replaces the broken variable reference in every factory-defined component.

Mixed factory/source configurations therefore work without writing a manual recipe:

```python
model = LanguageModel("my_model", base_model={
    "tok2vec": "en_core_web_sm",   # sourced
    "tagger":  "en_core_web_sm",   # sourced
    # morphologizer, trainable_lemmatizer, parser → trained from scratch
})
```

### Choosing source models

For guidance on which source models to use for your language and annotation scheme — including the OntoNotes vs UD compatibility issue, community UD-trained models, and training your own base model — see [Choosing and Setting Up Base Models](base_models.md).

---

## Using a different language

The module defaults to English (`lang="en"`). When `lang` is set to a non-English, non-multilingual code without also changing the `base_model`, the constructor issues a `UserWarning` — the English base models will produce incorrect results for other languages.

For a complete guide to base model selection for other languages, see [Choosing and Setting Up Base Models](base_models.md). The short version:

1. Find a spaCy model for your language at [https://spacy.io/models](https://spacy.io/models) and check whether it covers `morphologizer` and `parser` with UD labels. If so, use it for all components but `trainable_lemmatizer`, spaCy's are rule-based.
2. If not, use it for `tok2vec` and `tagger`, and source `morphologizer`, `trainable_lemmatizer`, and `parser` from a UD-trained model (community or self-trained).
3. Set `lang="xx"` for a language-agnostic multilingual model.

---

## GPU acceleration

The module defaults to CPU (`gpu=False`), which works on any machine. GPU speeds up training significantly — roughly 5–10× for a small corpus — but requires NVIDIA hardware.

### Enabling GPU

```python
model = LanguageModel("my_model", gpu=True)
```

If `gpu=True` is set but no NVIDIA GPU is detected, the constructor falls back to CPU with a warning.

### Installing GPU dependencies

```bash
pip install .[gpu]
```

This installs `cupy-cuda12x` and the required NVIDIA CUDA libraries. The warning `CUDA path could not be detected` that appears on import is harmless.

> **Note:** Evaluation always runs on CPU regardless of the `gpu` setting.

---

## Transformer-based training

The module can train the same five-component UD pipeline on top of a pre-trained transformer instead of the CNN backbone. The bundled `transformer_ud.cfg` recipe defaults to [MacBERTh](https://huggingface.co/emanjavacas/MacBERTh), a BERT model pre-trained on ~3.9B tokens of historical English (1450–1950) — the best available starting point for Early Modern and other historical English text.

```bash
pip install .[gpu,transformers]
```

```python
model = LanguageModel("my_model", gpu=True, recipe="transformer_ud.cfg")
```

Notes:

- **Transformer training is recipe-only.** The `base_model=` component-sourcing mechanism is specific to tok2vec pipelines and is not used with transformers.
- **A GPU is effectively required** — CPU transformer training is impractically slow. The constructor warns if the recipe is loaded with `gpu=False`.
- The first training run downloads the transformer weights (~450 MB for MacBERTh) from Hugging Face and caches them.
- To use a different transformer, edit `model.config["components"]["transformer"]["model"]["name"]` and `save_config()`.

The full walkthrough — including GPU checks, VRAM guidance, and a score comparison against the CNN pipeline — is in [Transformer-based Training](../../tutorials/language_model/transformer_tutorial.ipynb).

---

## Known issues and limitations

**Lemmatizer regression with small data** — The `trainable_lemmatizer` learns edit-tree patterns from examples. With fewer than ~500 training sentences it tends to make more errors than a rule-based lemmatizer. If LEMMA scores regress after fine-tuning, the training corpus is too small for the lemmatizer to generalise. Add more data or switch to a rule-based lemmatizer recipe.

**Packaging fails on Windows (path too long)** — Windows has a 260-character MAX_PATH limit. The sdist builder nests the package directory several levels deep; if `output_dir` is already a long path this limit is exceeded. Use a short output path (e.g. `C:/tmp/lm_pkg`).

**Jupyter idiosyncrasies** — spaCy functions assume command-line context and may produce unexpected output in notebooks (references to CLI flags, unusual exit behaviour). The wrapper reduces but does not eliminate this.

**Single config per model** — Each `LanguageModel` instance manages exactly one `config.cfg`. To compare training runs with different configs, use separate `model_dir` directories.
