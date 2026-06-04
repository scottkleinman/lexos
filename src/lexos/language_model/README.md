# language_model

A Python wrapper around spaCy's training workflow for fine-tuning language models on custom corpora. Designed for historical and low-resource text — the primary target is Early Modern English (Shakespeare), but it works for any language with a Universal Dependencies treebank.

Users interact with the `LanguageModel` class and `split_conllu` utility; they never need to edit config files or run command-line tools. Developers who want to add new training configurations do so by adding recipe `.cfg` files to the `recipes/` folder.

---

## Architecture

The module has two layers:

**`LanguageModel` class** — manages the directory structure, generates or loads the spaCy config, and exposes the lifecycle as simple method calls. It is a thin orchestration layer; it does not contain any ML logic.

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

```
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

| Component | Predicts | spaCy factory |
| --- | --- | --- |
| `tok2vec` | Token vectors (shared encoder backbone — no direct output) | `tok2vec` |
| `tagger` | Penn Treebank POS tags (XPOS column in UD: `NN`, `VBD`, …) | `tagger` |
| `morphologizer` | UD morphological features (`Tense=Past`, `Number=Plur`, …) and universal POS | `morphologizer` |
| `trainable_lemmatizer` | Lemmas via learned edit-tree patterns | `trainable_lemmatizer` |
| `parser` | UD dependency structure and labels (`nsubj`, `obj`, `case`, …) | `parser` |

`tok2vec` is the shared backbone — all other components read their token representations from it via `Tok2VecListener`. During training, gradients from all five tasks flow back through tok2vec, making its representations richer than any single-task model.

**UD vs OntoNotes:** spaCy's built-in English models (`en_core_web_sm/md/lg`) use a different annotation scheme (OntoNotes) with different dependency labels (`dobj`, `prep`, `pobj` instead of `obj`, `case`, `obl`). This module uses Universal Dependencies throughout, which matches the annotation of available Shakespeare treebanks and is more linguistically transparent.

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
| `multilingual_tagger.cfg` | Tagger-only model for any language, trained from scratch |
| `default_ud.cfg` | Full five-component UD pipeline, trained from scratch |
| `finetune_ud.cfg` | Structural base used internally by `_generate_finetune_config()` (see below) |

To add a new recipe: write a valid spaCy config and drop it in `recipes/`. Leave `[paths]` values as `null` — they are injected by `copy_assets()`.

### Modifying the config

After instantiation, the config can be modified as a nested dict:

```python
model.config["training"]["max_steps"] = 5000
model.config["training"]["optimizer"]["learn_rate"] = 0.0001
model.save_config()
```

Call `save_config()` to write changes to disk before training, since `train()` re-reads `config.cfg` from disk.

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
model = LanguageModel("my_model", base_model="en_core_web_sm")
```

**Per-component source mapping** — full control:
```python
model = LanguageModel("my_model", base_model={
    "tok2vec":              "en_core_web_sm",
    "tagger":               "en_core_web_sm",
    "morphologizer":        "training/UD_English-EWT/model-best",
    "trainable_lemmatizer": "training/UD_English-EWT/model-best",
    "parser":               "training/UD_English-EWT/model-best",
})
```

Components omitted from the dict are initialised from scratch using the architecture defined in `default_ud.cfg`.

### The tok2vec constraint

If `tok2vec` is included in the `base_model` dict, **all other components must also have a source**. This is because factory-defined components reference tok2vec's output width via the config variable `${components.tok2vec.model.encode.width}`, which no longer exists once tok2vec is sourced. Violating this constraint raises a `ValueError` with a clear message.

If you need a mixed factory/source configuration (e.g. sourced tok2vec + freshly-defined morphologizer with a custom architecture), write a manual `recipe=` `.cfg` file.

### What gets generated

For the dict form above, `_generate_finetune_config()` produces a config where the `[components]` section looks like:

```ini
[components.tok2vec]
source = "en_core_web_sm"

[components.tagger]
source = "en_core_web_sm"

[components.morphologizer]
source = "training/UD_English-EWT/model-best"
...
```

The rest (corpora, training loop, optimizer, etc.) comes from `recipes/finetune_ud.cfg`.

### Choosing source models

`en_core_web_sm` has `tok2vec` and `tagger` but **not** `morphologizer` or `trainable_lemmatizer` (it uses a rule-based lemmatizer). For those components, the only currently available UD-trained source is a model trained on UD_English-EWT. Better UD-trained English source models are an open research question — see the TODO in `NOTES.md`.

**Annotation scheme compatibility:** source models should use the same annotation scheme as your training data. If your treebank uses UD labels (`obj`, `case`, `obl`), sourcing the parser from `en_core_web_sm` (which uses OntoNotes labels: `dobj`, `prep`, `pobj`) means the output layer must be rebuilt from scratch. For parser and morphologizer, prefer UD-trained source models.

---

## GPU setup

spaCy uses `cupy` for GPU acceleration. The CUDA *driver* alone is not sufficient — you also need the CUDA *libraries* (cuBLAS, cuRAND, etc.). On Windows these can be installed from PyPI without the full CUDA Toolkit:

```powershell
pip install cupy-cuda12x
pip install nvidia-cublas-cu12
pip install nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cuda-runtime-cu12
```

The warning `CUDA path could not be detected` that appears on import is harmless — cupy finds its DLLs through Python's import system rather than the system PATH.

Set `gpu=0` in the constructor to use the first GPU; `gpu=-1` (default) uses CPU.

---

## Known issues and limitations

**`debug_data` calls `sys.exit(1)`** — This is spaCy's behaviour: if any data error is found, it exits the process. There is no way to catch this within the wrapper. Run `debug_data()` in isolation, not inside a larger script.

**Lemmatizer regression with small data** — The `trainable_lemmatizer` learns edit-tree patterns from examples. With fewer than ~500 training sentences it tends to make more errors than a rule-based lemmatizer. If LEMMA scores regress after fine-tuning, the training corpus is too small for the lemmatizer to generalise. Add more data or switch to a rule-based lemmatizer recipe.

**Jupyter idiosyncrasies** — spaCy functions assume command-line context and may produce unexpected output in notebooks (references to CLI flags, unusual exit behaviour). The wrapper reduces but does not eliminate this.

**`fill_config` bug in Scott's alpha (fixed here)** — The original `LanguageModel.fill_config()` method referenced an undefined variable `output_path`. This has been fixed in the module-level `fill_config()` function.

**Single config per model** — Each `LanguageModel` instance manages exactly one `config.cfg`. To compare training runs with different configs, use separate `model_dir` directories.
