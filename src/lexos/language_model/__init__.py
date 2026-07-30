"""__init__.py.

Light wrapper around spaCy's training workflow for fine-tuning language models
on custom corpora. Handles directory setup, config generation (including
fine-tuning via component sourcing), data conversion, training, and evaluation
without requiring the user to edit config files or use the command line.

Last Updated: July 29, 2026
Last Tested: July 29, 2026
"""

import importlib.util
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from time import time
from typing import Any

import spacy

from lexos.exceptions import LexosException
from smart_open import open as smart_open
from spacy.cli import convert
from spacy.cli import debug_config as spacy_debug_config
from spacy.cli import debug_data as spacy_debug_data
from spacy.cli import debug_model as spacy_debug_model
from spacy.cli import evaluate as spacy_evaluate
from spacy.cli import package as spacy_package
from spacy.cli._util import import_code, setup_gpu, show_validation_error
from spacy.cli.init_config import fill_config as spacy_fill_config
from spacy.cli.init_config import init_config
from spacy.schemas import ConfigSchemaTraining
from spacy.training.initialize import init_nlp
from spacy.training.loop import train as spacy_train
from spacy.util import load_config, load_model_from_config, registry
from thinc.api import Config, fix_random_seed, set_gpu_allocator
from wasabi import Printer

# The five-component Universal Dependencies pipeline used for full linguistic
# annotation (POS, morphology, lemmas, dependency structure).
FULL_UD_PIPELINE: list[str] = [
    "tok2vec",
    "tagger",
    "morphologizer",
    "trainable_lemmatizer",
    "parser",
]

_RECIPES_DIR = Path(__file__).parent / "recipes"


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


class _Timer:
    """Measure elapsed wall-clock time during training."""

    def __init__(self) -> None:
        self._start = time()

    def elapsed(self) -> str:
        """Return elapsed time formatted as HH:MM:SS."""
        m, s = divmod(time() - self._start, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)


def _has_nvidia_gpu() -> bool:
    """Return True if an NVIDIA GPU driver and nvidia-smi are accessible."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode == 0 and "NVIDIA" in result.stdout
    except Exception:
        return False


def _get_tok2vec_width(source: str) -> int:
    """Read tok2vec output width from a model's config.cfg without loading weights.

    Supports both local directory paths and installed spaCy package names.

    Args:
        source: An installed spaCy model name (e.g. `"en_core_web_sm"`) or a
            local path to a model directory containing `config.cfg`.

    Returns:
        The integer width of the tok2vec encoder's output.

    Raises:
        LexosException: If the config.cfg cannot be located or the width key
            is missing.
    """
    source_path = Path(source)
    if source_path.exists():
        cfg_path = source_path / "config.cfg"
        if not cfg_path.exists():
            raise LexosException(f"No config.cfg found at {source_path}")
    else:
        import spacy.util as _spacy_util

        try:
            pkg_path = _spacy_util.get_package_path(source)
            candidates = sorted(pkg_path.glob("*/config.cfg"))
            if not candidates:
                raise LexosException(
                    f"No config.cfg found in installed package '{source}'"
                )
            cfg_path = candidates[0]
        except LexosException:
            raise
        except Exception as e:
            raise LexosException(
                f"Could not locate config.cfg for source model '{source}': {e}\n"
                "Use recipe= with an explicit width instead."
            ) from e

    raw = Config().from_disk(cfg_path)
    width = (
        raw.get("components", {})
        .get("tok2vec", {})
        .get("model", {})
        .get("encode", {})
        .get("width")
    )
    if width is None:
        raise LexosException(
            f"Source model '{source}' has no tok2vec component or its width is "
            "not at components.tok2vec.model.encode.width."
        )
    return width


def _patch_tok2vec_width(component_cfg: dict[str, Any], width: int) -> None:
    """Replace the tok2vec width variable reference with a concrete integer.

    Thinc stores `${components.tok2vec.model.encode.width}` as a literal
    string until interpolation.  When tok2vec is sourced that config path
    disappears, so we walk the component dict and substitute the integer.

    Args:
        component_cfg: A single component's config sub-dict (mutated in place).
        width: The integer width to substitute for the variable reference.
    """
    _VAR = "${components.tok2vec.model.encode.width}"
    for key, value in component_cfg.items():
        if value == _VAR:
            component_cfg[key] = width
        elif isinstance(value, dict):
            _patch_tok2vec_width(value, width)


# ---------------------------------------------------------------------------
# Standalone utility: split_conllu
# ---------------------------------------------------------------------------


def split_conllu(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
    include_test: bool = True,
) -> dict[str, Path]:
    """Split a single CONLL-U file into train / dev / (optionally) test files.

    Sentences are the unit of splitting — sentence boundaries are blank lines
    in the CONLL-U format.

    Args:
        input_path: Path to the source CONLL-U file.
        output_dir: Directory where split files will be written.
        train_ratio: Fraction of sentences for the training split (default 0.8).
        dev_ratio: Fraction for the dev split (default 0.1).  The test split
            receives whatever remains (1 - train_ratio - dev_ratio).
        seed: Random seed for reproducible shuffling (default 42).
        shuffle: Whether to shuffle sentences before splitting.  Set to False
            to preserve document order (e.g. split by act / chapter).
        include_test: Whether to write a test file.  Set to False for workflows
            that evaluate manually or with an external test set.

    Returns:
        Dict with keys `"train"`, `"dev"`, and (if include_test) `"test"`
        mapping to the Path of each written file.  The return value is designed
        to be unpacked directly into :meth:`LanguageModel.copy_assets`::

            splits = split_conllu("corpus.conllu", "model/assets/en/")
            model.copy_assets(**splits)
    """
    import random

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    sentences = [s.strip() for s in text.split("\n\n") if s.strip()]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(sentences)

    n = len(sentences)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    stem = input_path.stem
    splits: dict[str, list[str]] = {
        "train": sentences[:n_train],
        "dev": sentences[n_train : n_train + n_dev],
    }
    if include_test:
        splits["test"] = sentences[n_train + n_dev :]

    msg = Printer()
    paths: dict[str, Path] = {}
    for name, data in splits.items():
        out = output_dir / f"{stem}-{name}.conllu"
        out.write_text("\n\n".join(data) + "\n\n", encoding="utf-8")
        paths[name] = out
        msg.good(f"  {name:5s}: {len(data):4d} sentences  →  {out}")

    return paths


def export_to_conllu(
    model_path: str | Path,
    texts: list[str],
    output_path: str | Path,
) -> Path:
    """Run a trained model on texts and write predictions to a CONLL-U file.

    Each element of `texts` can be a single sentence or a longer passage;
    the model's sentence segmenter splits passages into individual sentences
    automatically.

    Args:
        model_path: Path to a trained spaCy model directory, or an installed
            model name (e.g. `"en_core_web_sm"`).
        texts: List of strings to annotate.  Each element may contain multiple
            sentences — the model's sentence segmenter handles splitting.
        output_path: Path where the CONLL-U output file will be written.

    Returns:
        Path to the written output file.
    """
    # Using str() keeps installed-name inputs (e.g. "en_core_web_sm") loadable;
    # wrapping in Path() would force spaCy to treat them as directories.
    nlp = spacy.load(str(model_path))
    output_path = Path(output_path)
    msg = Printer()
    sent_id = 0
    with output_path.open("w", encoding="utf-8") as f:
        for text in texts:
            doc = nlp(text)
            for sent in doc.sents:
                # Collect only printable tokens; whitespace-only tokens
                # (newlines, tabs spaCy preserves as tokens) break tab-separated rows.
                tokens = [t for t in sent if t.text.strip()]
                if not tokens:
                    continue
                sent_id += 1
                # Normalise whitespace in the comment so newlines in the source
                # text don't split the # text = line across multiple file lines.
                text_comment = " ".join(sent.text.split())
                f.write(f"# sent_id = auto-{sent_id}\n")
                f.write(f"# text = {text_comment}\n")
                # Build spaCy-index → output-row-number map for correct head refs.
                index_map: dict[int, int] = {
                    t.i: row for row, t in enumerate(tokens, 1)
                }
                for j, token in enumerate(tokens, 1):
                    if token.head == token:
                        head = 0
                    else:
                        head = index_map.get(token.head.i, 0)
                    feats = str(token.morph) if token.morph else "_"
                    lemma = token.lemma_ if token.lemma_ else "_"
                    f.write(
                        f"{j}\t{token.text}\t{lemma}\t{token.pos_}\t"
                        f"{token.tag_}\t{feats}\t{head}\t{token.dep_}\t_\t_\n"
                    )
                f.write("\n")
    msg.good(f"{sent_id} sentences written to {output_path}")
    return output_path


def combine_conllu(
    round_files: list[str | Path],
    output_path: str | Path,
) -> Path:
    """Concatenate multiple CONLL-U files into a single training file.

    Training on the full accumulated corpus each round (not just the latest
    batch) produces more stable models.  Use this before each fine-tuning
    round to merge all corrected annotation batches.

    Args:
        round_files: List of paths to corrected CONLL-U files, in the order
            they should be concatenated.
        output_path: Path where the combined output file will be written.

    Returns:
        Path to the written output file.
    """
    output_path = Path(output_path)
    msg = Printer()
    with output_path.open("w", encoding="utf-8") as out:
        for filepath in round_files:
            text = Path(filepath).read_text(encoding="utf-8")
            out.write(text)
            if not text.endswith("\n\n"):
                out.write("\n")
    msg.good(f"Combined {len(round_files)} file(s) → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# LanguageModel class
# ---------------------------------------------------------------------------


class LanguageModel:
    """Manage the full lifecycle of a spaCy fine-tuning workflow.

    Creates and maintains a self-contained model directory with the following
    structure::

        model_dir/
        ├── config.cfg          spaCy training configuration
        ├── assets/{lang}/      raw input data (CONLL-U files)
        ├── corpus/{lang}/      converted spaCy binary (.spacy) files
        ├── training/{lang}/    trained model checkpoints
        └── metrics/{lang}/     evaluation output (JSON)

    The config is generated automatically unless a `recipe` path is supplied.
    When `base_model` is provided the config uses spaCy's component-sourcing
    mechanism to warm-start from existing model weights instead of training from
    random initialisation.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        lang: str = "en",
        gpu: bool = False,
        components: list[str] | None = None,
        base_model: str | dict | None = None,
        recipe: str | None = None,
        force: bool = False,
    ) -> None:
        """Initialise the LanguageModel and create its directory structure.

        Args:
            model_dir: Root folder for all model artefacts.
            lang: BCP-47 language code (default `"en"`).  Use `"xx"` for
                a language-agnostic multilingual model.
            gpu: Use GPU for training (default `False` — CPU).  Set to
                `True` to enable GPU (device 0).  Requires cupy and the CUDA
                libraries; see README.md for setup instructions.  If
                `gpu=True` but no NVIDIA GPU is detected, falls back to CPU
                with a warning.
            components: spaCy pipeline components to train.  Defaults to the
                full Universal Dependencies pipeline
                ``["tok2vec", "tagger", "morphologizer",
                "trainable_lemmatizer", "parser"]``.
            base_model: Starting point for fine-tuning.  Three forms are
                accepted:

                - `None` — train from random initialisation (scratch).
                - `str` — source every component from this model
                  (installed name like `"en_core_web_sm"` or a local path).
                - `dict[str, str]` — map each component name to its own
                  source model, e.g.
                  ``{"tok2vec": "en_core_web_sm", "tagger": "en_core_web_sm",
                  "morphologizer": "training/UD_English-EWT/model-best", ...}``.
                  Components absent from the dict are initialised from scratch.
                  Mixed factory/source configurations are supported even when
                  `tok2vec` is sourced — the module reads the tok2vec output
                  width from the source model's `config.cfg` and patches it
                  into factory-defined component configs automatically.

            recipe: Path to a `.cfg` file, or the filename of a bundled
                recipe (e.g. `"transformer_ud.cfg"`, resolved against the
                module's `recipes/` folder).  When provided, the file is
                loaded as-is, `base_model` is ignored for config generation,
                and `components` is replaced by the recipe's
                `[nlp] pipeline`.  Transformer-based training is only
                available through recipes — `base_model` sourcing is
                specific to tok2vec pipelines.
            force: Overwrite an existing `config.cfg` if one already exists.
        """
        self.model_dir = Path(model_dir)
        self.lang = lang
        self.base_model = base_model
        self.components = (
            components if components is not None else FULL_UD_PIPELINE.copy()
        )
        self.config: Config | None = None

        # --- GPU setup ---
        if gpu and not _has_nvidia_gpu():
            warnings.warn(
                "gpu=True was requested but no NVIDIA GPU was detected "
                "(nvidia-smi not found or returned no NVIDIA devices). "
                "Falling back to CPU. To enable GPU install the extras: "
                "pip install .[gpu]",
                UserWarning,
                stacklevel=2,
            )
            self.gpu = False
        else:
            self.gpu = gpu
        # Device id per spaCy: 0 = GPU, -1 = CPU
        self._use_gpu: int = 0 if self.gpu else -1

        # --- Non-English warning ---
        if lang not in ("en", "xx"):
            warnings.warn(
                f"lang='{lang}': the default base_model entries (en_core_web_sm "
                "and the bundled UD English model) are English-specific. "
                f"Make sure your base_model entries point to models trained for '{lang}'. "
                "See README.md for guidance on finding models for other languages.",
                UserWarning,
                stacklevel=2,
            )

        self._config_path = self.model_dir / "config.cfg"
        self._assets_dir = self.model_dir / "assets" / lang
        self._corpus_dir = self.model_dir / "corpus" / lang
        self._metrics_dir = self.model_dir / "metrics" / lang
        self._training_dir = self.model_dir / "training" / lang

        msg = Printer()
        for d in [
            self._assets_dir,
            self._corpus_dir,
            self._metrics_dir,
            self._training_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        if self._config_path.exists() and not force:
            msg.warn(
                f"{self._config_path} already exists. Pass force=True to regenerate it."
            )
            self.config = Config().from_disk(self._config_path)
            return

        # --- Generate or load config ---
        if recipe is not None:
            self._load_recipe(recipe, msg)
        elif base_model is not None:
            sources = self._resolve_sources(base_model)
            self.config = self._generate_finetune_config(sources)
        else:
            self.config = init_config(
                lang=self.lang,
                pipeline=self.components,
                optimize="efficiency",
                gpu=self.gpu,
            )

        self._apply_config_defaults()
        self.config.to_disk(self._config_path)
        msg.good(f"Config saved to {self._config_path}")
        msg.text("Next: copy_assets() → convert_assets() → train()")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_recipe(self, recipe: str, msg: Printer) -> None:
        """Load a config from a recipe file path.

        The recipe's `[nlp] pipeline` becomes `self.components` so that
        validation output and the score-weight calculation reflect the actual
        pipeline being trained, not the constructor default.

        Args:
            recipe: Path to a `.cfg` file, or the filename of a bundled
                recipe in the module's `recipes/` folder.
            msg: Printer used for status output.

        Raises:
            LexosException: If the recipe file cannot be found, or if the
                recipe uses a transformer component and `spacy-transformers`
                is not installed.
        """
        recipe_path = Path(recipe)
        if not recipe_path.exists():
            recipe_path = _RECIPES_DIR / recipe
        if not recipe_path.exists():
            raise LexosException(
                f"Recipe not found: {recipe}\n"
                f"Checked: {Path(recipe).resolve()} and {_RECIPES_DIR / recipe}"
            )
        self.config = Config().from_disk(recipe_path)

        pipeline = list(self.config.get("nlp", {}).get("pipeline", []))
        if pipeline:
            self.components = pipeline

        if "transformer" in self.components:
            if importlib.util.find_spec("spacy_transformers") is None:
                raise LexosException(
                    "This recipe uses a transformer component, but "
                    "spacy-transformers is not installed. Install the "
                    "transformer extras:\n"
                    "    pip install .[transformers]\n"
                    "or, for GPU training (strongly recommended):\n"
                    "    pip install .[gpu,transformers]"
                )
            if not self.gpu:
                warnings.warn(
                    "Transformer training on CPU is impractically slow "
                    "(hours become days). Pass gpu=True if an NVIDIA GPU "
                    "is available.",
                    UserWarning,
                    stacklevel=2,
                )

        msg.good(f"Loaded recipe: {recipe_path}")

    def _resolve_sources(self, base_model: str | dict) -> dict[str, str]:
        """Normalise base_model to a component→source mapping."""
        if isinstance(base_model, str):
            return {comp: base_model for comp in self.components}
        return base_model

    def _generate_finetune_config(self, sources: dict[str, str]) -> Config:
        """Build a Thinc Config that sources each component from an existing model.

        Loads default_ud.cfg as the structural base (providing corpora,
        training, and initialize sections), then replaces each component block
        with a `source = "..."` entry.  For factory-defined components
        alongside a sourced tok2vec, the broken
        `${components.tok2vec.model.encode.width}` variable reference is
        replaced with the actual integer read from the source model's config.
        """
        base = Config().from_disk(_RECIPES_DIR / "default_ud.cfg")

        base["nlp"]["lang"] = self.lang
        base["nlp"]["pipeline"] = self.components

        # Read tok2vec width once if tok2vec is sourced, so factory-defined
        # listener components can have the variable reference patched in.
        tok2vec_source = sources.get("tok2vec")
        tok2vec_width = _get_tok2vec_width(tok2vec_source) if tok2vec_source else None

        for comp in self.components:
            source = sources.get(comp)
            if source is not None:
                base["components"][comp] = {"source": source}
            elif tok2vec_width is not None:
                # Factory-defined component with sourced tok2vec: patch the
                # width variable so config interpolation doesn't break.
                _patch_tok2vec_width(base["components"][comp], tok2vec_width)

        return base

    def _apply_config_defaults(self) -> None:
        """Set values that spaCy's config generator leaves unset or unusable.

        Specifically:
        - `corpora.train.max_length = 2000` prevents runaway memory use on
          very long documents during training.
        - `training.before_update = null` silences spaCy's debug warning
          about the missing key.
        - `training.score_weights` assigns equal weight to each active
          component's accuracy metric so the best-model checkpoint reflects
          overall pipeline quality.
        """
        if self.config is None:
            return

        if "corpora" in self.config and "train" in self.config["corpora"]:
            self.config["corpora"]["train"]["max_length"] = 2000

        # Embedding backbones (tok2vec, transformer) produce no scored
        # output, so they intentionally have no entry here.
        _component_metrics: dict[str, list[str]] = {
            "tagger": ["tag_acc"],
            "morphologizer": ["pos_acc", "morph_acc"],
            "trainable_lemmatizer": ["lemma_acc"],
            "parser": ["dep_uas", "dep_las"],
        }
        active_metrics: list[str] = []
        for comp in self.components:
            active_metrics.extend(_component_metrics.get(comp, []))

        weights: dict[str, Any] = {
            "morph_per_feat": None,
            "dep_las_per_type": None,
            "sents_p": None,
            "sents_r": None,
            "sents_f": 0.0,
        }
        if active_metrics:
            per = round(1.0 / len(active_metrics), 4)
            for key in active_metrics:
                weights[key] = per

        if "training" in self.config:
            self.config["training"]["before_update"] = None
            self.config["training"]["score_weights"] = weights

    # ------------------------------------------------------------------
    # Public config interface
    # ------------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        """Path to the config.cfg file on disk."""
        return self._config_path

    def save_config(self, *, filepath: str | Path | None = None) -> None:
        """Write the in-memory config to disk.

        Args:
            filepath: Destination path.  Defaults to the model's config.cfg.
        """
        path = Path(filepath) if filepath else self._config_path
        self.config.to_disk(path)

    def load_config(self, *, filepath: str | Path | None = None) -> None:
        """Replace the current config by loading from a file.

        Args:
            filepath: Source path.  Defaults to the model's config.cfg.
        """
        path = Path(filepath) if filepath else self._config_path
        self.config = Config().from_disk(path)
        if path != self._config_path:
            self.config.to_disk(self._config_path)

    # ------------------------------------------------------------------
    # Workflow methods
    # ------------------------------------------------------------------

    def copy_assets(
        self,
        *,
        train: str | Path | None = None,
        dev: str | Path | None = None,
        test: str | Path | None = None,
    ) -> None:
        """Copy CONLL-U data files into the model's assets folder.

        Accepts local paths or URLs (via `smart_open`).  Also updates
        `config["paths"]["train"]` and `config["paths"]["dev"]` to point
        at the expected post-conversion `.spacy` files so that `train()`
        can find them automatically.

        The return value of :func:`split_conllu` can be unpacked directly::

            splits = split_conllu("corpus.conllu", "model/assets/en/")
            model.copy_assets(**splits)

        Args:
            train: Path or URL to the training CONLL-U file.
            dev: Path or URL to the development CONLL-U file.
            test: Path or URL to the test CONLL-U file.
        """
        msg = Printer()
        path_overrides: dict[str, str] = {}

        for label, filepath in [("train", train), ("dev", dev), ("test", test)]:
            if filepath is None:
                continue
            filepath = str(filepath)
            try:
                with smart_open(filepath, "rb") as f:
                    content = f.read()
            except Exception as e:
                raise LexosException(f"Could not read {label} file: {filepath}") from e

            dest = self._assets_dir / Path(filepath).name
            with open(dest, "wb") as f:
                f.write(content)

            # Record the expected post-conversion .spacy path for train and dev.
            # Test path is discovered automatically at evaluate() time.
            if label in ("train", "dev"):
                spacy_path = str(
                    (
                        self._corpus_dir / Path(filepath).with_suffix(".spacy").name
                    ).resolve()
                )
                # Set both [paths] and [corpora.*.path] directly. Thinc serialises
                # the resolved value of ${paths.train} (null at first save), so the
                # variable reference is lost — corpora paths must be set explicitly.
                path_overrides[f"paths.{label}"] = spacy_path
                path_overrides[f"corpora.{label}.path"] = spacy_path

        if path_overrides:
            self.config = load_config(self._config_path, overrides=path_overrides)
            self.config.to_disk(self._config_path)

        msg.good(f"Assets copied to {self._assets_dir}")

    def convert_assets(
        self, *, n_sents: int = 10, merge_subtokens: bool = True
    ) -> None:
        """Convert CONLL-U files in assets/ to spaCy's binary format in corpus/.

        Groups every `n_sents` sentences into a single spaCy Doc.  Larger
        groups give the model more context during training but use more memory.

        Args:
            n_sents: Sentences per Doc (0 to keep each sentence as its own Doc).
            merge_subtokens: Merge CONLL-U multi-word tokens into single tokens.
        """
        msg = Printer()
        success = True
        files = list(self._assets_dir.glob("*.conllu"))
        if not files:
            msg.warn(f"No .conllu files found in {self._assets_dir}")
            return
        for filepath in files:
            try:
                convert(
                    input_path=filepath,
                    output_dir=self._corpus_dir,
                    file_type="spacy",
                    converter="conllu",
                    n_sents=n_sents,
                    merge_subtokens=merge_subtokens,
                )
            except Exception as e:
                success = False
                msg.fail(f"Error converting {filepath.name}: {e}")
        if success:
            msg.good(f"Assets converted and saved to {self._corpus_dir}")
        else:
            msg.fail("One or more assets failed to convert. Check CONLL-U formatting.")

    def validate(self) -> None:
        """Run pre-training preflight checks and print a summary.

        Verifies that:

        - Assets exist in `assets/{lang}/` and are non-empty.
        - Converted `.spacy` corpus files are present (warns if missing, since
          `convert_assets()` may not have been called yet).
        - The config file exists and passes spaCy's `debug_config` check.
        - Training data passes spaCy's `debug_data` check (if corpus exists).

        Raises:
            LexosException: If any check fails.  All failures are reported
                before raising so the user can fix them in one round.
        """
        msg = Printer()
        errors: list[str] = []

        assets = list(self._assets_dir.glob("*.conllu"))
        if not assets:
            errors.append(
                f"No .conllu files found in {self._assets_dir}. "
                "Run copy_assets() first."
            )
        else:
            for f in assets:
                if f.stat().st_size == 0:
                    errors.append(f"Asset file is empty: {f}")

        spacy_files = list(self._corpus_dir.glob("*.spacy"))
        if not spacy_files:
            msg.warn(
                f"No .spacy files found in {self._corpus_dir}. "
                "Run convert_assets() before train()."
            )
        else:
            for f in spacy_files:
                if f.stat().st_size == 0:
                    errors.append(f"Corpus file is empty: {f}")

        if not self._config_path.exists():
            errors.append(f"Config not found: {self._config_path}")
        else:
            try:
                debug_config(self._config_path)
            except Exception as e:
                errors.append(f"Config validation failed: {e}")
            if spacy_files:
                try:
                    debug_data(self._config_path)
                except LexosException as e:
                    errors.append(str(e))

        if self.lang not in ("en", "xx"):
            msg.warn(
                f"lang='{self.lang}': confirm your base_model components are "
                "appropriate for this language."
            )

        if errors:
            for err in errors:
                msg.fail(err)
            raise LexosException(
                f"Validation failed with {len(errors)} issue(s). See output above."
            )

        msg.good("All checks passed.")
        msg.info(f"  Language:   {self.lang}")
        msg.info(f"  Pipeline:   {self.components}")
        msg.info(f"  Device:     {'GPU (device 0)' if self.gpu else 'CPU'}")
        msg.info(f"  Base model: {self.base_model}")
        msg.info(f"  Assets:     {[f.name for f in assets]}")
        msg.info(f"  Output:     {self._training_dir}")

    def train(self, *, skip_validation: bool = False) -> None:
        """Train the model using the current config.

        Reads `config.cfg` from disk (so any manual edits to that file are
        respected), runs a preflight check via :meth:`validate` (unless
        `skip_validation=True`), then initialises the spaCy pipeline and
        runs the training loop.  Progress is logged to stdout.

        The trained model is saved to `training/{lang}/model-best` (best dev
        score) and `training/{lang}/model-last` (final step).

        Args:
            skip_validation: Skip the pre-training preflight check (default
                `False`).  Set to `True` to skip validation and go straight
                to training.
        """
        if not skip_validation:
            self.validate()
        timer = _Timer()
        # Acquire the GPU (or configure CPU) before building the pipeline,
        # exactly as spaCy's `train` CLI does. init_nlp() and spacy_train() do
        # NOT call require_gpu themselves — spaCy's loop.train docstring says
        # "Make sure to call require_gpu" — so without this, gpu=True silently
        # trains on CPU: thinc stays on NumpyOps and the transformer is never
        # moved onto the GPU.
        setup_gpu(self._use_gpu)
        config = load_config(self._config_path)
        nlp = init_nlp(config, use_gpu=self._use_gpu)
        spacy_train(
            nlp=nlp,
            output_path=self._training_dir,
            use_gpu=self._use_gpu,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        msg = Printer()
        msg.text(f"Training complete. Time elapsed: {timer.elapsed()}")

    def evaluate(
        self,
        *,
        model: str | None = None,
        test_file: str | Path | None = None,
        gpu: bool = False,
        silent: bool = False,
    ) -> None:
        """Evaluate a trained model against a test set.

        Defaults to CPU (`gpu=False`).  Pass `gpu=True` to use GPU if
        available — evaluation is fast and rarely needs it, but the option
        is there.

        Results are printed to stdout and saved as JSON to
        `metrics/{lang}/{lang}.json`.

        Args:
            model: Path to a trained model directory.  Defaults to
                `training/{lang}/model-best`.
            test_file: Path to the test `.spacy` file.  If omitted,
                searches `corpus/{lang}/` for a file matching `*test*.spacy`.
            gpu: Use GPU for evaluation (default `False` — CPU).
            silent: Suppress console output (results are still saved to disk).
        """
        if model is None:
            model = str(self._training_dir / "model-best")

        if test_file is None:
            candidates = sorted(self._corpus_dir.glob("*test*.spacy"))
            if not candidates:
                raise LexosException(
                    f"No test .spacy file found in {self._corpus_dir}. "
                    "Pass test_file= explicitly."
                )
            test_file = candidates[0]

        output = self._metrics_dir / f"{self.lang}.json"
        spacy_evaluate(
            model=model,
            data_path=Path(test_file),
            output=output,
            use_gpu=0 if gpu else -1,
            gold_preproc=False,
            displacy_path=None,
            displacy_limit=25,
            silent=silent,
        )

    def package(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        name: str,
        version: str,
        *,
        force: bool = False,
        silent: bool = False,
    ) -> None:
        """Package a trained model as a pip-installable distribution.

        Creates a source distribution (`.tar.gz`) that can be installed with
        `pip install` and then loaded by package name with `spacy.load()`.

        Args:
            input_dir: Path to a trained model directory (e.g.
                `training/en/model-best`).
            output_dir: Directory where the package will be written.
            name: Short name for the package (e.g. `"shakespeare_sm"`).
            version: Semantic version string (e.g. `"1.0.0"`).
            force: Overwrite an existing package with the same name/version.
            silent: Suppress console output.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        try:
            spacy_package(
                input_dir=in_path,
                output_dir=out_path,
                name=name,
                version=version,
                create_meta=False,
                create_sdist=True,
                create_wheel=False,
                force=force,
                silent=silent,
            )
        except SystemExit as e:
            if e.code != 0:
                raise LexosException(
                    "Packaging failed (see output above). "
                    "The most common cause is a missing 'build' package: "
                    "pip install build"
                ) from e
        msg = Printer()
        tarfile = f"{self.lang}_{name}-{version}.tar.gz"
        dist = out_path / f"{self.lang}_{name}-{version}" / "dist" / tarfile
        msg.good(
            f"Model packaged.\n"
            f"  Install: pip install {dist}\n"
            f"  Load:    spacy.load('{self.lang}_{name}')"
        )


# ---------------------------------------------------------------------------
# Module-level debugging utilities (no LanguageModel instance required)
# ---------------------------------------------------------------------------


def debug_config(
    config_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
    code_path: str | Path | None = None,
    show_funcs: bool = False,
    show_vars: bool = False,
) -> None:
    """Validate a spaCy config file and report any errors.

    Creates all registered objects described by the config and checks that
    every function reference is resolvable.  Note: some validation errors are
    blocking — you may need to fix errors one round at a time.

    Args:
        config_path: Path to the `.cfg` file to validate.
        overrides: Dict of config key overrides to test (e.g.
            `{"training.max_steps": 100}`).
        code_path: Path to a Python file containing custom registered functions.
        show_funcs: Print all registered functions used by the config.
        show_vars: Print all config variables and their resolved values.
    """
    if overrides is None:
        overrides = {}
    config_path = Path(config_path)
    if isinstance(code_path, str):
        code_path = Path(code_path)
    import_code(code_path)
    try:
        spacy_debug_config(
            config_path, overrides=overrides, show_funcs=show_funcs, show_vars=show_vars
        )
    except SystemExit as e:
        if e.code != 0:
            raise LexosException(
                "debug_config found errors in your config (see output above). "
                "Fix them before training."
            ) from e


def debug_data(
    config_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
    code_path: str | Path | None = None,
    ignore_warnings: bool = False,
    verbose: bool = False,
    no_format: bool = False,
) -> None:
    """Analyse and validate training and dev data, reporting stats and issues.

    Useful for catching problems like missing labels, data imbalance, or
    invalid annotations before a long training run.  Raises `LexosException`
    if spaCy's data checker finds errors.

    Args:
        config_path: Path to the `.cfg` file that references the data.
        overrides: Dict of config key overrides.
        code_path: Path to a Python file with custom registered functions.
        ignore_warnings: Show only errors, not warnings.
        verbose: Print additional explanations alongside stats.
        no_format: Plain-text output without colour formatting.
    """
    if overrides is None:
        overrides = {}
    config_path = Path(config_path)
    if isinstance(code_path, str):
        code_path = Path(code_path)
    import_code(code_path)
    try:
        spacy_debug_data(
            config_path,
            config_overrides=overrides,
            ignore_warnings=ignore_warnings,
            verbose=verbose,
            no_format=no_format,
            silent=False,
        )
    except SystemExit as e:
        if e.code != 0:
            raise LexosException(
                "debug_data found errors in your training data (see output above). "
                "Fix them before training."
            ) from e


def debug_model(
    config_path: str | Path,
    *,
    config_overrides: dict[str, Any] | None = None,
    component: str = "tagger",
    layers: list[int] | None = None,
    dimensions: bool = False,
    parameters: bool = False,
    gradients: bool = False,
    attributes: bool = False,
    P0: bool = False,
    P1: bool = False,
    P2: bool = False,
    P3: bool = False,
    use_gpu: int = -1,
) -> None:
    """Inspect a trained model's internal layer structure and weights.

    Args:
        config_path: Path to the `.cfg` file.
        config_overrides: Dict of config key overrides.
        component: Pipeline component to inspect (default `"tagger"`).
        layers: Layer IDs to examine in detail.
        dimensions: Print layer dimensions.
        parameters: Print parameter counts.
        gradients: Print gradient information.
        attributes: Print component attributes.
        P0: Print model state before training.
        P1: Print model state after initialisation.
        P2: Print model state after training.
        P3: Print final predictions.
        use_gpu: GPU device ID or `-1` for CPU.
    """
    if config_overrides is None:
        config_overrides = {}
    if layers is None:
        layers = []
    config_path = Path(config_path)
    setup_gpu(use_gpu)
    print_settings = {
        "dimensions": dimensions,
        "parameters": parameters,
        "gradients": gradients,
        "attributes": attributes,
        "layers": [int(x) for x in layers],
        "print_before_training": P0,
        "print_after_init": P1,
        "print_after_training": P2,
        "print_prediction": P3,
    }
    with show_validation_error(config_path):
        raw_config = load_config(
            config_path, overrides=config_overrides, interpolate=False
        )
    config = raw_config.interpolate()
    allocator = config["training"]["gpu_allocator"]
    if use_gpu >= 0 and allocator:
        set_gpu_allocator(allocator)
    with show_validation_error(config_path):
        nlp = load_model_from_config(raw_config)
        config = nlp.config.interpolate()
        T = registry.resolve(config["training"], schema=ConfigSchemaTraining)
    seed = T["seed"]
    if seed is not None:
        fix_random_seed(seed)
    pipe = nlp.get_pipe(component)
    spacy_debug_model(config, T, nlp, pipe, print_settings=print_settings)


def fill_config(
    config_path: str | Path,
    output_file: str | Path,
    *,
    pretraining: bool = False,
    diff: bool = False,
    code_path: str | Path | None = None,
) -> None:
    """Fill a partial config file with spaCy defaults and save it.

    Useful for debugging or understanding what a minimal config expands to.

    Args:
        config_path: Path to the partial `.cfg` file to fill.
        output_file: Path where the filled config will be written.
        pretraining: Include pretraining config section.
        diff: Print a visual diff of changes made.
        code_path: Path to a Python file with custom registered functions.
    """
    config_path = Path(config_path)
    output_file = Path(output_file)
    if isinstance(code_path, str):
        code_path = Path(code_path)
    import_code(code_path)
    spacy_fill_config(output_file, config_path, pretraining=pretraining, diff=diff)
