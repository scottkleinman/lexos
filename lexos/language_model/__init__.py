"""__init__.py.

This is a light wrapper around the spaCy training workflow functions. Most functions
are methods of the `LanguageModel` class, but debugging is done through separate functions.
"""
import sys
from pathlib import Path
from time import time
from wasabi import Printer
from typing import List, Optional, Union

from smart_open import open
from spacy.cli import convert, evaluate, package
from spacy.cli import debug_config as spacy_debug_config
from spacy.cli import debug_data as spacy_debug_data
from spacy.cli import debug_model as spacy_debug_model
from spacy.cli.init_config import fill_config, init_config
from spacy.cli._util import import_code, setup_gpu, show_validation_error
from spacy.schemas import ConfigSchemaTraining
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from spacy.util import is_in_jupyter, load_config, load_model_from_config, registry, run_command
from thinc.api import Config, fix_random_seed, set_gpu_allocator


class Timer:
    """Create a timer object."""

    def __init__(self):
        """Initialise the timer object."""
        self.start = time()

    def get_time_elapsed(self):
        """Get the elapsed time and format it as hours, minutes, and seconds."""
        end = time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str


class LanguageModel:
    """Create a LanguageModel object."""

    def __init__(self,
                 model_dir: str = "language_model",
                 config_file: str = "config.cfg",
                 training_file: str = "train.conllu",
                 dev_file: str = "dev.conllu",
                 test_file: str = "test.conllu",
                 gpu: int = -1,
                 lang: str = "xx",
                 package_name: str = "model_sm",
                 package_version: str = "1.0.0",
                 components: List[str] = ["tagger"],
                 optimize: str = "efficiency",
                 exclusive_classes: bool = True,
                 force: bool = False,
                 recipe: str = None):
        """Initialise the LanguageModel object."""
        self.model_dir = model_dir
        self.config = None
        self.config_file = config_file
        self.config_filepath = f"{self.model_dir}/{self.config_file}"
        self.training_file = training_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.gpu = gpu
        self.lang = lang
        self.package_name = package_name
        self.package_version = package_version
        self.components = components
        self.optimize = optimize
        self.exclusive_classes = exclusive_classes

        msg = Printer()

        # Issue a warning if the user is working in a Jupyter notebook
        if is_in_jupyter and Path.cwd().name == "lexos" and Path(self.model_dir).resolve().parts[-2] == "lexos":
            msg.warn("It looks like you're calling `LanguageModel()` from within a "
            "Jupyter notebook or a similar environment. If you set the system path "
            "to the Lexos API folder to import locally, you should configure "
            "your model directory with an absolute path or a path relative. "
            "to the lexos directory.")
        # Otherwise, we're safe to proceed
        else:
            # Create the model directory if it doesn't exist
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

            # Create model folders
            self.assets_dir = f"{self.model_dir}/assets/{self.lang}"
            self.corpus_dir = f"{self.model_dir}/corpus/{self.lang}"
            self.metrics_dir = f"{self.model_dir}/metrics/{self.lang}"
            self.training_dir = f"{self.model_dir}/training/{self.lang}"
            for dir in [self.assets_dir, self.corpus_dir, self.metrics_dir, self.training_dir]:
                Path(dir).mkdir(parents=True, exist_ok=True)
            msg.good(f"Created assets directory: {self.assets_dir}.")
            msg.good(f"Created corpus directory: {self.corpus_dir}.")
            msg.good(f"Created metrics directory: {self.metrics_dir}.")
            msg.good(f"Created training directory: {self.training_dir}.")

            # Create the base config file if it doesn't exist
            if not Path(self.config_filepath).is_file() or force == True:
                # Load a recipe if one was provided
                if recipe:
                    try:
                        self.load_config(recipe)
                    except FileNotFoundError:
                        msg.fail(f"Recipe file {recipe} not found.")
                else:
                    if self.gpu == -1:
                        gpu = False
                    else:
                        gpu = True
                    self.config = init_config(
                        lang=self.lang,
                        pipeline=self.components,
                        optimize=self.optimize,
                        gpu=gpu)
                # Add the file paths to the config
                files = {"train": self.training_file, "dev": self.dev_file, "test": self.test_file}
                for k, file in files.items():
                    if file is not None:
                        self.config["paths"][k] = f'{self.corpus_dir}/{file.replace(".conllu", ".spacy")}'
                self.config.to_disk(self.config_filepath)
                self._fix_config()
                msg.good(f"Saved config file to {self.config_filepath}.")
                msg.text("You can now add your data and train your pipeline.")
            else:
                msg.warn(f"{self.config_filepath} already exists. "
                         "You can now add your data and train your pipeline.")

    def _fix_config(self) -> None:
        """Fix the config file.

        This is a hack because init_config() does not generate
        useable max_length and score_weights values.
        """
        self.config["corpora"]["train"]["max_length"] = 2000
        self.config["training"]["score_weights"] = {
            "morph_per_feat": None,
            "dep_las_per_type": None,
            "sents_p": None,
            "sents_r": None,
            "tag_acc": 0.33,
            "pos_acc": 0.17,
            "morph_acc": 0.17,
            "dep_uas": 0.17,
            "dep_las": 0.17,
            "sents_f": 0.0
        }
        self.config.to_disk(self.config_filepath)

    def convert_assets(self,
                       n_sents=10,
                       merge_subtokens=True):
        """Convert CONLLU assets to spaCy DocBins.

        Args:
            n_sents (int): The number of sentences per doc (0 to disable).
            merge_subtokens (bool): Whether to merge CoNLL-U subtokens.
        """
        msg = Printer()
        success = True
        files = [self.training_file, self.dev_file, self.test_file]
        for file in files:
            filepath = Path(f"{self.assets_dir}/{file}")
            if filepath.exists():
                try:
                    convert(
                        input_path=filepath,
                        output_dir=self.corpus_dir,
                        file_type="spacy",
                        converter="conllu",
                        n_sents=n_sents,
                        merge_subtokens=merge_subtokens
                    )
                except Exception:
                    success = False
                    msg.fail(f"Error converting {file}. Check that the CONLLU file formatting is valid.")
        if success:
            msg.good(f"Assets converted and saved to {self.corpus_dir}.")
            msg.text("You can now train your pipeline.")
        else:
            msg.fail("Failed to convert one or more assets.")

    def copy_assets(self,
                    training_file: str = None,
                    dev_file: str = None,
                    test_file: str = None):
        """Copy assets to the assets folder.

        Args:
            training_file (str): The path to the training file to copy.
            dev_file (str): The path to the dev file to copy.
            test_file (str): The path to the test file to copy.
        """
        for asset in [training_file, dev_file, test_file]:
            if asset is not None:
                try:
                    with open(asset, "rb") as f:
                        content = f.read()
                except IOError:
                    raise IOError(f"{asset} not found.")
                save_path = f"{self.assets_dir}/{Path(asset).name}"
                with open(save_path, "wb") as f:
                    f.write(content)
        msg = Printer()
        msg.good(f"Copied assets to {self.assets_dir}.")

    def evaluate(self,
                 model: str = None,
                 testfile: Union[Path, str] = None,
                 output: Path = None,
                 use_gpu: int = -1,
                 gold_preproc: bool = False,
                 displacy_path: Optional[Path] = None,
                 displacy_limit: int = 25,
                 silent: bool = False) -> None:
        """Evaluate a spaCy model.

        Args:
            model (str): The path to the model to evaluate.
            testfile (Union[Path, str]): The path to the test file to evaluate.
            output (Path): The path to the output file.
            use_gpu (int): The GPU to use.
            gold_preproc (bool): Whether to use gold preprocessing.
            displacy_path (Optional[Path]): The path to the displacy package.
            displacy_limit (int): The number of tokens to display.
            silent (bool): Whether to suppress output.

        Returns:
            Dict[str, Any]: The evaluation results.
        """
        if isinstance(testfile, str):
            testfile = Path(testfile)
        output = Path(f"{self.metrics_dir}/{self.lang}").with_suffix(".json")
        if not use_gpu:
            use_gpu = self.gpu
        evaluate(
            model=model,
            data_path=testfile,
            output=output,
            use_gpu=use_gpu,
            gold_preproc=gold_preproc,
            displacy_path=displacy_path,
            displacy_limit=displacy_limit,
            silent=silent
        )

    def fill_config(self,
        path: Union[str, Path],
        output_file: Union[str, Path] = None,
        pretraining: bool = False,
        diff: bool = False,
        code_path: Union[str, Path] = None) -> None:
        """
        Fill partial config file with default values.

        Adds all missing settings from the current config and creates all objects,
        checks the registered functions for their default values, and updates the
        config file. Although the `LanguageModel` class automatically generates a
        full config file, this method may be useful for debugging.
        DOCS: https://spacy.io/api/cli#init-fill-config

        Args:
            path (Union[str, Path]): Path to the config file to fill.
            output_file (Union[str, Path]): Path to output .cfg file.
            pretraining (bool): Include config for pretraining (with "spacy pretrain").
            diff (bool): Print a visual diff highlighting the changes.
            code_path (Union[str, Path]): Path to Python file with additional code
                (registered functions) to be imported.
        """
        if not output_path:
            output_path = self.config_filepath
        if isinstance(path, str):
            path = Path(path)
        if isinstance(output_file, str):
            output_file = Path(output_file)
        if isinstance(code_path, str):
            code_path = Path(code_path)
        import_code(code_path)
        fill_config(output_file, path, pretraining=pretraining, diff=diff)
        self.config = Config().from_disk(self.config_filepath)

    def load_config(self, filepath: Path = None):
        """Load the config from the config file.

        Use this method if you wish to load the config from a different file.
        But use with caution, as the old config file will be overwritten, and
        the new config file may have different paths or a different pipeline
        from those you used to instantiate the `LanguageModel`.

        Args:
            filepath (Path): The path to the config file.
        """
        if not filepath:
            filepath = self.config_filepath
        self.config = load_config(filepath)
        self.config.to_disk(self.config_filepath)

    def package(self,
                input_dir: str = None,
                output_dir: str = None,
                meta_path: Optional[str] = None,
                code_paths: Optional[List[str]] = [],
                name: str = None,
                version: str = None,
                create_meta: bool = False,
                create_sdist: bool = True,
                create_wheel: bool = False,
                force: bool = False,
                silent: bool = False) -> None:
        """Package the model so that it can be installed."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        if meta_path:
            meta_path = Path(meta_path)
        if code_paths:
            code_paths = [Path(p) for p in code_paths]
        self.packages_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        package(
            input_dir,
            output_dir,
            meta_path=meta_path,
            code_paths=code_paths,
            name=name,
            version=version,
            create_meta=create_meta,
            create_sdist=create_sdist,
            create_wheel=create_wheel,
            force=force,
            silent=silent
        )
        # Print the paths to the package
        tarfile = f"{self.lang}_{name}-{version}.tar.gz"
        print(
            f"Model Directory for spacy.load(): {output_dir}/{self.lang}-{version}")
        print(
            f"Binary file (for pip install): {output_dir}/{self.lang}-{version}/dist/{tarfile}")

    def save_config(self, filepath: Path = None):
        """Save the config from the config file.

        Use this method to save a config file after making modifications.
        But use with caution, as the old config file will be overwritten, and
        the new config file may have different paths or a different pipeline
        from those you used to instantiate the `LanguageModel`.

        Args:
            filepath (Path): The path to a config file (for saving a copy).
        """
        if not filepath:
            filepath = self.config_filepath
        self.config.to_disk(self.config_filepath)

    def train(self):
        """Train the corpus."""
        timer = Timer()
        config = load_config(self.config_filepath)
        nlp = init_nlp(config, use_gpu=self.gpu)
        train(nlp=nlp, output_path=Path(self.training_dir),
                    use_gpu=self.gpu, stdout=sys.stdout, stderr=sys.stderr)
        msg = Printer()
        msg.text(f"Time elapsed: {timer.get_time_elapsed()}")


def debug_config(
    config_path: Union[str, Path],
    overrides: dict = {},
    code_path: Union[str, Path] = None,
    show_funcs: bool = False,
    show_vars: bool = False
):
    """Debug a config file and show validation errors.

    The function will create all objects in the tree and validate them. Note that
    some config validation errors are blocking and will prevent the rest of the
    config from being resolved. This means that you may not see all validation errors
    at once and some issues are only shown once previous errors have been fixed.
    As with the 'train' command, you can override settings from the config by passing
    arguments in the `overrides` dict.
    DOCS: https://spacy.io/api/cli#debug-config

    Args:
        config_path (Union[str, Path]): Path to the configuration file.
        overrides (dict): A dictionary of config overrides.
        code_path (Union[str, Path]): Path to Python file with additional code (registered functions)
            to be imported.
        show_funcs (bool): Show an overview of all registered functions used in the config and where they
            come from (modules, files etc.).
        show_vars (bool): Show an overview of all variables referenced in the config and their values.
            This will also reflect variables overwritten in the function call.
    """
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if isinstance(code_path, str):
        code_path = Path(code_path)
    import_code(code_path)
    spacy_debug_config(
        config_path, overrides=overrides, show_funcs=show_funcs, show_vars=show_vars
    )


def debug_data(
    config_path: Union[str, Path],
    overrides: dict = {},
    code_path: Union[str, Path] = None,
    ignore_warnings: bool = False,
    verbose: bool = False,
    no_format: bool = False
):
    """Analyze, debug and validate your training and development data.

    Outputs useful stats, and can help you find problems like invalid entity
    annotations, cyclic dependencies, low data labels and more.
    DOCS: https://spacy.io/api/cli#debug-data

    Args:
        config_path: Path to the configuration file.
        overrides: A dictionary of config overrides.
        code_path: Path to Python file with additional code (registered functions) to be imported.
        ignore_warnings: Ignore warnings, only show stats and errors.
        verbose: Print additional information and explanations.
        no_format: Don't pretty-print the results.

    Note:
        The only way to avoid the `SystemExit: 1` error is to make a copy of the module and
        remove the `sys.exit()` call at the end.
    """
    msg = Printer()
    msg.info("Note: If at least one error is found at the end of the analysis, "
             "the script will terminate with a `SystemExit: 1` error code.")
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if isinstance(code_path, str):
        code_path = Path(code_path)
    import_code(code_path)
    spacy_debug_data(
        config_path,
        config_overrides=overrides,
        ignore_warnings=ignore_warnings,
        verbose=verbose,
        no_format=no_format,
        silent=False
    )


def debug_model(
    config_path: Union[str, Path],
    config_overrides: dict = {},
    component: str = "tagger",
    layers: List[int] = [],
    dimensions: bool = False,
    parameters: bool = False,
    gradients: bool = False,
    attributes: bool = False,
    P0: bool = False,
    P1: bool = False,
    P2: bool = False,
    P3: bool = False,
    use_gpu: int = -1,
):
    """Debug a trained model.

    Args:
        config_path (Union[str, Path]): Path to the config file.
        config_overrides (dict): A dictionary of config overrides.
        component (str): Name of the pipeline component of which the model should be analysed
        layers (str): List of layer IDs to print.
        dimensions (bool): Whether to show dimensions.
        parameters (bool): Whether to show parameters.
        gradients (bool): Whether to show gradients.
        attributes (bool): Whether to show attributes.
        P0 (bool): Whether to print the model before training.
        P1 (bool): Whether to print the model after initialization.
        P2 (bool): Whether to print the model after training.
        P3 (bool): Whether to print final predictions.
        use_gpu (int): GPU ID or -1 for CPU
    """
    if isinstance(config_path, str):
        config_path = Path(config_path)
    setup_gpu(use_gpu)
    layers = [int(x) for x in layers]
    print_settings = {
        "dimensions": dimensions,
        "parameters": parameters,
        "gradients": gradients,
        "attributes": attributes,
        "layers": layers,
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
    msg = Printer()
    if seed is not None:
        msg.info(f"Fixing random seed: {seed}")
        fix_random_seed(seed)
    pipe = nlp.get_pipe(component)
    spacy_debug_model(config, T, nlp, pipe, print_settings=print_settings)

