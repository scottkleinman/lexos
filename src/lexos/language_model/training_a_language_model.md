The Lexos [tokenizer](../tokenising_texts) module now uses language models for a richer and more flexible way of accessing the linguistic features of texts. Token segmentation and feature assignment are performed under the hood by the <a href="https://spacy.io/" target="_blank">spaCy</a>, and, for some users, spaCy's pre-built language models may be sufficient. For others, models trained on their data may be necessary, especially if they are working with a language for which no pre-built model exists. The spaCy library provides a rich environment for training language models, and the Lexos API provides a thin wrapper around its functionality for greater ease of use and integration into applications.

Language models in spaCy are built on pipelines of configurable components that handle tagging different types of linguistic features. The most basic is the "tagger", which labels parts of speech, but other pipeline components handle morphological analysis, lemmatization, and named entity recognition. Components are registered in a configuration (config) file which exposes the dizzying array of options available in a machine learning workflow and serves as the "single source of truth" for how the model was trained. The Lexos API maintains and provides access to the config file whilst allowing the user to train a language model with minimal need to edit it. Language models can be instantiated with configuration "recipes" that provide recommended configuration values for various use cases.

!!! note
    In spaCy, the term "pipeline" refers to a configuration of components, rules or algorithms for assigning feature labels to texts, whereas the term "model" refers to the statistical weights (vectors) these algorithms produce that are used to make predictions about labels. Since trained pipelines generate statistical models, there is inevitably some overlap in the terminology.

## The `LanguageModel` Class and the Training Workflow

Most of the work can be performed using the `LanguageModel` class, which can be instantiated with `model = LanguageModel()`. This will create a new language model folder with various subfolders and a configuration file. In most cases, the user will want to pass certain information on instantiation, such as the desired path to the model folder. This procedure will be illustrated below.

Once a `LanguageModel` object is instantiated, the next step is to copy assets into the model's `assets` folder. The assets are normally three files, one for training, one for development, and one for testing. These three files are then converted to binary spaCy format, stored in the `corpus` folder, with [LanguageModel.convert_assets][lexos.language_model.LanguageModel.convert_assets].

Once the assets are in spaCy format, the training and development files are used to train the model with `LanguageModel.train()`. The trained model is stored in the `training` folder. Once complete, the model can be evaluated with `LanguageModel.evaluate()`. This provides various statistical measures of the model's accuracy by testing its predictions against the labels in the testing file. Results are saved in the `metrics` folder.

If the user deems the model to be sufficiently accurate, they can then package it with `LanguageModel.package()`. This saves the model, its configuration file, and all its pipeline components in the `packages` folder. The path to the new model can now be passed to the `Tokenizer`, and the new model will be loaded for parsing texts into spaCy docs.

A more in-depth, step-by-step explanation of the procedure is described below.

## Before Getting Started

Training a model requires you to have a set of training data, which is a series of sentences (or sentence-like units) with pre-assigned labels. Producing training data can be a time-consuming process. If the language you are working in is close to that of a pre-built model, it is probably easiest to use `Tokenizer` to generate labels on some sample texts from your data and then correct the labels as appropriate. For most purposes, it will be easiest to work in the <a href="https://universaldependencies.org/format.html" target="_blank">CONLL-U format</a> because of its convenient columnar format and because spaCy has a built-in converter for CONLL-U. This tutorial does not cover data preparation for training features, such as named entities, which are not included in CONLL-U. That said, the conversion process converts the CONLL-U text to a set of spaCy docs, so, if the desired labels are available in a spaCy doc, it is possible to proceed. More information on this is given below.

!!! important
    Note that spaCy functions sometimes assume that you are working on the command line and can provide output that does not make sense if you are working in, say, a Jupyter notebook. In most cases, this will take the form of references to command-line flags, but other idiosyncratic behaviours may occur. For instance, when evaluating a model, if there is a single error, the spaCy function calls `sys.exit()` causing `SystemExit: 1` error code with no indication to the user that the script did not exit in the middle of execution! However, such glitches are rare.

## Instantiating a `LanguageModel` Object

By default, calling `model = LanguageModel()` will generate a folder called `language_model` in the current directory. However, it is possible to add arguments to override the default settings. Here is an example:

```python
model = LanguageModel(
    project_dir="update_en_core_web_sm",
    recipe="project/recipes/update_en_core_web_sm.cfg",
    force=True
)
```

This will create the "update_en_core_web_sm" folder. Instead of auto-generating a default config file, it will copy the "recipe" config file for updating spaCy's "en_core_web_sm" model. Setting `force=True` will cause Lexos to overwrite and pre-existing configuration if the model folder was created previously. See the API documentation for the full set of arguments that can be passed to the `LanguageModel` class. We will demonstrate some of them below.

!!! note "A note on model folder paths"
    When a `LanguageModel` object is instantiated, most of the folders necessary for managing the workflow are created. By default, each subfolder contains the name of the model's language (the default is multilingual "xx"). Most of the model's methods will then assume that files are written to this language-specific subfolder. This organizational structure is largely based on the spaCy <a href="https://github.com/explosion/projects/tree/v3/pipelines/tagger_parser_ud" target="_blank">tagger_parser_ud project</a>, and it remains to be seen if it should be a best practice for a more generic application. spaCy projects also store configuration files in a `configs` folder, which has not been deemed necessary for the `LanguageModel` class. As an instance should only have one configuration at a time, the `LanguageModel`'s config file is stored in its folder root. These decisions are subject to re-evaluation.

## Working with the Configuration

Once the `LanguageModel` object is instantiated, the model folder will contain a configuration file with default configuration values or values you have assigned using arguments passed during instantiation. Although the configuration file is the "single source of truth" for the model, the `LanguageModel` object maintains a copy as a class attribute. Internally, this copy is parsed as a modified Python dict, which can be accessed by calling `model.config`. To see a string representation, call `model.config.to_str()`.

!!! note
    The "modified Python dict" is actually a Thinc `Config` object (Thinc is the machine-learning library used internally by spaCy), so any of the `Config` object's methods as described in the <a href="https://thinc.ai/docs/api-config" target="_blank">Thinc documentation</a> can be used.

If you wish to change a value in your configuration you can simply modify the dictionary. For instance, if you wanted to change the language to English, you would use: `model.config["nlp"]["lang"] = "en"`. This only modifies the `config` attribute. To ensure that your changes are made in the config _file_, you must then call `model.save_config()`.

You can also replace the current config file and (`config` attribute) by calling `model.load_config()` with a path to the desired config file. This is the equivalent of instantiating the `LanguageModel` object with a `recipe` path and `force=True`.

## Copying Assets

Copying assets involves copying your training and testing data from another location into the models `assets` folder. This is done with the following code:

```python
project.copy_assets(
    training_file="path/to/train.conllu",
    dev_file="path/to/dev.conllu",
    test_file="path/to/test.conllu"
)
```

In addition to local filepaths, `copy_assets()` also accepts urls.

Asset files are not used directly when training a model; they are primarily for archival purposes. This is especially useful if you have downloaded them since you do not have to download them again if you need to re-instantiate the model object.

Asset files are assumed to be in CONLL-U format, which will be converted to spaCy binary format by `LanguageModel.convert_assets()` (see below). If your training and testing files are not in CONLL-U format, you may still manually copy them into the `assets` folder for archival purposes, and you do not need to call `LanguageModel.copy_assets()`.

## Converting Assets to spaCy Format

The `LanguageModel.convert_assets()` method assumes that you have CONLL-U formatted files in the model's `assets` folder and automatically converts them to spaCy binary formatted files in the model's `corpus` folder. If your assets are not in CONLL-U format, you can skip this step, but you are responsible for converting them to spaCy binary format and depositing them in the `corpus` folder by some other means. Information on how to do that can be found in the <a href="https://spacy.io/usage/training#training-data" target="_blank">spaCy documentation</a>.

!!! important
    The model's `training_file`, `dev_file`, and `test_file` attributes are used to locate files for conversion in the `assets` folder. If the files do not have the same names, an error will occur. You can see (or set) these attribute values with `model.training_file`, `model.dev_file`, and `model.test_file`. If you have not set them when you instantiated the `LanguageModel` object.

## Debugging Configuration and Data

Before training your model, or if you encounter a problem, it is a good idea to try debugging with `debug_config(path/to/config/file)` or `debug_data(path/to/config/file)`. This will generate reports identifying potential problems.

!!! important
    `LanguageModel.debug_config()` and `LanguageModel.debug_data()` are not part of the `LanguageModel` class, so do not call them without the dot notation. This means that you can debug a config file without an instantiated `LanguageModel` object.

## Selecting Pipeline Components

Before training your model, you must also ensure that you have specified which pipeline components you wish to train. By default, the `LanguageModel` object configures just the "tagger", which labels parts of speech. Other common pipeline components are "tok2vec", "attribute_ruler", "lemmatizer", "parser", "ner". For an overview of pipeline components, see the <a href="https://spacy.io/usage/processing-pipelines" target="_blank">spaCy documentation</a>.

You can set pipeline components when instantiating a `LanguageModel` object with the `components` argument, which takes a list of components. You can also set them afterwards by modifying `model.config["nlp"]["pipeline"]`, or even by editing the config file itself.

!!! warning
    Setting pipeline components opens a bewildering away of possibilities for configuring the components, and it is very likely to generate invalid config files. It is highly recommended that you use `LanguageModel.debug_config()` if this happens but even more highly recommended that you rely on trusted recipes, especially if you are developing an interface for user-defined configuration. This will minimize the chance that you will encounter errors during training.

Most components are independent of each other, but some share a "token-to-vector" component like "tok2vec". It is possible to speed up training by setting some components to be "frozen" and by indicating which components will be used for annotation. The following example assumes a model with the pipeline components "tagger", "attribute_ruler", "lemmatizer", "parser", and "ner". However, it skips training "attribute_ruler", "lemmatizer", "parser", and "ner". If the source of the components is a pre-existing model, the pre-existing state is simply passed to the new model. However, in the new model, only the "tagger", "attribute_rule", and "lemmatizer" components will have annotations.

```python
model.config["training"]["frozen_components"] = [
    "attribute_ruler",
    "lemmatizer",
    "parser",
    "ner"
]

model.config["training"]["annotating_components"] = [
    "tagger",
    "attribute_ruler",
    "lemmatizer"
]

model.save_config()
```

Again, it is safest to use a trusted recipe to ensure that all components are properly configured.

## Training a Model

If everything looks good, you are ready to train your model. To do this, simply call `LanguageModel.train()`. Although it is possible to pass arguments to override the settings in the config file, in most cases you will not want to do so.

Once training begins, its progress will be logged to the screen, and you can watch it train. Be aware that training can take a very (and unpredictably) long time for large data sets. Be patient!

### Balancing Efficiency and Accuracy

By default, models are generated with efficiency in mind, but you can change this by setting `optimize="accuracy"` when instantiating your model. This will cause the `LanguageModel` object to generate a different config file, one that may produce more accurate models, but with potentially longer training times and bigger output model sizes. Internally, these config files are generated by an evolving set of <a href="https://github.com/explosion/spaCy/blob/master/spacy/cli/templates/quickstart_training_recommendations.yml" target="_blank">language-specific recommendations</a> maintained by spaCy. You can, of course, override these recommendations or use a recipe config file to bypass them entirely. If you are using a language for which spaCy has a pre-built model, it can be useful to generate a config file with the default recommendations and then modify it for your own purposes as necessary.

### Training with a GPU and Transformer Models

Using a GPU to train your model can be 2-3 times faster than using your computer's CPU. You can substitute a GPU for your computer's CPU simply by instantiating your `LanguageModel` object with `gpu` set to your GPU's id. However, this may not be worth it, given that setting up a GPU can be very fiddly. Where it truly becomes necessary is if you want to take advantage of pre-built transformer models such as BERT or GPT-2. For an introduction to the use of transformer models, see <a href="https://spacy.io/usage/embeddings-transformers" target="_blank">spaCy's documentation</a> and the <a href="https://github.com/explosion/spacy-transformers" target="_blank">spacy-transformers</a> plugin.

!!!warning
    Setting up a GPU and transformer models is not for the faint of heart. Most likely, this will be done by application developers on the back end so that individual users do not have to do so on their own machines.

## Evaluating a Model

Once a model has completed training, the output is stored in the `training` folder. You can then evaluate it by calling `LanguageModel.evaluate()` with the path to the trained model folder and the path to your test file in the `corpus` folder.

```python
model.evaluate(
    model="training/en/model-best",
    testfile="corpus/en/test.spacy"
)
```

You will receive a report showing how accurately the model predicted the labels in your test data. A copy of the report in json format is saved to the `metrics` folder.

!!! note
    If you run into a problem, you can also debug your model by calling `LanguageModel.debug_model("path/to/config/file")`.

If you are unhappy with the level of accuracy, you can try tweaking your configuration or, with a greater likelihood of success, adding more training data. Re-train the model and test again.

## Packaging the Model

To make the model useable, it must first be packaged. Once packaged, the model can be used for parsing texts using either the Lexos API or using spaCy directly:

```python
# Using the Lexos API
from lexos import tokenizer
doc = tokenizer.make_doc(text, model="path/to/model")

# Using spaCy
import spacy
nlp = spacy.load("path/to/model")
doc = nlp(text)
```

The path to the model package should point to the directory containing the packages `config.cfg` file.

!!! note
    So far, I have only tried loading model packages using the path the package folder. It should also be possible to use the package name, but this needs further testing.

To package a model, call `LanguageModel.package()`, providing the input folder for the model in the `training` folder and the path to the output folder. You can also provide a `name` and `version` for the model if you did not instantiation the `LanguageModel` object with `package_name` and `package_version` values. Setting `force=True` will overwrite any previous package in the output directory.

```python
model.package(
    input_dir="path/to/model-best",
    output_dir="path/to/output/directory",
    name="en_test_sm",
    version="1.0.0",
    force=True
)
```

It is a good idea to store packages in a folder called `packages` inside the model folder, although they can be stored in a separate location for distribution.
