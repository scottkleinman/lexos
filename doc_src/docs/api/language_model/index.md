# Language Model

The `language_model` module is a wrapper around spaCy's training workflow for fine-tuning language models on custom corpora. It handles directory setup, config generation (including fine-tuning via component sourcing and transformer-based training via recipes), data conversion, training, evaluation, and packaging without requiring the user to edit config files or use the command line.

For a user-friendly overview, see [Training Language Models](../../user_guide/language_model/index.md) in the User Guide. For hands-on walkthroughs, see the tutorial notebooks listed in [Tutorials](../../tutorials/index.md).

## Constants

### ::: lexos.language_model.FULL_UD_PIPELINE

    rendering:
      show_root_heading: true
      heading_level: 3

## CONLL-U Utilities

Standalone functions for preparing and managing CONLL-U training data.

### ::: lexos.language_model.split_conllu

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.export_to_conllu

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.combine_conllu

    rendering:
      show_root_heading: true
      heading_level: 3

## The LanguageModel Class

The main entry point for the module. Manages the model directory, generates or loads the spaCy training config, and exposes the training lifecycle as method calls.

### ::: lexos.language_model.LanguageModel

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.__init__

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.config_path

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.save_config

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.load_config

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.copy_assets

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.convert_assets

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.validate

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.train

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.evaluate

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel.package

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel._load_recipe

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel._resolve_sources

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel._generate_finetune_config

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.LanguageModel._apply_config_defaults

    rendering:
      show_root_heading: true
      heading_level: 3

## Debugging Utilities

Wrappers around spaCy's debugging commands for inspecting a model's config and data before training.

### ::: lexos.language_model.debug_config

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.debug_data

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.debug_model

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model.fill_config

    rendering:
      show_root_heading: true
      heading_level: 3

## Internal Helpers

### ::: lexos.language_model._has_nvidia_gpu

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model._get_tok2vec_width

    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.language_model._patch_tok2vec_width

    rendering:
      show_root_heading: true
      heading_level: 3
