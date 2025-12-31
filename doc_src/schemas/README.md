# `mkdocs.yml` Schema Validation

The `mkdocs.yml` file is validated using a JSON schema loaded from mkdocs-material. However, that schema does not possess a reference to the autorefs plugin. The plugin still works, but its presence triggers a validation warning. This folder contains a modified version of the mkdocs-material schema which points to a schema for autorefs. However, it won't seem to load from a local folder.

Part of the problem may relate to the fact that the mkdocs-material schema has to be added to to the user settings file. I have tried to add it to the workspace and project folder settings files, and it does not load. This is something to be investigated further. I wouldn't like to have to add the entire schema to the user settings file.

Additionally, storing a private copy of the schema will mean that it will periodically have to be checked for updates. The mkdocs-material team is not interested in keeping track of every external plugin, so requesting that autorefs be added may not be worth it (but it is highly rated, so maybe so?). Otherwise, we may have to live with the validation warning.
