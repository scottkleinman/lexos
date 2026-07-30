
# Mallet

Topic modeling is a statistical method for discovering abstract themes or "topics" within a collection of documents. MALLET is a mature tool for topic modeling used widely in the Humanities. It is a Java package that needs to be installed separately from Lexos. The Lexos `mallet` module provides a straightforward wrapper for running MALLET, managing outputs, and creating visualizations of your topic model.

## Main Functions and MALLET Class

### ::: lexos.topic_modeling.mallet.MALLET_BINARY_PATH
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.read_file
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.read_dirs
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.import_files
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.import_docs
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.Mallet
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.Mallet._metadata_get
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.Mallet._metadata_has
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.Mallet._import_training_data
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.Mallet._setup_wordcloud
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.Mallet._track_progress
    rendering:
      show_root_heading: true
      heading_level: 3

## LLM Topic Labeling

`llm_labeler` is an experimental module for using LLMs to automatically label topics produced by MALLET. It requires an existing MALLET `topic-keys.txt` file.

### ::: lexos.topic_modeling.mallet.llm_labeler.label_mallet_topics
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.llm_labeler.TopicLabelerConfig
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.llm_labeler.TopicLabelerConfig.__init__
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.llm_labeler.TopicLabelerClient
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.llm_labeler.TopicLabelerClient.generate_label
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.llm_labeler.TopicLabelerClient._call_openai_compatible
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.llm_labeler.TopicLabelerClient._call_gemini
    rendering:
      show_root_heading: true
      heading_level: 3

### ::: lexos.topic_modeling.mallet.llm_labeler.TopicLabelerClient._call_claude
    rendering:
      show_root_heading: true
      heading_level: 3
