"""__init__.py."""

from typing import List, Union

import spacy

from lexos import utils

from . import extensions, lexosdoc

default_model = spacy.load("xx_sent_ud_sm")


def _get_excluded_components(
    exclude: List[str] = None, pipeline_components: dict = None
) -> List[str]:
    """Get a list of components to exclude from the pipeline."""
    if exclude is None:
        exclude = []
    custom_exclude = []
    if "exclude" in pipeline_components:
        for component in pipeline_components["exclude"]:
            custom_exclude.append(component)
    exclude.extend(custom_exclude)
    return list(set(exclude))


def _get_disabled_components(
    disable: List[str] = None, pipeline_components: dict = None
) -> List[str]:
    """Get a list of components to disable in the pipeline."""
    if disable is None:
        disable = []
    custom_disable = []
    if "disable" in pipeline_components:
        for component in pipeline_components["disable"]:
            custom_disable.append(component)
    disable.extend(custom_disable)
    return list(set(disable))


def make_doc(
    text: str,
    model: object = "xx_sent_ud_sm",
    max_length: int = 2000000,
    disable: List[str] = [],
    exclude: List[str] = [],
    add_stopwords: List[str] = [],
    remove_stopwords: Union[List[str], str] = [],
    pipeline_components: List[dict] = [],
) -> object:
    """Return a doc from a text.

    Args:
        text (str): The text to be parsed.
        model (object): The model to be used.
        max_length (int): The maximum length of the doc.
        disable (List[str]): A list of spaCy pipeline components to disable.
        exclude (List[str]): A list of spaCy pipeline components to exclude.
        add_stopwords (List[str]): A list of stop words to add to the model.
        remove_stopwords (Union[List[str], str]): A list of stop words to remove
            from the model. If "all" is specified, all stop words will be removed.
        pipeline_components (List[dict]): A list custom component dicts to add
            to the pipeline. See https://spacy.io/api/language/#add_pipe for
            more information.

    Returns:
        object: A spaCy doc object.
    """
    disable = _get_disabled_components(disable, pipeline_components)
    exclude = _get_excluded_components(exclude, pipeline_components)
    nlp = spacy.load(model, disable=disable, exclude=exclude)
    nlp.max_length = max_length
    if add_stopwords:
        nlp.Defaults.stop_words |= set(add_stopwords)  # A set, e.g. {"and", "the"}
    if remove_stopwords:
        if remove_stopwords == "all":
            nlp.Defaults.stop_words |= {}
        else:
            nlp.Defaults.stop_words |= set(
                remove_stopwords
            )  # A set, e.g. {"and", "the"}
    if pipeline_components and "custom" in pipeline_components:
        for component in pipeline_components["custom"]:
            nlp.add_pipe(**component)
    return nlp(text)


def make_docs(
    texts: Union[List[str], str],
    model: object = "xx_sent_ud_sm",
    max_length: int = 2000000,
    disable: List[str] = [],
    exclude: List[str] = [],
    add_stopwords: List[str] = [],
    remove_stopwords: Union[List[str], str] = [],
    pipeline_components: List[dict] = [],
) -> List[object]:
    """Return a list of docs from a text or list of texts.

    Args:
        texts (Union[List[str], str]): The text(s) to be parsed.
        model (object): The model to be used.
        max_length (int): The maximum length of the doc.
        disable (List[str]): A list of spaCy pipeline components to disable.
        exclude (List[str]): A list of spaCy pipeline components to exclude.
        add_stopwords (List[str]): A list of stop words to add to the model.
        remove_stopwords (Union[List[str], str]): A list of stop words to remove
            from the model. If "all" is specified, all stop words will be removed.
        pipeline_components (List[dict]): A list custom component dicts to add
            to the pipeline. See https://spacy.io/api/language/#add_pipe for
            more information.

    Returns:
        List[object]: A list of spaCy doc objects.
    """
    disable = _get_disabled_components(disable, pipeline_components)
    exclude = _get_excluded_components(exclude, pipeline_components)
    nlp = spacy.load(model, disable=disable, exclude=exclude)
    nlp.max_length = max_length
    if add_stopwords:
        nlp.Defaults.stop_words |= set(add_stopwords)  # A set, e.g. {"and", "the"}
    if remove_stopwords:
        if remove_stopwords == "all":
            nlp.Defaults.stop_words |= {}
        else:
            nlp.Defaults.stop_words |= set(
                remove_stopwords
            )  # A set, e.g. {"and", "the"}
    if pipeline_components and "custom" in pipeline_components:
        for component in pipeline_components["custom"]:
            nlp.add_pipe(**component)
    return list(nlp.pipe(utils.ensure_list(texts)))


def doc_from_ngrams(ngrams: list, model="xx_sent_ud_sm", strict=False) -> object:
    """Generate spaCy doc from a list of ngrams.

    Args:
        ngrams (list): A list of ngrams.
        model (object): The language model to use for tokenisation.
        strict (bool): Whether to preserve token divisions, include whitespace in the source.

    Returns:
        object: A spaCy doc

    Notes:
        The `strict=False` setting will allow spaCy's language model to remove whitespace from
        ngrams and split punctuation into separate tokens. `strict=True` will preserve the
        sequences in the source list.
    """
    nlp = spacy.load(model)
    if strict:
        spaces = [False for token in ngrams if token != ""]
        doc = spacy.tokens.doc.Doc(nlp.vocab, words=ngrams, spaces=spaces)
        # Run the standard pipeline against the doc
        for _, proc in nlp.pipeline:
            doc = proc(doc)
        return doc
    else:
        text = " ".join([x.replace(" ", "") for x in ngrams])
        return nlp(text)


def docs_from_ngrams(
    ngrams: List[list], model="xx_sent_ud_sm", strict=False
) -> List[object]:
    """Generate spaCy doc from a list of ngram lists.

    Args:
        ngrams (List[list]): A list of ngram lists.
        model (object): The language model to use for tokenisation.
        strict (bool): Whether to preserve token divisions, include whitespace in the source.

    Returns:
        List[object]: A list of spaCy docs
    """
    docs = []
    for ngram_list in ngrams:
        doc = doc_from_ngrams(ngram_list, model, strict)
        docs.append(doc)
    return docs


def generate_character_ngrams(
    text: str, size: int = 1, drop_whitespace: bool = True
) -> List[str]:
    """Generate character n-grams from raw text.

    Args:
        text (str): The source text.
        size (int): The size of the ngram.
        drop_whitespace (bool): Whether to preserve whitespace in the ngram list.

    Returns:
        List[str]: A list of ngrams
    """
    from textwrap import wrap

    return wrap(text, size, drop_whitespace=drop_whitespace)


def ngrams_from_doc(doc: object, size: int = 2) -> List[str]:
    """Generate a list of ngrams from a spaCy doc.

    A wrapper for `textacy.extract.basics.ngrams`. With basic functionality.
    Further functionality can be accessed by calling `textacy` directly.

    Args:
        doc (object): A spaCy doc
        size (int): The size of the ngrams.

    Returns:
        List[str]: A list of ngrams.
    """
    import textacy.extract.basics.ngrams as textacy_ngrams

    ngrams = list(textacy_ngrams(doc, size, min_freq=1))
    # Ensure quoted strings are returned
    return [token.text for token in ngrams]
