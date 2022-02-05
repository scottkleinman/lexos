"""__init__.py."""

from typing import List, Union

import spacy

from lexos import utils

from . import extensions, lexosdoc

default_model = spacy.load("xx_sent_ud_sm")

def make_doc(text: str,
			  model: object = "xx_sent_ud_sm",
			  disable: List[str] = []) -> object:
    """Return a doc from a text.

	Args:
		text (str): The text to be parsed.
		model (object): The model to be used.
		disable (List[str]): A list of spaCy pipeline components to disable.

	Returns:
		object: A spaCy doc object.
	"""
    nlp = spacy.load(model)
    return nlp(text, disable=disable)

def make_docs(texts: Union[List[str], str],
			  model: object = "xx_sent_ud_sm",
			  disable: List[str] = []) -> List:
	"""Return a list of docs from a text or list of texts.

	Args:
		text (Union[List[str], str]): The text(s) to be parsed.
		model (object): The model to be used.
		disable (List[str]): A list of spaCy pipeline components to disable.

	Returns:
		list: A list of spaCy doc objects.
	"""
	nlp = spacy.load(model)
	return list(nlp.pipe(utils.ensure_list(texts), disable=disable))
