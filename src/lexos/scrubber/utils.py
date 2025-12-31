"""utils.py.

Last Update: 2025-01-15
Tested: 2025-01-15
"""

import json
import re
from xml.etree import ElementTree

from natsort import humansorted


def get_tags(text: str) -> dict:
    """Get information about the tags in a text.

    Args:
        text (str): The text to be analyzed.

    Returns:
        dict: A dict with the keys "tags" and "attributes". "Tags is a list of unique tag names
        in the data and "attributes" is a list of dicts containing the attributes and values
        for those tags that have attributes.

    Note:
        The procedure tries to parse the markup as well-formed XML using ETree; otherwise, it falls
        back to BeautifulSoup's parser.
    """
    tags = []
    attributes = []

    try:
        root = ElementTree.fromstring(text)
        for element in root.iter():
            if re.sub("{.+}", "", element.tag) not in tags:
                tags.append(re.sub("{.+}", "", element.tag))
            if element.attrib != {}:
                attributes.append({re.sub("{.+}", "", element.tag): element.attrib})
        tags = humansorted(tags)
        attributes = json.loads(json.dumps(attributes, sort_keys=True))
    except ElementTree.ParseError:
        import bs4
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "xml")
        for e in soup:
            if isinstance(e, bs4.element.ProcessingInstruction):
                e.extract()
        [tags.append(tag.name) for tag in soup.find_all()]
        [attributes.append({tag.name: tag.attrs}) for tag in soup.find_all()]
        tags = humansorted(tags)
        attributes = json.loads(json.dumps(attributes, sort_keys=True))
    return {"tags": tags, "attributes": attributes}
