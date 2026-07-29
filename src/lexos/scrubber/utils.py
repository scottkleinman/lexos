"""utils.py.

Last Update: 2026-06-26
Tested: 2026-06-26
"""

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
    seen_tags: set[str] = set()

    def normalize_tag_name(tag_name: str) -> str:
        return re.sub(r"{.+}", "", tag_name)

    def normalize_attribs(attribs: dict) -> dict:
        return {key: attribs[key] for key in sorted(attribs)}

    try:
        root = ElementTree.fromstring(text)
        for element in root.iter():
            tag_name = normalize_tag_name(element.tag)
            if tag_name not in seen_tags:
                seen_tags.add(tag_name)
                tags.append(tag_name)
            if element.attrib:
                attributes.append({tag_name: normalize_attribs(element.attrib)})
        tags = humansorted(tags)
        attributes = [
            {k: normalize_attribs(v)} for item in attributes for k, v in item.items()
        ]
    except ElementTree.ParseError:
        import bs4
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "xml")
        for e in list(soup):
            if isinstance(e, bs4.element.ProcessingInstruction):
                e.extract()
        for tag in soup.find_all():
            tags.append(tag.name)
            attributes.append({tag.name: normalize_attribs(tag.attrs)})
        tags = humansorted(tags)
        attributes = [
            {k: normalize_attribs(v)} for item in attributes for k, v in item.items()
        ]
    return {"tags": tags, "attributes": attributes}
