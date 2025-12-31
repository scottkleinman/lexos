"""tags.py.

This module provides functions to manipulate HTML/XML tags and attributes
using BeautifulSoup. It allows for removing, replacing, and modifying tags
and their attributes in HTML or XML documents.

It supports both exact, contains, and regex matching for selectors and
attributes, and can filter elements based on attributes and their values.

Last Updated: December 4, 2025
Last Tested: September 5, 2025
"""

import re
from typing import Optional

from bs4 import BeautifulSoup, Comment

# from wasabi import msg
from lexos.exceptions import LexosException


def _match_elements(
    selector: str,
    text: str,
    mode: str = "html",
    matcher_type: Optional[str] = "exact",
    attribute: Optional[str | list[str]] = None,
    attribute_value: Optional[str] = None,
    attribute_filter: Optional[str] = None,
) -> tuple[BeautifulSoup, list]:
    """Finds HTML/XML elements matching a selector.

    Args:
        selector: Tag name or CSS selector to match elements
        text: HTML or XML text to process
        mode: Parser mode, either "html" or "xml"
        matcher_type: Type of match to perform, either "exact", "contains", or "regex"
        attribute: Optional attribute name to filter elements
        attribute_value: Optional value for the attribute filter
        attribute_filter: Optional attribute name to filter elements

    Returns:
        A BeautifulSoup object and a list of matching elements.
    """
    # Validate mode
    if mode not in ["html", "xml"]:
        raise LexosException("Mode must be either 'html' or 'xml'.")

    # Parse the document
    parser = "lxml-xml" if mode == "xml" else "html.parser"
    soup = BeautifulSoup(text, parser)

    # Find elements matching the selector
    if not selector:
        elements = soup.find_all()  # Select all elements
    else:
        elements = (
            soup.select(selector)
            if selector.startswith(".") or selector.startswith("#")
            else soup.find_all(selector)
        )

    # Filter by attribute if specified using attribute_filter
    if attribute_filter:
        if attribute_value:
            # Filter elements that have the attribute with the specific value
            elements = [
                el
                for el in elements
                if el.has_attr(attribute_filter)
                and _match_value(el[attribute_filter], attribute_value, matcher_type)
            ]
        else:
            # Filter elements that have the attribute regardless of value
            elements = [el for el in elements if el.has_attr(attribute_filter)]

    # Filter by attribute if specified
    elif attribute:
        if attribute_value:
            # Filter elements that have the attribute with the specific value
            elements = [
                el
                for el in elements
                if el.has_attr(attribute)
                and _match_value(el[attribute], attribute_value, matcher_type)
            ]
        else:
            # Filter elements that have the attribute regardless of value
            elements = [el for el in elements if el.has_attr(attribute)]

    return soup, elements


def _match_value(
    attribute_value: str | list[str], pattern: str, match_type: str = "exact"
) -> bool:
    """Match attribute values using different matching strategies.

    Args:
        attribute_value: The attribute value(s) to match against (string or list)
        pattern: The pattern/string to match
        match_type: Type of matching - "exact", "contains", or "regex"

    Returns:
        True if the pattern matches according to the specified type, False otherwise.

    Raises:
        LexosException: If match_type is not one of the valid options.
    """
    # Convert list to space-separated string if needed
    if isinstance(attribute_value, list):
        attribute_value = " ".join(attribute_value)

    if match_type == "exact":
        # Exact match - pattern must match the entire attribute value
        return pattern == attribute_value

    elif match_type == "contains":
        # Includes match - pattern must exactly match one of the space-separated values
        values = attribute_value.split()
        return pattern in values

    elif match_type == "regex":
        # Regex match - pattern is treated as a regex
        try:
            return re.search(pattern, attribute_value) is not None
        except re.error:
            # Invalid regex pattern
            return False

    else:
        raise LexosException(
            f"match_type must be 'exact', 'contains', or 'regex', got '{match_type}'"
        )


def remove_attribute(
    text: str,
    selector: str,
    attribute: str = None,
    mode: str = "html",
    matcher_type: str = "exact",
    attribute_value: Optional[str] = None,
    attribute_filter: Optional[str] = None,
) -> str:
    """Removes attributes from HTML/XML elements.

    Removes specified attributes from elements matching the selector.
    Can filter elements by specific attribute or attribute value.

    Args:
        text: HTML or XML text to process
        selector: Tag name or CSS selector to match elements
        attribute: Attribute name to remove.
        mode: Parser mode, either "html" or "xml"
        matcher_type: Type of match to perform, either "exact", "contains", or "regex"
        attribute_value: Optional value for the attribute filter
        attribute_filter: Optional attribute name to filter elements

    Returns:
        Processed text with attributes removed from matching elements

    Raises:
        LexosException: If mode is not "html" or "xml"

    Examples:
        >>> text = '<div class="main" id="content">Text</div>'
        >>> remove_attributes(text, "div", "class")
        '<div id="content">Text</div>'

        >>> text = '<p class="a">Keep</p><p class="b" id="x">Remove attrs</p>'
        >>> remove_attributes(text, "p", attribute_filter="class", attribute_value="b")
        '<p class="a">Keep</p><p>Remove attrs</p>'
    """
    # Get matching elements
    soup, elements = _match_elements(
        selector, text, mode, matcher_type, attribute, attribute_value, attribute_filter
    )

    # Filter by attribute if specified
    # if attribute_filter:
    #     if attribute_value:
    #         elements = [
    #             el
    #             for el in elements
    #             if el.has_attr(attribute_filter)
    #             and _match_value(el[attribute_filter], attribute_value, matcher_type)
    #         ]
    #     else:
    #         # Filter elements that have the attribute regardless of value
    #         elements = [el for el in elements if el.has_attr(attribute_filter)]

    # Remove specified attributes from matching elements
    for element in elements:
        if attribute:
            # Remove only the specified attribute
            if element.has_attr(attribute):
                del element[attribute]
        else:
            # Remove all attributes
            element.attrs = {}

    # Return the processed document
    return str(soup)


def remove_comments(text: str, mode: str = "html") -> str:
    """Removes comments from HTML or XML text.

    Uses BeautifulSoup to find and remove all comments from HTML or XML content.

    Args:
        text: HTML or XML text to process
        mode: Parser mode, either "html" or "xml"

    Returns:
        String containing the HTML/XML content with all comments removed

    Raises:
        LexosException: If mode is not "html" or "xml"

    Examples:
        >>> html = '<!-- Header comment --><div>Content</div><!-- Footer -->'
        >>> remove_comments(html)
        '<div>Content</div>'

        >>> xml = '<?xml version="1.0"?><!-- Config --><root>Data</root>'
        >>> remove_comments(xml, mode="xml")
        '<?xml version="1.0"?><root>Data</root>'
    """
    # Validate mode
    if mode not in ["html", "xml"]:
        raise LexosException("Mode must be either 'html' or 'xml'.")

    # Parse the document
    parser = "lxml-xml" if mode == "xml" else "html.parser"
    soup = BeautifulSoup(text, parser)

    # Find all comment nodes
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    # Remove each comment
    for comment in comments:
        comment.extract()

    # Return the processed document
    return str(soup)


def remove_doctype(text: str) -> str:
    """Removes a document type declaration from HTML or XML text.

    Args:
        text: HTML or XML text to process

    Returns:
        String containing the HTML/XML content with document type declaration removed
    """
    # Remove HTML and XML doctype declarations
    html_doctype_pattern = re.compile(r"<!DOCTYPE[^>]*>", re.IGNORECASE | re.DOTALL)
    text = re.sub(html_doctype_pattern, "", text)

    xml_doctype_pattern = re.compile(r"<?xml[^>]*>", re.IGNORECASE | re.DOTALL)
    text = re.sub(xml_doctype_pattern, "", text)

    # Return the processed document
    return text


def remove_element(
    text: str,
    selector: str,
    mode: str = "html",
    matcher_type: str = "exact",
    attribute: str = None,
    attribute_value: str = None,
) -> str:
    """Removes HTML/XML elements using BeautifulSoup.

    Removes elements that match the given selector from HTML or XML text.
    Can further filter elements by specific attribute or attribute value.

    Args:
        text: HTML or XML text to process
        selector: Tag name or CSS selector to match elements for removal
        mode: Parser mode, either "html" or "xml"
        matcher_type: Type of match to perform, either "exact", "contains", or "regex"
        attribute: Optional attribute name to filter elements
        attribute_value: Optional value for the attribute filter

    Returns:
        Processed text with matching elements removed

    Raises:
        LexosException: If mode is not "html" or "xml"

    Examples:
        >>> text = "<p class='a'>Keep</p><p class='b'>Remove</p><div>Remove</div>"
        >>> remove_element(text, "div")
        '<p>Keep</p>'
        >>> remove_element("text", "p", attribute="class", attribute_value="b")
        "<p class='a'>Keep</p><div>Remove</div>"
    """
    # Get matching elements
    soup, elements = _match_elements(
        selector, text, mode, matcher_type, attribute, attribute_value
    )

    # Remove matching elements
    for element in elements:
        element.decompose()

    # Return the processed document
    return str(soup)


def remove_tag(
    text: str,
    selector: str,
    mode: str = "html",
    matcher_type: str = "exact",
    attribute: str = None,
    attribute_value: str = None,
) -> str:
    """Removes HTML/XML tags but keeps their inner content.

    Removes tags matching the selector while preserving their inner content.
    Can filter elements by specific attribute or attribute value.

    Args:
        text: HTML or XML text to process
        selector: Tag name or CSS selector to match elements for unwrapping
        mode: Parser mode, either "html" or "xml"
        matcher_type: Type of match to perform, either "exact", "contains", or "regex"
        attribute: Optional attribute name to filter elements
        attribute_value: Optional value for the attribute filter

    Returns:
        Processed text with matching tags unwrapped but content preserved

    Raises:
        LexosException: If mode is not "html" or "xml"

    Examples:
        >>> text = "<div><p>Keep this</p></div><span>And this</span>"
        >>> remove_tag(text, "div")
        '<p>Keep this</p><span>And this</span>'
        >>> text = "<p class='a'>Keep tag</p><p class='b'>Remove tag only</p>"
        >>> remove_tag(text, "p", attribute="class", attribute_value="b")
        "<p class='a'>Keep tag</p>Remove tag only"
    """
    # Get matching elements
    soup, elements = _match_elements(
        selector, text, mode, matcher_type, attribute, attribute_value
    )

    # Unwrap matching elements (remove tag but keep content)
    for element in elements:
        element.unwrap()

    # Return the processed document
    return str(soup)


def replace_attribute(
    text: str,
    selector: str,
    old_attribute: str,
    new_attribute: str,
    mode: str = "html",
    matcher_type: str = "exact",
    attribute_value: Optional[str] = None,
    replace_value: Optional[str] = None,
    attribute_filter: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> str:
    """Replaces HTML/XML element attributes or their values.

    This function finds elements matching the selector and replaces attribute names
    or attribute values. It can filter elements by a specific attribute/value.

    Args:
        text: HTML or XML text to process
        selector: Tag name or CSS selector to match elements
        old_attribute: Name of the attribute to replace
        new_attribute: Name of the new attribute (or same name if only changing value)
        mode: Parser mode, either "html" or "xml"
        matcher_type: Type of match to perform, either "exact", "contains", or "regex"
        attribute_value: Only replace attributes with this specific value
        replace_value: New value to use (keeps original value if None)
        attribute_filter: Optional attribute name to filter elements
        filter_value: Optional value for the attribute filter

    Returns:
        Processed text with attributes replaced in matching elements

    Raises:
        LexosException: If mode is not "html" or "xml"

    Examples:
        >>> # Replace class attribute with data-type, keeping the value
        >>> text = '<div class="main">Text</div>'
        >>> replace_attribute(text, "div", "class", "data-type")
        '<div data-type="main">Text</div>'

        >>> # Replace class="info" with class="highlight"
        >>> text = '<p class="info">Text</p><p class="data">More</p>'
        >>> replace_attribute(text, "p", "class", "class", filter_value="info", replace_value="highlight")
        '<p class="highlight">Text</p><p class="data">More</p>'

        >>> # Only replace attributes on elements with a specific attribute value
        >>> text = '<div class="main" id="content">Text</div><div class="sidebar">Side</div>'
        >>> replace_attribute(text, "div", "class", "role", attribute_filter="id", filter_value="content")
        '<div role="main" id="content">Text</div><div class="sidebar">Side</div>'
    """
    # Get matching elements
    soup, elements = _match_elements(
        selector, text, mode, matcher_type, old_attribute, attribute_value
    )

    # Filter by attribute if specified
    if attribute_filter:
        if filter_value:
            # Filter elements that have the attribute with the specific value
            elements = [
                el
                for el in elements
                if el.has_attr(attribute_filter)
                and el[attribute_filter] == filter_value
            ]
        else:
            # Filter elements that have the attribute regardless of value
            elements = [el for el in elements if el.has_attr(attribute_filter)]

    result = []

    # Replace attributes in matching elements
    for element in elements:
        result.append(element)
        if element.has_attr(old_attribute):
            # NOTE: It appears that this block is not needed
            # Only process attributes with the specific value if provided
            # if matcher_type == "regex":
            #     check_match = re.search(
            #         attribute_value, " ".join(element[old_attribute])
            #     )
            # else:
            #     check_match = " ".join(element[old_attribute])
            # if attribute_value is not None and check_match is None:
            #     continue # Never reached because check_match is always a string

            # Keep original value unless a replacement is specified
            if replace_value:
                # For debugging
                # msg.text(
                #     f"Detected attribute value '{attribute_value}' in '{element.name}'."
                # )
                # msg.text(f"Replaced '{old_attribute}' with '{new_attribute}'.")
                # msg.text(f"Replaced '{attribute_value}' with '{replace_value}'.")
                # If the old attribute is a string, split it into a list
                old_attribute_str = " ".join(element[old_attribute])
                if matcher_type == "regex":
                    new_values = []
                    for value in element[old_attribute]:
                        if re.search(attribute_value, value):
                            new_values.append(replace_value)
                        else:
                            new_values.append(value)
                    replace_value = new_values
                else:
                    if len(replace_value) == 1:
                        replace_value = replace_value[0]
                    # Use string replacement (current logic)
                    replace_value = old_attribute_str.replace(
                        attribute_value, replace_value
                    ).split(" ")

            value = (
                replace_value if replace_value is not None else element[old_attribute]
            )
            if isinstance(value, list):
                value = [str(v) for v in value if v]  # Remove empty strings
                value = " ".join(value)

            # Remove old attribute if the names are different
            if old_attribute != new_attribute:
                del element[old_attribute]

            # Set the new attribute with the appropriate value
            element[new_attribute] = value

    # Return the processed document
    return str(soup)


def replace_tag(
    text: str,
    selector: str,
    replacement: str,
    mode: str = "html",
    matcher_type: str = "exact",
    attribute: str = None,
    attribute_value: str = None,
    preserve_attributes: bool = True,
) -> str:
    """Replaces HTML/XML tags with another tag while preserving content.

    Args:
        text: HTML or XML text to process
        selector: Tag name or CSS selector to match elements for replacement
        replacement: New tag name to replace the matched elements with
        mode: Parser mode, either "html" or "xml"
        matcher_type: Type of match to perform, either "exact", "contains", or "regex"
        attribute: Optional attribute name to filter elements
        attribute_value: Optional value for the attribute filter
        preserve_attributes: Whether to preserve original tag attributes

    Returns:
        Processed text with matching tags replaced but content preserved

    Raises:
        LexosException: If mode is not "html" or "xml"

    Examples:
        >>> text = "<div><p>Keep this</p></div>"
        >>> replace_tag(text, "div", "section")
        '<section><p>Keep this</p></section>'

        >>> text = "<p class='a'>Keep</p><p class='b' id='x'>Replace tag</p>"
        >>> replace_tag(text, "p", "span", attribute="class", attribute_value="b")
        "<p class='a'>Keep</p><span class='b' id='x'>Replace tag</span>"
    """
    # Get matching elements
    soup, elements = _match_elements(
        selector, text, mode, matcher_type, attribute, attribute_value
    )

    # Replace matching elements with the new tag
    for element in elements:
        # Create a new tag with the same content
        new_element = soup.new_tag(replacement)

        # Copy all attributes if requested
        if preserve_attributes:
            for attr_name, attr_value in element.attrs.items():
                new_element[attr_name] = attr_value

        # Copy all child nodes
        for child in list(element.children):
            new_element.append(child)

        # Replace the old element with the new one
        element.replace_with(new_element)

    # Return the processed document
    return str(soup)
