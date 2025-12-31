"""test_tags.py.

Coverage: 99%. Missing: 447

Last Tested: June 8, 2025

Test suite for lexos.scrubber.tags module.

This suite tests the functionality of tag manipulation functions
provided in tags.py, ensuring they correctly process HTML/XML strings.

Although 'xml' with 'html.parser' might work for basic XML, 'lxml' is more robust for XML) and should probably be installed.

This suite provides a good starting point for testing your `tags.py` module. You can expand it with more specific edge cases or different combinations of parameters as needed.
"""

import pytest
from bs4 import BeautifulSoup

from lexos.exceptions import LexosException
from lexos.scrubber.tags import (
    _match_elements,
    _match_value,
    remove_attribute,
    remove_comments,
    remove_doctype,
    remove_element,
    remove_tag,
    replace_attribute,
    replace_tag,
)


# Helper function to normalize HTML/XML strings for comparison
def normalize_output(html_string: str, mode: str = "html") -> str:
    """Normalizes an HTML/XML string for consistent comparison.

    Args:
        html_string: The HTML/XML string to normalize.
        mode: The parsing mode, 'html' or 'xml'.

    Returns:
        A normalized string representation.
    """
    parser = "lxml-xml" if mode == "xml" else "html.parser"
    # Handle empty or whitespace-only strings gracefully
    if not html_string or html_string.isspace():
        return ""
    soup = BeautifulSoup(html_string, parser)
    # For XML, prettify can sometimes add an XML declaration if one isn't present
    # and the input was just a fragment. str(soup) is generally more direct.
    return str(soup)


# --- Test Data ---
HTML_WITH_COMMENTS = "<!-- comment1 --><p>Hello</p><!-- comment2 -->"
HTML_NO_COMMENTS = "<p>Hello</p>"
XML_WITH_COMMENTS = '<?xml version="1.0"?><!-- comment1 --><root><item>Hello</item><!-- comment2 --></root>'
XML_NO_COMMENTS = '<?xml version="1.0"?><root><item>Hello</item></root>'
EMPTY_STRING = ""

HTML_WITH_DOCTYPE = "<!DOCTYPE html><html><body><p>Test</p></body></html>"
HTML_WITHOUT_DOCTYPE = "<html><body><p>Test</p></body></html>"
HTML_WITH_COMPLEX_DOCTYPE = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd"><html><body>Test</body></html>'
XML_WITH_DOCTYPE_AND_DECL = '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE note SYSTEM "Note.dtd"><note><to>Tove</to></note>'
XML_WITHOUT_DOCTYPE_AND_DECL = (
    '<?xml version="1.0" encoding="UTF-8"?><note><to>Tove</to></note>'
)

HTML_FOR_UNWRAP = "<div><p>Content <span>inside</span> p</p></div><span>Sibling</span>"
EXPECTED_UNWRAP_HTML = "<p>Content <span>inside</span> p</p><span>Sibling</span>"
XML_FOR_UNWRAP = (
    "<wrapper><item><name>Test</name></item></wrapper><another>Data</another>"
)
EXPECTED_UNWRAP_XML = "<item><name>Test</name></item><another>Data</another>"
HTML_UNWRAP_ATTR = (
    "<div class='unwrap-me'><p>Content</p></div><div class='keep-me'>Keep</div>"
)
EXPECTED_UNWRAP_ATTR_HTML = "<p>Content</p><div class='keep-me'>Keep</div>"

HTML_FOR_ELEMENT_REMOVE = (
    "<div><p>Remove this paragraph.</p><span>Keep this span.</span></div>"
)
EXPECTED_ELEMENT_REMOVE_HTML = "<div><span>Keep this span.</span></div>"
XML_FOR_ELEMENT_REMOVE = "<root><item_to_remove><name>Old</name></item_to_remove><item_to_keep>New</item_to_keep></root>"
EXPECTED_ELEMENT_REMOVE_XML = "<root><item_to_keep>New</item_to_keep></root>"
HTML_ELEMENT_REMOVE_ATTR = "<p class='remove'>Remove</p><p class='keep'>Keep</p>"
EXPECTED_ELEMENT_REMOVE_ATTR_HTML = "<p class='keep'>Keep</p>"

HTML_FOR_ATTR_REPLACE = '<div class="old" id="main"><p class="another">Text</p></div>'

XML_FOR_ATTR_REPLACE = '<item status="initial" code="123"><name>Test</name></item>'

XML_FOR_ATTR_REPLACE = '<item status="initial" code="123"><name>Test</name></item>'

HTML_FOR_ATTR_REMOVE = (
    '<div class="main" id="content" data-info="test"><p>Text</p></div>'
)
XML_FOR_ATTR_REMOVE = (
    '<config enabled="true" version="1.2"><option>Debug</option></config>'
)

XML_FOR_ATTR_REMOVE = (
    '<config enabled="true" version="1.2"><option>Debug</option></config>'
)

HTML_FOR_TAG_REPLACE = '<div><p class="foo">Hello</p><span>World</span></div>'
XML_FOR_TAG_REPLACE = (
    '<outer><inner id="1">Content</inner><another>Data</another></outer>'
)
HTML_TAG_REPLACE_PRESERVE_ATTR = '<b class="bold-text" id="important">Important</b>'
HTML_TAG_REPLACE_NO_PRESERVE_ATTR = '<i class="italic-text">Emphasized</i>'


# --- Tests for remove_comments ---
class TestRemoveComments:
    """Tests for the remove_comments function."""

    def test_match_invalid_mode(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException, match="Mode must be either 'html' or 'xml'"):
            _match_elements("p", HTML_FOR_MATCH, mode="invalid")

    def test_remove_html_comments(self):
        """Ensures HTML comments are removed."""
        processed = remove_comments(HTML_WITH_COMMENTS, mode="html")
        assert normalize_output(processed, "html") == normalize_output(
            HTML_NO_COMMENTS, "html"
        )

    def test_remove_xml_comments(self):
        """Ensures XML comments are removed."""
        processed = remove_comments(XML_WITH_COMMENTS, mode="xml")
        assert normalize_output(processed, "xml") == normalize_output(
            XML_NO_COMMENTS, "xml"
        )

    def test_no_comments_html(self):
        """Ensures no change if no HTML comments exist."""
        processed = remove_comments(HTML_NO_COMMENTS, mode="html")
        assert normalize_output(processed, "html") == normalize_output(
            HTML_NO_COMMENTS, "html"
        )

    def test_no_comments_xml(self):
        """Ensures no change if no XML comments exist."""
        processed = remove_comments(XML_NO_COMMENTS, mode="xml")
        assert normalize_output(processed, "xml") == normalize_output(
            XML_NO_COMMENTS, "xml"
        )

    def test_empty_string_comments(self):
        """Ensures empty string is handled correctly."""
        processed = remove_comments(EMPTY_STRING, mode="html")
        assert processed == EMPTY_STRING

    def test_invalid_mode_comments(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException):
            remove_comments(HTML_WITH_COMMENTS, mode="invalid")


# --- Tests for remove_doctype ---
class TestRemoveDoctype:
    """Tests for the remove_doctype function."""

    def test_remove_html_doctype(self):
        """Ensures HTML DOCTYPE is removed."""
        processed = remove_doctype(HTML_WITH_DOCTYPE)
        assert normalize_output(processed, "html") == normalize_output(
            HTML_WITHOUT_DOCTYPE, "html"
        )

    def test_remove_complex_html_doctype(self):
        """Ensures complex HTML DOCTYPE is removed."""
        processed = remove_doctype(HTML_WITH_COMPLEX_DOCTYPE)
        expected = "<html><body>Test</body></html>"
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_remove_xml_doctype(self):
        """Ensures XML DOCTYPE is removed, preserves XML declaration."""
        processed = remove_doctype(XML_WITH_DOCTYPE_AND_DECL)
        assert normalize_output(processed, "xml") == normalize_output(
            XML_WITHOUT_DOCTYPE_AND_DECL, "xml"
        )

    def test_no_doctype_html(self):
        """Ensures no change if no HTML DOCTYPE exists."""
        processed = remove_doctype(HTML_WITHOUT_DOCTYPE)
        assert normalize_output(processed, "html") == normalize_output(
            HTML_WITHOUT_DOCTYPE, "html"
        )

    def test_no_doctype_xml(self):
        """Ensures no change if no XML DOCTYPE exists."""
        processed = remove_doctype(XML_WITHOUT_DOCTYPE_AND_DECL)
        assert normalize_output(processed, "xml") == normalize_output(
            XML_WITHOUT_DOCTYPE_AND_DECL, "xml"
        )

    def test_empty_string_doctype(self):
        """Ensures empty string is handled correctly."""
        processed = remove_doctype(EMPTY_STRING)
        assert processed == EMPTY_STRING


# --- Tests for remove_tag (unwrap) ---
class TestRemoveTag:
    """Tests for the remove_tag (unwrap) function."""

    def test_match_invalid_mode(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException, match="Mode must be either 'html' or 'xml'"):
            _match_elements("p", HTML_FOR_MATCH, mode="invalid")

    def test_unwrap_html_tag(self):
        """Ensures HTML tag is unwrapped, content preserved."""
        processed = remove_tag(HTML_FOR_UNWRAP, "div", mode="html")
        assert normalize_output(processed, "html") == normalize_output(
            EXPECTED_UNWRAP_HTML, "html"
        )

    def test_unwrap_xml_tag(self):
        """Ensures XML tag is unwrapped, content preserved."""
        processed = remove_tag(XML_FOR_UNWRAP, "wrapper", mode="xml")
        assert normalize_output(processed, "xml") == normalize_output(
            EXPECTED_UNWRAP_XML, "xml"
        )

    def test_unwrap_html_tag_with_attribute_filter(self):
        """Ensures HTML tag is unwrapped based on attribute filter."""
        processed = remove_tag(
            HTML_UNWRAP_ATTR,
            "div",
            mode="html",
            attribute="class",
            attribute_value="unwrap-me",
        )
        assert normalize_output(processed, "html") == normalize_output(
            EXPECTED_UNWRAP_ATTR_HTML, "html"
        )

    def test_unwrap_non_matching_attribute_filter(self):
        """Ensures no change if attribute filter doesn't match."""
        processed = remove_tag(
            HTML_UNWRAP_ATTR,
            "div",
            mode="html",
            attribute="class",
            attribute_value="nonexistent",
        )
        assert normalize_output(processed, "html") == normalize_output(
            HTML_UNWRAP_ATTR, "html"
        )

    def test_unwrap_nonexistent_tag(self):
        """Ensures no change if tag to unwrap doesn't exist."""
        processed = remove_tag(HTML_FOR_UNWRAP, "nonexistent", mode="html")
        assert normalize_output(processed, "html") == normalize_output(
            HTML_FOR_UNWRAP, "html"
        )

    def test_empty_string_unwrap(self):
        """Ensures empty string is handled correctly."""
        processed = remove_tag(EMPTY_STRING, "div", mode="html")
        assert processed == EMPTY_STRING

    def test_invalid_mode_unwrap(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException):
            remove_tag(HTML_FOR_UNWRAP, "div", mode="invalid")


# --- Tests for remove_element ---
class TestRemoveElement:
    """Tests for the remove_element function."""

    def test_match_invalid_mode(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException, match="Mode must be either 'html' or 'xml'"):
            _match_elements("p", HTML_FOR_MATCH, mode="invalid")

    def test_remove_html_element(self):
        """Ensures HTML element and its content are removed."""
        processed = remove_element(HTML_FOR_ELEMENT_REMOVE, "p", mode="html")
        assert normalize_output(processed, "html") == normalize_output(
            EXPECTED_ELEMENT_REMOVE_HTML, "html"
        )

    def test_remove_xml_element(self):
        """Ensures XML element and its content are removed."""
        processed = remove_element(XML_FOR_ELEMENT_REMOVE, "item_to_remove", mode="xml")
        assert normalize_output(processed, "xml") == normalize_output(
            EXPECTED_ELEMENT_REMOVE_XML, "xml"
        )

    def test_remove_html_element_with_attribute_filter(self):
        """Ensures HTML element is removed based on attribute filter."""
        processed = remove_element(
            HTML_ELEMENT_REMOVE_ATTR,
            "p",
            mode="html",
            attribute="class",
            attribute_value="remove",
        )
        assert normalize_output(processed, "html") == normalize_output(
            EXPECTED_ELEMENT_REMOVE_ATTR_HTML, "html"
        )

    def test_remove_non_matching_attribute_filter_element(self):
        """Ensures no change if attribute filter doesn't match for element removal."""
        processed = remove_element(
            HTML_ELEMENT_REMOVE_ATTR,
            "p",
            mode="html",
            attribute="class",
            attribute_value="nonexistent",
        )
        assert normalize_output(processed, "html") == normalize_output(
            HTML_ELEMENT_REMOVE_ATTR, "html"
        )

    def test_remove_nonexistent_element(self):
        """Ensures no change if element to remove doesn't exist."""
        processed = remove_element(HTML_FOR_ELEMENT_REMOVE, "nonexistent", mode="html")
        assert normalize_output(processed, "html") == normalize_output(
            HTML_FOR_ELEMENT_REMOVE, "html"
        )

    def test_empty_string_remove_element(self):
        """Ensures empty string is handled correctly."""
        processed = remove_element(EMPTY_STRING, "div", mode="html")
        assert processed == EMPTY_STRING

    def test_invalid_mode_remove_element(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException):
            remove_element(HTML_FOR_ELEMENT_REMOVE, "p", mode="invalid")


# --- Tests for replace_attribute ---
class TestReplaceAttribute:
    """Tests for the replace_attribute function."""

    def test_match_invalid_mode(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException, match="Mode must be either 'html' or 'xml'"):
            _match_elements("p", HTML_FOR_MATCH, mode="invalid")

    def test_replace_attribute_name_html(self):
        """Ensures attribute name is replaced, value preserved in HTML."""
        processed = replace_attribute(
            HTML_FOR_ATTR_REPLACE,
            "div",
            "class",
            "data-type",
            mode="html",
            attribute_value="old",
        )
        expected = '<div data-type="old" id="main"><p class="another">Text</p></div>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_attribute_name_xml(self):
        """Ensures attribute name is replaced, value preserved in XML."""
        processed = replace_attribute(
            XML_FOR_ATTR_REPLACE, "item", "status", "state", mode="xml"
        )
        expected = '<item code="123" state="initial"><name>Test</name></item>'
        assert normalize_output(processed, "xml") == normalize_output(expected, "xml")

    def test_replace_attribute_value_html(self):
        """Ensures attribute value is replaced in HTML."""
        processed = replace_attribute(
            HTML_FOR_ATTR_REPLACE,
            "div",
            old_attribute="class",
            new_attribute="new_class",
            mode="html",
        )
        expected = '<div new_class="old" id="main"><p class="another">Text</p></div>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_attribute_name_and_value_html(self):
        """Ensures attribute name and value are replaced in HTML."""
        processed = replace_attribute(
            HTML_FOR_ATTR_REPLACE,
            "div",
            "class",
            "data-role",
            mode="html",
            attribute_value="old",
            replace_value="content_holder",
        )
        expected = '<div data-role="content_holder" id="main"><p class="another">Text</p></div>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_attribute_with_value_filter_html(self):
        """Ensures attribute is replaced only if value matches."""
        html = '<p class="one">First</p><p class="two">Second</p>'
        processed = replace_attribute(
            html, "p", "class", "style", mode="html", attribute_value="one"
        )
        expected = '<p style="one">First</p><p class="two">Second</p>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_attribute_with_element_filter_html(self):
        """Ensures attribute is replaced on elements matching attribute_filter."""
        html = '<div id="target" class="a">Target</div><div id="other" class="a">Other</div>'
        processed = replace_attribute(
            html,
            "div",
            "class",
            "role",
            mode="html",
            attribute_filter="id",
            filter_value="target",
        )
        expected = '<div id="target" role="a">Target</div><div id="other" class="a">Other</div>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_nonexistent_attribute(self):
        """Ensures no change if attribute to replace doesn't exist."""
        processed = replace_attribute(
            HTML_FOR_ATTR_REPLACE, "div", "nonexistent", "newattr", mode="html"
        )
        assert normalize_output(processed, "html") == normalize_output(
            HTML_FOR_ATTR_REPLACE, "html"
        )

    def test_empty_string_replace_attribute(self):
        """Ensures empty string is handled correctly."""
        processed = replace_attribute(EMPTY_STRING, "div", "class", "id", mode="html")
        assert processed == EMPTY_STRING

    def test_invalid_mode_replace_attribute(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException):
            replace_attribute(
                HTML_FOR_ATTR_REPLACE, "div", "class", "id", mode="invalid"
            )

    def test_replace_attribute_contains_single_value(self):
        """Ensures that contains works with a single attribute value."""
        html = '<div class="a">Target</div><div class="a b">Other</div>'
        processed = replace_attribute(
            html,
            "div",
            "class",
            "class",
            mode="html",
            attribute_value="a",
            replace_value="replaced",
            matcher_type="contains",
        )
        expected = (
            '<div class="replaced">Target</div><div class="replaced b">Other</div>'
        )
        assert normalize_output(processed, "html") == normalize_output(expected, "html")


# --- Tests for remove_attribute ---
class TestRemoveAttribute:
    """Tests for the remove_attribute function."""

    def test_match_invalid_mode(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException, match="Mode must be either 'html' or 'xml'"):
            _match_elements("p", HTML_FOR_MATCH, mode="invalid")

    def test_remove_single_attribute_html(self):
        """Ensures a single specified attribute is removed from an HTML element."""
        processed = remove_attribute(
            HTML_FOR_ATTR_REMOVE, "div", attribute="class", mode="html"
        )
        expected = '<div id="content" data-info="test"><p>Text</p></div>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    # def test_remove_multiple_attributes_html(self):
    #     """Ensures multiple specified attributes are removed from an HTML element."""
    #     processed = remove_attribute(
    #         HTML_FOR_ATTR_REMOVE, "div", attributes=["id", "data-info"], mode="html"
    #     )
    #     expected = '<div class="main"><p>Text</p></div>'
    #     assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_remove_all_attributes_html(self):
        """Ensures all attributes are removed from an HTML element if attributes list is None."""
        processed = remove_attribute(
            HTML_FOR_ATTR_REMOVE, "div", attribute=None, mode="html"
        )
        expected = "<div><p>Text</p></div>"
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_remove_attribute_xml(self):
        """Ensures specified attributes are removed from an XML element."""
        processed = remove_attribute(
            XML_FOR_ATTR_REMOVE, "config", attribute="version", mode="xml"
        )
        expected = '<config enabled="true"><option>Debug</option></config>'
        assert normalize_output(processed, "xml") == normalize_output(expected, "xml")

    def test_remove_all_attributes_xml(self):
        """Ensures all attributes are removed from an XML element."""
        processed = remove_attribute(
            XML_FOR_ATTR_REMOVE, "config", attribute=None, mode="xml"
        )
        expected = "<config><option>Debug</option></config>"
        assert normalize_output(processed, "xml") == normalize_output(expected, "xml")

    def test_remove_attribute_with_element_filter_html(self):
        """Ensures attributes are removed only from elements matching the attribute_filter."""
        html = '<p class="important" id="one">First</p><p class="important" id="two">Second</p>'
        processed = remove_attribute(
            html,
            "p",
            attribute="class",
            mode="html",
            attribute_filter="id",
        )
        expected = '<p id="one">First</p><p id="two">Second</p>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_remove_nonexistent_attribute(self):
        """Ensures no change if specified attribute to remove doesn't exist."""
        processed = remove_attribute(
            HTML_FOR_ATTR_REMOVE, "div", attribute="style", mode="html"
        )
        assert normalize_output(processed, "html") == normalize_output(
            HTML_FOR_ATTR_REMOVE, "html"
        )

    def test_remove_attribute_from_nonexistent_tag(self):
        """Ensures no change if tag from which to remove attributes doesn't exist."""
        processed = remove_attribute(
            HTML_FOR_ATTR_REMOVE, "span", attribute="class", mode="html"
        )
        assert normalize_output(processed, "html") == normalize_output(
            HTML_FOR_ATTR_REMOVE, "html"
        )

    def test_empty_string_remove_attribute(self):
        """Ensures empty string is handled correctly."""
        processed = remove_attribute(
            EMPTY_STRING, "div", attribute="class", mode="html"
        )
        assert processed == EMPTY_STRING

    def test_invalid_mode_remove_attribute(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException):
            remove_attribute(
                HTML_FOR_ATTR_REMOVE, "div", attribute="class", mode="invalid"
            )


# --- Tests for replace_tag ---
class TestReplaceTag:
    """Tests for the replace_tag function."""

    def test_match_invalid_mode(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException, match="Mode must be either 'html' or 'xml'"):
            _match_elements("p", HTML_FOR_MATCH, mode="invalid")

    def test_replace_html_tag_simple(self):
        """Ensures a simple HTML tag is replaced."""
        processed = replace_tag(HTML_FOR_TAG_REPLACE, "div", "section", mode="html")
        expected = '<section><p class="foo">Hello</p><span>World</span></section>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_xml_tag_simple(self):
        """Ensures a simple XML tag is replaced."""
        processed = replace_tag(XML_FOR_TAG_REPLACE, "inner", "item", mode="xml")
        expected = '<outer><item id="1">Content</item><another>Data</another></outer>'
        assert normalize_output(processed, "xml") == normalize_output(expected, "xml")

    def test_replace_html_tag_preserve_attributes(self):
        """Ensures HTML tag is replaced and attributes are preserved by default."""
        processed = replace_tag(
            HTML_TAG_REPLACE_PRESERVE_ATTR, "b", "strong", mode="html"
        )
        expected = '<strong class="bold-text" id="important">Important</strong>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_html_tag_no_preserve_attributes(self):
        """Ensures HTML tag is replaced and attributes are not preserved when specified."""
        processed = replace_tag(
            HTML_TAG_REPLACE_NO_PRESERVE_ATTR,
            "i",
            "em",
            mode="html",
            preserve_attributes=False,
        )
        expected = "<em>Emphasized</em>"
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_html_tag_with_attribute_filter(self):
        """Ensures HTML tag is replaced only if it matches the attribute filter."""
        html = '<p class="replace_me">Replace</p><p class="keep_me">Keep</p>'
        processed = replace_tag(
            html,
            "p",
            "span",
            mode="html",
            attribute="class",
            attribute_value="replace_me",
        )
        expected = '<span class="replace_me">Replace</span><p class="keep_me">Keep</p>'
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_nonexistent_tag(self):
        """Ensures no change if tag to replace doesn't exist."""
        processed = replace_tag(
            HTML_FOR_TAG_REPLACE, "nonexistent", "article", mode="html"
        )
        assert normalize_output(processed, "html") == normalize_output(
            HTML_FOR_TAG_REPLACE, "html"
        )

    def test_replace_tag_empty_string(self):
        """Ensures empty string is handled correctly."""
        processed = replace_tag(EMPTY_STRING, "div", "section", mode="html")
        assert processed == EMPTY_STRING

    def test_replace_tag_invalid_mode(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException):
            replace_tag(HTML_FOR_TAG_REPLACE, "div", "section", mode="invalid")

    def test_replace_tag_multiple_instances(self):
        """Ensures all instances of a tag are replaced."""
        html = "<div>First</div><div>Second</div>"
        processed = replace_tag(html, "div", "article", mode="html")
        expected = "<article>First</article><article>Second</article>"
        assert normalize_output(processed, "html") == normalize_output(expected, "html")

    def test_replace_tag_nested_structure_attribute_preservation(self):
        """Tests replacing a tag in a nested structure, preserving attributes."""
        html_nested = (
            '<div id="outer"><span class="inner-span"><em>Content</em></span></div>'
        )
        processed = replace_tag(
            html_nested, "span", "strong", mode="html", preserve_attributes=True
        )
        expected = (
            '<div id="outer"><strong class="inner-span"><em>Content</em></strong></div>'
        )
        assert normalize_output(processed, "html") == normalize_output(expected, "html")


### PRIVATE FUNCTION TESTS ###


# --- Tests for _match_value ---
class TestMatchValue:
    """Tests for the _match_value helper function."""

    @pytest.mark.parametrize(
        "text_list, pattern, match_type, expected",
        [
            # Exact matches
            (["hello", "world"], "hello world", "exact", True),
            (["hello", "world"], "hello", "exact", False),
            (["hello", "world"], "world", "exact", False),
            (["singleword"], "singleword", "exact", True),
            (["hello", "world"], "goodbye", "exact", False),
            (["test"], "testing", "exact", False),
            ([], "anything", "exact", False),
            (["content"], "", "exact", False),
            ([], "", "exact", True),
            # Contains matches
            (["hello", "world"], "hello", "contains", True),
            (["hello", "world"], "world", "contains", True),
            (["hello", "world"], "o w", "contains", False),
            (["hello", "world"], "o x", "contains", False),
            (["singleword"], "single", "contains", False),
            (["hello", "world"], "goodbye", "contains", False),
            (["test"], "testing", "contains", False),
            ([], "anything", "contains", False),
            (["content"], "", "contains", False),
            ([], "", "contains", False),
            # Regex matches
            (["hello", "world", "123"], r"world", "regex", True),
            (["hello", "world", "123"], r"\d+", "regex", True),
            (["item-001"], r"item-\d{3}", "regex", True),
            (["item-001"], r"item-\d{2}a", "regex", False),
            (["apple", "banana"], r"^apple", "regex", True),
            (["apple", "banana"], r"na$", "regex", True),  # Matches "banana"
            (["apple", "banana"], r"orange", "regex", False),
            ([], r"anything", "regex", False),
            (["content"], r"", "regex", True),  # Empty pattern matches
            ([], r"", "regex", True),  # Empty pattern matches
        ],
    )
    def test_matches(
        self, text_list: list[str], pattern: str, match_type: str, expected: bool
    ):
        """Tests various scenarios for exact and regex matching.

        Args:
            text_list: The list of strings to join and match against.
            pattern: The pattern to match.
            match_type: The type of match ('exact' or 'regex').
            expected: The expected boolean result.
        """
        assert _match_value(text_list, pattern, match_type) == expected

    def test_invalid_match_type(self):
        """Tests that an invalid match type raises LexosException."""
        with pytest.raises(LexosException) as excinfo:
            _match_value(["test"], "pattern", "invalid_type")
        assert "match_type must be 'exact', 'contains', or 'regex'" in str(
            excinfo.value
        )

    def test_match_value_invalid_regex(self):
        """Test _match_value with invalid regex pattern (lines 126-128)."""
        # Test with invalid regex patterns that will raise re.error
        invalid_patterns = [
            "[",  # Unclosed character class
            "(",  # Unclosed group
            "*",  # Nothing to repeat
            "\\",  # Trailing backslash
        ]

        test_value = "bold italic"

        for pattern in invalid_patterns:
            # Should return False instead of raising an exception
            result = _match_value(test_value, pattern, "regex")
            assert result is False, (
                f"Invalid regex pattern '{pattern}' should return False"
            )

    def test_exact_match_substring_behavior(self):
        """Explicitly tests the 'pattern in joined_text' behavior for exact matches."""
        assert _match_value(["longstring"], "str", "exact") is False
        assert _match_value(["longstring"], "longstring", "exact") is True
        assert _match_value(["not", "in", "list"], "list", "exact") is False
        assert _match_value(["not", "in", "list"], "notin", "exact") is False

    def test_contains_match_substring_behavior(self):
        """Explicitly tests the 'pattern in joined_text' behavior for exact matches."""
        assert _match_value(["longstring"], "str", "contains") is False
        assert _match_value(["longstring"], "longstring", "contains") is True
        assert _match_value(["in", "list"], "list", "contains") is True
        assert _match_value(["not", "in", "list"], "notin", "contains") is False

    def test_regex_match_special_characters(self):
        """Tests regex matching with special characters."""
        assert _match_value(["version 1.0"], r"version \d\.\d", "regex") is True
        assert _match_value(["(brackets)"], r"\(brackets\)", "regex") is True
        assert _match_value(["test?"], r"test\?", "regex") is True

    def test_default_match_type_is_exact(self):
        """Tests that the default match type is 'exact'."""
        assert (
            _match_value(["hello", "world"], "hello world") is True
        )  # type defaults to "exact"
        assert (
            _match_value(["hello", "world"], "goodbye") is False
        )  # type defaults to "exact"
        # This regex pattern would match if type="regex", but should fail with default "exact"
        assert _match_value(["item-001"], r"item-\d{3}") is False


# --- Test Data for _match_elements ---
HTML_FOR_MATCH = """
<html>
  <head><title>Test Page</title></head>
  <body>
    <div id="main" class="container">
      <p class="para first">First paragraph.</p>
      <p class="para second" data-custom="value1">Second paragraph.</p>
      <span>A span</span>
      <div class="container">
        <p class="para third">Third paragraph.</p>
      </div>
    </div>
    <div id="footer" class="container">
      <p>Footer p</p>
    </div>
  </body>
</html>
"""

XML_FOR_MATCH = """
<root>
  <item id="1" type="A">
    <name>Item A1</name>
    <value>100</value>
  </item>
  <item id="2" type="B" status="active">
    <name>Item B2</name>
    <value>200</value>
  </item>
  <item id="3" type="A" status="inactive">
    <name>Item A3</name>
    <value>300</value>
  </item>
</root>
"""


# --- Tests for _match_elements ---
class TestMatchElements:
    """Tests for the _match_elements helper function."""

    def test_match_html_tag_basic(self):
        """Tests basic tag selection in HTML."""
        soup, elements = _match_elements("p", HTML_FOR_MATCH, mode="html")
        assert len(elements) == 4
        assert all(el.name == "p" for el in elements)

    def test_match_xml_tag_basic(self):
        """Tests basic tag selection in XML."""
        soup, elements = _match_elements("item", XML_FOR_MATCH, mode="xml")
        assert len(elements) == 3
        assert all(el.name == "item" for el in elements)

    def test_match_html_css_class_selector(self):
        """Tests CSS class selector in HTML."""
        soup, elements = _match_elements(".para", HTML_FOR_MATCH, mode="html")
        assert len(elements) == 3
        assert all("para" in el.get("class", []) for el in elements)

    def test_match_html_css_id_selector(self):
        """Tests CSS ID selector in HTML."""
        soup, elements = _match_elements("#main", HTML_FOR_MATCH, mode="html")
        assert len(elements) == 1
        assert elements[0].name == "div"
        assert elements[0]["id"] == "main"

    def test_match_html_attribute_presence(self):
        """Tests filtering by attribute presence in HTML."""
        soup, elements = _match_elements(
            "p", HTML_FOR_MATCH, mode="html", attribute="data-custom"
        )
        assert len(elements) == 1
        assert elements[0].name == "p"
        assert elements[0].has_attr("data-custom")
        assert "second" in elements[0].get("class", [])

    def test_match_xml_attribute_presence(self):
        """Tests filtering by attribute presence in XML."""
        soup, elements = _match_elements(
            "item", XML_FOR_MATCH, mode="xml", attribute="status"
        )
        assert len(elements) == 2
        assert all(el.has_attr("status") for el in elements)

    def test_match_html_attribute_exact_value(self):
        """Tests filtering by exact attribute value in HTML."""
        soup, elements = _match_elements(
            "p",
            HTML_FOR_MATCH,
            mode="html",
            attribute="class",
            attribute_value="para first",
            matcher_type="exact",
        )
        assert len(elements) == 1
        assert "first" in elements[0].get("class", [])

    def test_match_xml_attribute_exact_value(self):
        """Tests filtering by exact attribute value in XML."""
        soup, elements = _match_elements(
            "item",
            XML_FOR_MATCH,
            mode="xml",
            attribute="type",
            attribute_value="A",
            matcher_type="exact",
        )
        assert len(elements) == 2
        assert all(el["type"] == "A" for el in elements)

    def test_match_html_attribute_regex_value(self):
        """Tests filtering by regex attribute value in HTML."""
        # Assuming _match_value and _match_elements correctly use matcher_type="regex"
        # for attribute values.
        soup, elements = _match_elements(
            "p",
            HTML_FOR_MATCH,
            mode="html",
            attribute="class",
            attribute_value=r"para\s(first|second)",
            matcher_type="regex",
        )
        assert len(elements) == 2
        classes = {" ".join(el.get("class", [])) for el in elements}
        assert "para first" in classes
        assert "para second" in classes

    def test_match_xml_attribute_regex_value(self):
        """Tests filtering by regex attribute value in XML."""
        soup, elements = _match_elements(
            "item",
            XML_FOR_MATCH,
            mode="xml",
            attribute="status",
            attribute_value=r"active|inactive",
            matcher_type="regex",
        )
        assert len(elements) == 2
        statuses = {el["status"] for el in elements}
        assert "active" in statuses
        assert "inactive" in statuses

    def test_match_html_attribute_filter_presence(self):
        """Tests attribute_filter for attribute presence in HTML."""
        soup, elements = _match_elements(
            "div", HTML_FOR_MATCH, mode="html", attribute_filter="id"
        )
        assert len(elements) == 2  # main and footer
        assert all(el.has_attr("id") for el in elements)

    def test_match_html_attribute_filter_with_value(self):
        """Tests attribute_filter with a specific value in HTML."""
        soup, elements = _match_elements(
            "div",
            HTML_FOR_MATCH,
            mode="html",
            attribute_filter="class",
            attribute_value="container",
            matcher_type="exact",
        )
        soup, elements = _match_elements(
            "div",
            HTML_FOR_MATCH,
            mode="html",
            attribute="class",  # This is used by the _match_value call
            attribute_filter="class",  # This is used for el.has_attr(attribute_filter)
            attribute_value="container",
            matcher_type="exact",
        )
        assert len(elements) == 3  # main, inner div, footer
        assert all("container" in el.get("class", []) for el in elements)

    def test_match_no_elements_found_tag(self):
        """Tests scenario where no elements match the tag."""
        soup, elements = _match_elements("nonexistent", HTML_FOR_MATCH, mode="html")
        assert len(elements) == 0

    def test_match_no_elements_found_attribute(self):
        """Tests scenario where no elements match the attribute filter."""
        soup, elements = _match_elements(
            "p",
            HTML_FOR_MATCH,
            mode="html",
            attribute="data-nonexistent",
            attribute_value="any",
        )
        assert len(elements) == 0

    def test_match_invalid_mode(self):
        """Tests LexosException for invalid mode."""
        with pytest.raises(LexosException, match="Mode must be either 'html' or 'xml'"):
            _match_elements("p", HTML_FOR_MATCH, mode="invalid")

    def test_match_empty_string_input(self):
        """Tests behavior with empty string input."""
        soup, elements = _match_elements("p", EMPTY_STRING, mode="html")
        assert len(elements) == 0
        assert str(soup) == ""  # Or based on how BS4 handles empty string

    def test_match_selector_priority_over_attribute_filter(self):
        """Tests that the main selector is applied first."""
        # Select only 'span' tags, then filter by class (which no span has)
        soup, elements = _match_elements(
            "span",
            HTML_FOR_MATCH,
            mode="html",
            attribute="class",
            attribute_value="container",
        )
        assert len(elements) == 0  # No spans have class "container"

    def test_attribute_vs_attribute_filter_precedence(self):
        """Tests the precedence: if attribute_filter is present, its block is used.

        The `elif attribute:` block should not be reached if `attribute_filter` is set.
        """
        # `attribute_filter` is "id", `attribute` is "class".
        # The filtering should happen based on "id".
        soup, elements = _match_elements(
            "div",
            HTML_FOR_MATCH,
            mode="html",
            attribute_filter="id",  # This condition will be true
            attribute="class",  # This should be ignored for has_attr check if attribute_filter is used
            attribute_value="main",  # This value will be checked against el[attribute] (i.e. el["class"])
        )

        assert len(elements) == 1

    def test_invalid_matcher_type_raises_exception(self):
        """Tests that an exception is raised if matcher_type is None."""
        # `matcher_type` is None, so `_match_value` should use "exact"
        with pytest.raises(
            LexosException, match="match_type must be 'exact', 'contains', or 'regex'"
        ):
            soup, elements = _match_elements(
                "p",
                HTML_FOR_MATCH,
                mode="html",
                attribute="class",
                attribute_value="para first",
                matcher_type=None,
            )


# Additional tests for coverage


def test_remove_attribute_with_attribute_filter_no_value():
    """Test remove_attribute with attribute_filter but no attribute_value (line 162)."""
    text = '<p class="a" id="x">Keep</p><p class="b">Remove attrs</p><div class="c" id="y">Keep</div>'
    result = remove_attribute(text, "p", attribute="class", attribute_filter="id")
    expected = '<p id="x">Keep</p><p class="b">Remove attrs</p><div class="c" id="y">Keep</div>'
    assert result == expected


def test_replace_attribute_regex_no_match():
    """Test replace_attribute with regex that doesn't match (line 412)."""
    text = '<p class="hello world">Test</p>'
    result = replace_attribute(
        text,
        selector="p",
        old_attribute="class",
        new_attribute="class",
        attribute_value="xyz.*",  # This won't match "hello" or "world"
        replace_value="matched",
        matcher_type="regex",
    )
    # Should remain unchanged because no regex match
    expected = '<p class="hello world">Test</p>'
    assert result == expected


def test_replace_attribute_regex_match():
    """Test replace_attribute with regex that matches (line 419)."""
    text = '<p class="bolder italic">Test</p>'
    result = replace_attribute(
        text,
        selector="p",
        old_attribute="class",
        new_attribute="class",
        attribute_value="bold.*",  # This will match "bolder"
        replace_value="happy",
        matcher_type="regex",
    )
    expected = '<p class="happy italic">Test</p>'
    assert result == expected


def test_replace_attribute_exact_no_replace_value():
    """Test replace_attribute without replace_value (line 425)."""
    text = '<p class="bold italic" id="test">Test</p>'
    # Change class to data-class but keep the same value
    result = replace_attribute(
        text,
        selector="p",
        old_attribute="class",
        new_attribute="data-class",
        # No replace_value specified - should keep original value
    )
    expected = '<p data-class="bold italic" id="test">Test</p>'
    assert result == expected


def test_replace_attribute_exact_string_replacement():
    """Test replace_attribute with exact matching and string replacement (lines 438-444)."""
    text = '<p class="bold">Test</p>'
    result = replace_attribute(
        text,
        selector="p",
        old_attribute="class",
        new_attribute="class",
        attribute_value="bold",  # Exact match
        replace_value="happy",
        matcher_type="exact",
    )
    expected = '<p class="happy">Test</p>'
    assert result == expected


def test_replace_attribute_contains_string_replacement():
    """Test replace_attribute with contains matching and string replacement (lines 438-444)."""
    text = '<p class="bold italic">Test</p>'
    result = replace_attribute(
        text,
        selector="p",
        old_attribute="class",
        new_attribute="class",
        attribute_value="bold",  # Contains match
        replace_value="happy",
        matcher_type="contains",
    )
    expected = '<p class="happy italic">Test</p>'
    assert result == expected


def test_replace_attribute_exact_no_match():
    """Test replace_attribute with exact matching that doesn't match."""
    text = '<p class="bolder italic">Test</p>'
    result = replace_attribute(
        text,
        selector="p",
        old_attribute="class",
        new_attribute="class",
        attribute_value="bold",  # Won't match "bolder" exactly
        replace_value="happy",
        matcher_type="exact",
    )
    # Should remain unchanged because "bold" != "bolder"
    expected = '<p class="bolder italic">Test</p>'
    assert result == expected


def test_remove_attribute_with_filter_and_value_match():
    """Test remove_attribute with attribute_filter and attribute_value using _match_value (lines 182-187)."""
    # Test HTML with elements that have different attribute values
    text = """
    <div class="container main" data-role="primary">Keep this div</div>
    <p class="text bold" data-role="secondary">Remove class from this p</p>
    <p class="text italic" data-role="primary">Keep this p unchanged</p>
    <span class="highlight" data-role="tertiary">Keep this span</span>
    """

    # Remove the 'class' attribute only from elements that have data-role="primary"
    # This should trigger the highlighted code path with _match_value
    result = remove_attribute(
        text,
        selector=None,  # Select all elements, regardless of tag name
        attribute="class",
        attribute_filter="data-role",
        attribute_value="primary",  # This triggers the _match_value call
        matcher_type="regex",
    )

    # Parse the result to check what happened
    soup = BeautifulSoup(result, "html.parser")

    # Elements with data-role="primary" should have their class attribute removed
    div = soup.find("div")
    assert not div.has_attr("class")  # class was removed
    assert div.get("data-role") == "primary"  # data-role preserved

    p_primary = soup.find("p", {"data-role": "primary"})
    assert not p_primary.has_attr("class")  # class was removed
    assert p_primary.get("data-role") == "primary"  # data-role preserved

    # Elements without data-role="primary" should keep their class
    p_secondary = soup.find("p", {"data-role": "secondary"})
    assert p_secondary.get("class") == ["text", "bold"]  # class preserved

    span = soup.find("span")
    assert span.get("class") == ["highlight"]  # class preserved


def test_remove_attribute_with_filter_regex_match():
    """Test remove_attribute with attribute_filter using regex matching."""
    text = """
    <div id="header-main" class="container">Header</div>
    <div id="header-sub" class="subcontainer">Subheader</div>
    <div id="content-main" class="content">Content</div>
    <div id="footer" class="footer">Footer</div>
    """

    # Remove class attribute from elements whose id starts with "header"
    result = remove_attribute(
        text,
        selector="div",
        attribute="class",
        attribute_filter="id",
        attribute_value="header.*",  # Regex pattern
        matcher_type="regex",
    )

    soup = BeautifulSoup(result, "html.parser")

    # Elements with id starting with "header" should have class removed
    header_main = soup.find("div", {"id": "header-main"})
    assert not header_main.has_attr("class")

    header_sub = soup.find("div", {"id": "header-sub"})
    assert not header_sub.has_attr("class")

    # Elements not matching the regex should keep their class
    content = soup.find("div", {"id": "content-main"})
    assert content.get("class") == ["content"]

    footer = soup.find("div", {"id": "footer"})
    assert footer.get("class") == ["footer"]


def test_remove_attribute_with_filter_contains_match():
    """Test remove_attribute with attribute_filter using contains matching."""
    text = """
    <p class="text bold important">Paragraph 1</p>
    <p class="text italic">Paragraph 2</p>
    <p class="header bold">Paragraph 3</p>
    <div class="text normal">Div</div>
    """

    # Remove class attribute from elements that have "bold" in their class
    result = remove_attribute(
        text,
        selector="p",
        attribute="class",
        attribute_filter="class",
        attribute_value="bold",  # Should match elements with "bold" as one of the class values
        matcher_type="contains",
    )

    soup = BeautifulSoup(result, "html.parser")

    # Elements with "bold" in their class should have class removed
    paragraphs = soup.find_all("p")

    # First and third paragraphs had "bold", so no class attribute
    assert not paragraphs[0].has_attr("class")  # had "text bold important"
    assert paragraphs[1].get("class") == ["text", "italic"]  # didn't have "bold"
    assert not paragraphs[2].has_attr("class")  # had "header bold"

    # Div didn't have "bold" so should keep class
    div = soup.find("div")
    assert div.get("class") == ["text", "normal"]


def test_replace_attribute_with_filter_no_value():
    """Test replace_attribute with attribute_filter but no filter_value (line 434)."""
    text = """
    <div class="container" id="main">Has both class and id</div>
    <p class="text">Has class only</p>
    <span id="sidebar">Has id only</span>
    <article>Has neither</article>
    """

    # Replace 'class' with 'data-class' only on elements that have an 'id' attribute
    # This should trigger line 434: elements = [el for el in elements if el.has_attr(attribute_filter)]
    result = replace_attribute(
        text,
        selector=None,  # Select all elements
        old_attribute="class",
        new_attribute="data-class",
        attribute_filter="id",  # Only elements that have an 'id' attribute
        # No filter_value specified - this triggers line 434
    )

    soup = BeautifulSoup(result, "html.parser")

    # Elements with both 'id' and 'class' should have class → data-class
    div = soup.find("div")
    assert not div.has_attr("class")  # class was removed
    assert div.has_attr("data-class")  # data-class was added
    assert div.get("data-class") == "container"  # value was preserved
    assert div.get("id") == "main"  # id unchanged

    # Elements with 'class' but no 'id' should be unchanged
    p = soup.find("p")
    assert p.has_attr("class")  # class still there
    assert p.get("class") == ["text"]  # value unchanged
    assert not p.has_attr("data-class")  # no data-class added

    # Elements with 'id' but no 'class' should be unchanged (no class to replace)
    span = soup.find("span")
    assert span.get("id") == "sidebar"  # id unchanged
    assert not span.has_attr("class")  # no class to begin with
    assert not span.has_attr("data-class")  # no data-class added

    # Elements with neither should be unchanged
    article = soup.find("article")
    assert not article.has_attr("class")  # no class
    assert not article.has_attr("data-class")  # no data-class
    assert not article.has_attr("id")  # no id


###
def test_replace_attribute_continue_with_empty_attribute():
    """Test replace_attribute continues when attribute is empty (line 447)."""
    text = """
    <div class="">Empty class</div>
    <div class="content">Has content</div>
    <div>No class attribute</div>
    """

    # Try to replace class attribute, but the first div has an empty class=""
    # This should trigger line 447 because check_match will be an empty string (falsy)
    result = replace_attribute(
        text,
        selector="div",
        old_attribute="class",
        new_attribute="css-class",
        attribute_value="content",  # Looking for "content"
        replace_value="main-content",
        matcher_type="exact",  # Non-regex path
    )

    soup = BeautifulSoup(result, "html.parser")
    divs = soup.find_all("div")

    # First div: empty class="" - should trigger continue (line 447)
    assert divs[0].get("class") == []  # Empty but still has class attribute
    assert not divs[0].has_attr("css-class")

    # Second div: has "content" - should be processed
    assert not divs[1].has_attr("class")
    assert divs[1].get("css-class") is not None

    # Third div: no class attribute - not processed by this logic path
    assert not divs[2].has_attr("class")
    assert not divs[2].has_attr("css-class")


def test_replace_attribute_continue_regex_no_match():
    """Test replace_attribute continues when regex doesn't match (line 447)."""
    text = """
    <span data-id="user-123">User</span>
    <span data-id="admin-456">Admin</span>
    <span data-id="guest">Guest</span>
    """

    # Use regex that won't match "guest" (no numbers)
    result = replace_attribute(
        text,
        selector="span",
        old_attribute="data-id",
        new_attribute="user-id",
        attribute_value=r"\d+",  # Regex: one or more digits
        replace_value="matched",
        matcher_type="regex",
    )
    # print(result)
    # assert False

    soup = BeautifulSoup(result, "html.parser")
    spans = soup.find_all("span")

    # First span: "user-123" matches \d+ - should be processed
    assert not spans[0].has_attr("data-id")
    assert spans[0].get("user-id") is not None

    # Second span: "admin-456" matches \d+ - should be processed
    assert not spans[1].has_attr("data-id")
    assert spans[1].get("user-id") is not None

    # Third span: "guest" doesn't match \d+ - should trigger continue (line 447)
    assert spans[2].get("data-id") == "guest"  # Original attribute preserved
    assert not spans[2].has_attr("user-id")  # No new attribute
