"""replace.py."""
from __future__ import annotations

import html
import re
import sys
import unicodedata
from typing import Collection, List, Optional, Union

from lexos import utils

from . import resources


def currency_symbols(text: str, repl: str = "_CUR_") -> str:
    """Replace all currency symbols in `text` with `repl`.

    Args:
        text (str): The text in which currency symbols will be replaced.
        repl (str): The replacement value for currency symbols.

    Returns:
        str: The text with currency symbols replaced.
    """
    return resources.RE_CURRENCY_SYMBOL.sub(repl, text)


def digits(text: str, repl: str = "_DIGIT_") -> str:
    """Replace all digits in `text` with `repl`.

    Args:
        text (str): The text in which digits will be replaced.
        repl (str): The replacement value for digits.

    Returns:
        str: The text with digits replaced.
    """
    return resources.RE_NUMBER.sub(repl, text)


def emails(text: str, repl: str = "_EMAIL_") -> str:
    """Replace all email addresses in `text` with `repl`.

    Args:
        text (str): The text in which emails will be replaced.
        repl (str): The replacement value for emails.

    Returns:
        str: The text with emails replaced.
    """
    return resources.RE_EMAIL.sub(repl, text)


def emojis(text: str, repl: str = "_EMOJI_") -> str:
    """
    Replace all emoji and pictographs in `text` with `repl`.

    Args:
        text (str): The text in which emojis will be replaced.
        repl (str): The replacement value for emojis.

    Returns:
        str: The text with emojis replaced.

    Note:
        If your Python has a narrow unicode build ("USC-2"), only dingbats
        and miscellaneous symbols are replaced because Python isn't able
        to represent the unicode data for things like emoticons. Sorry!
    """
    return resources.RE_EMOJI.sub(repl, text)


def hashtags(text: str, repl: str = "_HASHTAG_") -> str:
    """Replace all hashtags in `text` with `repl`.

    Args:
        text (str): The text in which hashtags will be replaced.
        repl (str): The replacement value for hashtags.

    Returns:
        str: The text with currency hashtags replaced.
    """
    return resources.RE_HASHTAG.sub(repl, text)


def pattern(
    text: str,
    *,
    pattern: Union[dict, Collection[dict]]
) -> str:
    """Replace strings from `text` using a regex pattern.

    Args:
        text (str): The text in which a pattern or pattern will be replaced.
        pattern: (Union[dict, Collection[dict]]): A dictionary or list of dictionaries
            containing the pattern(s) and replacement(s).

    Returns:
        str: The text with pattern(s) replaced.
    """
    pattern = utils.ensure_list(pattern)
    for pat in pattern:
        k = str(*pat)
        match = re.compile(k)
        text = re.sub(match, pat[k], text)
    return text


def phone_numbers(text: str, repl: str = "_PHONE_") -> str:
    """Replace all phone numbers in `text` with `repl`.

    Args:
        text (str): The text in which phone numbers will be replaced.
        repl (str): The replacement value for phone numbers.

    Returns:
        str: The text with phone numbers replaced.
    """
    return resources.RE_PHONE_NUMBER.sub(repl, text)


def process_tag_replace_options(orig_text: str, tag: str, action: str,
                                attribute: str) -> str:
    """Replace html-style tags in text files according to user options.

    Args:
        orig_text: The user's text containing the original tag.
        tag: The particular tag to be processed.
        action: A string specifying the action to be performed on the tag.
        attribute: Replacement value for tag when "replace_with_attribute" is specified.

        Action options are:
        - "remove_tag": Remove the tag
        - "remove_element": Remove the element and contents
        - "replace_element": Replace the tag with the specified attribute

    Returns:
        str: The text after the specified tag is processed.

    Note: The replacement of a tag with the value of an attribute may not be supported. This needs a second look.
    """
    if action == "remove_tag":
        # searching for variants this specific tag:  <tag> ...
        pattern = re.compile(
            r'<(?:' + tag + r'(?=\s)(?!(?:[^>"\']|"[^"]*"|\'[^\']*\')*?(?<=\s)'
                            r'\s*=)(?!\s*/?>)\s+(?:".*?"|\'.*?\'|[^>]*?)+|/?'
            + tag + r'\s*/?)>', re.MULTILINE | re.DOTALL | re.UNICODE)

        # substitute all matching patterns with one space
        processed_text = re.sub(pattern, " ", orig_text)

    elif action == "remove_element":
        # <[whitespaces] TAG [SPACE attributes]> contents </[whitespaces]TAG>
        # as applied across newlines, (re.MULTILINE), on re.UNICODE,
        # and .* includes newlines (re.DOTALL)
        pattern = re.compile(
            r"<\s*" + re.escape(tag) + r"( .+?>|>).+?</\s*" + re.escape(tag) +
            ">", re.MULTILINE | re.DOTALL | re.UNICODE)

        processed_text = re.sub(pattern, " ", orig_text)

    elif action == "replace_element":
        pattern = re.compile(
            r"<\s*" + re.escape(tag) + r".*?>.+?</\s*" + re.escape(tag) +
            ".*?>", re.MULTILINE | re.DOTALL | re.UNICODE)

        processed_text = re.sub(pattern, attribute, orig_text)

    else:
        processed_text = orig_text  # Leave Tag Alone

    return processed_text


def punctuation(
    text: str,
    *,
    exclude: Optional[str | Collection[str]] = None,
    only: Optional[str | Collection[str]] = None,
) -> str:
    """Replace punctuation from `text`.

    Replaces all instances of punctuation (or a subset thereof specified by `only`)
    with whitespace.

    Args:
        text (str): The text in which punctuation will be replaced.
        exclude: Remove all punctuation except designated characters.
        only: Remove only those punctuation marks specified here. For example,
            `"."` removes only periods, while `[",", ";", ":"]` removes commas,
            semicolons, and colons; if None, all unicode punctuation marks are removed.

    Returns:
        str

    Note:
        When `only=None`, Python's built-in `str.translate()` is used;
        otherwise, a regular expression is used. The former's performance
        can be up to an order of magnitude faster.
    """
    if only is not None:
        only = utils.to_collection(only, val_type=str, col_type=set)
        return re.sub("[{}]+".format(re.escape("".join(only))), " ", text)
    else:
        if exclude:
            exclude = utils.ensure_list(exclude)
            translation_table = dict.fromkeys(
                (
                    i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith("P")
                    and chr(i) not in exclude), " "
            )
        else:
            translation_table = resources.PUNCT_TRANSLATION_TABLE
        return text.translate(translation_table)


def special_characters(
    text: str,
    *,
    is_html: bool = False,
    ruleset: dict = None,
) -> str:
    """Replace strings from `text` using a regex pattern.

    Args:
        text (str): The text in which special characters will be replaced.
        is_html (bool): Whether to replace HTML entities.
        ruleset (dict): A dict containing the special characters to match and their replacements.

    Returns:
        str
    """
    if is_html:
        text = html.unescape(text)
    else:
        for k, v in ruleset.items():
            match = re.compile(k)
            text = re.sub(match, v, text)
    return text

from typing import Dict


def tag_map(text: str,
    # xmlhandlingoptions: List[dict],
    map: Dict[str],
    remove_comments: bool = True,
    remove_doctype: bool = True,
    remove_whitespace: bool = False) -> str:
    """Handle tags that are found in the text.

    Args:
        text (str): The text in which tags will be replaced.
        remove_comments (bool): Whether to remove comments.
        remove_doctype (bool): Whether to remove the doctype or xml declaration.
        remove_whitespace (bool): Whether to remove whitespace.

    Returns:
        str: The text after tags have been replaced.
    """
    if remove_whitespace:
        text = re.sub(r"[\n\s\t\v ]+", " ", text, re.UNICODE)  # Remove extra white space
    if remove_doctype:
        doctype = re.compile(r"<!DOCTYPE.*?>", re.DOTALL)
        text = re.sub(doctype, "", text)  # Remove DOCTYPE declarations
        text = re.sub(r"(<\?.*?>)", "", text)  # Remove xml declarations
    if remove_comments:
        text = re.sub(r"(<!--.*?-->)", "", text)  # Remove comments

    # This matches the DOCTYPE and all internal entity declarations
    doctype = re.compile(r"<!DOCTYPE.*?>", re.DOTALL)
    text = re.sub(doctype, "", text)  # Remove DOCTYPE declarations

    # Visit each tag:
    for tag, opts in map.items():
        action = opts["action"]
        attribute = opts["attribute"]
        text = process_tag_replace_options(text, tag, action, attribute)

    # One last catch-all removes extra whitespace from all the removed tags
    if remove_whitespace:
        text = re.sub(r"[\n\s\t\v ]+", " ", text, re.UNICODE)

    return text


def urls(text: str, repl: str = "_URL_") -> str:
    """Replace all URLs in `text` with `repl`.

    Args:
        text (str): The text in which urls will be replaced.
        repl (str): The replacement value for urls.

    Returns:
        str: The text with urls replaced.
    """
    return resources.RE_SHORT_URL.sub(repl, resources.RE_URL.sub(repl, text))


def user_handles(text: str, repl: str = "_USER_") -> str:
    """Replace all (Twitter-style) user handles in `text` with `repl`.

    Args:
        text (str): The text in which user handles will be replaced.
        repl (str): The replacement value for user handles.

    Returns:
        str: The text with user handles replaced.
    """
    return resources.RE_USER_HANDLE.sub(repl, text)
