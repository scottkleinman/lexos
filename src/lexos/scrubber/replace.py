"""replace.py.

Last Update: 2025-06-08
Tested: 2025-06-08
"""

import html
import re
import sys
import unicodedata
from typing import Collection, Optional

from pydantic import ConfigDict, validate_call

from lexos.util import ensure_list, to_collection

from . import resources

validation_config = ConfigDict(arbitrary_types_allowed=True)


@validate_call(config=validation_config)
def currency_symbols(text: str, repl: str = "_CUR_") -> str:
    """Replace all currency symbols in `text` with `repl`.

    Args:
        text (str): The text in which currency symbols will be replaced.
        repl (str): The replacement value for currency symbols.

    Returns:
        str: The text with currency symbols replaced.
    """
    return resources.RE_CURRENCY_SYMBOL.sub(repl, text)


@validate_call(config=validation_config)
def digits(text: str, repl: str = "_DIGIT_") -> str:
    """Replace all digits in `text` with `repl`.

    Args:
        text (str): The text in which digits will be replaced.
        repl (str): The replacement value for digits.

    Returns:
        str: The text with digits replaced.
    """
    return resources.RE_NUMBER.sub(repl, text)


@validate_call(config=validation_config)
def emails(text: str, repl: str = "_EMAIL_") -> str:
    """Replace all email addresses in `text` with `repl`.

    Args:
        text (str): The text in which emails will be replaced.
        repl (str): The replacement value for emails.

    Returns:
        str: The text with emails replaced.
    """
    return resources.RE_EMAIL.sub(repl, text)


@validate_call(config=validation_config)
def emojis(text: str, repl: str = "_EMOJI_") -> str:
    """Replace all emoji and pictographs in `text` with `repl`.

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


@validate_call(config=validation_config)
def hashtags(text: str, repl: str = "_HASHTAG_") -> str:
    """Replace all hashtags in `text` with `repl`.

    Args:
        text (str): The text in which hashtags will be replaced.
        repl (str): The replacement value for hashtags.

    Returns:
        str: The text with currency hashtags replaced.
    """
    return resources.RE_HASHTAG.sub(repl, text)


@validate_call(config=validation_config)
def pattern(text: str, *, pattern: Optional[dict | Collection[dict]]) -> str:
    """Replace strings from `text` using a regex pattern.

    Args:
        text (str): The text in which a pattern or pattern will be replaced.
        pattern: (Optional[dict | Collection[dict]]): A dictionary or list of dictionaries
            containing the pattern(s) and replacement(s).

    Returns:
        str: The text with pattern(s) replaced.
    """
    pattern = ensure_list(pattern)
    for pat in pattern:
        k = str(*pat)
        match = re.compile(k)
        text = re.sub(match, pat[k], text)
    return text


@validate_call(config=validation_config)
def phone_numbers(text: str, repl: str = "_PHONE_") -> str:
    """Replace all phone numbers in `text` with `repl`.

    Args:
        text (str): The text in which phone numbers will be replaced.
        repl (str): The replacement value for phone numbers.

    Returns:
        str: The text with phone numbers replaced.
    """
    return resources.RE_PHONE_NUMBER.sub(repl, text)


@validate_call(config=validation_config)
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
        exclude (Optional[str | Collection[str]]): Remove all punctuation except designated characters.
        only (Optional[str | Collection[str]]): Remove only those punctuation marks specified here.
            For example, `"."` removes only periods, while `[",", ";", ":"]` removes commas,
            semicolons, and colons; if None, all unicode punctuation marks are removed.

    Returns:
        str: The text with punctuation replaced.

    Note:
        When `only=None`, Python's built-in `str.translate()` is used;
        otherwise, a regular expression is used. The former's performance
        can be up to an order of magnitude faster.
    """
    if only is not None:
        only = to_collection(only, val_type=str, col_type=set)
        return re.sub("[{}]+".format(re.escape("".join(only))), " ", text)
    else:
        if exclude:
            exclude = ensure_list(exclude)
            translation_table = dict.fromkeys(
                (
                    i
                    for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith("P")
                    and chr(i) not in exclude
                ),
                " ",
            )
        else:
            translation_table = resources.PUNCT_TRANSLATION_TABLE
        return text.translate(translation_table)


@validate_call(config=validation_config)
def special_characters(
    text: str,
    *,
    is_html: Optional[bool] = False,
    ruleset: Optional[dict] = None,
) -> str:
    """Replace strings from `text` using a regex pattern.

    Args:
        text (str): The text in which special characters will be replaced.
        is_html (Optional[bool]): Whether to replace HTML entities.
        ruleset (Optional[dict]): A dict containing the special characters to match and their replacements.

    Returns:
        str: The text with special characters replaced.
    """
    if is_html:
        text = html.unescape(text)
    else:
        for k, v in ruleset.items():
            match = re.compile(k)
            text = re.sub(match, v, text)
    return text


@validate_call(config=validation_config)
def urls(text: str, repl: str = "_URL_") -> str:
    """Replace all URLs in `text` with `repl`.

    Args:
        text (str): The text in which urls will be replaced.
        repl (str): The replacement value for urls.

    Returns:
        str: The text with urls replaced.
    """
    return resources.RE_SHORT_URL.sub(repl, resources.RE_URL.sub(repl, text))


@validate_call(config=validation_config)
def user_handles(text: str, repl: str = "_USER_") -> str:
    """Replace all (Twitter-style) user handles in `text` with `repl`.

    Args:
        text (str): The text in which user handles will be replaced.
        repl (str): The replacement value for user handles.

    Returns:
        str: The text with user handles replaced.
    """
    return resources.RE_USER_HANDLE.sub(repl, text)
