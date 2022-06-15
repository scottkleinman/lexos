"""validators.py."""

import re
from pathlib import Path

ip_middle_octet = "(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5]))"
ip_last_octet = "(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"

regex = re.compile(
    "^"
    # protocol identifier
    "(?:(?:https?|ftp)://)"
    # user:pass authentication
    "(?:\S+(?::\S*)?@)?" "(?:" "(?P<private_ip>"
    # IP address exclusion
    # private & local networks
    "(?:(?:10|127)" + ip_middle_octet + "{2}" + ip_last_octet + ")|"
    "(?:(?:169\.254|192\.168)" + ip_middle_octet + ip_last_octet + ")|"
    "(?:172\.(?:1[6-9]|2\d|3[0-1])" + ip_middle_octet + ip_last_octet + "))"
    "|"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    "(?P<public_ip>"
    "(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    "" + ip_middle_octet + "{2}"
    "" + ip_last_octet + ")"
    "|"
    # host name
    "(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
    # domain name
    "(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
    # TLD identifier
    "(?:\.(?:[a-z\u00a1-\uffff]{2,}))" ")"
    # port number
    "(?::\d{2,5})?"
    # resource path
    "(?:/\S*)?"
    # query string
    "(?:\?\S*)?" "$",
    re.UNICODE | re.IGNORECASE,
)

pattern = re.compile(regex)


def is_path_or_url(value: str) -> bool:
    """Return whether or not given value is a valid path or url.

    Args:
        value (str): The value to validate.

    Returns:
        bool: Whether or not the value is a valid path or url.
    """
    if is_valid_path(value):
        return True
    if is_valid_url(value):
        return True
    else:
        return False


def is_valid_path(value: str) -> bool:
    """Return whether or not given value is a valid path.

    Args:
        value (str): The value to validate.

    Returns:
        bool: Whether or not the value is a valid path.
    """
    return Path(value).exists()


def is_valid_url(value: str, public: bool = False) -> bool:
    """Return whether or not given value is a valid URL.

    This validator is based on the Python validators package, which is
    itself based on Diego Perini's URL validator: https://gist.github.com/dperini/729294.

    Args:
        value (str): The value to validate.
        public (bool): Whether or not to allow only public IP addresses.

    Returns:
        bool: Whether or not the value is a valid URL.
    """
    result = pattern.match(value)
    if not public:
        return result

    return result and not result.groupdict()["private_ip"]
