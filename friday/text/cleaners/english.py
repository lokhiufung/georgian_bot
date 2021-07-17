import re

from .common import base_cleaner


def english_cleaner(text: str) -> str:
    """clean text in english style

    :param text: text
    :type text: str
    :return: cleaned text
    :rtype: str
    """
    text = base_cleaner(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    return text

