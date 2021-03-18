import re

from .common import base_cleaner


def english_cleaner(text):
    text = base_cleaner(text)
    text = text.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    return text

