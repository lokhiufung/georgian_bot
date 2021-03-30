import re
from typing import Union 

from phonemizer import phonemize


def base_cleaner(text):
    text = text.strip()
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\t', '', text)
    return text


def normalize_to_str(text_like: Union[str, float, int]):
    text = str(text_like)
    return text


def normalize_to_phonemes(text, lang):
    text = phonemize(text, language=lang, backend='espeak')
    return text