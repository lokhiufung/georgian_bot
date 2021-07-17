import re
import os
from typing import Union 

from phonemizer import phonemize


def base_cleaner(text: str) -> str:
    """minimally clean text 

    :param text: text
    :type text: str
    :return: cleaned text
    :rtype: str
    """
    text = text.strip()
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\t', '', text)
    return text


def remove_special_characters(text: str) -> str:
    # text = re.sub(r'')
    return text


def normalize_to_str(text_like: Union[str, float, int]) -> str:
    """normalize input variable to str, can be str, float, int

    :param text_like: variable that can be casted to str 
    :type text_like: Union[str, float, int]
    :return: normalized text
    :rtype: str
    """
    text = str(text_like)
    return text


def normalize_to_phonemes(text: str, lang: str) -> str:
    """normalize text to phonemes using phonemizer backend

    :param text: text
    :type text: str
    :param lang: 'yue', 'en', etc
    :type lang: str
    :return: phonemized text
    :rtype: str
    """
    text = phonemize(text, language=lang, backend='espeak', preserve_punctuation=True, njobs=os.cpu_count())
    return text