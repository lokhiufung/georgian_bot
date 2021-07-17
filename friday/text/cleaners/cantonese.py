
from opencc import OpenCC

from .common import base_cleaner, normalize_to_phonemes


__s2hk = OpenCC('s2hk')


def normalize_to_hk(text: str) -> str:
    """normalize text to hk style

    :param text: text
    :type text: str
    :return: normalized text
    :rtype: str
    """
    text = __s2hk.convert(text)
    return text


def normalize_hk_code_swicth(text: str) -> str:
    """normalize phonemes if they contains code switch flag. '#' indicate the start of code switch; '%' indicates the end of code switch

    :param text: text, phonemized
    :type text: str
    :return: normalized text
    :rtype: str
    """
    text = text.replace('(en)', '#')
    text = text.replace('(zhy)', '%')
    return text


def cantonese_cleaner(text: str) -> str:
    """clean text in cantonese style

    :param text: text
    :type text: str
    :return: cleaned text
    :rtype: str
    """
    text = base_cleaner(text)
    text = normalize_to_hk(text)
    return text