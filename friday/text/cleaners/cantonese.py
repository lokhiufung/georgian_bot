
from opencc import OpenCC

from .common import base_cleaner, normalize_to_phonemes


__s2hk = OpenCC('s2hk')


def normalize_to_hk(text):
    text = __s2hk.convert(text)
    return text


def normalize_hk_code_swicth(text):
    text = text.replace('(en)', '[BOCS]')
    text = text.replace('(zhy)', '[EOCS]')
    return text


def cantonese_cleaner(text):
    text = base_cleaner(text)
    text = normalize_to_hk(text)
    return text