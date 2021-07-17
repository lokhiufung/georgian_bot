from typing import List, Callable, Tuple
from functools import partial


class TextPipeline(object):
    def __init__(self, callable_tuples: List[Tuple[str, Callable]]):
        """pipeline for processing text

        :param callable_tuples: callables: List of tuples with (name, callable). Callables should preserve the input dtype, i.e str
        :type callable_tuples: List[Tuple[str, Callable]]
        """
        self.callable_tuples = callable_tuples

    def __call__(self, text: str) -> str:
        """process text

        :param text: text
        :type text: str
        :return: processed text
        :rtype: str
        """
        for _, callable_ in self.callable_tuples:
            text = callable_(text)
        return text
    
    def append_step(self, callable_tuple: Tuple[str, Callable]):
        """append node to the pipeline

        :param callable_tuple: tuple with (name, callable)
        :type callable_tuple: Tuple[str, Callable]
        """
        self.callable_tuples.append(callable_tuple)


def build_cantonese_tts_text_pipeline() -> TextPipeline:
    """build defualt pipeline for cantonese tts

    :return: text pipeline for cantonese tts
    :rtype: TextPipeline
    """
    from .cleaners.cantonese import (
        cantonese_cleaner,
        normalize_to_phonemes,
        normalize_hk_code_swicth
    )
    
    cleaner_tuples = []
    cleaner_tuples.append(
        ('cantonese_cleaner', cantonese_cleaner)
    )
    cleaner_tuples.append(
        ('phonemize', partial(normalize_to_phonemes, lang='yue'))
    )
    cleaner_tuples.append(
        ('normalize_code_switch', normalize_hk_code_swicth)
    )
    return TextPipeline(callable_tuples=cleaner_tuples)


def build_english_tts_text_pipeline() -> TextPipeline:
    """build defualt pipeline for cantonese tts

    :return: text pipeline for english tts
    :rtype: TextPipeline
    """
    from .cleaners.common import normalize_to_phonemes
    from .cleaners.english import english_cleaner
    
    cleaner_tuples = []
    cleaner_tuples.append(
        ('english_cleaner', english_cleaner)
    )
    cleaner_tuples.append(
        ('phonemize', partial(normalize_to_phonemes, lang='en-us'))
    )
    return TextPipeline(callable_tuples=cleaner_tuples)
   
    