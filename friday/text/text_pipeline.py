from typing import List, Callable, Tuple
from functools import partial


class TextPipeline(object):
    def __init__(self, callable_tuples: List[Tuple[str, Callable]]):
        """callables: List of tuples with (name, callable). Callables should preserve the input dtype, i.e str"""
        self.callable_tuples = callable_tuples

    def __call__(self, text):
        for _, callable_ in self.callable_tuples:
            text = callable_(text)
        return text
    
    def append_step(self, callable_tuple: Tuple[str, Callable]):
        self.callable_tuples.append(callable_tuple)


def build_cantonese_tts_text_pipeline():
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


def build_english_tts_text_pipeline():
    from .cleaners.english import (
        english_cleaner,
        normalize_to_phonemes,
        normalize_hk_code_swicth
    )
    
    cleaner_tuples = []
    cleaner_tuples.append(
        ('english_cleaner', english_cleaner)
    )
    cleaner_tuples.append(
        ('phonemize', partial(normalize_to_phonemes, lang='en'))
    )
    return TextPipeline(callable_tuples=cleaner_tuples)
   
    