from typing import List, Callable, Tuple



class TextPipeline(object):
    def __init__(self, callables: List[Tuple[str, Callable]]):
        """callables: List of tuples with (name, callable). Callables should preserve the input dtype, i.e str"""
        self.callables = callables

    def __call__(self, text):
        for callable_ in self.callables:
            text = callable_(text)
        return text
    
    def append_step(self, callable_: Callable):
        self.callables.append(callable_)


def build_cantonese_tts_text_pipeline():
    from .cleaners.cantonese import (
        cantonese_cleaner,
        normalize_to_phonemes,
        normalize_hk_code_swicth
    )
    
    cleaner_tuples = []
    cleaner_tuples.append(
        ('cantonese_cleaner', cantonese_cleaner(text))
    )
    cleaner_tuples.append(
        ('phonemize', normalize_to_phonemes(text, 'yue'))
    )
    cleaner_tuples.append(
        ('normalize_code_switch', normalize_hk_code_swicth(text))
    )
    return TextPipeline(callables=cleaners_tuples)


def build_english_tts_text_pipeline():
    from .cleaners.english import (
        english_cleaner,
        normalize_to_phonemes,
        normalize_hk_code_swicth
    )
    
    cleaner_tuples = []
    cleaner_tuples.append(
        ('english_cleaner', english_cleaner(text))
    )
    cleaner_tuples.append(
        ('phonemize', normalize_to_phonemes(text, 'en'))
    )
    return TextPipeline(callables=cleaners_tuples)
   
    