import os
import re
from typing import List, Union, Dict, Optional

import sentencepiece as spm
from nemo.collections.common.tokenizers import TokenizerSpec, SentencePieceTokenizer


__all__ = ['HalfSentencePieceTokenizer']


def clean_text(func):
    def wrapper(ref, text):
        text = re.sub(r'[\u200b|\u200c|\u200d|\u200e]', '', text)
        return func(ref, text)
    return wrapper


class HalfSentencePieceTokenizer(SentencePieceTokenizer):
    def __init__(self, model_path: str, model_vocab_path: str, chinese_vocabulary: str, special_tokens: Optional[Union[Dict[str, str], List[str]]] = None):
        """A tokenizer that only tokenize english words

        :param model_path: model path of a sentence piece english tokenizer
        :type model_path: str
        :param model_vocab_path: path of vocabulary file of the tokenizer
        :type model_vocab_path: str
        :param chinese_vocabulary: path of vocabulary file of the chinese words
        :type chinese_vocabulary: str
        :raises ValueError: raise ValueError if the model_path does not exist
        """
        if not os.path.exists(model_path):
            raise ValueError(f"model_path: {model_path} is invalid")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model_path)
        self._tokenizer_vocabulary = [line.split('\t')[0] for line in open(model_vocab_path, 'r')]
        if isinstance(chinese_vocabulary, str):
            self._chinese_vocabulary = [char.strip() for char in open(chinese_vocabulary, 'r')]
        else:
            self._chinese_vocabulary = chinese_vocabulary

        # remove intersection chars from chinese_vocabulary
        intersection = set(self._tokenizer_vocabulary).intersection(self._chinese_vocabulary) 
        if intersection:
            self._chinese_vocabulary = [char for char in self._chinese_vocabulary if char not in intersection]

        self.vocabulary = self._tokenizer_vocabulary + self._chinese_vocabulary
        self.original_vocab_size = len(self.vocabulary)
        self.vocab_size = len(self.vocabulary)
        self.special_token_to_id = {}
        self.id_to_sepcial_token = {}

        self.vocabs_to_ids = {vocab: id_ for id_, vocab in enumerate(self.vocabulary)}
        self._ids_to_vocabs = {id_: vocab for vocab, id_ in self.vocabs_to_ids.items()}

        if special_tokens:
            self.add_special_tokens(special_tokens)
        # self.tokenizer.set_vocabulary(self.vocabulary)
    
    @staticmethod
    def _is_contain_chinese(text):
        if re.search(r'[\u4e00-\u9fff]', text):
            return True
        else:
            return False

    @staticmethod
    def _is_contain_int(text):
        if re.search(r'[0-9]', text):
            return True
        else:
            return False

    @clean_text
    def text_to_tokens(self, text: str):
        tokens = []
        for token in self.tokenizer.encode(text, out_type=str):
            if self._is_contain_chinese(token) or self._is_contain_int(token):
                chinese_tokens = [char for char in token]  # tokenized by characters
                tokens.extend(chinese_tokens)
            else:
                tokens.append(token)
        return tokens
    
    def tokens_to_text(self, tokens: List[str]):
        return self.tokenizer.decode(tokens)

    def tokens_to_ids(self, tokens):
        return [self.vocabs_to_ids[token] for token in tokens]

    def ids_to_tokens(self, ids: List[int]):
        try:
            tokens = [self._ids_to_vocabs[id_] for id_ in ids]
            return tokens
        except:
            print(ids)
            print(self._ids_to_vocabs)
            raise Exception

    @clean_text
    def text_to_ids(self, text: str):
        tokens = self.text_to_tokens(text)
        ids = [self.vocabs_to_ids[token] if token in self.vocabulary else 0 for token in tokens]  # 0 is the unk token
        return ids
    
    def ids_to_text(self, ids: List[int]):
        tokens = self.ids_to_tokens(ids)
        text = self.tokens_to_text(tokens)
        return text
    
    def add_special_tokens(self, special_tokens):
        super().add_special_tokens(special_tokens)
        
        # extend the voca
        self.vocabs_to_ids = {**self.vocabs_to_ids, **self.special_token_to_id}
        self._ids_to_vocabs = {**self._ids_to_vocabs, **self.id_to_special_token}


class JiebaTokenizer(TokenizerSpec):
    def __init__(self, userdict=None):
        import importlib

        self.jieba = importlib.import_module('jieba')
        if userdict:
            self.jieba.load_userdict(userdict)
            
    def text_to_tokens(self, text):
        return [token for token in self.jieba.cut(text)]

    def tokens_to_text(self, tokens):
        return ''.join(tokens)
    
    def tokens_to_ids(self, tokens):
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        raise NotImplementedError

    def text_to_ids(self, text):
        raise NotImplementedError

    def ids_to_text(self, text):
        raise NotImplementedError
    
