import os
import copy

from omegaconf import OmegaConf, DictConfig
import nemo.collections.asr as nemo_asr
from nemo.utils import logging


class EncDecCTCModelHalfBPE(nemo_asr.models.EncDecCTCModelBPE):
    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        # Prevent tokenizer parallelism (unless user has explicitly set it)
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.tokenizer_cfg = OmegaConf.to_container(tokenizer_cfg, resolve=True)  # type: dict
        self.tokenizer_dir = self.tokenizer_cfg.pop('dir')  # Remove tokenizer directory
        self.tokenizer_type = self.tokenizer_cfg.pop('type').lower()  # Remove tokenizer_type

        if self.tokenizer_type not in ['hbpe', 'hwpe']:
            raise ValueError(
                "`tokenizer.type` must be either `hbpe` for SentencePiece tokenizer or "
                "`hwpe` for BERT based tokenizer"
            )

        if self.tokenizer_type == 'hbpe':
            from friday.common.tokenizer import HalfSentencePieceTokenizer

            # This is a BPE Tokenizer
            model_path = os.path.join(self.tokenizer_dir, 'tokenizer.model')
            model_path = self.register_artifact('tokenizer.model_path', model_path)
            self.model_path = model_path

            if 'special_tokens' in self.tokenizer_cfg:
                special_tokens = self.tokenizer_cfg['special_tokens']
            else:
                special_tokens = None

            vocab_path = os.path.join(self.tokenizer_dir, 'vocab.txt')
            vocab_path = self.register_artifact('tokenizer.vocab_path', vocab_path)
            self.vocab_path = vocab_path
            
            # addtional chinese vocabulary path
            chinese_vocab_path = self.tokenizer_cfg['chinese_vocab_path']
            chinese_vocab_path = self.register_artifact('tokenizer.chinese_vocab_path', chinese_vocab_path)
            self.chinese_vocab_path = chinese_vocab_path

            # Update special tokens
            self.tokenizer = HalfSentencePieceTokenizer(
                model_path=model_path,
                model_vocab_path=self.vocab_path,
                chinese_vocabulary=chinese_vocab_path,
                special_tokens=special_tokens
            )

            # wrapper method to get vocabulary conveniently
            def get_vocab():
                return self.tokenizer.vocabs_to_ids

            # attach utility values to the tokenizer wrapper
            self.tokenizer.tokenizer.vocab_size = self.tokenizer.vocab_size
            self.tokenizer.tokenizer.get_vocab = get_vocab
            self.tokenizer.tokenizer.all_special_tokens = self.tokenizer.special_token_to_id

        else:
            raise NotImplementedError('Only hbpe is supported.')
            # This is a WPE Tokenizer
            # vocab_path = os.path.join(self.tokenizer_dir, 'vocab.txt')
            # self.tokenizer_dir = self.register_artifact('tokenizer.vocab_path', vocab_path)
            # self.vocab_path = self.tokenizer_dir

            # self.tokenizer = tokenizers.AutoTokenizer(
            #     pretrained_model_name='bert-base-cased', vocab_file=self.tokenizer_dir, **self.tokenizer_cfg
            # )

        logging.info(
            "Tokenizer {} initialized with {} tokens".format(
                self.tokenizer.__class__.__name__, self.tokenizer.vocab_size
            )
        )

   