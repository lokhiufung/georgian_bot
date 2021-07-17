# from collections import OrderedDict

# import torch
# from torch.utils.data import DataLoader
# import nemo.collections.asr as nemo_asr
# from nemo.collections.asr.modules.audio_preprocessing import AudioPreprocessor
# from nemo.core import NeuralModule, Exportable, typecheck
# from nemo.core.neural_types import NeuralType
# from nemo.core.neural_types import *
# from nemo.utils import logging
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


# __all__ = ['Wave2Vec2Encoder', 'Wave2Vec2Preprocessor', 'EncDecCTCModelWave2Vec2']


# class Wave2Vec2Encoder(NeuralModule, Exportable):
#     def __init__(self, pretrained_model_name, downsampling=3266):
#         super().__init__()

#         self.pretrained_model_name = pretrained_model_name
#         self.downsampling = downsampling
#         # self.sample_rate = sample_rate
#         # self.padding_value = padding_value
#         # self.pad_token_id = pad_token_id
#         # self.feature_extractor = Wav2Vec2FeatureExtractor(
#         #     feature_size=1,  # since it is raw audio input
#         #     sampling_rate=self.sample_rate,
#         #     padding_value=self.padding_value,
#         #     return_attention_mask=True,
#         #     do_normalize=True,
#         # )
#         logging.info(f'loading transformer pretrained model: `{self.pretrained_model_name}`')
#         self.encoder = Wav2Vec2Model.from_pretrained(
#             self.pretrained_model_name,
#             gradient_checkpointing=False,
#             pad_token_id=0
#         )
#         logging.info(f'loaded transformer pretrained model: `{self.pretrained_model_name}`')
    
#     @property
#     def input_types(self):
#         """Returns definitions of module input ports.
#         """
#         return OrderedDict(
#             {
#                 "audio_signal": NeuralType(('B', 'T'), AudioSignal()),
#                 "audio_lengths": NeuralType(tuple('B'), LengthsType()),
#                 # "attention_mask": NeuralType(tuple('B', 'T'), MaskType())
#             }
#         )

#     @property
#     def output_types(self):
#         """Returns definitions of module output ports.
#         """
#         return OrderedDict(
#             {
#                 "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
#                 "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
#             }
#         )
    
#     @staticmethod
#     def get_attention_mask(audio_lengths):
#         maxlen = torch.max(audio_lengths)
#         batch_size = audio_lengths.size(0)

#         attention_mask = torch.ones((batch_size, maxlen), dtype=audio_lengths.dtype)
#         for i in range(batch_size):
#             attention_mask[i, audio_lengths[i]:] = 0.0
#         return attention_mask

#     @typecheck()
#     def forward(self, audio_signal, audio_lengths):
#         # x = self.feature_extractor(audio_signal, sampling_rate=self.sample_rate, return_tensor='pt', return_attention_mask=True)
#         attention_mask = self.get_attention_mask(audio_lengths)
#         # print(attention_mask)
#         x = self.encoder(audio_signal, attention_mask)
#         ecoded_len = audio_lengths // self.downsampling
#         return torch.transpose(x['last_hidden_state'], 1, 2), ecoded_len  # RemindMe: probably buggy

#     def freeze(self):
#         """override the parent's method with transformers api
#         """
#         self.encoder.feature_extractor._freeze_parameters()
        

# from nemo.collections.asr.modules.audio_preprocessing import AudioPreprocessor


# class Wave2Vec2Preprocessor(AudioPreprocessor):

#     def __init__(self, sample_rate=16000, padding_value=0.0, do_normalize=True, pad_to=16, dither=1e-5):
#         super().__init__(win_length=-1, hop_length=-1)  # remindme: hacky
        
#         self.sample_rate = sample_rate
#         self.padding_value = padding_value
#         self.do_normalize = do_normalize

#         class _Featurizer(object):
#             pass
#         self.featurizer = _Featurizer()
#         # self.featurizer = Wav2Vec2FeatureExtractor(
#         #     feature_size=1,
#         #     sampling_rate=self.sample_rate,
#         #     padding_value=self.padding_value,
#         #     return_attention_mask=True
#         # )

#         # for compatibility with nemo 1.x api
#         self.featurizer.dither_value = dither
#         self.featurizer.pad_to = pad_to

#     @property
#     def input_types(self):
#         """Returns definitions of module input ports.
#         """
#         return OrderedDict(
#             {
#                 "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self.sample_rate)),
#                 "length": NeuralType(tuple(('B',)), LengthsType()),
#             }
#         ) 

#     @property
#     def output_types(self):
#         """Returns definitions of module output ports.
#         """
#         return OrderedDict(
#             {
#                 "audio_signal": NeuralType(('B', 'T'), AudioSignal()),
#                 "audio_lengths": NeuralType(tuple(('B',)), LengthsType()),
#                 # "attention_mask": NeuralType(tuple(('B', 'T')), MaskType())
#             }
#         )

#     def get_features(self, input_signal, length):
#         # x = self.featurizer(
#         #     input_signal,
#         #     sampling_rate=self.sample_rate,
#         #     return_tensor='pt',
#         #     return_attention_mask=True
#         # )
#         """No operation

#         :param input_signal: batched audio singal
#         :type input_signal: torch.Tensor
#         :param length: [description]
#         :type length: [type]
#         :return: [description]
#         :rtype: [type]
#         """
#         return input_signal, length
    
#     @typecheck()
#     def forward(self, input_signal, length):
#         processed_signal, processed_length = self.get_features(input_signal, length)
#         return processed_signal, processed_length
    


# # class EncDecCTCModelWave2Vec2(nemo_asr.models.EncDecCTCModel):

# #     def forward(
# #         self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
# #     ):
# #         has_input_signal = input_signal is not None and input_signal_length is not None
# #         has_processed_signal = processed_signal is not None and processed_signal_length is not None
# #         if (has_input_signal ^ has_processed_signal) == False:
# #             raise ValueError(
# #                 f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
# #                 " with ``processed_signal`` and ``processed_signal_len`` arguments."
# #             )

# #         if not has_processed_signal:
# #             processed_signal, processed_signal_length = self.preprocessor(
# #                 input_signal=input_signal, length=input_signal_length,
# #             )
        
# #         encoded, encoded_len = self.encoder(audio_signal=processed_signal, audio_lengths=processed_signal_length)
# #         log_probs = self.decoder(encoder_output=encoded)
# #         greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

# #         return log_probs, encoded_len, greedy_predictions

# #     def __setup_dataloader_from_config(self, config):
# #         shuffle = config['shuffle']
# #         # device = 'gpu' if torch.cuda.is_available() else 'cpu'

# #         if 'manifest_filepath' in config and config['manifest_filepath'] is None:
# #             logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
# #             return None

# #         dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

# #         return torch.utils.data.DataLoader(
# #             dataset=dataset,
# #             batch_size=config['batch_size'],
# #             collate_fn=None,
# #             drop_last=config.get('drop_last', False),
# #             shuffle=shuffle,
# #             num_workers=config.get('num_workers', 0),
# #             pin_memory=config.get('pin_memory', False),
# #         )
# import os
# import tempfile

# from omegaconf import DictConfig, OmegaConf, open_dict

# from nemo.collections.asr.metrics.wer import WER
# from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


# class TransformerCTCModelWave2Vec2(nemo_asr.models.ASRModel):
#     def __init__(self, cfg, trainer=None):
#         self.global_rank = 0
#         self.world_size = 1
#         self.local_rank = 0
#         if trainer is not None:
#             self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
#             self.world_size = trainer.num_nodes * trainer.num_gpus
#             self.local_rank = trainer.local_rank
        
#         super().__init__(cfg=cfg, trainer=trainer)
        
#         tokenizer = Wav2Vec2CTCTokenizer(
#             self._cfg.vocab_file,
#             unk_token='[UNK]',
#             pad_token='[PAD]',
#             word_delimiter_token='|',
#         )
#         feature_extractor = Wav2Vec2FeatureExtractor(
#             feature_size=1,
#             sampling_rate=self._cfg.sample_rate,
#             padding_value=self._cfg.pad_value,
#             do_normalize=self._cfg.do_normalize,
#             return_attention_mask=True,
#         )
#         self.processor = Wav2Vec2Processor(
#             feature_extractor,
#             tokenizer,
#         )
#         self._wer = WER(
#             vocabulary=self._cfg.vocabulary,
#             batch_dim_index=0,
#             user_cer=self._cfg.get('user_cer', False),
#             ctc_decoder=True,
#             dist_sync_on_step=True,
#             log_prediction=self._cfg.get('log_prediction', False),
#         )

#     @torch.no_grad()
#     def transcribe(self, paths2audio_files, batch_size, logprobs=False):
#         if paths2audio_files is None or len(paths2audio_files) == 0:
#             return {}
#         # We will store transcriptions here
#         hypotheses = []
#         # Model's mode and device
#         mode = self.training
#         device = next(self.parameters()).device

#         try:
#             self.eval()

# if __name__ == '__main__':


#     dataset = nemo_asr.data.audio_to_text_dataset.get_char_dataset(
#         config={
#             'manifest_filepath': '/home/xxx/data/cantonese/src/common_voice/commonvoice_dev_manifest.json',
#             'sample_rate': 16000,
#             'labels': [char.strip() for char in open('/home/xxx/data/cantonese/src/common_voice/commonvoice_labels.txt', 'r')],
#             'batch_size': 4,
#             'trim_scilence': True
#         },
#     )

#     encoder = Wave2Vec2Encoder(
#         'facebook/wav2vec2-large-xlsr-53'
#     )

#     decoder = nemo_asr.modules.ConvASRDecoder(
#         feat_in=1024,
#         num_classes=3000,
#     )

#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)

#     audio_signal, audio_lengths, tokens, tokens_lengths = next(iter(dataloader))
    
#     encoder_output = encoder(audio_signal=audio_signal, audio_lengths=audio_lengths)
#     log_softmax = decoder(encoder_output=encoder_output)

#     print(log_softmax.size())