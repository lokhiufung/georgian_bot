import os
import json
from typing import List, Union

import torch
import numpy as np
import nemo.collections.asr as nemo_asr

from friday.asr.base_recognizer import BaseRecognizer


# def softmax(logits):
#     e = np.exp(logits - np.max(logits))
#     return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


# def _manifests_to_paths(manifests: List[str]) -> List[str]:
#     """extract audio_filepaths from list of manifest

#     :param manifests: list of manifest files
#     :type manifests: List[str]
#     :return: list of audio_filepaths
#     :rtype: List[str]
#     """
#     audio_filepaths = []
#     for manifest in manifests:
#         with open(manifest, 'r') as f:
#             item = json.load(f)
#             audio_filepaths.append(item['audio_filepath'])
#     return audio_filepaths


class Recognizer(BaseRecognizer):
    def __init__(self, asr_model, model_type: str='ctc', device='cpu', use_lm=False, lm_path=''):
        """Object that transcrible an audio

        :param asr_model: .nemo file path
        :type asr_model: str
        :param asr_model: model type; can be one of the `ctc`, `ctc_hbpe`, 'ctc_bpe'
        :type asr_model: NemoModel
        :param device: device to be used, defaults to 'cpu'
        :type device: str, optional
        :param use_lm: whether to use lm or not, defaults to False
        :type use_lm: bool, optional
        :param lm_path: path of .lm file, defaults to ''
        :type lm_path: str, optional
        """
        if model_type.lower() == 'ctc_hbpe':
            from friday.asr.models.ctc_half_bpe_model import EncDecCTCModelHalfBPE
            self.model = EncDecCTCModelHalfBPE.restore_from(restore_path=asr_model, map_location=torch.device(device))
        elif model_type.lower() == 'ctc_bpe':
            self.model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=asr_model, map_location=torch.device(device))
        else:
            self.model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=asr_model, map_location=torch.device(device))
            
        self.use_lm = use_lm
        self.beam_search_decoder = None
        if self.use_lm:
            self.beam_search_decoder = nemo_asr.modules.BeamSearchDecoderWithLM(
                vocab=list(self.model.cfg.decoder.vocabulary),
                beam_width=64,
                alpha=2,
                beta=1.5,
                lm_path=lm_path,
                num_cpus=min(os.cpu_count(), 2),
                input_tensor=False
            )

    def wav_to_text(self, manifests: Union[List[str], str], batch_size=1) -> List[str]:
        """transcribe audio with manifest file(s)

        :param manifests: list of manifest files or a single manifest file
        :type manifests: Union[List[str], str]
        :param batch_size: batch size for inference, defaults to 1
        :type batch_size: int, optional
        :return: list of transcriptions
        :rtype: List[str]
        """
        if not isinstance(manifests, list):
            manifests = [manifests]
        
        audio_filepaths = self._manifests_to_paths(manifests)

        if self.use_lm:
            logits = self.model.transcribe(
                audio_filepaths,
                logprobs=True,
                batch_size=batch_size
            )[0].cpu().numpy()
            probs = softmax(logits)
            # return list of list of tuples with (score, transcription)
            transcriptions = self.beam_search_decoder.forward(
                log_probs=np.expand_dims(probs, axis=0),
                log_probs_length=None
            )
        else:
            transcriptions = self.model.transcribe(
                audio_filepaths,
                batch_size
            )
        return transcriptions