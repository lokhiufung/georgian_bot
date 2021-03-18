import os
import json
from typing import List

import torch
import numpy as np
import nemo.collections.asr as nemo_asr


__all__ = ['Recognizer']


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


def _manifests_to_paths(manifests: List[str]) -> List[str]:
    """extract audio_filepaths from list of manifest"""
    audio_filepaths = []
    for manifest in manifests:
        with open(manifest, 'r') as f:
            item = json.load(f)
            audio_filepaths.append(item['audio_filepath'])
    return audio_filepaths


class Recognizer:
    def __init__(self, asr_model, device='cpu', use_lm=False, lm_path=''):
        self.model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=asr_model, map_location=torch.device(device))

        # self.model.setup_test_data(
        #     test_data_config={
        #         'sample_rate': 16000,
        #         'manifest_filepath': '',
        #         'labels': self.model.decoder.vocabulary,
        #         'batch_size': 1,
        #         'normalize_transcripts': False
        #     }
        # )
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

    def wav_to_text(self, manifests, batch_size=1):
        if not isinstance(manifests, list):
            manifests = [manifests]
        
        audio_filepaths = _manifests_to_paths(manifests)

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