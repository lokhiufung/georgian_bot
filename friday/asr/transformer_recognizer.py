import os

import torch
import torchaudio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC

from friday.asr.base_recognizer import BaseRecognizer


class TransformerRecognizer(BaseRecognizer):
    def __init__(self, transformer_model, device='cpu'):

        self.sampling_rate = 16000
        self._device = torch.device(device)

        tokenizer = Wav2Vec2CTCTokenizer(
            os.path.join(transformer_model, 'vocab.json'),
            unk_token='[UNK]',
            pad_token='[PAD]',
            word_delimiter_token='|',
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor,
            tokenizer,
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            transformer_model,
        ).to(self._device)

        self.model.eval()

    @torch.no_grad()
    def wav_to_text(self, manifests, batch_size=1):
        if not isinstance(manifests, list):
            manifests = [manifests]
        
        audio_filepath = self._manifests_to_paths(manifests)[0]
        speech = torchaudio.load(audio_filepath)[0][0].numpy()
        input_dict = self.processor(speech, return_tensors='pt', padding=True, sampling_rate=self.sampling_rate)

        logits = self.model(input_dict.input_values.to(self._device)).logits

        pred_ids = torch.argmax(logits, dim=-1)[0]  # greedy decoder
        transcription = self.processor.decode(pred_ids)

        return [transcription]
