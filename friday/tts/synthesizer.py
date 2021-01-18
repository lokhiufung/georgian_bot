# import time

# import numpy as np
# import nemo
# import nemo.collections.asr as nemo_asr
import torch
import nemo.collections.tts as nemo_tts

# # from assist.utils import get_logger
from assist.interface import Interface


class Synthesizer(Interface):
    def __init__(self, text2mel_model, mel2audio_model, device='cpu'):
        self.text2mel_model = nemo_tts.models.Tacotron2Model.restore_from(text2mel_model)
        self.mel2audio_model = nemo_tts.models.WaveGlowModel.restore_from(mel2audio_model)

    def preprocess_text(self, text):
        # TODO: should be detached from flask server
        return text

    def text_to_wav(self, manifest):
        text = self.preprocess_text(manifest['text'])
        with torch.no_grad():
            parsed = self.text2mel_model.parse(manifest['text'])
            melspec = self.text2mel_model.generate_spectrogram(tokens=parsed)
            audio = self.mel2audio_model.convert_spectrogram_to_audio(spec=melspec)
            audio = audio.to('cpu').numpy()
        return audio

    
# class Synthesizer(Interface):
#     def __init__(
#             self, labels, tacotron2_model_definition, waveglow_model_definition, 
#             checkpoints, device="cpu"  
#         ):
#         self.labels = labels
#         self.device = device.lower()
#         self.sample_rate = waveglow_model_definition['sample_rate']
#         self._n_stride = tacotron2_model_definition['n_stride']
        
#         self._neural_factory = nemo.core.NeuralModuleFactory(placement=DEVICE_MAP[self.device]) 
#         self._neural_modules = self._create_neural_modules(tacotron2_model_definition, waveglow_model_definition)
#         if checkpoints:
#             self._restore_checkpoints(checkpoints)
#         self.setup_denoiser()  # denoiser of waveglow
#         # if logger is None:
#         #     self.logger = get_logger(__name__, fh_lv=fh_lv, ch_lv=ch_lv)

#     @classmethod
#     def from_yaml(cls, yaml_config):
#         from ruamel.yaml import YAML
        
#         yaml = YAML(typ="safe")
#         with open(yaml_config, 'r') as f:
#             config = yaml.load(f)
        
#         labels = config['labels']
#         if isinstance(labels, str):
#             labels = [char.rstrip('\n') for char in open(labels, 'r')]
#         else:
#             assert isinstance(labels, list)
        
#         labels += ['<BOS>', '<EOS>', '<PAD>']  # added default tokens
        
#         with open(config['tacotron2_model'], 'r') as f:
#             tacotron2_model_definition = yaml.load(f)
#         with open(config['waveglow_model'], 'r') as f:
#             waveglow_model_definition = yaml.load(f)
        
#         checkpoints = {
#             'encoder': config['checkpoint_encoder'],
#             'decoder': config['checkpoint_decoder'],
#             'text_embedding': config['checkpoint_embedding'],
#             'postnet': config['checkpoint_postnet'],
#             'waveglow': config['checkpoint_waveglow'],
#         }

#         return cls(
#             labels,
#             tacotron2_model_definition,
#             waveglow_model_definition,
#             checkpoints=checkpoints
#         )

#     @property
#     def sample_rate(self):
#         return self._sample_rate

#     @sample_rate.setter
#     def sample_rate(self, sample_rate):
#         self._sample_rate = sample_rate

#     def setup_denoiser(self):
#         self._neural_modules['waveglow'].setup_denoiser()

#     def _restore_checkpoints(self, checkpoints):
#         for name, checkpoint_path in checkpoints.items():
#             self._neural_modules[name].restore_from(checkpoint_path)

#     def _create_neural_modules(self, tacotron2_model_definition, waveglow_model_definition):
#         neural_modules = dict()
#         neural_modules['text_embedding'] = nemo_tts.TextEmbedding(
#             n_symbols=len(self.labels),
#             **tacotron2_model_definition['TextEmbedding'],
#         )
#         neural_modules['encoder'] = nemo_tts.Tacotron2Encoder(
#             **tacotron2_model_definition['Tacotron2Encoder'],
#         )
#         neural_modules['decoder'] = nemo_tts.Tacotron2DecoderInfer(
#             **tacotron2_model_definition['Tacotron2Decoder'],
#         )
#         neural_modules['postnet'] = nemo_tts.Tacotron2Postnet(
#             **tacotron2_model_definition['Tacotron2Postnet'],   
#         )
#         neural_modules['waveglow'] = nemo_tts.WaveGlowInferNM(
#             sample_rate=self._sample_rate,
#             **waveglow_model_definition['WaveGlowNM']
#         )
#         return neural_modules

#     def text_to_wav(self, manifest, denoiser_strength):
#         data_layer = nemo_asr.TranscriptDataLayer(
#             path=manifest,
#             labels=self.labels,
#             batch_size=1,
#             # num_workers=5,
#             bos_id=len(self.labels) - 3,
#             eos_id=len(self.labels) - 2,
#             pad_id=len(self.labels) - 1,
#             shuffle=False
#         )
#         transcript, transcript_len = data_layer()
#         transcript_embedded = self._neural_modules['text_embedding'](char_phone=transcript)
#         transcript_encoded = self._neural_modules['encoder'](char_phone_embeddings=transcript_embedded, embedding_length=transcript_len,)

#         mel_decoder, gate, alignments, mel_len = self._neural_modules['decoder'](
#                     char_phone_encoded=transcript_encoded, 
#                     encoded_length=transcript_len,
#                 )
#         mel_postnet = self._neural_modules['postnet'](mel_input=mel_decoder)

#         start = time.perf_counter()
#         tacotron2_output_tensors = self._neural_factory.infer(
#             tensors=[mel_postnet, gate, alignments, mel_len],
#             use_cache=False
#         )
#         # tacotron2_infer_time = time.perf_counter() - start
#         # self.logger.debug('tacotron2_infer_time: {}'.format(tacotron2_infer_time))
        
#         audio_pred = self._neural_modules['waveglow'](mel_spectrogram=mel_postnet)   

#         start = time.perf_counter()
#         waveglow_output_tensors = self._neural_factory.infer(
#             tensors=[audio_pred],
#         )
#         # waveglow_infer_time = time.perf_counter() - start
#         # self.logger.debug('waveglow_infer_time: {}'.format(waveglow_infer_time))

#         audio = np.squeeze(waveglow_output_tensors[0][0].cpu().numpy())
#         sample_len = tacotron2_output_tensors[-1][0] * self._n_stride
#         # print(sample_len)
#         sample = audio[:sample_len]
#         if denoiser_strength > 0:
#             sample, _ = self._neural_modules['waveglow'].denoise(sample, strength=denoiser_strength)  # start from 0
#         return sample
