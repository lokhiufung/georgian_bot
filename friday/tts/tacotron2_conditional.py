# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import nemo.collections.tts as nemo_tts
# import nemo.collections.asr as nemo_asr
# from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
# from nemo.core.classes import typecheck, NeuralModule
# from nemo.core.neural_types.elements import (
#     EncodedRepresentation,
#     EmbeddedTextType,
#     LengthsType,
#     AudioSignal,
#     MelSpectrogramType
# )
# from nemo.core.neural_types import NeuralType
# from nemo.core.classes.module import Module


# class SpeakerEmbeddingProjectionLayer(NeuralModule):
#     def __init__(self, embedding_size, projected_size):
#         super().__init__()
#         self.embedding_size = embedding_size  # size of encoder embedding + spk embedding
#         self.projected_size = projected_size
        
#         self.layer = nn.Linear(self.embedding_size, self.projected_size)

#     @property
#     def input_types(self):
#         return {
#             "spk_enc_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
#         }

#     @property
#     def output_types(self):
#         return {
#             "projected_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
#         }

#     @typecheck()
#     def forward(self, spk_enc_embedding):
#         projected_embedding = F.relu(self.layer(spk_enc_embedding))
#         return projected_embedding


# class ConditionalTacotron2(nemo_tts.models.Tacotron2Model):
#     """
#     Tacotron2 Model that can generate spectrogram conditioning on speaker embeddings.
#     Simply concatenate the speaker embedding to the embedding produced by the encoder 
#     """
#     def __init__(self, cfg, trainer=None):
#         super().__init__(cfg=cfg, trainer=trainer)
#         self.spk_encoder = nemo_asr.models.ExtractSpeakerEmbeddingsModel.restore_from(self.cfg.spk_encoder.restore_path)  # initiate a spk_encoder outside
#         self.spk_encoder.eval()  # freeze the spk_encoder. This model should be well trained 

#         ### 2. concatenation + projection
#         self.spk_projection_layer = SpeakerEmbeddingProjectionLayer(
#             embedding_size=self.cfg.spk_encoder.spk_emb_dim + self.cfg.decoder.encoder_embedding_dim,
#             projected_size=self.cfg.decoder.encoder_embedding_dim
#         )

#     @property
#     def input_types(self):
#         if self.training:
#             return {
#                 "tokens": NeuralType(('B', 'T'), EmbeddedTextType()),
#                 "token_len": NeuralType(('B'), LengthsType()),
#                 "audio": NeuralType(('B', 'T'), AudioSignal()),
#                 "audio_len": NeuralType(('B'), LengthsType()),
#             }
#         else:
#             return {
#                 "tokens": NeuralType(('B', 'T'), EmbeddedTextType()),
#                 "token_len": NeuralType(('B'), LengthsType()),
#                 "audio": NeuralType(('B', 'T'), AudioSignal(), optional=True),
#                 "audio_len": NeuralType(('B'), LengthsType(), optional=True),
#             }

#     @typecheck()
#     def forward(self, *, tokens, token_len, audio=None, audio_len=None):
#         # if audio is not None and audio_len is not None:
#         # audio and audio_len must not be None at any time
#         self.spk_encoder.eval()  # ensure to freeze the spk_encoder. This model should be well trained 

#         spec_target, spec_target_len = self.audio_to_melspec_precessor(audio, audio_len)
#         token_embedding = self.text_embedding(tokens).transpose(1, 2)
#         encoder_embedding = self.encoder(token_embedding=token_embedding, token_len=token_len)
        
#         with torch.no_grad():
#             # get speaker embedding
#             _, spk_embedding = self.spk_encoder.forward(input_signal=audio, input_signal_length=audio_len)
#             spk_embedding = spk_embedding.unsqueeze(1).expand(-1, encoder_embedding.size()[1], -1)  # expand spk_embedding (B, D) -> (B, T, D) 

#         #########################################################################################################
#         # 1. use sum instead of concatenation: in order to preserve the dimension 512 dim of embeddings
#         # if self.training:
#         #     spec_pred_dec, gate_pred, alignments = self.decoder(
#         #         memory=encoder_embedding + spk_embedding, decoder_inputs=spec_target, memory_lengths=token_len
#         #     )
#         # else:
#         #     spec_pred_dec, gate_pred, alignments, pred_length = self.decoder(
#         #         memory=encoder_embedding + spk_embedding, memory_lengths=token_len
#         #     )
#         #########################################################################################################
#         #########################################################################################################
#         # 2. concatenation: 512 encoder embedding + 512 speaker embedding -> projection layer
#         projected_embedding = self.spk_projection_layer(
#             spk_enc_embedding=torch.cat([encoder_embedding, spk_embedding], dim=2)
#         )
#         if self.training:
#             spec_pred_dec, gate_pred, alignments = self.decoder(
#                 memory=projected_embedding, decoder_inputs=spec_target, memory_lengths=token_len
#             )
#         else:
#             spec_pred_dec, gate_pred, alignments, pred_length = self.decoder(
#                 memory=projected_embedding, memory_lengths=token_len
#             )
#         #########################################################################################################
        
#         spec_pred_postnet = self.postnet(mel_spec=spec_pred_dec)
        

#         if not self.calculate_loss:
#             return spec_pred_dec, spec_pred_postnet, gate_pred, alignments, pred_length
#         return spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments

#     @typecheck(
#         input_types={
#             "audio": NeuralType(('B', 'T'), AudioSignal()),
#             "tokens": NeuralType(('B', 'T'), EmbeddedTextType())
#         },
#         output_types={"spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType())},
#     )
#     def generate_spectrogram(self, *, tokens, audio):
#         self.eval()
#         self.calculate_loss = False
#         token_len = torch.tensor([len(i) for i in tokens]).to(self.device)
#         audio_len = torch.tensor([len(i) for i in audio]).to(self.device)
#         tensors = self(tokens=tokens, token_len=token_len, audio=audio, audio_len=audio_len)
#         spectrogram_pred = tensors[1]

#         if spectrogram_pred.shape[0] > 1:
#             # Silence all frames past the predicted end
#             mask = ~get_mask_from_lengths(tensors[-1])
#             mask = mask.expand(spectrogram_pred.shape[1], mask.size(0), mask.size(1))
#             mask = mask.permute(1, 0, 2)
#             spectrogram_pred.data.masked_fill_(mask, self.pad_value)

#         return spectrogram_pred
