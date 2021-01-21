import torch
from hydra.utils import instantiate
import nemo.collections.tts as nemo_tts
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths, tacotron2_log_to_tb_func
from nemo.collections.common import typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    EmbeddedTextType,
    LengthsType,
    AudioSignal,
    MelSpectrogramType
)
from nemo.core.neural_types import NeuralType



class ConditionalTacotron2(nemo_tts.models.Tacotron2Model):
    """
    Tacotron2 Model that can generate spectrogram conditioning on speaker embeddings.
    Simply concatenate the speaker embedding to the embedding produced by the encoder 
    """
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg, trainer)
        self.spk_encoder = instantiate(self._cfg.spk_encoder)
        self.spk_encoder.eval()  # freeze the spk_encoder. This model should be well trained 

    @property
    def input_types(self):
        if self.training:
            return {
                "seapkers_embedding": NeuralType(('B', 'C'), EncodedRepresentation()),
                "tokens": NeuralType(('B', 'T'), EmbeddedTextType()),
                "token_len": NeuralType(('B'), LengthsType()),
                "audio": NeuralType(('B', 'T'), AudioSignal()),
                "audio_len": NeuralType(('B'), LengthsType()),
            }
        else:
            return {
                "tokens": NeuralType(('B', 'T'), EmbeddedTextType()),
                "token_len": NeuralType(('B'), LengthsType()),
                "audio": NeuralType(('B', 'T'), AudioSignal(), optional=True),
                "audio_len": NeuralType(('B'), LengthsType(), optional=True),
            }

    @typecheck
    def forward(self, *, tokens, token_len, audio=None, audio_len=None):
        # if audio is not None and audio_len is not None:
        spec_target, spec_target_len = self.audio_to_melspec_precessor(audio, audio_len)
        token_embedding = self.text_embedding(tokens).transpose(1, 2)
        encoder_embedding = self.encoder(token_embedding=token_embedding, token_len=token_len)
        with torch.no_grad():
            spk_embedding = self.spk_encoder(spec_target)

        if self.training:
            spec_pred_dec, gate_pred, alignments = self.decoder(
                memory=torch.cat((encoder_embedding, spk_embedding), dim=1), decoder_inputs=spec_target, memory_lengths=token_len
            )
        else:
            spec_pred_dec, gate_pred, alignments, pred_length = self.decoder(
                memory=torch.cat((encoder_embedding, spk_embedding), dim=1), memory_lengths=token_len
            )
        spec_pred_postnet = self.postnet(mel_spec=spec_pred_dec)

        if not self.calculate_loss:
            return spec_pred_dec, spec_pred_postnet, gate_pred, alignments, pred_length
        return spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments

    @typecheck(
        input_types={
            "spk_embedding": NeuralType(('B', 'C'), EncodedRepresentation()),
            "tokens": NeuralType(('B', 'T'), EmbeddedTextType())
        },
        output_types={"spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType())},
    )
    def generate_spectrogram(self, *, audio, tokens):
        self.eval()
        self.calculate_loss = False
        token_len = torch.tensor([len(i) for i in tokens]).to(self.device)
        tensors = self(speaker_embedding=speaker_embedding, tokens=tokens, token_len=token_len)
        spectrogram_pred = tensors[1]

        if spectrogram_pred.shape[0] > 1:
            # Silence all frames past the predicted end
            mask = ~get_mask_from_lengths(tensors[-1])
            mask = mask.expand(spectrogram_pred.shape[1], mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            spectrogram_pred.data.masked_fill_(mask, self.pad_value)

        return spectrogram_pred
