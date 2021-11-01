import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

from friday.constants import SensorType
from friday.sensors.base_sensor import BaseSensor



class AudioSensor(BaseSensor):
    def __init__(self, sampling_rate: int=16000, model_name: str='facebook/wav2vec2-base-960h', backend: str='transformers'):
        self.sampling_rate =sampling_rate
        self.model_name = model_name
        self.backend = backend

        if backend == 'transformers':
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        else:
            raise NotImplementedError('Only `transformers` is supported: {}'.format(self.backend))

    @property
    def type_(self):
        return SensorType.AUDIO

    def process(self, audio_signal: np.ndarray):
        inputs = self.processor(audio_signal, sampling_rate=self.sampling_rate, return_tensors='pt')  # return pytorch's tensor        
        logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(predicted_ids)

        return {
            'transcription': transcription[0]
        }



if __name__ == '__main__':

    from friday.common import Audio
    audio_sensor = AudioSensor()

    audio = Audio.from_ogg('./329319724-2021_11_01_23_55_1635782148.oga')

    result= audio_sensor.process(
        audio_signal=audio.get_np_array()[:, 0]
    )

    print(result['transcription'])
    
    

    

