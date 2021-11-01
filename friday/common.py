from typing import Union

import numpy as np

from pydub import AudioSegment, audio_segment


class Audio:
    """"""
    def __init__(self, signal: Union[np.ndarray, AudioSegment], sampling_rate=16000):
        
        if isinstance(signal, np.ndarray):
            self.audio_segment = AudioSegment(
                data=np.asarray(signal),
                frame_rate=sampling_rate,
                channels=signal.shape[1],
            )
        elif isinstance(signal, AudioSegment):
            self.audio_segment = signal
        else:
            raise ValueError('Expected `np.ndarray` or `AudioSegment`: {}'.format(type(signal)))
        
    @property
    def sampling_rate(self):
        return self.audio_segment.frame_rate

    @classmethod
    def from_ogg(cls, filepath, sampling_rate=16000):
        audio_segment = AudioSegment.from_ogg(filepath)
        audio_segment = audio_segment.set_frame_rate(sampling_rate)
        
        return cls(
            signal=audio_segment,
            sampling_rate=sampling_rate,
        ) 

    def get_np_array(self) -> np.ndarray:
        """ return the audio signal in np.ndarray with shape (n_channels, n_samples) and type `float32`

        :return: audio signal
        :rtype: np.ndarray
        """
        channel_sounds = self.audio_segment.split_to_mono()
        samples = [s.get_array_of_samples() for s in channel_sounds]

        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        return fp_arr

    

if __name__ == '__main__':

    import os

    HOME = os.environ['HOME']
    audio = Audio.from_ogg(os.path.join(HOME, '329319724-2021_11_01_23_55_1635782148.oga'), sampling_rate=16000)

    signal = audio.get_np_array()

    print(signal.shape)