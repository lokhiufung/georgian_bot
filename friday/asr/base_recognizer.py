from abc import ABC, abstractmethod
from typing import List
import json


class BaseRecognizer(ABC):

    @abstractmethod
    def wav_to_text(self, manifests, batch_size=1):
        raise NotImplementedError

    @staticmethod
    def _manifests_to_paths(manifests: List[str]) -> List[str]:
        """extract audio_filepaths from list of manifest

        :param manifests: list of manifest files
        :type manifests: List[str]
        :return: list of audio_filepaths
        :rtype: List[str]
        """
        audio_filepaths = []
        for manifest in manifests:
            with open(manifest, 'r') as f:
                item = json.load(f)
                audio_filepaths.append(item['audio_filepath'])
        return audio_filepaths

    @staticmethod
    def softmax(logits):
        e = np.exp(logits - np.max(logits))
        return e / e.sum(axis=-1).reshape([logits.shape[0], 1])
