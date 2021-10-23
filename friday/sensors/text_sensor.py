import sentence_transformers import SentenceTransformer

from friday.sensors.base_sensor import APICallingSensor, BaseSensor
from friday.constants import SensorType
# from friday.data_classes import SensorOutput



class RequestTextEmbeddingSensor(APICallingSensor):
    """"""
    def __init__(self, embedding_endpoint):
        super().__init__()
        self.embedding_endpoint = embedding_endpoint

    def type_(self):
        return SensorType.TEXT

    def process(self, text):
        response = self._sess.post(self.embedding_endpoint, json={'text': text})
        
        data = response.json()

        return {
            'embedding': data['embedding']
        }


class TextEmbeddingSensor(BaseSensor):
    """"""
    def __init__(self, model_name='distiluse-base-multilingual-cased'):
        self.model = SentenceTransformer(model_name)

    def type_(self):
        return SensorType.TEXT

    def process(self, text):
        return {
            'embedding': self.model.encode([text])
        }

