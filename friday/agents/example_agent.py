from friday.agents.base_agent import CompositionalAgent
from friday.constants import SensorType
from friday.data_classes import DialogOutput


class ExampleAgent(CompositionalAgent):
    def __init__(self, fulfillment_router, sensors, confidence_threshold=0.5):
        super().__init__(fulfillment_router, sensors)

        self.confidence_threshold = confidence_threshold 
    
    def _default(self):
        return {
            'answer': "Sorry, I cannot answer this question. ><"
        }

    def fulfillment_adaptor(self, sensor_output):
        # TODO: re-implementation needed
        if SensorType.TEXT in sensor_output:
            return (0, sensor_output[SensorType.TEXT]['embedding'])  # `0` means always choose the first one
        elif SensorType.AUDIO in sensor_output:
            transcription = sensor_output[SensorType.AUDIO]['transcription'] 
            result = self.sensors[SensorType.TEXT].process(transcription)
            return(0, result['embedding'])
        else:
            raise Exception

    def dialog_adaptor(self, sensor_output, fulfillment, confidence):
        if confidence > self.confidence_threshold: 
            doc = fulfillment.data[0]
            reply = doc['answer']
        else:
            reply = self._default['answer']

        return DialogOutput(
            reply=reply,
        )
