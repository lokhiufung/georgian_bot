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
        return (0, sensor_output[SensorType.TEXT])  # `0` means always choose the first one

    def dialog_adaptor(self, sensor_output, fulfillment, confidence):
        if confidence > self.confidence_threshold: 
            doc = fulfillment.data[0]
            reply = doc['answer']
        else:
            reply = self._default['answer']

        return DialogOutput(
            reply=reply,
        )
