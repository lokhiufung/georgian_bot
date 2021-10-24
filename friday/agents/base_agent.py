from abc import ABC, abstractmethod


class CompositionalAgent(ABC):
    
    def __init__(self, fulfillment_router, sensors):
        self.fulfillment_router = fulfillment_router
        # self.fulfillment_adaptor = fulfillment_adaptor
        # self.dialog_adaptor = dialog_adaptor 
        self.sensors = sensors
    
    @abstractmethod
    def fulfillment_adaptor(self, sensor_output):
        raise NotImplementedError 
    
    @abstractmethod
    def dialog_adaptor(self, sensor_output, fulfillment, confidence):
        raise NotImplementedError

    @classmethod
    def from_json(self, filepath: dict):
        # import json

        # with open(filepath, 'r') as f:
        #     config = json.load(f)
        raise NotImplementedError  
 
    def act(self, obs, fulfillment_key=None):
        
        # embedding = send_embedding_server(text)
        sensor_output = {sensor.type_: sensor.process(obs[sensor.type_]) for sensor in self.sensors}
        
        if fulfillment_key is None:
            fulfillment_key, fulfillment_args = self.fulfillment_adaptor(sensor_output)
        
        fulfillment, confidence = self.fulfillment_router.route(fulfillment_key, fulfillment_args)
        dialog = self.dialog_adaptor(sensor_output, fulfillment, confidence)

        return dialog, fulfillment
