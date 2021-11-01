
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np

from friday.fulfillments.routers.base_router import BaseFulfillmentRouter
from friday.sensors.base_sensor import BaseSensor
from friday.data_classes import Fulfillment


class BaseAgent(ABC):
    """The base base class of agent ~"""


class CompositionalAgent(BaseAgent):
    
    def __init__(self, fulfillment_router: BaseFulfillmentRouter, sensors: Dict[str, BaseSensor]):
        """Base class of compositional agent

        A Compositional agent is composed of different operating units. It breaks a agent into different units, like sensors, dialog-making, fulfillment router.  

        :param fulfillment_router: [description]
        :type fulfillment_router: BaseFulfillmentRouter
        :param sensors: [description]
        :type sensors: BaseSensor
        """
        self.fulfillment_router = fulfillment_router
        # self.fulfillment_adaptor = fulfillment_adaptor
        # self.dialog_adaptor = dialog_adaptor 
        self.sensors = sensors
    
    @abstractmethod
    def fulfillment_adaptor(self, sensor_output: Dict[Union[str, int, np.ndarray], dict]):
        raise NotImplementedError 
    
    @abstractmethod
    def dialog_adaptor(self, sensor_output: dict, fulfillment: Fulfillment, confidence: float):
        raise NotImplementedError

    # @classmethod
    # def from_json(self, filepath: str):
    #     # import json

    #     # with open(filepath, 'r') as f:
    #     #     config = json.load(f)
    #     raise NotImplementedError  
 
    def act(self, obs: dict, fulfillment_key: Union[int, str]=None):
        
        # embedding = send_embedding_server(text)
        sensor_output = {type_: sensor.process(obs[type_]) for type_, sensor in self.sensors.items() if type_ in obs}
        
        if fulfillment_key is None:
            fulfillment_key, fulfillment_args = self.fulfillment_adaptor(sensor_output)
        
        fulfillment, confidence = self.fulfillment_router.route(fulfillment_key, fulfillment_args)
        dialog = self.dialog_adaptor(sensor_output, fulfillment, confidence)

        return dialog, fulfillment
