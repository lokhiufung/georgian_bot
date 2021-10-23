from abc import ABC, abstractmethod


class BaseFulfillmentEngine(ABC):
    NAME = 'base_fulfillment'

    def __init__(self):
        self.name = self.NAME
        
    @abstractmethod
    def run(self):
        raise NotImplementedError



