from abc import ABC, abstractmethod, abstractproperty

import requests


class BaseSensor(ABC):
    """Base class of `Sensor`

    A `Sensor` is the primary data processing layer of an agent. Different sensors process different types of inputs. Outputs from different sensors can be integrated. 
    """
    @abstractproperty
    def type_(self):
        raise NotImplementedError

    @abstractmethod
    def process(self):
        pass


class APICallingSensor(BaseSensor):
    def __init__(self):
        self._sess = requests.Session()

    
