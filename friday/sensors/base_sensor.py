from abc import ABC, abstractmethod, abstractproperty

import requests


class BaseSensor(ABC):
    """"""
    @abstractproperty
    def type_(self):
        raise NotImplementedError

    @abstractmethod
    def process(self):
        pass


class APICallingSensor(BaseSensor):
    def __init__(self):
        self._sess = requests.Session()

    
