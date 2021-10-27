from abc import ABC, abstractmethod


class BasePlatformAdaptor(ABC):
    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    def start_server(self):
        raise NotImplementedError


