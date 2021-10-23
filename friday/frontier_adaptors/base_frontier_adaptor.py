from abc import ABC


class BaseFrontierAdaptor(ABC):
    def __init__(self, agent):
        self.agent = agent

    def frontier_to_agent(self):
        pass

    def agent_to_frontier(self):
        pass


