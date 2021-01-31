from friday.task import Task


class BaseAgent(object):
    """Base class of all agents"""
    def register_task(self, task: Task):
        raise NotImplementedError


class CompositionalAgent(BaseAgent):
    """Abstract class of all compositional agents, which can be broken down into separate functional components"""
    
    def dialog_flow(self):
        """define the dialogflow in this method"""
        raise NotImplementedError

    def text_infer(self):
        """define the """
        raise NotImplementedError