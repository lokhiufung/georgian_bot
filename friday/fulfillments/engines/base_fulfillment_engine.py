from abc import ABC, abstractmethod


class BaseFulfillmentEngine(ABC):
    NAME = 'base_fulfillment'

    def __init__(self):
        """Base class of `FulfillmentEngine`

        `FulfillmentEngine` runs task assigned by the agent and return the results of the assigned task to the agent. A single `FulfillmentEngine` is specialized for doing a single task.
        
        You can override the NAME to assign the name of this engine. Otherwise, {your_class_name}_fulfillment will be the name of your object.
        ```
        class MyFulfillmentEngine(BaseFulfillmentEngine):
            NAME = 'my_fulfillment'

        ```
        """
        self.name = self.NAME
        if self.name == 'base_fulfillment':
            self.name = f'{self.__name__}_fulfillment'
        

    @abstractmethod
    def run(self):
        raise NotImplementedError



