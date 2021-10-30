from typing import Union, List
from abc import ABC, abstractmethod

import numpy as np


class BaseFulfillmentRouter:
    """Base class of `FulfillmentRouter`

        A `FulfillmentRouter` is a key-value based router for retriving fulfillment engines. A agent will apply a key to the fulfillment router to retrive the target fulfillment engine.
    """
    
    @abstractmethod
    def route(self, key: Union[int, str, List[float], np.ndarray]):
        pass


