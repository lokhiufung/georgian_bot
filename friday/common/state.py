from typing import Union

import redis


class StateStorage(object):
    """Base class of StateStorage object"""
    def set_variable(self, name, value):
        raise NotImplementedError

    def get_variable(self, name):
        raise NotImplementedError


class SimpleStateStorage(StateStorage):
    def __init__(self, state_dict: dict):
        self.state_dict = state_dict

    def set_variable(self, name, value):
        self.state_dict[name] = value

    def get_variable(self, name):
        return self.state_dict[name]
    

class RedisStateStorage(StateStorage):
    def __init__(self, state_dict, redis_host='localhost', redis_port=6379):
        self._redis_server = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        for key, value in state_dict.items():
            self.set_variable(name, value)

    def set_variable(self, name: str, value: Union[str, int, float]):
        self._redis_server.set(name, value)

    def get_variable(self, name):
        return self._redis_server.get(name)
    

