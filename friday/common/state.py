from typing import Union

import redis


class ClientStateStorage(object):
    """Base class of StateStorage object"""
    def set_variable(self, client_id: str, name: str, value):
        raise NotImplementedError

    def get_variable(self, client_id: str, name: str):
        raise NotImplementedError


class SimpleClientStateStorage(ClientStateStorage):
    def __init__(self):
        """
        simple state storage using dict
        """
        self.client_states = {}  # store using states

    def set_variable(self, client_id, name, value):
        if client_id in self.state_dict:
            self.client_states[cleint_id] = {}
        self.client_states[cleint_id][name] = value

    def get_variable(self, client, name):
        return self.cleint_states[client_id][name]
    

class RedisStateStorage(ClientStateStorage):
    def __init__(self, redis_host='localhost', redis_port=6379, key_lifetime=500):
        self._redis_server = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.key_lifetime = key_lifetime

    def set_variable(self, client_id: str, name: str, value):
        if client_id in self._redis_server.scan_iter():
            state_dict = self.get_client_state_dict(client_id)
        else:
            state_dict = {}
        state_dict[name] = value
        self._redis_server.hmset(client_id, state_dict)
        self._redis_server.expire(name=client_id, time=self.key_lifetime)
    
    def get_variable(self, client_id, name):
        state_dict = self.get_client_state_dict(client_id)
        # print(state_dict)
        return state_dict[name]
    
    def get_client_state_dict(self, client_id: str):
        return self._redis_server.hgetall(client_id)


