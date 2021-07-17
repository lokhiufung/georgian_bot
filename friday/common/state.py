from typing import Union

import redis


class ClientStateStorage(object):
    """abstract class of client state storage

    :param object: [description]
    :type object: [type]
    """
    def set_variable(self, client_id: str, name: str, value):
        raise NotImplementedError

    def get_variable(self, client_id: str, name: str):
        raise NotImplementedError


class SimpleClientStateStorage(ClientStateStorage):
    def __init__(self):
        """simple storage using python dict
        """
        self.client_states = {}  # store using states

    def set_variable(self, client_id, name, value):
        if client_id in self.state_dict:
            self.client_states[client_id] = {}
        self.client_states[client_id][name] = value

    def get_variable(self, client_id, name):
        return self.client_states[client_id][name]
    

class RedisStateStorage(ClientStateStorage):
    def __init__(self, redis_host='localhost', redis_port=6379, key_lifetime=500):
        """storage using redis server

        :param redis_host: redis server host, defaults to 'localhost'
        :type redis_host: str, optional
        :param redis_port: redis server port, defaults to 6379
        :type redis_port: int, optional
        :param key_lifetime: how long the key-value will keep alive, defaults to 500s
        :type key_lifetime: int, optional
        """
        self._redis_server = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.key_lifetime = key_lifetime

    def set_variable(self, client_id: str, name: str, value):
        """store/update variable to redis server

        :param client_id: client_id to identify a specific client
        :type client_id: str
        :param name: key name
        :type name: str
        :param value: value
        :type value: [type]
        """
        if client_id in self._redis_server.scan_iter():
            state_dict = self.get_client_state_dict(client_id)
        else:
            state_dict = {}
        state_dict[name] = value
        self._redis_server.hmset(client_id, state_dict)
        self._redis_server.expire(name=client_id, time=self.key_lifetime)
    
    def get_variable(self, client_id: str, name: str):
        """get variable from redis server

        :param client_id: client_id to identify a specific client
        :type client_id: str
        :param name: key name
        :type name: str
        :return: retrieved value
        :rtype: [type]
        """
        state_dict = self.get_client_state_dict(client_id)
        # print(state_dict)
        return state_dict[name]
    
    def get_client_state_dict(self, client_id: str):
        return self._redis_server.hgetall(client_id)


