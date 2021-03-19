
from typing import Callable

from omegaconf import DictConfig
# from hydra.utils import instantiate
import requests

from friday.response.server_response import *


class CompositionalAgent(object):
    """Base class: An agent that is built from compositional blocks: asr, tts, nlp_qa, ...etc"""
    def __init__(self, cfg: DictConfig):
        self.is_voice = cfg.is_voice
        self.validate_dl_endpoint(cfg.dl_endpoint)

        if cfg.use_redis:
            from friday.common.state import RedisClientStateStorage

            self.state_storage = RedisStateStorage(**cfg.state_storage)
        else:
            from friday.common.state import SimpleClientStateStorage
            from friday.common.dialog_history import SimpleDialogHistory

            self.state_storage = SimpleStateStorage(cfg.state_dict)
            self.dialog_history = SimpleDialogHistory(**cfg.dialog_history)
        
        self.dl_endpoints = cfg.dl_endpoints
        self._sess = requests.Session()

        self.task = None  # prepare for assistant jobs

        
    def validate_dl_endpoint(self, dl_endpoint):
        if self.is_voice:
            for dl_server in ['asr', 'tts']:
                if dl_server.lower() not in dl_endpoint.keys():
                    raise ValueError('Invalid dl_endpoint: {}'.format(dl_enpoint))
            
    def request(self, endpoint: str, json: dict, data: dict=None, method: str='POST', callback=None) -> FridayServerResponse:
        """
        request for response from friday servers
        """
        response = self._sess.request(method, endpoint, json=json, data=data)

        if callback:
            response = callback(response)
        return response

    def handle_asr_response(self, response: requests.Response):
        """
        return:
            payload: dict
            {"transcription": str, **}
        """
        payload = response.json()
        return ASRServerResponse(**payload)

    def handle_tts_response(self, response: requests.Response):
        payload = response.json()
        """
        return:
            payload: dict
            {"audio": str, **}
        """
        return TTSServerResponse(**payload)

    def handle_nlp_qa_response(self, response: requests.Response):
        """
        return:
            payload: dict
            {"answers": [{"answer", ...},], "max_scroe": float, "size": int**}
        """
        payload = response.json()
        return NLPQAServerResponse(**payload)

    def dialog_history_flow(self, text):
        """control input text with state objects"""
        return text

    def dialog_flow(self):
        """aggregate the results from other friday servers and task servers and return the AgentResponse"""
        raise NotImplementedError

    def get_voice_response(self):
        raise NotImplementedError

    def get_text_response(self):
        raise NotImplementedError
        