
from typing import Callable

from omegaconf import DictConfig
# from hydra.utils import instantiate
import requests


class CompositionalAgent(object):
    """Base class: An agent that is built from compositional blocks: asr, tts, nlp_qa, ...etc"""
    def __init__(self, cfg: DictConfig):
        self.is_voice = cfg.is_voice
        self.validate_dl_endpoint(cfg.dl_endpoint)

        if cfg.use_redis:
            from friday.common.state import RedisStateStorage

            self.state_storage = RedisStateStorage(**cfg.state_storage)
        else:
            from friday.common.state import SimpleStateStorage
            from friday.common.dialog_history import InMemoryDialogHistory

            self.state_storage = SimpleStateStorage(**cfg.state_dict)
            self.dialog_history = SimpleDialogHistory(**cfg.dialog_history)
        
        self.dl_endpoints = cfg.dl_endpoints
        self._sess = requests.Session()

        self.task = None  # prepare for assistant jobs

        
    def validate_dl_endpoint(self, dl_endpoint):
        if self.is_voice:
            for dl_server in ['asr', 'tts']:
                if dl_server.lower() not in dl_endpoint.keys():
                    raise ValueError('Invalid dl_endpoint: {}'.format(dl_enpoint))
            
    def request(self, endpoint, data, method, callback=None):
        response = self._sess.request(method, endpoint, data=data)

        if callback:
            response = callback(response)
        return response

    def handle_asr_response(self, response):
        raise NotImplementedError('handle_asr_response is required for is_voice=True')

    def handle_tts_response(self, response):
        raise NotImplementedError('handle_tts_response is required for is_voice=True')

    def dialog_flow(self):
        raise NotImplementedError

    def dialog_history_flow(self):
        raise NotImplementedError


