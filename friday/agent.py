from typing import Callable

import requests


class CompositionalAgent(object):
    """An agent that is built from compositional blocks: asr, tts, nlp_qa, ...etc"""
    def __init__(self, dl_endpoints: dict, is_voice: bool=True):
        self.is_voice
        self.validate_dl_endpoint(dl_endpoint)
        
        self.dl_endpoints = dl_endpoints

        # self._dialog_flow = dialog_flow
        # self._dialog_history_flow = dialog_history_flow
        self._sess = requests.Session() 
        
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
        pass

    def handle_tts_response(self, response):
        pass



