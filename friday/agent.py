
from friday.response.action_response import ActionResponse
from typing import Callable
import inspect

from omegaconf import DictConfig
# from hydra.utils import instantiate
import requests

from friday.decorators import ensure_register_action
from friday.response.server_response import *
from friday.action import ActionRequest


class CompositionalAgent(object):
    def __init__(self, cfg: DictConfig):
        """An agent that is built from compositional blocks: asr, tts, nlp_qa, ...etc

        :param cfg: DictConfig object for building blocks of the agent
        :type cfg: DictConfig
        """
        self.is_voice = cfg.is_voice
        self.validate_dl_endpoint(cfg.dl_endpoints)

        if cfg.use_redis:
            from friday.common.state import RedisClientStateStorage

            self.state_storage = RedisClientStateStorage(**cfg.state_storage)
        else:
            from friday.common.state import SimpleClientStateStorage
            from friday.common.dialog_history import SimpleDialogHistory

            self.state_storage = SimpleClientStateStorage()
            self.dialog_history = SimpleDialogHistory(**cfg.dialog_history)
        
        self.dl_endpoints = cfg.dl_endpoints
        self._sess = requests.Session()

        self.action = None  # prepare for assistant jobs

    def register_action_class(self, action_class):
        """attach and initiate an action object 

        :param action_class: customized action class
        :type action_class: [type]
        """
        self.action = action_class(agent=self)

    def validate_dl_endpoint(self, dl_endpoint: dict):
        if self.is_voice:
            for dl_server in ['asr', 'tts']:
                if dl_server.lower() not in dl_endpoint.keys():
                    raise ValueError('Invalid dl_endpoint: {}'.format(dl_enpoint))
            
    def request(self, endpoint: str, json: dict, data: dict=None, method: str='POST', callback=None) -> FridayServerResponse:
        """request for response from friday servers

        :param endpoint: server endpoint
        :type endpoint: str
        :param json: json data to be sent to the server
        :type json: dict
        :param data: data, defaults to None
        :type data: dict, optional
        :param method: request method, defaults to 'POST'
        :type method: str, optional
        :param callback: callback for handling the response from server, defaults to None
        :type callback: [type], optional
        :return: response object
        :rtype: FridayServerResponse
        """
        response = self._sess.request(method, endpoint, json=json, data=data)

        if callback:
            response = callback(response)
        return response

    def handle_asr_response(self, response: requests.Response) -> ASRServerResponse:
        """handle asr response from friday server

        :param response: response from friday server
        :type response: requests.Response
        :return: server response
        :rtype: ASRServerResponse
        """
        payload = response.json()
        return ASRServerResponse(**payload)

    def handle_tts_response(self, response: requests.Response) -> TTSServerResponse:
        """handle tts response from friday server

        :param response: response from friday server
        :type response: requests.Response
        :return: server response
        :rtype: TTSServerResponse
        """

        payload = response.json()
        return TTSServerResponse(**payload)

    def handle_nlp_qa_response(self, response: requests.Response) -> NLPQAServerResponse:
        """handle nlp qa response from friday server

        :param response: response from friday server
        :type response: requests.Response
        :return: server response
        :rtype: NLPQAServerResponse
        """
        payload = response.json()
        return NLPQAServerResponse(**payload)

    def handle_nlp_faq_response(self, response: requests.Response) -> NLPQAServerResponse:
        """handle nlp faq response from friday server

        :param response: response from friday server
        :type response: requests.Response
        :return: server response
        :rtype: NLPQAServerResponse
        """
        payload = response.json()
        return NLPQAServerResponse(**payload)

    def dialog_history_flow(self, text: str):
        """abstract method for controlling input text with dialog history

        :param text: [description]
        :type text: str
        :return: [description]
        :rtype: [type]
        """
        return text

    # def dialog_flow(self):
    #     """abstract method for aggregating the results from other friday servers and task servers and return the AgentResponse

    #     :raises NotImplementedError: [description]
    #     """
    #     raise NotImplementedError

    def _nlu(self, text, client_id=None, is_analyze=False) -> ActionRequest:
        # handle logic of nlp engine
        raise NotImplementedError 

    def nlu(self, text, client_id=None, is_analyze=False) -> ActionRequest:
        #####
        # check signature of self._handle_dialog
        signatures = inspect.signature(self._handle_dialog)
        required_signatures = ('text', 'client_id', 'is_analysis')  # remindMe: hard-coded

        missing_signatures = set(required_signatures) - set(signatures)
        if len(missing_signatures) > 0:
            raise ValueError('some of the required signatures are missing: {}'.format(missing_signatures))
        #####

        return self._nlu(text=text, client_id=client_id, is_analyze=is_analyze)

    @ensure_register_action
    def get_voice_response(self, voice_request, is_analyze=False):
        """
        """
        asr_response = self.request(
            endpoint=self.dl_endpoints['asr'],
            json=voice_request,
            callback=self.handle_asr_response,
        )
        
        agent_response = self.get_text_response(
            text=asr_response.transcription,
            client_id=asr_response.client_id,
            is_analyze=is_analyze,
        ) 
        
        tts_response = self.request(
            endpoint=self.dl_endpoints['tts'],
            json={'text': agent_response.text_answer, 'is_analyze': is_analyze},
            callback=self.handle_tts_response,
        )

        agent_response.voice_answer = tts_response.audio
        if is_analyze:
            agent_response.analysis = {
                'tts': tts_response.analysis,
                'asr': asr_response.analysis,
                **agent_response.analysis,
            }

        return agent_response

    @ensure_register_action
    def get_text_response(self, text, client_id, is_analyze=False):
        """
        """
        action_request = self._nlu(
            text=text,
            client_id=client_id,
            is_analyze=is_analyze
        )
        # action_response = self.action.execute(
        #     command=action_request.command,
        #     client_id=action_request.client_id,
        #     nlu_data=nlu_data,
        # )
        action_response = self.action.execute_v2(
            action_request=action_request
        )
        agent_response = AgentResponse(
            input_text=text,
            is_fallout=action_response.is_fallout,
            has_action=action_response.has_action_data if action_response else False,
            text_answer=action_response.text_answer,
            is_voice=self.is_voice,
            action_response=action_response,
            additional_answers=action_response.additional_answers,
        )
        return agent_response
    
    
