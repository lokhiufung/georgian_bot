import copy

from friday.agent import CompositionalAgent
from friday.decorators import ensure_register_action
from friday.common.graph import KeyTermGraph
from friday.response.agent_response import AgentResponse


class MasterAgent(CompositionalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.worker_endpoints = cfg.worker_endpoints
        self.threshold = cfg.threshold
        self.keyterm_graph = KeyTermGraph(**cfg.keyterm_graph)

    @ensure_register_action
    def get_voice_response(self, voice_request):
        """
        Foward voice_request to asr server. voice_request must contain the field client_id
        args:
            voice_request: dict, request obtained from server
        return:
            voice_response: AgentResponse  
        """
        asr_server_response = self.request(
            endpoint=self.dl_endpoints['asr'],
            data=voice_request,
            callback=self.handle_asr_response,
        )
        
        # consider the dialog_history
        input_text = self.dialog_history_flow(
            text=asr_server_response.transcription,
            client_id=asr_server_response.client_id,
        )
        
        # use faq as its core
        faq_response = self.request(
            endpoint=self.dl_endpoints['nlp_faq'],
            data={'text': asr_server_response.transcription},
            callback=self.handle_nlp_embedding_response,
        )

        if doc['action'] == 'switch_domain' and doc['score'] > self.threshold:
            action_response = self.action.execute(command=doc['command'])
        elif self.state_storage.get('current_domain') is not None:
            worker_response = self.request(
                endpoint=self.worker_endpoints[self.state_storage['current_domain']],
                data={'text': text},
            )
        else:
            if retrieved['score'] > self.threshold:
                action_response = self.action.execute(command=doc['command'])
            else:
                bot_response = {
                    'fallout': True,
                    'answer': 'fallout' 
                }
        
        response = self.dialog_flow(input_text)

        voice_reponse = self.request(
            endpoint=self.dl_endpoints['tts'],
            data=retrieved,
            callback=self.handle_tts_response,
        )
        return voice_reponse

    def dialog_history_flow(self, text, client_id):
        tokens = self.tokenizer.text_to_tokens(text)
        keyterms = self.token_filter.filter(tokens)
        
        # in case text do not contain any keyterms, don't update keyterm_pool and don't consider keyterm_pool
        if keyterms:
            # 1. update keyterm_pool by the relation of new input keyterms
            keyterm_pool = copy.deepcopy(self.state_dict[client_id].keyterm_pool)  # consider different clients
            for keyterm_in_pool in keyterm_pool:
                neighbors = self.keyterm_graph.get_neighbors_of(keyterm_in_pool)
                if not set(neighbors).intersection(set(keyterms)):
                    self.keyterm_pool.pop(keyterm_in_pool)
            # 2. join keyterms in keyterm_pool
            keyterm_string = ' '.join(self.keyterm_pool)
        
            # 3. join keyterms with input_text
            text = ' '.join([keyterm_string, text])
        return text

    def dialog_flow(self, ) -> AgentResponse:
        pass
        
        