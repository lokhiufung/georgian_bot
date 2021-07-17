import copy

from friday.agent import CompositionalAgent
from friday.decorators import ensure_register_action
from friday.common.graph import KeyTermGraph
from friday.response.agent_response import AgentResponse
from friday.common.tokenizer import JiebaTokenizer


class MasterAgent(CompositionalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.worker_endpoints = cfg.worker_endpoints
        self.threshold = cfg.threshold
        self.keyterm_graph = KeyTermGraph(**cfg.keyterm_graph)
        self.tokenizer = JiebaTokenizer(**cfg.tokenizer)

    def _handle_dialog(self, text, client_id, is_analyze=False):
        text = self.process_keyterm(text, client_id)

        nlp_faq_response = self.request(
            endpoint=self.dl_endpoints['nlp_faq'],
            json={'text': text, 'is_anaylze': is_analyze},
            callback=self.handle_nlp_faq_response,
        )
        ##################
        # action logic
        top_answer = nlp_faq_response.answers[0]

        if nlp_faq_response.max_score > self.threshold:
            
            action_response = self.action.execute(top_answer.command, client_id=client_id, text=text)
        else:
            action_response = self.action.worker_agent_response(client_id=client_id, text=text)
            

        ##################
        
        ##################
        # polish action_response
        agent_response = AgentResponse(
            input_text=text,
            is_fallout=is_fallout,
            has_action=True if not is_fallout else False,
            text_answer=action_response.answer,
        )
        return agent_response
        ##################

    @ensure_register_action
    def get_voice_response(self, voice_request):
        """
        Foward voice_request to asr server. voice_request must contain the field client_id
        args:
            voice_request: dict, request obtained from server
        return:
            voice_response: AgentResponse  
        """
        # asr_server_response = self.request(
        #     endpoint=self.dl_endpoints['asr'],
        #     data=voice_request,
        #     callback=self.handle_asr_response,
        # )
        
        # # consider the dialog_history
        # input_text = self.dialog_history_flow(
        #     text=asr_server_response.transcription,
        #     client_id=asr_server_response.client_id,
        # )
        
        # # use faq as its core
        # faq_response = self.request(
        #     endpoint=self.dl_endpoints['nlp_faq'],
        #     data={'text': asr_server_response.transcription},
        #     callback=self.handle_nlp_embedding_response,
        # )

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

    def process_keyterm(self, text, client_id):
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

        
        