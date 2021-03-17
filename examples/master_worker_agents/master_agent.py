import copy

from friday.agent import CompositionalAgent
from friday.decorators import ensure_register_task
from friday.common.graph import KeyTermGraph


class MasterAgent(CompositionalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.threshold = cfg.threshold
        self.keyterm_graph = KeyTermGraph(**cfg.keyterm_graph)

    @ensure_register_task
    def get_voice_response(self, signal, client_id):
        transcript = self.request(
            endpoint=self.dl_endpoints['asr'],
            data=data,
            callback=self.handle_asr_response,
        )
        
        # consider the dialog_history
        input_text = self.dialog_history_flow(transcript)
        # use faq as its core
        retrieved = self.request(
            endpoint=self.dl_endpoints['nlp_faq'],
            data={'transcript': transcript},
            callback=self.handle_nlp_embedding_response,
        )

        if doc['action'] == 'switch_domain' and doc['score'] > self.threshold:
            task_response = self.task.execute(command=doc['command'])
        elif self.state_storage.get('current_domain') is not None:
            worker_response = self.send_to_worker(input_text)
        else:
            if retrieved['score'] > self.threshold:
                task_response = self.task.execute(command=doc['command'])
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

    def handle_nlp_faq_response(self, response):
        data = response.json
        retrieved = {'score': data['score'], **data['doc']}
        return retrieved

    def dialog_history_flow(self, text):
        tokens = self.tokenizer.text_to_tokens(text)
        keyterms = self.token_filter.filter(tokens)
        
        # in case text do not contain any keyterms, don't update keyterm_pool and don't consider keyterm_pool
        if keyterms:
            # 1. update keyterm_pool by the relation of new input keyterms
            keyterm_pool = copy.deepcopy(self.keyterm_pool)
            for keyterm_in_pool in keyterm_pool:
                neighbors = self.keyterm_graph.get_neighbors_of(keyterm_in_pool)
                if not set(neighbors).intersection(set(keyterms)):
                    self.keyterm_pool.pop(keyterm_in_pool)
            # 2. join keyterms in keyterm_pool
            keyterm_string = ' '.join(self.keyterm_pool)
        
            # 3. join keyterms with input_text
            text = ' '.join([keyterm_string, text])
        return text



            
         
        
        