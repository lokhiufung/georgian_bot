from friday.agent import CompositionalAgent
from friday.decorators import ensure_register_task


class MasterAgent(CompositionalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.threshold = cfg.threshold

    @ensure_register_task
    def get_voice_response(self, signal, client_id):
        transcript = self.request(
            endpoint=self.dl_endpoints['asr'],
            data=data,
            callback=self.handle_asr_response,
        )
        
        # use faq as its core
        retrieved = self.request(
            endpoint=self.dl_endpoints['nlp_faq'],
            data={'transcript': transcript}
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
            
            
         
        
        