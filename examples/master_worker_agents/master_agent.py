from friday.agent import CompositionalAgent


class MasterAgent(CompositionalAgent):
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

        voice_reponse = self.request(
            endpoint=self.dl_endpoints['tts'],
            data=retrieved,
            callback=self.handle_tts_response,
        )

    def handle_nlp_faq_response(self, response):
        data = response.json
        score, doc = data['score'], data['doc']

        if score > self.threshold:
            task_response = self.task.execute(command=doc['command'])
        else:
            task_response = self.task.execute()
         
        
        