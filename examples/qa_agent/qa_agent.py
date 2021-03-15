from friday.agent import CompositionalAgent


class QAAgent(CompositionalAgent):
    def get_text_response(self, text, client_id):
        retrieved = self.request(
            endpoint=self.dl_endpoints['nlp_qa'],
            data={'text': text},
            callback=self.handle_nlp_qa_response
        )
        return retrieved
    
    def handle_nlp_qa_response(self, response):
        data = response.data
        return data
    
    def dialog_flow(self, input_text, answer, context):
        return {
            'input_text': input_text,
            'answer': answer,
            'context': context,
        }
