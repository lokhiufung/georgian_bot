from friday.agent import CompositionalAgent


class QAAgent(CompositionalAgent):
    def get_text_response(self, text):
        qa_response = self.request(
            endpoint=self.dl_endpoints['nlp_qa'],
            data={'text': text},
            callback=self.handle_nlp_qa_response
        )

        qa_response
        
        return qa_response
    
    def dialog_flow(self, input_text, answer, context):
        return {
            'input_text': input_text,
            'answer': answer,
            'context': context,
        }
