from friday.agent import CompositionalAgent
from friday.response.server_response import AgentResponse, NLPQAServerResponse


class QAAgent(CompositionalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.threshold = cfg.threshold
        self.k = cfg.k

    def get_text_response(self, text, client_id):
        qa_response = self.request(
            endpoint=self.dl_endpoints['nlp_qa'],
            json={'text': text, 'k': self.k},
            callback=self.handle_nlp_qa_response
        )
        
        return self.dialog_flow(text, qa_response)
    
    def dialog_flow(self, input_text: str, qa_response: NLPQAServerResponse):
        if qa_response.max_score > self.threshold:
            is_fallout = False
            text_answer = qa_response.answers[0]['context']
        else:
            is_fallout = True
            text_answer = 'no answer'
        return AgentResponse(
            input_text=input_text,
            is_fallout=True,
            text_answer=text_answer,
            has_task=False,
        )