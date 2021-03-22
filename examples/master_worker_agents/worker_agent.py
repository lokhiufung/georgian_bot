from friday.agent import CompositionalAgent
from friday.decorators import ensure_register_task
from friday.response.agent_response import AgentResponse


class WorkerAgent(CompositionalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.threshold = cfg.threshold

    @ensure_register_task
    def get_text_response(self, text, client_id):
        # use faq as its core
        nlp_faq_response = self.request(
            endpoint=self.dl_endpoints['nlp_faq'],
            data={'text': text},
            callback=self.handle_nlp_faq_response,
        )

        if nlp_qa_response.max_score > self.threshold:
            task_response = self.task.execute(command=retrieved['command'])
        else:
            nlp_qa_response = self.request(
                endpoint=self.dl_endpoints['nlp_qa'],
                data={'text': text},
                callback=self.handle_nlp_qa_response,
            )
                
        return self.dialog_flow(text, nlp_qa_response, nlp_faq_response, task_response)
    
    def dialog_flow(self, input_text, nlp_qa_response, nlp_faq_response, task_response):
        return AgentResponse(
            input_text=input_text,
            is_fallout=False,
            has_task=True if task_response else False,
            text_answer=nlp_qa_response.answer,
            is_voice=self.is_voice,
            voice_answer=nlp_qa_response.answer,
            task_response=task_response,
        )