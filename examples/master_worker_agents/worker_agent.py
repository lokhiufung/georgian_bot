from friday.agent import CompositionalAgent
from friday.decorators import ensure_register_task


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

            if qa_answer['answer'] == 'no answer':
                bot_response = {
                    'fallout': True,
                    'answer': 'fallout'
                }
            else:
                bot_response = qa_answer
                
        return self.dialog_flow(task_response)
        