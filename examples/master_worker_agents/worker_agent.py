import numpy as np

from friday.agent import CompositionalAgent
from friday.decorators import ensure_register_task
from friday.response.agent_response import AgentResponse



def gap_mean_filter(answers, n_std: float=1.0):
    scores = [answer['score'] for answer in answers]
    gaps = [next_score - score for next_score, score in zip(score[1:], score[:-1])]
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps, ddof=1)

    if gaps[0] > mean_gap + n_std * std_gap:
        return [answers[0]]        
    return answers  


def gap_iqr_filter(answers, n_iqr: float=1.5):
    scores = [answer['score'] for answer in answers]
    gaps = [next_score - score for next_score, score in zip(score[1:], score[:-1])]
    q75 = np.quantile(gaps, q=0.75)
    q25 = np.quantile(gaps, q=0.25)
    iqr = q75 - q25

    if gaps[0] > q75 + n_iqr * iqr:
        return [answers[0]]        
    return answers  
    

class WorkerAgent(CompositionalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.threshold = cfg.threshold
        self.default_answers = cfg.default_answers

    @ensure_register_task
    def get_text_response(self, text, client_id):
        # use faq as its core
        nlp_faq_response = self.request(
            endpoint=self.dl_endpoints['nlp_faq'],
            data={'text': text},
            callback=self.handle_nlp_faq_response,
        )

        if nlp_faq_response.max_score > self.threshold:
            task_response = self.task.execute(command=retrieved['command'])
            return self.dialog_flow(text, nlp_faq_response, task_response)
        else:
            nlp_qa_response = self.request(
                endpoint=self.dl_endpoints['nlp_qa'],
                data={'text': text},
                callback=self.handle_nlp_qa_response,
            )
            return self.dialog_flow(text, nlp_faq_response, nlp_qa_response)
    
    def dialog_flow(self, input_text, nlp_faq_response, nlp_qa_response=None, task_response=None):
        is_fallout = False
        additional_response = None
        if nlp_qa_response:
            answers = [
                answer for answer in gap_mean_filter(nlp_qa_response.answers, n_std=1.0) if answer['score'] > self.threshold
            ]  # filter answers by gaps and output only the answers that have scores greater than self.threshold
            if len(answers) > 1:
                additional_response = answers
                text_answer = self.default_answers.clarifying_quesiton
            elif len(answers) == 1:
                text_answer = answers['answer']
            else:
                text_answer = self.default_answers.fallout_answer
        else:
            answers = [
                answer for answer in gap_mean_filter(nlp_faq_response.answers, n_std=1.0) if answer['score'] > self.threshold
            ]  # filter answers by gaps and output only the answers that have scores greater than self.threshold
            if len(answers) > 1:
                additional_response = answers
                text_answer = self.default_answers.clarifying_quesiton
            elif len(answers) == 1:
                text_answer = answers['answer']
            else:
                is_fallout = True
                text_answer = self.default_answers.fallout_answer
        return AgentResponse(
            input_text=input_text,
            is_fallout=is_fallout,
            has_task=True if task_response else False,
            text_answer=text_answer,
            is_voice=self.is_voice,
            task_response=task_response,
            additional_response=addtional_response
        )

