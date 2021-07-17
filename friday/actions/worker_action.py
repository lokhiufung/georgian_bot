from friday.agents.master_worker_agents.worker_agent_v2 import gap_mean_filter
from friday.action import Action
from friday.response.action_response import ActionResponse
from friday.agents.utils import gap_mean_filter


class WorkerAction(Action):
    def __init__(self, agent, nlp_qa_endpoint, nlp_qa_threshold, clarifying_answer, fallout_answer):
        super().__init__(agent=agent)
        self.clarifying_answer = clarifying_answer
        self.fallout_answer = fallout_answer
        self.nlp_qa_endpoint = nlp_qa_endpoint
        self.nlp_qa_threshold = nlp_qa_threshold
        
    def fallout_response(self, client_id, text, nlu_data):
        nlp_qa_response = self.agent.request(
            endpoint=self.nlp_qa_endpoint,
            json={'text': text},
            callback=self.agent.handle_nlp_qa_response,
        )
        if nlp_qa_response.max_score > self.nlp_qa_threshold:
            return ActionResponse(
                action_name='fallout_response',
                text_answer=nlp_qa_response.answers[0]['answer'],
                has_action_data=True,
                action_data=nlp_qa_response.answers,
            )
        else:
            return ActionResponse(
                action_name='fallout_response',
                text_answer=self.fallout_answer,
                has_action_data=True,
                action_data=nlp_qa_response.answers,
            )
    
    def clarifying_response(self, client_id: str, text, nlu_data):
        return ActionResponse(
            action_name='clarifying_response',
            text_answer=self.clarifying_answer,
            has_action_data=True,
            action_data=nlu_data['nlu_data']
        )
