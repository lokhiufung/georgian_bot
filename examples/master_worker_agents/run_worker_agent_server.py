from omegaconf import DictConfig

from worker_agent import WorkerAgent
from worker_action import WorkerAction

from friday_server.agent import create_agent_server

agent_cfg = {
    'dl_endpoints': {
        'asr': 'http:///transcribe',
        'tts': 'http:///synthesize',
        'nlp_qa': 'http:///qa',
        'nlp_faq': 'http:///faq',
    },
    'is_voice': True,
    'dialog_history': {
        'capacity': 10
    },
    'threshold': 0.85,
    'default_answers': {
        'fallout_answer': '對唔住我唔明你問咩',
        'clarifying_question': '你喺咪問緊呢啲問題'
    }
}

server_cfg = {
    'name': 'worker_agent-server'
}

agent = WorkerAgent(cfg=DictConfig(agent_cfg))
agent.register_action_class(WorkerAction)


app = create_agent_server(DictConfig(server_cfg), agent)

if __name__ == '__main__':
    app.run()
