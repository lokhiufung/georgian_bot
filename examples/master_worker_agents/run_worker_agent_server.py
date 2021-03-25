from omegaconf import DictConfig

from worker_agent import WorkerAgent
from worker_action import WorkerAction

from friday_server.agent import create_agent_server

agent_cfg = {
    'dl_endpoints': {
        'nlp_faq': 'http://localhost:3001/faq'
    },
    'is_voice': False,
    'dialog_history': {
        'capacity': 10
    },
    'threshold': 0.6,
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

app.run()
