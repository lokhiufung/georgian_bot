from omegaconf import DictConfig

from friday_server.agent import create_agent_server

from qa_agent import QAAgent


agent_cfg = {
    'dl_endpoints': {
        'nlp_qa': 'http://localhost:5000/qa'
    },
    'is_voice': False,
    'state_dict': {},
    'dialog_history': {
        'capacity': 10
    },
    'threshold': 0.1
}

server_cfg = {
    'name': 'nlp_qa_agent-server'
}

agent = QAAgent(cfg=DictConfig(agent_cfg))
app = create_agent_server(
    server_cfg=DictConfig(server_cfg),
    agent=agent,
)

app.run(port=3000)