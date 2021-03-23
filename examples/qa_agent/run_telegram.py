import sys
from omegaconf import DictConfig

from friday_server.telegram_agent import create_telegram_bot_agent_server

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
    'name': 'nlp_qa_agent-server',
    'default_answers': {
        'welcome_message': 'Ask me any questions in deep learning!'
    },
    'telegram': {
        'token': sys.argv[1]
    }
}

agent = QAAgent(cfg=DictConfig(agent_cfg))

app = create_telegram_bot_agent_server(DictConfig(server_cfg), agent)

app.run(port=3000)
