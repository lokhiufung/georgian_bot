from dataclasses import asdict
import time

from omegaconf import DictConfig
from flask import Flask, request
from hydra.utils import instantiate

from friday.agent import CompositionalAgent


def create_agent_server(server_cfg: DictConfig, agent: CompositionalAgent):
    """helper function for creating a flask server for agent

    :param server_cfg: server_cfg
    :type server_cfg: DictConfig
    :param agent: friday agent object
    :type agent: CompositionalAgent
    :return: Flask server
    :rtype: Flask
    """
    # agent = instantiate(agent_server_cfg.agent)  # must provide a _target_ in cfg

    app = Flask(server_cfg.name)

    @app.route('/voice_bot', methods=['POST'])
    def voice_bot():
        data = request.get_json()
        client_id = data.get('client_id', '')
        is_analyze = data.get('is_analyze', False)

        start = time.perf_counter()
        response = agent.get_voice_response(
            voice_request=data,
            is_analyze=is_analyze,
        )
        end = time.perf_counter()
        
        payload = {
            'client_id': client_id,
            'time': end - start,
            **asdict(response)
        }
        return payload
    
    @app.route('/text_bot', methods=['POST'])
    def text_bot():
        data = request.get_json()
        client_id = data.get('client_id', '')

        start = time.perf_counter()
        response = agent.get_text_response(
            text=data['text'],
            client_id=client_id,
        )
        end = time.perf_counter()
        
        payload = {
            'client_id': client_id,
            'time': end - start,
            **asdict(response)
        }
        return payload
    
    return app
