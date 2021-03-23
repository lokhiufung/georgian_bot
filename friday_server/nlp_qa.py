import time

from omegaconf import DictConfig
from flask import Flask, request

from friday.nlp.qa import ExtractiveQA


def create_nlp_qa_server(qa_server_cfg: DictConfig):
    app = Flask(qa_server_cfg.server.name)

    qa_model = ExtractiveQA(**qa_server_cfg.extractive_qa)

    @app.route('/qa', methods=['POST'])
    def qa():
        data = request.get_json()
        client_id = data.get('client_id', '')
        k = data.get('k', 1)  # return only the top result if not specify k
        start = time.perf_counter()
        answers = qa_model.retrieve_top_k(data['text'], k=k)
        end = time.perf_counter()

        payload = {
            'client_id': client_id,
            'time': end - start,
            'answers': answers,
            'max_score': max([answer['score'] for answer in answers]) if len(answers) > 0 else 0.0,
            'size': len(answers), 
        }
        return payload

    return app

        
