from omegaconf import DictConfig
from flask import Flask, request

from friday.nlp.qa import ExtractiveQA


def create_nlp_qa_server(qa_cfg, server_cfg):
    app = Flask('nlp_qa')

    k = server_cfg.k
    qa_model = ExtractiveQA(**cfg.qa_cfg)

    @app.route('/qa', methods=['POST'])
    def qa():
        data = request.get_json()
        retrieved = qa_model.retrieve_top_k(data['text'], k=k)
        return retrieved

    return app

        
