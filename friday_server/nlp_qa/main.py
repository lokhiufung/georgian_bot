from omegaconf import DictConfig
from flask import Flask, request

from friday.nlp.qa import ExtractiveQA


def create_nlp_qa_server(cfg: DictConfig):
    app = Flask('nlp_qa')

    k = cfg.k
    qa_model = ExtractiveQA(**cfg.qa_model)

    @app.route('/qa', methods=['POST'])
    def qa():
        data = request.json
        retrieved = qa_model.retrieve_top_k(data['text'], k=k)
        return retrieved

    return app

        
