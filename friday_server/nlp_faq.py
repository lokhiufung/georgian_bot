from omegaconf import DictConfig
from flask import Flask, request

from friday.nlp.qa import EmbeddingQA


def create_nlp_faq_server(faq_cfg, server_cfg):
    app = Flask('nlp_faq')

    k = server_cfg.k
    faq_model = EmbeddingQA(**cfg.qa_cfg)

    @app.route('/faq', methods=['POST'])
    def qa():
        data = request.get_json()
        retrieved = faq_model.retrieve_top_k(data['text'], k=k)
        return retrieved

    return app

        
