import time

from omegaconf import DictConfig
from flask import Flask, request

from friday.nlp.qa import EmbeddingQA


def create_nlp_faq_server(faq_server_cfg):
    app = Flask('nlp_faq')

    k = faq_server_cfg.k
    faq_model = EmbeddingQA(**faq_server_cfg.embedding_qa)

    @app.route('/faq', methods=['POST'])
    def qa():
        data = request.get_json()
        client_id = data.get('client_id', '')
        k = data.get('k', 1)  # return only the top result if not specify k
        start = time.perf_counter()
        answers = faq_model.retrieve_top_k(data['text'], k=k)
        end = time.perf_counter()

        payload = {
            'client_id': client_id,
            'time': end - start,
            'answers': answers,
            'max_score': max([answer['score'] for answer in answers]),
            'size': len(answers), 
        }
        return payload

    return app

        
