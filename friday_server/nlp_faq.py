import time

from omegaconf import DictConfig
from flask import Flask, request

from friday.nlp.qa import EmbeddingQA


def create_nlp_faq_server(faq_server_cfg: DictConfig):
    """helper function for creating a flask server for nlp faq

    :param faq_server_cfg: contain cfg for server and faq
    :type faq_server_cfg: DictConfig
    :return: Flask server
    :rtype: Flask
    """
    app = Flask('nlp_faq')

    k = faq_server_cfg.k
    faq_model = EmbeddingQA(**faq_server_cfg.embedding_qa)

    @app.route('/faq', methods=['POST'])
    def qa():
        request_time = time.perf_counter()

        data = request.get_json()
        client_id = data.get('client_id', '')
        is_analyze = data.get('is_analyze', False)

        k = data.get('k', 1)  # return only the top result if not specify k
        model_start = time.perf_counter()
        answers = faq_model.retrieve_top_k(data['text'], k=k)
        mdoel_end = time.perf_counter()
        model_time = model_end - model_start

        payload = {
            'client_id': client_id,
            'time': model_time,
            'answers': answers,
            'max_score': max([answer['score'] for answer in answers]) if len(answers) > 0 else 0.0,
            'size': len(answers), 
        }
        request_end = time.perf_counter()
        if is_analyze:
            analysis = {
                'latency': {
                    'model_time': model_time,
                    'request_time': request_start - request_end,
                }
            }
            payload['analysis'] = analysis
        return payload

    return app

        
