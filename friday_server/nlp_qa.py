import time

from omegaconf import DictConfig
from flask import Flask, request

from friday.nlp.qa import ExtractiveQA


def create_nlp_qa_server(qa_server_cfg: DictConfig):
    """helper function for creating a flask server for nlp qa

    :param qa_server_cfg: contain server and nlp qa
    :type qa_server_cfg: DictConfig
    :return: Flask server
    :rtype: Flask
    """
    app = Flask(qa_server_cfg.server.name)

    qa_model = ExtractiveQA(**qa_server_cfg.extractive_qa)

    @app.route('/qa', methods=['POST'])
    def qa():
        request_start = time.perf_counter()

        data = request.get_json()
        client_id = data.get('client_id', '')
        is_analyze = data.get('is_analyze', False)

        k = data.get('k', 1)  # return only the top result if not specify k
        model_start = time.perf_counter()
        answers = qa_model.retrieve_top_k(data['text'], k=k)
        model_end = time.perf_counter()

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
                    'request_time': request_end - request_start 
                }
            }
            payload['analysis'] = analysis
        return payload

    return app

        
