import json

import numpy as np
from flask import Flask, request, abort
import annoy


def create_annoy_search_server(annoy_search_server_cfg):
    app = Flask(annoy_search_server_cfg.server.name)
    
    dim = annoy_search_server_cfg.annoy_search.dim
    n_trees = annoy_search_server_cfg.annoy_search.n_trees
    tree = annoy.AnnoyIndex(f=dim, metric='angular')
    buffer = []

    # seeder
    ############
    # DOCS = [
    #     {'question': 'How old are you?', 'answer': 'I am 21 years old.', 'vector': np.random.randn(512).tolist()},
    #     {'question': 'What is your name?', 'answer': 'My name is Friday.', 'vector': np.random.randn(512).tolist()},
    # ]
    with open(annoy_search_server_cfg.annoy_search.seeder_doc_filepath, 'r') as f:
        DOCS = json.load(f)
        
    buffer += DOCS

    for i, item in enumerate(buffer):
        tree.add_item(i, item['vector'])

    tree.build(n_trees)    
    ############


    def ensure_have_item(func):
        def wrapper(*args, **kwargs):
            if tree.get_n_items() < 1:
                abort(500, 'No items in the tree.')
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper


    @app.route('/query', methods=['POST'])
    @ensure_have_item
    def query():
        data = request.get_json()

        indexes, distances = tree.get_nns_by_vector(
            vector=data['vector'],
            n=data['n'],
            include_distances=True
        )
        results = [buffer[index] for index in indexes]
        
        docs = [{'score': distance, **item} for item, distance in zip(results, distances)]
        return {
            'message': 'ok',
            'result': {
                'max_score': max(distances),
                'docs': sorted(docs, key=lambda x: x['score']),    
            }
        
        }


    @app.route('/write', methods=['POST'])
    @ensure_have_item
    def write():
        """Only for writing single itme into the tree"""
        global tree  # update a global variable

        data = request.get_json()
        
        buffer.append(data['doc'])  # write item to the buffer

        tree = annoy.AnnoyIndex(f=dim, metric='angular')
        tree.build(n_trees)

        return {
            'message': 'ok',
            'result': {}     
        }


    @app.route('/', methods=['GET'])
    def home():

        return {
            'message': 'ok',
            'result': {
                'n_items': tree.get_n_items(),
            }     
        }

    return app