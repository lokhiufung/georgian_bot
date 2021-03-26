import os
import argparse

from omegaconf import OmegaConf, DictConfig

from friday_server.nlp_faq import create_nlp_faq_server


cfg = {
    'server': {
        'name': 'nlp_faq-server',
        'k': 5
    },
    'embedding_qa': {
        'model_path': '/home/lokhiufung/projects/hey-assist/pretrained_models/finetune-2020-11-24_04-09-29',
        'document_store_mode': 'es',
        'es_index': 'add_two_test',
        'device': 'cpu',
        'model_format': 'sentence_transformers'
    }
} 

app = create_nlp_faq_server(DictConfig(cfg))

if __name__ == '__main__':
    app.run(port=3001)
