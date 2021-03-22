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
        'model_path': 'deepset/sentence_bert',
        'document_store_mode': 'es',
        'es_index': 'faq-emsd',
        'device': 'cuda:0'
    }
} 

app = create_nlp_faq_server(DictConfig(cfg))

app.run()
