import os
import argparse

from omegaconf import OmegaConf, DictConfig

from friday_server.nlp_qa import create_nlp_qa_server


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, required=True)

#     return parser.parse_args()

# args = parse_args()

# cfg = OmegaConf.load(args.model_path)
cfg = {
    'server': {
        'name': 'nlp_qa-server'
    },
    'extractive_qa': {
        'model_path': 'deepset/roberta-base-squad2',
        'document_store_mode': 'in_memory',
        'document_dir': '/home/{}/Desktop/test_haystack_eng'.format(os.environ['USER'])
    }
} 

app = create_nlp_qa_server(DictConfig(cfg))

app.run()