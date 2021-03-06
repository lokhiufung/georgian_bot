import os
import argparse

from omegaconf import OmegaConf, DictConfig

from friday_server.nlp_qa import create_nlp_qa_server


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, required=True)

#     return parser.parse_args()

# args = parse_args()

cfg = OmegaConf.load('./configs/nlp_qa_server_config.yaml')
# cfg = {
#     'server': {
#         'name': 'nlp_qa-server'
#     },
#     'extractive_qa': {
#         'model_path': 'uer/roberta-base-chinese-extractive-qa',
#         'document_store_mode': 'es',
#         'es_index': 'drcd_test',
#         'device': 'cpu'
#         # 'document_dir': '/home/{}/Desktop/test_haystack_eng'.format(os.environ['USER'])
#     }
# } 

app = create_nlp_qa_server(DictConfig(cfg))


if __name__ == '__main__':
    app.run()
