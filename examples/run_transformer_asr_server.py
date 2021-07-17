from omegaconf import OmegaConf, DictConfig

from friday_server.asr import create_asr_server


asr_server_cfg = OmegaConf.load('./examples/configs/transformer_asr_server_config.yaml')
# asr_server_cfg = {
#     'recognizer': {
#         'asr_model': '/home/lokhiufung/pretrained_model/recorderdata-quartnet15x5.nemo',
#         'use_lm': False,
#         'device': 'cpu'
#     },
#     'server': {
#         'name': 'asr-server',
#         'constants': {
#             'sample_rate': 16000,
#             'nb_channels': 1
#         }
#     }
# }

app = create_asr_server(DictConfig(asr_server_cfg), use_transformer=True)


if __name__ == '__main__':
    app.run()

