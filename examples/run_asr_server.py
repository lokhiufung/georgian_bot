from omegaconf import OmegaConf, DictConfig

from friday_server.asr import create_asr_server


# asr_server_cfg = OmegaConf.load('./configs/asr_server_config.yaml')
asr_server_cfg = {
    'recognizer': {
        'asr_model': '/media/lokhiufung/Storage/experiments/hey-friday/asr/recorderdata-hbpe_2021-04-16.nemo',
        'use_lm': False,
        'device': 'cuda',
        'model_type': 'ctc_hbpe',
    },
    'server': {
        'name': 'asr-server',
        'constants': {
            'sample_rate': 16000,
            'nb_channels': 1
        }
    }
}

# asr_server_cfg = {
#     'recognizer': {
#         'transformer_model': '/media/lokhiufung/Storage/experiments/hey-friday/archives/wav2vec2-xlsr_transformer_common-voice_2020-05-10/checkpoint-23600',
#         'device': 'cuda'
#     },
#     'server': {
#         'name': 'asr-server',
#         'constants': {
#             'sample_rate': 16000,
#             'nb_channels': 1
#         }
#     }
# }

app = create_asr_server(DictConfig(asr_server_cfg), use_transformer=False)

if __name__ == '__main__':
    app.run()

