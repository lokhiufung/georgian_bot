from omegaconf import DictConfig

from friday_server.asr import create_asr_server


# asr_server_cfg = OmegaConf.load('./example_configs/asr_server.yaml')
asr_server_cfg = {
    'recognizer': {
        'asr_model': '/home/einstein/friday/pretrained_models/recorderdata-quartnet15x5.nemo',
        'use_lm': False,
        'device': 'cpu'
    },
    'server': {
        'name': 'asr-server',
        'constants': {
            'sample_rate': 16000,
            'nb_channels': 1
        }
    }
}
app = create_asr_server(DictConfig(asr_server_cfg))

app.run()
