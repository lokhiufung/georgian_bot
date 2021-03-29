from omegaconf import DictConfig

from friday_server.tts import create_tts_server


# tts_server_cfg = OmegaConf.load('./base_configs/friday_server/tts_server.yaml')
tts_server_cfg = {
    'synthesizer': {
        'text2mel_model': '',
        'mel2audio_model': '',
        'device': 'cpu'
    },
    'server': {
        'name': 'tts-server',
        'lang': 'cantonese',
        'constants': {
            'sample_rate': 22050,
            'denoiser_strength': 1.0
        }
    }
}
app = create_tts_server(DictConfig(tts_server_cfg))


if __name__ == '__main__':
    app.run()
