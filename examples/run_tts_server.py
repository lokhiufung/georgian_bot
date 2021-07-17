from omegaconf import DictConfig

from friday_server.tts import create_tts_server


# tts_server_cfg = OmegaConf.load('./base_configs/friday_server/tts_server.yaml')
tts_server_cfg = {
    'synthesizer': {
        'text2mel_model': '/home/lokhiufung/Downloads/nemo_experiments/Tacotron2/2021-04-30_01-28-00/checkpoints/Tacotron2.nemo',
        'mel2audio_model': '/home/lokhiufung/pretrained_model/nemottsmodels_1.0.0a5/WaveGlow-22050Hz.nemo',
        'mel2audio_model_type': 'waveglow',
        'device': 'cuda:0'
    },
    'server': {
        'name': 'tts-server',
        'lang': 'cantonese',
        'constants': {
            'sample_rate': 22050,
            'denoiser_strength': 1.0,
            'post_processing': False
        }
    }
}
app = create_tts_server(DictConfig(tts_server_cfg))


if __name__ == '__main__':
    app.run()
