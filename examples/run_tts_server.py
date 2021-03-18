from omegaconf import OmegaConf, DictConfig

from friday_server.tts import create_tts_server


tts_server_cfg = OmegaConf.load('./base_configs/friday_server/tts_server.yaml')

app = create_tts_server(tts_server_cfg)

app.run()
