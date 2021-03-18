from omegaconf import OmegaConf, DictConfig

from friday_server.asr import create_asr_server


asr_server_cfg = OmegaConf.load('./example_configs/asr_server.yaml')

app = create_asr_server(asr_server_cfg)

app.run()
