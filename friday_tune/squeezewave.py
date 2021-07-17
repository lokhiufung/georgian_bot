# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf, DictConfig

import torch
import pytorch_lightning as pl
import nemo.collections.tts as nemo_tts
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_name='squeezewave.yaml', config_path='base_config')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    if cfg.get('ckpt', None):
        device = 'cuda:0' if  cfg.trainer.gpus > 1 else 'cpu'  
        model = nemo_tts.models.SqueezeWaveModel.restore_from(cfg.ckpt, map_location=torch.device(device))

        model.setup_optimization(DictConfig(cfg.model.optim))
        model.setup_training_data(DictConfig(cfg.model.train_ds))
        model.setup_validation_data(DictConfig(cfg.model.validation_ds))
    else:
        model = nemo_tts.models.SqueezeWave(cfg=cfg.model)
        
    epoch_time_logger = LogEpochTimeCallback()
    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    trainer.callbacks.extend([epoch_time_logger, lr_logger])
    trainer.fit(model)


if __name__ == '__main__':
    main()