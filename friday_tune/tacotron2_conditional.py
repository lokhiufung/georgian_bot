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
import os

import torch
import pytorch_lightning as pl
from ruamel.yaml import YAML
from omegaconf import OmegaConf

from nemo.collections.common.callbacks import LogEpochTimeCallback
# from nemo.collections.tts.models import Tacotron2Model
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

from friday.tts.models import ConditionalTacotron2

yaml = YAML(typ='safe')


@hydra_runner(config_name='tacotron2_nemo.yaml', config_path='base_config')
def main(cfg):
    if cfg.get('user_config', None):
        print(f'Overriding the base yaml with: {cfg.user_config} file...')
        with open(cfg.user_config, 'r') as f:
            user_cfg = yaml.load(f)
        cfg = OmegaConf.merge(cfg, user_cfg)

    trainer = pl.Trainer(**cfg.trainer)  # prepare a pyotrch lightning trianer
    exp_manager(trainer, cfg.get("exp_manager", None))  # for tensorboard logger
    
    model = ConditionalTacotron2(cfg=cfg.model, trainer=trainer)
    
    tacotron2_ckpt_dir = cfg.get('tacotron2_ckpt_dir', None)
    asr_ckpt_dir = cfg.get('asr_ckpt_dir', None) 
    if tacotron2_ckpt_dir and asr_ckpt_dir:
        print('Loading pretrained_model from: {}'.format(ckpt_dir))
        model.encoder.load_state_dict(torch.load(
            os.path.join(tacotron2_ckpt_dir, 'encoder.ckpt')
        ))
        model.decoder.load_state_dict(torch.load(
            os.path.join(tacotron2_ckpt_dir, 'decoder.ckpt')
        ))
        model.postnet.load_state_dict(torch.load(
            os.path.join(tacotron2_ckpt_dir, 'postnet.ckpt')
        ))
        # load weights of speaker encoder
        model.spk_encoder.encoder.load_state_dict(torch.load(
            os.path.join(asr_ckpt_dir, 'encoder.ckpt')
        ))
        model.spk_encoder.decoder.load_state_dict(torch.load(
            os.path.join(asr_ckpt_dir, 'decoder.ckpt')
        ))
        model.spk_encoder.preprocessor.load_state_dict(torch.load(
            os.path.join(asr_ckpt_dir, 'preprocessor.ckpt')
        ))
        print('Successfully loaded pretrained_model from: {}'.format(ckpt_dir))

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
