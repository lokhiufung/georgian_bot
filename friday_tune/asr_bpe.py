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

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"bpe" or "wpe"> \
        --log
```

# Training the model
```sh
python speech_to_text_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    trainer.gpus=2 \
    trainer.accelerator="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.1 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_ratio=0.05 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="AN4-BPE-1024" \
    exp_manager.wandb_logger_kwargs.project="AN4_BPE_1024"
```
"""
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def extract_state_dict_from_nemo(nemo_file):
    pass


@hydra_runner(config_path="base_config", config_name="quartznet_15x5_bpe_nemo.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    print(OmegaConf.to_yaml(cfg))
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)
    # load pretrained encoder
    if cfg.get('pretrained_encoder', None):
        asr_model.encoder.load_state_dict(torch.load(cfg.pretrained_encoder))
        logging.info('loaded pretrained encoder from {}'.format(cfg.pretrained_encoder))
        
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu, precision=trainer.precision, amp_level=trainer.amp_level, amp_backend=trainer.amp_backend,
        )
        if asr_model.prepare_test(test_trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
