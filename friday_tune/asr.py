import argparse
import pprint

from omegaconf import OmegaConf, DictConfig
# from ruamel.yaml import YAML
from nemo.core.config import hydra_runner
import torch
import pytorch_lightning as pl
import nemo
from nemo.collections.asr.metrics.wer import WER
import nemo.collections.asr as nemo_asr
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


# NEMO_VER = str(nemo.__version__)


@hydra_runner(config_path='base_config', config_name='quartznet_15x5_nemo.yaml')
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    print(OmegaConf.to_yaml(cfg))
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    vocabulary = []
    if cfg.get('vocabulary', None):
        vocabulary = [char.strip() for char in open(cfg.vocabulary, 'r')]
        if vocabulary[0] != '':
            logging.info('add "" to the first char of vocabulary')
            vocabulary = [''] + vocabulary
        assert vocabulary[0] == '', 'Must leave a space for the first char in vocabulary for pad_id: {}'.format(vocabulary[:5])

        if ' ' not in vocabulary:
            logging.info('add " " to the vocabulary')
            vocabulary.append(' ')

        if vocabulary[-1] != '"':  # temp fix for the blank token; may not be necessary
            logging.info('add """ to the last char of vocabulary')
            vocabulary += ['"']
        assert vocabulary[-1] == '"', 'Must leave blank token e.g """ for the last char in vocabulary for blank token: {}'.format(vocabulary[-5:])

    if vocabulary:
        logging.info('changing vocabularies of train_ds/validation_ds...')
        cfg.model.train_ds['labels'] = vocabulary
        cfg.model.validation_ds['labels'] = vocabulary

    asr_model = nemo_asr.models.EncDecCTCModel(cfg=cfg.model, trainer=trainer)
    
    if vocabulary:
        logging.info('changing vocabularies of asr_model using asr_model.change_vocabulary()...')
        original_vocabulary_size = len(asr_model.decoder.vocabulary)
        asr_model.change_vocabulary(vocabulary)
        logging.info('changed vocabularies with file {}; vocabulary_size: {}; original_vocabulary_size: {}'.format(cfg.vocabulary, len(vocabulary), original_vocabulary_size))

        logging.info('just to ensure that the setup of training and validation data loader...')
        asr_model.setup_optimization(optim_config=DictConfig(cfg.model.optim))
        asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
        asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    if cfg.get('pretrained_encoder', None):
        asr_model.encoder.load_state_dict(torch.load(cfg.pretrained_encoder))
        logging.info('loaded pretrained encoder from {}'.format(cfg.pretrained_encoder))
    
    for batch in asr_model._train_dl:
        print(batch[2])
        prediction = asr_model._wer.ctc_decoder_predictions_tensor(batch[2])
        print(prediction)
        break        
    # UGLY: reset metric objects
    # model._wer = WER(
    #     vocabulary=model.decoder.vocabulary,
    #     batch_dim_index=0,
    #     use_cer=cfg.model.get('use_cer', False),
    #     ctc_decode=True,
    #     dist_sync_on_step=True,
    #     log_prediction=cfg.model.get("log_prediction", False),
    # )

    trainer.fit(asr_model)


if __name__ == '__main__':
    main()