import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)


def nemo_to_transformer(vocabs, output_dir='./', **nemo_manifests):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    docs_dict = {}
    for name, manifest in nemo_manifests.items():
        docs = []
        with open(manifest, 'r') as f:
            for line in f:
                doc = json.loads(line)
                del doc['duration']
                docs.append(doc)
        docs_dict[name] = docs

    # labels
    vocab_dict = {vocab.strip(): index for index, vocab in enumerate(open(vocabs, 'r'))}
    vocab_dict['[UNK]'] = len(vocab_dict)
    vocab_dict['[PAD]'] = len(vocab_dict)
    
    with open(os.path.join(output_dir, 'vocab.json'), 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)
    return docs_dict, vocab_dict


@dataclass
class DataCollatorCTCWithPadding:
    # to be continued
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features):
        speech = [{'input_values': feature['input_values']} for feature in features]
        input_features = [{'input_values': feature['input_values']} for feature in features]
        label_features = [{'input_ids': feature['labels']} for feature in features]

        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors='pt'
            )
        
        labels = labels_batch['input_ids'].masked_fill(label_batch.attention_mask.ne(1), -100)
        
        batch['labels'] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# class Wave2VecCTCModel(torch.nn.Module):
#     def __init__(self, vocab_size, pretrained_model_name='facebook/wav2vec2-large-xlsr-53'):
#         super().__init__()
#         self.pretrained_model_name = pretrained_model_name
#         self.vocab_size = vocab_size
#         self.hidden_size = 1024
#         self.encoder = Wav2Vec2Model.from_pretrained(self.pretrained_model_name)
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(self.hidden_size, self.vocab_size)
#         )

#         self.encoder.feature_extractor._freeze_parameters()

#     def forward(self, batch):
        
def main():

    config = {
        'manifest_dict': {
            'train': '/media/lokhiufung/Storage/recorder-webapp/train-1.json',
            'test': '/media/lokhiufung/Storage/recorder-webapp/validation-1.json',
        },
        'vocab': '/media/lokhiufung/Storage/recorder-webapp/labels-1.txt',
        'data_dir': './wav2vec2_huggingface'
    }
     
    docs_dict, vocab_dict = nemo_to_transformer(
        config['vocab'],
        config['data_dir'],
        **config['manifest_dict']
    )
    
    vocab_size = len(vocab_dict)

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    tokenizer = Wav2Vec2CTCTokenizer(
        os.path.join(config['data_dir'], 'vocab.json'),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    processor = Wav2Vec2Processor(
        feature_extractor,
        tokenizer=tokenizer,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        'facebook/wav2vec2-large-xlsr-53',
        gradient_checkpointing=True,
        ctc_loss_reduction='mean',
        pad_token_id=processor.tokenizer.pad_token_id
    )
    model.lm_head = torch.nn.Linear(1024, vocab_size)
    
    print(model)

    # training_args = TrainingArguments(
    #     output_dir='./wav2vec2-ctc-test',
    #     group_by_length=True,
    #     per_device_train_batch_size=32,
    #     evaluation_strategy='steps',
    #     num_train_epochs=30,
    #     fp16=True,
    #     save_steps=500,
    #     eval_steps=500,
    #     logging_steps=500,
    #     learning_rate=1e-4,
    #     weight_decay=0.005,
    #     warmup_steps=1000,
    #     save_total_limit=2
    # )

    # trainer = Trainer(
    #     model=model,
    #     data_collator=data_collator,
    #     args=training_args,
    #     compute_metrics=compute_metrics,
    #     train_dataset=docs_dict['train'],
    #     eval_dataset=docs_dict['test'],
    #     tokenizer=processor.feature_extractor,
    # )

main()