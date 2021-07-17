import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# from friday.helpers.nlp.text_encoder import TextEncoder
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import LabelSentenceReader
from sentence_transformers.datasets import SentencesDataset


class TextClassificationAdaptor(nn.Module):
    def __init__(self, dim, n_class):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(dim, n_class),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.output_layer(x)
        return x
        

class TextClassifier(pl.LightningModule):
    def __init__(self, encoder, dim, n_class):
        super().__init__()
        self.encoder = encoder
        # self.encoder.freeze()
        # freeze parameters of encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.classification_adpator = TextClassificationAdaptor(dim=dim, n_class=n_class)

    def forward(self, x):
        encoder_output = self.encoder(x)
        embeddings = encoder_output['sentence_embedding'].detach()
        output = self.classification_adpator(embeddings)
        return output

    def training_step(self, batch, batch_idx):
        """"""
        features = batch.get('features')[0]
        labels = batch.get('labels')

        logits = self(features)
        loss = F.nll_loss(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    parser = argparse.ArgumentParser() 
    # parser.add_argument('--train', type=str, default='./test/data/text_classification.tsv', help='train data file with (label, text) as its row')
    # parser.add_argument('--eval', type=str, default='./test/data/text_classification.tsv', help='train data file with (label, text) as its row')
    parser.add_argument('--data_folder', type=str, default='./test/data', help='data folder')
    parser.add_argument('--encoder', type=str, required=True, help='used encoder ckpt path')
    args = parser.parse_args()

    encoder = SentenceTransformer(model_name_or_path=args.encoder)

    train_data = LabelSentenceReader(
        folder=args.data_folder,
        label_col_idx=0,
        sentence_col_idx=1,
    )
    train_dataset = SentencesDataset(
        train_data.get_examples('text_classification_data.tsv'),
        model=encoder
    )
    eval_data = LabelSentenceReader(
        folder=args.data_folder,
        label_col_idx=0,
        sentence_col_idx=1
    )
    eval_dataset = SentencesDataset(
        eval_data.get_examples('text_classification_data.tsv'),
        model=encoder
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=encoder.smart_batching_collate)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, collate_fn=encoder.smart_batching_collate)

    # x = next(iter(train_dataloader))
    # print(x)
    # encoded_input = encoder.tokenizer(x, padding=True, truncation=True, return_tensors='pt')

    # output = encoder(x['features'][0])
    # print(output)
    # print(output)
    text_classifier = TextClassifier(
        encoder=encoder,
        dim=768,
        n_class=10
    )
    # print(dir(text_classifier))
    trainer = pl.Trainer(gpus=None, max_epochs=5)
    trainer.fit(text_classifier, train_dataloader, eval_dataloader)


if __name__ == '__main__':
    main()