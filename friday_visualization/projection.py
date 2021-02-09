############################
# reference: https://dash.plotly.com/layout
############################

import os
import argparse
import collections
import pickle
import re

import nemo.collections.asr as nemo_asr
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from sklearn.manifold import TSNE
from omegaconf import DictConfig


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_files', '-F', type=str, required=True, help='manifest_files with at least these entries: audio_filepath, spk_id')
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--embedding_dir', type=str, help='embedding dir; embeddings key-value pairs')
    parser.add_argument('--encoder', type=str)
    args = parser.parse_args()
    return args


# class RecorderDataDB(object):
#     def __init__(self, manifest_all, is_print_info=True, to_hk=True):
#         if to_hk:
#             print('text will be converted to hk...')

#         self.samples = self._load_samples(manifest_all, to_hk=to_hk)
#         self.total_duration = self._calculate_total_duration()
#         self.vocabulary = self._get_vocabulary()
#         self.vocabulary_size = len(self.vocabulary)
#         self.speakers = self._get_speakers()
#         self.n_speakers = len(self.speakers)

#         if is_print_info:
#             self.print_info()
    
#     @staticmethod
#     def _load_samples(manifest_all, to_hk=True):
#         _Sample = collections.namedtuple(
#             'Sample',
#             ['spk_id', 'audio_filepath', 'duration', 'text', 'tags']
#         )
#         with open(manifest_all, 'r') as f:
#             sample_list = json.load(f)
#         samples = []
#         for sample in sample_list:
#             # older sample do not have "tags"
#             if to_hk:
#                 text = S2HK.convert(sample['text'])
#             samples.append(_Sample(spk_id=sample['spk_id'], audio_filepath=sample['audio_filepath'], duration=sample['duration'], text=text, tags=sample.get('tags', [])))
#         return samples        

#     def _calculate_total_duration(self):
#         total_duration = sum([sample.duration for sample in self.samples])
#         return total_duration / 3600.0

#     def _get_vocabulary(self):
#         vocabulary = []
#         for sample in self.samples:
#             # text = S2HK.convert(sample.text)
#             tokens = [char.strip().lower() for char in sample.text]
#             vocabulary.extend(list(set(tokens)))
#         vocabulary = [char for char in vocabulary]  # normalize vocabulary to hk
#         vocabulary = list(set(vocabulary))
#         return vocabulary
    
#     def _get_speakers(self):
#         _Speaker = collections.namedtuple(
#             'Speaker',
#             ['spk_id', 'n_samples']
#         )

#         speaker_counts = collections.Counter([sample.spk_id for sample in self.samples])
#         speakers = []
#         for spk_id, count in speaker_counts.items():
#             speakers.append(_Speaker(spk_id=spk_id, n_samples=count))
#         speakers = sorted(speakers, key=lambda x: x.n_samples, reverse=True)
#         return speakers

#     def print_info(self):
#         print('------------RecorderData Summary------------')
#         print('total duration: {}'.format(self.total_duration))
#         print('total vocabulary size: {}'.format(self.vocabulary_size))
#         print('number of speakers: {}'.format(self.n_speakers))
#         print('top 5 count:')
#         print('\n'.join(['{}: {}'.format(speaker.spk_id, speaker.n_samples) for speaker in self.speakers[:5]]))
#         print('--------------------------------------------')

#     def shuffle_samples(self):
#         random.shuffle(self.samples)


def run_dash_app():
    args = parser_args()

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }


    app = dash.Dash('Speaker Embeddings', external_stylesheets=external_stylesheets)
    
    if not os.path.exits(args.embedding_dir):
        model = nemo_asr.models.ExtractSpeakerEmbeddingsModel.restore_from(args.encoder)
        
        test_config = OmegaConf.create(dict(
            manifest_filepath=,
            sample_rate= 16000,
            labels= None,
            batch_size=8,
            shuffle= False,
            time_length= 8,
            embedding_dir=args.embedding_dir
        ))
        print(OmegaConf.to_yaml(test_config))
        model.setup_test_data(test_config)
        trainer.test(model)

    # load embeddings from dir

    with open('./embeddings/spk_label_manifest_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    labels = []
    embedding_array = []
    for key, embedding in embeddings.items():
        labels.append(key)
        embedding_array.append(embedding)
    embedding_array = np.array(embedding_array)
    labels = [re.match(r'spk\-[0-9]+', name)[0] for name in labels]

    random_index = np.random.choice(range(len(labels)), size=min(args.size, len(labels)))

    # tsne = 


    fig = px.scatter(x=)