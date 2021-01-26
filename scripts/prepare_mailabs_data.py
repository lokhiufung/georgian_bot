"""
Parser for MaiLab's datasets: https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/
"""

import os
import re
import json
import random
import argparse

import librosa
import pandas as pd
import scipy.io.wavfile as wave
import tqdm


def get_parser():
    parser = argparse.ArgumentParser('Prepare dataset for training TTS english with M-AILabs')
    parser.add_argument('--data_root', type=str, required=True, help='directory of source data')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')
    parser.add_argument('--version_name', type=str, required=True, help='dataset_name')
    parser.add_argument('--books', type=str, required=False, help='e.g comma-seperated str of books: ozma_of_oz,rinkitink_in_oz,sky_island,the_master_key')
    parser.add_argument('--gender', type=str, required=False, help='gender of spk')
    parser.add_argument('--spk', type=str, required=False, help='name of spk')
    parser.add_argument('--p', type=float, default=0.99, required=False, help='proportion of training data')
    parser.add_argument('--resample', action='store_true', required=False, help='whether to resample wav file')
    parser.add_argument('--sr', type=float, default=22050, required=False, help='target sample rate of training audio')
    return parser


class JsonWriter(object):
    def __init__(self, output_dir, name, mode='w'):
        self.name = name
        self.output_dir = output_dir
        # self.mode = 'w'  
        # if os.path.exists(os.path.join(self.output_dir, self.name)):
        #     self.mode = 'a'
        assert mode == 'w' or mode == 'a'  
        self.writer = open(os.path.join(self.output_dir, self.name), mode)

    def write(self, item):
        self.writer.write(json.dumps(item, ensure_ascii=False) + '\n')

    def close(self):
        self.writer.close()


def resample_audio(audio_filename, sr):
    y, sr = librosa.load(audio_filename, sr=sr)
    librosa.output.write_wav(audio_filename, y, sr)


def prepare_item(audio_filepath, duration, text):
    # folder = os.path.join(output_dir, 'wav/{}'.format(metadata['user_id']))
    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    item = dict()
    item['audio_filepath'] = audio_filepath
    item['duration'] = duration
    item['text'] = text
    
    return item


class Speaker:
    def __init__(self, name, gender, books):
        self.name = name
        self.gender = gender
        self.books = books


class MaiLabDataset(object):
    def __init__(self, data_root, dataset_name):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.dataset_root = os.path.join(self.data_root, dataset_name)
        self.metadata_schema = ['wav_filename', 'text', 'normalized_text']
        self.output_schema = ['wav_filepath', 'duration', 'text']
        self.speakers = self.prepare_speakers()

    def prepare_speakers(self):
        speakers = []
        genders = [path for path in os.listdir(os.path.join(self.dataset_root, 'by_book')) if not path.endswith('.txt')]
        for gender in genders:
            speaker_names = os.listdir(os.path.join(self.dataset_root, f'by_book/{gender}'))
            # speaker_names = list(filter(lambda x: (not x.endswith('.csv')) or not (x.endswith('.json')), speaker_names))
            for speaker_name in speaker_names:
                books = os.listdir(os.path.join(self.dataset_root, f'by_book/{gender}/{speaker_name}'))
                books = list(filter(lambda x: not x.endswith('.txt'), books))
                speakers.append(
                    Speaker(name=speaker_name, gender=gender, books=books)
                )
        return speakers

    @staticmethod
    def _get_duration(wav_filepath):
        sr, y = wave.read(wav_filepath)
        # assert sr == 22050
        duration = len(y) / sr
        return duration

    def aggregate_metadata(self, speaker_names: list=None, books: list=None, gender: str=None):
        """
        not_found_wav_filenames: in case we may ne
        """
        meta_df = []
        # not_found_wav_filenames = []
        for speaker in filter(lambda x: (x.name in speaker_names) and speaker.gender == gender, speaker_names): 
            for book in filter(lambda x: x in books, speaker.books):
                metadata_filepath = os.path.join(self.dataset_root, 'by_book/{}/{}/{}/metadata.csv'.format(speaker.gender, speaker.name, book))
                df = pd.read_csv(metadata_filepath, header=None, sep='|', names=self.metadata_schema)
                wav_filepaths = []
                for row in df.itertuples():
                    wav_filepath = os.path.join(
                        self.dataset_root,
                        'by_book/{}/{}/{}/wavs/{}.wav'.format(speaker.gender, speaker.name, book, row.wav_filename)
                        )
                    if not os.path.exists(wav_filepath):
                        wav_filepath = None
                        # not_found_wav_filenames.append(wav_filepath)
                    wav_filepaths.append(wav_filepath)

                df['wav_filepath'] = wav_filepaths
                df = df.dropna()  # drop those rows without wav files        
                meta_df.append(df)
         
        meta_df = pd.concat(meta_df, axis=0)
        meta_df['duration'] = meta_df['wav_filepath'].apply(self._get_duration)
        meta_df = meta_df[self.output_schema]
        return meta_df

 
def main():
    parser = get_parser()
    
    args = parser.parse_args()
    data_root = args.data_root
    dataset_name = args.dataset_name
    version_name = args.version_name
    books = args.books.split(',')
    spk = args.spk
    gender = args.gender
    p = args.p
    resample = args.resample
    target_sr = args.sr

    # data_root = '/home/lokhiufung/data/english/src/mailabs'
    # dataset_name = 'en_US'
    # dataset_root = os.path.join(data_root, dataset_name)
    
    # 1. aggregate datasets to a dataframe
    dataset = MaiLabDataset(data_root, dataset_name)
    meta_df = dataset.aggregate_metadata(spk, gender, books)
    
    print('total_duration: ', meta_df['duration'].sum() / 3600.0)
    
    # 2. write to json files
    train_writer = JsonWriter(data_root, f'{dataset_name}-{version_name}-train.json', mode='w')
    valid_writer = JsonWriter(data_root, f'{dataset_name}-{version_name}-valid.json', mode='w')

    for row in tqdm.tqdm(meta_df.itertuples(), total=len(meta_df)):
        item = prepare_item(
            audio_filepath=row.wav_filepath,
            duration=row.duration,
            text=row.normalized_text
            )
        if random.random() < p:
            train_writer.write(item)
        else:
            valid_writer.write(item)
    train_writer.close()
    valid_writer.close()

    # 3. resample
    if resample:
        for row in tqdm.tqdm(meta_df.itertuples(), total=len(meta_df)):
            resample_audio(row.wav_filepath, sr=target_sr)



if __name__ == '__main__':
    main()