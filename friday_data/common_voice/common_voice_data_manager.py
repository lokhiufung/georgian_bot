import os
import tarfile
import json
import glob

import pandas as pd  # may switch to dask later
import scipy.io.wavfile as wav


def raw_to_data_dir(raw_filepath, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if raw_filepath.endswith('.zip'):
        raise NotImplementedError('.zip is not yet supported.')
    elif raw_filepath.endswith('.tar') or raw_filepath.endswith('.tar.gz'):
        # raise NotImplementedError('.zip is not yet supported.')
        if raw_filepath.endswith('.gz'):
            open_mode = 'r:gz'
        else:
            open_mode = 'r'
        with tarfile.open(raw_filepath, open_mode) as tar:
            tar.extractall(data_dir)
    else:
        pass


def convert_to_wav(audio_folder, data_dir, wav_audio_folder='wav/', src_format='mp3', sample_rate=16000):
    wav_audio_dir = os.path.join(data_dir, wav_audio_folder)
    if not os.path.exists(wav_audio_dir):
        os.mkdir(wav_audio_dir)

    audio_filepaths = glob.glob(os.path.join(data_dir, audio_folder, f'*.{src_format}'))
    for audio_filepath in audio_filepaths:
        wav_audio_filename = os.path.basename(audio_filepath)
        wav_audio_filepath = os.path.join(wav_audio_dir, wav_audio_filename)
        
        os.system(f'ffmpeg -i {audio_filepath} -acodec pcm_s16le -ar {sample_rate} {wav_audio_filepath}')


def tsv_to_nemo_speech_json(tsv_filenames, data_dir, header=0, text='text', duration='duration', audio_filepath='audio_filepath', sep='\t', audio_folder='wav'):

    def __get_duration(wav_filepath):
        sr, data = wav.read(wav_filepath)
        duration = len(data) / sr
        return durations

    for tsv_filename in tsv_filenames:
        tsv_filepath = os.path.join(data_dir, tsv_filename)
        df = pd.read_csv(tsv_filepath, header=header, sep='\t')

        # if header:
        #     for field in schema:
        #         if field not in df.columns:
        #             raise ValueError('Required field `{}` is not in the df: `{}`'.format(field, tsv_filename))

        # rename columns with nemo schema
        if duration in df.columns:
            df = df.rename(columns={
                text: 'text',
                duration: 'duration',
                audio_filepath: 'audio_filepath',
            })
        else:
            # if there is no duration columns
            df = df.rename(columns={
                text: 'text',
                audio_filepath: 'audio_filepath',
            })
        df['audio_filepath'] = df['audio_filepath'].apply(lambda audio_filename: os.path.join(data_dir, audio_folder, audio_filename))
        if 'duration' not in df.columns:
            print('getting durations from audio files...')
            df['duration'] = df['path'].apply(__get_duration)
        df = df[['text', 'duration', 'audio_filepath']]  # reduce schema to nemo format
        docs = df.to_dict(orient='record')
        nemo_json_filename = tsv_filename.replace('.tsv', '.json')
        with open(os.path.join(data_dir, nemo_json_filename), 'w') as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')



if __name__ == '__main__':
    # raw_to_nemo(
    #     raw_filepath='/home/lokhiufung/data/cantonese/src/common_voice/zh-HK.tar',
    #     data_dir='/home/lokhiufung/data/cantonese/src/common_voice/zh-HK_common_voice'
    # )
    # tsv_to_nemo_speech_json(
    #     tsv_filenames=[
    #         'cv-corpus-6.1-2020-12-11/zh-HK/train.tsv',
    #         'cv-corpus-6.1-2020-12-11/zh-HK/dev.tsv',
    #         'cv-corpus-6.1-2020-12-11/zh-HK/test.tsv',
    #     ],
    #     data_dir='/home/lokhiufung/data/cantonese/src/common_voice/zh-HK_common_voice',
    #     text='sentence',
    #     audio_filepath='path',
    # )
    convert_to_wav(
        audio_folder='clips',
        data_dir='/home/lokhiufung/data/cantonese/src/common_voice/zh-HK_common_voice/cv-corpus-6.1-2020-12-11/zh-HK',
    )