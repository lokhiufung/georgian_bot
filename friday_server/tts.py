
import json
import os
import tempfile
import time
import base64
import random
import string
import os

from omegaconf import DictConfig
from flask import Flask, request, abort, jsonify
from scipy.io.wavfile import write

from friday.tts.synthesizer import Synthesizer


#########################
# tmp
import librosa

def trim_silence(sample, top_db=60, frame_length=2048, hop_length=512):
    sample, _ = librosa.effects.trim(sample, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return sample


def extract_segnment(sample, i=0, top_db=60, frame_length=2048, hop_length=512):
    intervals = librosa.effects.split(sample, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    sample = sample[intervals[i][0]:intervals[i][1]]
    return sample
#########################

    
def create_tts_server(tts_server_cfg: DictConfig):
    """helper function for creating a flask server for tts

    :param tts_server_cfg: contain cfg for server and tts
    :type tts_server_cfg: DictConfig
    :raises ValueError: Supported languages: "cantonese", "english"
    :return: Flask server
    :rtype: Flask
    """
    server_cfg = tts_server_cfg.server
    
    app = Flask(server_cfg.name)

    CONSTANTS = tts_server_cfg.server.constants
    tts_model = Synthesizer(**tts_server_cfg.synthesizer)

    if server_cfg.lang == 'cantonese':
        from friday.text import build_cantonese_tts_text_pipeline

        text_pipeline = build_cantonese_tts_text_pipeline()
    elif server_cfg.lang == 'english':
        from friday.text import build_english_tts_text_pipeline

        text_pipeline = build_english_tts_text_pipeline()
    else:
        raise ValueError('server.cfg.lang {} is not supported.')

    
    @app.route('/synthesize', methods=['POST'])
    def synthesize():
        request_start = time.perf_counter()  # start getting result of request

        payload = dict() 
        data = request.get_json()
        # LOGGER.debug('receiced data: {}'.format(data))
        text = data.get('text', '')
        payload['client_id'] = data.get('client_id', '')
        is_analyze = data.get('is_analyze', False)  # get more info for inference analysis

        denoiser_strength = data.get('denoiser_strength', CONSTANTS["denoiser_strength"])
        
        experimental = data.get('experimental', {})
        break_to_segments = experimental.get('break_to_segments', False)
        print('break_to_segments: {}'.format(break_to_segments))
        if text:
            text_pipeline_start = time.perf_counter()
            text = text_pipeline(text)
            text_pipeline_end = time.perf_counter()

            audio_temp = './audio_temp-{}-{}.wav'.format(str(time.time()).replace('.', ''), _random_string())
            # initialize temp file for audio and manifest
            # manifest_temp = tempfile.NamedTemporaryFile(suffix='.json')

            # manifest = dict()
            # manifest['audio_filepath'] = audio_temp
            # manifest['duration'] = 1.0
            # manifest['text'] = text

            # with open(manifest_temp.name, 'w') as f:
            #     json.dump(manifest, f)
            model_start = time.perf_counter()
            sample = tts_model.text_to_wav(text=text, break_to_segments=break_to_segments)  # scipy.io.wav write needs a 1-d array 
            # librosa: trim silence/ get the first split
            if CONSTANTS['post_processing']:
                sample = trim_silence(sample)
                sample = extract_segnment(sample)
            model_end = time.perf_counter()
            model_time = model_end - model_start
            # LOGGER.debug('Successful time: {} manifest: {}'.format(total_t, manifest))

            # manifest_temp.close()  # close and remove manifest_temp
            write(audio_temp, CONSTANTS.sample_rate, sample)

            # format payload
            with open(audio_temp, 'rb') as f:
                data = f.read()
            os.remove(audio_temp)  # explicitly remove audio_temp

            # payload = dict()
            payload['audio'] = base64.b64encode(data).decode('utf-8')
            payload['time'] = model_time
            
            request_end = time.perf_counter()
            if is_analyze:
                analysis = {
                    'latency': {
                        'model_time': model_time,
                        'request_time': request_end - request_start,
                        'text_pipeline_time': text_pipeline_end - text_pipeline_start,
                    }
                }
                payload['analysis'] = analysis 
            return jsonify(payload)
        else:
            abort(400, 'text cannot be empty')
    return app


def _random_string(string_length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))
