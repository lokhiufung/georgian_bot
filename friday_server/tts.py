
import json
import os
import tempfile
import time
import base64
import random
import string
import os

from flask import Flask, request, abort, jsonify
from scipy.io.wavfile import write

from friday.tts.synthesizer import Synthesizer

# tmp
os.environ['PHONEMIZER_ESPEAKER_PATH'] = '/usr/bin/espeak-ng'


def create_tts_server(tts_server_cfg):
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
        payload = dict() 
        data = request.get_json()
        # LOGGER.debug('receiced data: {}'.format(data))
        text = data.get('text', '')
        payload['client_id'] = data.get('client_id', '')
        denoiser_strength = data.get('denoiser_strength', CONSTANTS["denoiser_strength"])
        if text:
            text = text_pipeline(text)
            audio_temp = './audio_temp-{}-{}.wav'.format(str(time.time()).replace('.', ''), _random_string())
            # initialize temp file for audio and manifest
            manifest_temp = tempfile.NamedTemporaryFile(suffix='.json')

            manifest = dict()
            manifest['audio_filepath'] = audio_temp
            manifest['duration'] = 1.0
            manifest['text'] = text

            with open(manifest_temp.name, 'w') as f:
                json.dump(manifest, f)
            start = time.perf_counter()
            sample = tts_model.text_to_wav(manifest=manifest)[0]  # scipy.io.wav write needs a 1-d array 
            # librosa: trim silence/ get the first split
            # if CONSTANTS['POST_PROCESSING']:
            #     sample = helpers.trim_silence(sample)
            #     sample = helpers.extract_segnment(sample)
            total_t = time.perf_counter() - start
            # LOGGER.debug('Successful time: {} manifest: {}'.format(total_t, manifest))

            manifest_temp.close()  # close and remove manifest_temp
            write(audio_temp, CONSTANTS.sample_rate, sample)

            # format payload
            with open(audio_temp, 'rb') as f:
                data = f.read()
            os.remove(audio_temp)  # explicitly remove audio_temp

            # payload = dict()
            payload['audio'] = base64.b64encode(data).decode('utf-8')
            
            return jsonify(payload)
        else:
            abort(400, 'text cannot be empty')

    return app


def _random_string(string_length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))
