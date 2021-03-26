import io
import base64
import tempfile
import json
import time
import string
import random
import librosa
import soundfile

from flask import Flask, abort, request
import scipy.io.wavfile as wave

from friday.asr.recognizer import Recognizer


__all__ = ['create_asr_server']


def create_asr_server(asr_server_cfg):
    recognizer = Recognizer(**asr_server_cfg.recognizer)

    CONSTANTS = asr_server_cfg.server.constants  # constants, nb channels, sample rate 
    app = Flask(asr_server_cfg.server.name)

    @app.route('/transcribe', methods=['POST'])
    def transcribe_file():
        """
        """
        payload = {}  # initialize a payload object for response
        # payload['request_id'] = get_request_id()
        body = request.get_json()
        payload['client_id'] = body.get('client_id', '')
        # LOGGER.debug('id: {}'.format(payload['request_id']))
        # LOGGER.debug('client_id: {}'.format(payload['client_id']))

        try:
            f = io.BytesIO(base64.b64decode(body['content'].encode('utf-8')))
        except Exception as err:
            abort(500, 'decode error err: {}'.format(err)) 
        lang = body['lang'].upper()
        if audio_file_checker(f, sample_rate=CONSTANTS['sample_rate'], num_channels=CONSTANTS['nb_channels']):
            audio_temp = tempfile.NamedTemporaryFile(suffix='.wav')
            manifest_temp = tempfile.NamedTemporaryFile(suffix='.json')

            with open(audio_temp.name, 'wb') as f_bytes:
                f_bytes.write(f.read())

            # temp: trim silence
            trim_silence(audio_temp.name, sr=CONSTANTS['sample_rate'])
            # 1. stream-in audio
            #### TO BE IMPLEMENTED

            # 2. temp file
            manifest = dict()
            manifest['audio_filepath'] = audio_temp.name
            manifest['duration'] = 18000
            manifest['text'] = 'todo'
            with open(manifest_temp.name, 'w') as f:
                json.dump(manifest, f)

            start_t = time.time()
            try:
                transcription = recognizer.wav_to_text(manifests=manifest_temp.name)  # list with single transcription string
            except Exception as err:
                abort(500, err)
            total_t = time.time() - start_t

            audio_temp.close()
            manifest_temp.close()
            
            payload['time'] = total_t
            payload['transcription'] = transcription[0] # wav_to_text return a list of transcriptions
            
            # LOGGER.debug('Successful payload: {}'.format(payload))
            return payload
        else:
            abort(400, 'sample rate and number of channels should be {}Hz and {}'.format(CONSTANTS['sample_rate'], CONSTANTS['nb_channels']))

    return app
    

def audio_file_checker(audio_file, sample_rate, num_channels):
    sr, signal = wave.read(audio_file)
    if sr != sample_rate or len(signal.shape) != num_channels:
        is_pass = False
    else:
        is_pass = True
    return is_pass


def get_random_string(string_length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


def get_request_id():
    ts = str(time.time())
    random_string = get_random_string()
    request_id = random_string + '-' + ts.split('.')[0]
    return request_id 


def trim_silence(audio_filename, sr=16000, top_db=30):
    sample, sr = librosa.load(audio_filename, sr=sr)
    sample, _ = librosa.effects.trim(sample, top_db=top_db)
    soundfile.write(audio_filename, sample, sr)
