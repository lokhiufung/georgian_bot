# import librosa


def process_audio(audio_signal, transforms=[]):
    for transform in transforms:
        audio_signal = transform(audio_signal)
    return audio_signal



