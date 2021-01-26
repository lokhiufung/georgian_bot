
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_dir', type=str, required=True, help='directory of pretrained models in .nemo format')
    parser.add_argument('--output_dir', type=str, required=True, help='directory of extracted state_dict')
    parser.add_argument('--tts', action='store_true', default=False)
    parser.add_argument('--asr', action='store_true', default=False)
    # parser.add_argument('--nlp', action='store_true')
    return parser.parse_args()


def extract_state_dict_tts(nemo_dir, output_dir):
    import nemo.collections.tts as nemo_tts

    nemo_tts.models.Tacotron2Model.extract_state_dict_from(
        os.path.join(nemo_dir, 'Tacotron2-22050Hz.nemo'),
        save_dir=os.path.join(output_dir, 'tacotron2_ckpt'),
        split_by_module=True
    )
    nemo_tts.models.GlowTTSModel.extract_state_dict_from(
        os.path.join(nemo_dir, 'GlowTTS-22050Hz.nemo'),
        save_dir=os.path.join(output_dir, 'glowtts_ckpt'),
        split_by_module=True
    )
    nemo_tts.models.WaveGlowModel.extract_state_dict_from(
        os.path.join(nemo_dir, 'WaveGlow-22050Hz.nemo'),
        save_dir=os.path.join(output_dir, 'waveglow_ckpt'),
        split_by_module=True
    )
    nemo_tts.models.SqueezeWaveModel.extract_state_dict_from(
        os.path.join(nemo_dir, 'SqueezeWave-22050Hz.nemo'),
        save_dir=os.path.join(output_dir, 'squeezewave_ckpt'),
        split_by_module=True
    )


def extract_state_dict_asr(nemo_dir, output_dir):
    import nemo.collections.asr as nemo_asr

    nemo_asr.models.EncDecCTCModel.extract_state_dict_from(
        os.path.join(nemo_dir, 'QuartzNet15x5Base-En.nemo'),
        save_dir=os.path.join(output_dir, 'quartznet15x5base_en_ckpt'),
        split_by_module=True
    )
    nemo_asr.models.EncDecCTCModel.extract_state_dict_from(
        os.path.join(nemo_dir, 'SpeakerNet_verification.nemo'),
        save_dir=os.path.join(output_dir, 'speakernet_verification'),
        split_by_module=True
    )


def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.tts:
        print('Extracting state dicts of tts models...')
        extract_state_dict_tts(args.nemo_dir, args.output_dir)
    if args.asr:
        print('Extracting state dicts of asr models...')
        extract_state_dict_asr(args.nemo_dir, args.output_dir)
    print('Finish extracting state dicts.')


if __name__ == '__main__':
    main()
    