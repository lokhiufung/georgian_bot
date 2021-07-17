import argparse

from haystack.reader.farm import FARMReader


def main():
    """finetune extractive qa model
        e.g python friday_tune/nlp_qa_roberta_farm.py --model_path uer/roberta-base-chinese-extractive-qa \
            --train_dir .. --train_filename .. --gpu --save_dir .. --n_epochs 10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-M', type=str, required=True, help='transformer model card or pretrained model path')
    parser.add_argument('--train_dir', type=str, required=True, help='train directory')
    parser.add_argument('--train_filename', type=str, help='train dataset filename')
    parser.add_argument('--dev_filename', type=str, default=None, help='dev dataset filename')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    parser.add_argument('--save_dir', type=str, default='./nlp_qa_roberta_farm', help='checkpoint save directory')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='number of epochs')

    args = parser.parse_args()

    reader = FARMReader(
        model_name_or_path=args.model_path,
        use_gpu=args.gpu,
    )

    reader.train(
        data_dir=args.train_dir,
        train_filename=args.train_filename,
        dev_filename=args.dev_filename,
        use_gpu=args.gpu,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()