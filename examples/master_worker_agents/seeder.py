import argparse

from friday.utils import index_faq_documents_to_es, json_to_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', '-F', type=str, help='docs')
    parser.add_argument('--model', '-M', type=str, help='model name')
    parser.add_argument('--gpu', action='store_true', default=False, help='whether to use gpu or not')
    args = parser.parse_args()
    
    json_file = args.json_file
    embedding_model = args.model
    use_gpu = args.gpu

    docs = json_to_docs(json_file)
    index_faq_documents_to_es(
        docs,
        index='add_two_test',
        embedding_column='question',
        embedding_model=embedding_model,
        use_gpu=use_gpu,
        model_format='sentence_transformers'
    )


if __name__ == '__main__':
    main()
