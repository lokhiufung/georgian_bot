import argparse

from friday.utils import index_faq_documents_to_es, index_qa_documents_to_es, json_to_docs


def prepare_faq(json_file, embedding_model, use_gpu):
    docs = json_to_docs(json_file, mode='faq')
    index_faq_documents_to_es(
        docs,
        index='add_two_test',
        embedding_column='question',
        embedding_model=embedding_model,
        use_gpu=use_gpu,
        model_format='sentence_transformers'
    )


def prepare_qa(json_file):
    docs = json_to_docs(json_file)
    index_qa_documents_to_es(
        docs,
        index='drcd_test',
        host='localhost'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faq_file', type=str, help='docs')
    parser.add_argument('--qa_file', type=str, help='docs')
    parser.add_argument('--model', '-M', type=str, help='model name')
    parser.add_argument('--gpu', action='store_true', default=False, help='whether to use gpu or not')
    args = parser.parse_args()
    
    json_file = args.json_file
    embedding_model = args.model
    use_gpu = args.gpu
    if args.faq_file:
        prepare_faq(
            json_file=args.faq_file,
            embedding_model=args.model,
            use_gpu=args.gpu
        )
    if args.qa_file:
        prepare_qa(json_file=args.qa_file)

if __name__ == '__main__':
    main()
