import json
from typing import List

from haystack.retriever.dense import EmbeddingRetriever
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore



def index_faq_documents_to_es(
    docs: List[dict],
    index: str,
    embedding_column: str,
    host='localhost',
    embedding_dim: int=768,
    embedding_model: str='deepset/sentence_bert',
    use_gpu=False,
    model_format='farm'
):
    embedding_field = embedding_column + '_emb'
    document_store = ElasticsearchDocumentStore(
        host,
        index=index,
        embedding_field=embedding_field,
        embedding_dim=embedding_dim,
        excluded_meta_data=[embedding_field]
    )
    # use model_format='sentence_transformers' for customzied type
    retriever = EmbeddingRetriever(
        document_store,
        embedding_model,
        use_gpu=use_gpu,
        model_format=model_format
    )
    # create embeddings for docs
    embeddings = retriever.embed_queries(texts=[doc[embedding_column] for doc in docs])
    for embedding, doc in zip(embeddings, docs):
        doc[embedding_field] = embedding
        doc['text'] = doc[embedding_column]

        del doc[embedding_column]
    
    # print(docs[0])
    document_store.write_documents(docs)


def check_faq_schema(docs):
    required_fields = ['question', 'answer', 'action']
    for doc in docs:
        if any([key not in required_fields for key, _ in doc.items()]):
            raise ValueError('doc must contain the following fields: {}'.format(required_fields))


def json_to_docs(json_file):
    with open(json_file, 'r') as f:
        docs = json.load(f)
    
    check_faq_schema(docs)
    return docs