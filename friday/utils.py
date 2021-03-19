from haystack.retriever.dense import EmbeddingRetriever
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore



def index_faq_documents_to_es(
    docs: dict,
    index: str,
    embedding_column: str,
    host='localhost',
    embedding_dim: int=768,
    embedding_model: str='deepset/sentence_bert',
    use_gpu=False,
):
    embedding_field = embedding_column + '_emb'
    document_store = ElasticsearchDocumentStore(
        host,
        index,
        embedding_field=embedding_field,
        embedding_dim=embedding_dim,
        excluded_meta_data=[embedding_field]
    )

    retriever = EmbeddingRetriever(
        document_store,
        embedding_model,
        use_gpu,
    )

    document_store.write_documents(docs)

