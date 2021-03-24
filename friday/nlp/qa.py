from haystack.reader.farm import FARMReader


class ExtractiveQA:
    def __init__(
        self,
        model_path: str,
        document_store_mode: str='es',
        document_dir: str=None,
        top_k_retriever: int=10,
        top_k_reader: int=5,
        device: str='cpu',
        es_index: str='document',
        es_host: str='localhost'
    ):
        from haystack.pipeline import ExtractiveQAPipeline
        
        self.device = device
        self.es_index = es_index
        self.es_host = es_host
        # retriever and reader args
        self.top_k_reader = top_k_reader
        self.top_k_retriever = top_k_retriever

        if document_store_mode == 'in_memory':
            from haystack.document_store.memory import InMemoryDocumentStore
            from haystack.retriever.sparse import TfidfRetriever

            self.document_store = InMemoryDocumentStore()
            self._load_documents(document_dir=document_dir)  # must have document_dir if use InMemorydocumentStore
            self.retriever = TfidfRetriever(document_store=self.document_store)
        else:
            from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
            from haystack.retriever.sparse import ElasticsearchRetriever

            self.document_store = ElasticsearchDocumentStore(host=self.es_host, index=self.es_index)
            self.retriever = ElasticsearchRetriever(document_store=self.document_store)
        
        use_gpu = True if self.device == 'cuda:0' else False
        self.reader = FARMReader(model_name_or_path=model_path, use_gpu=use_gpu)
        self.pipe = ExtractiveQAPipeline(self.reader, self.retriever)
    
    def retrieve_top_k(self, text, k=1):
        prediction = self.pipe.run(
            query=text,
            top_k_retriever=self.top_k_retriever,
            top_k_reader=self.top_k_reader,
        )
        retrieved = prediction['answers'][:k]  # top k answers
        answers = []
        # formatting on the retrieved docs
        for doc in retrieved: 
            answer = {}
            answer['answer'] = doc['answer']
            answer['context'] = doc['context']
            answer['document_id'] = doc['document_id']
            answer['title'] = doc['meta']['name']
            answer['score'] = doc['score']
            answer['probability'] = doc['probability']
            answers.append(answer)     
        return answers
    
    def _load_documents(self, document_dir):
        """
        load documents only for InMemory mode
        """
        from haystack.preprocessor.utils import convert_files_to_dicts
        from haystack.preprocessor.preprocessor import PreProcessor
        
        all_docs = convert_files_to_dicts(document_dir)
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False,
            split_by='word',
            split_length=100,
            split_respect_sentence_boundary=True
        )
        nested_docs = [preprocessor.process(d) for d in all_docs]
        docs = [d for x in nested_docs for d in x]
        self.document_store.write_documents(docs)


class EmbeddingQA:
    def __init__(
        self,
        model_path,
        document_store_mode: str='es',
        top_k_retriever: int=10,
        device: str='cpu',
        es_index: str='str',
        es_host: str='localhost',
        embedding_dim: int=768,
        embedding_field: str='question_emb',
        similarity: str='cosine',
        model_format: str='farm'
    ):
        """
        A virtual assistant with question and answering type as backend of natural language understanding and inference.
        Use Haystack as nlu backend 
        """
        from haystack.pipeline import FAQPipeline
        
        self.device = device
        self.es_host = es_host
        self.es_index = es_index
        self.embedding_dim = embedding_dim 
        self.embedding_field = embedding_field

        if document_store_mode == 'es':
            from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
            from haystack.retriever.dense import EmbeddingRetriever

            self.document_store = ElasticsearchDocumentStore(
                host=self.es_host,
                index=self.es_index,
                embedding_dim=self.embedding_dim,
                embedding_field=self.embedding_field,
                excluded_meta_data=[self.embedding_field],
                similarity=similarity
            )
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=model_path,
                use_gpu=True if self.device == 'cuda:0' else False,
                model_format=model_format,
            )
        else:
            raise ValueError(f'Only "es" is supported currently: {document_store_mode}')
        
        self.pipe = FAQPipeline(retriever=self.retriever) 
        
        self.top_k_retriever = top_k_retriever
    
    def retrieve_top_k(self, text, k=1):
        prediction = self.pipe.run(
            query=text,
            top_k_retriever=self.top_k_retriever,
        )
        retrieved = prediction['answers'][:k]  # top k answers
        answers = []
        # formatting on the retrieved docs
        for doc in retrieved:
            # print(doc) 
            answer = {}
            answer['question'] = doc['query']
            answer['answer'] = doc['answer']
            answer['context'] = doc['context']
            answer['document_id'] = doc['document_id']
            answer['score'] = doc['score']
            answer['probability'] = doc['probability']
            answer = {**answer, **doc['meta']}
            answers.append(answer)     
        return answers
    