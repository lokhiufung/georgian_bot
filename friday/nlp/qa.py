from haystack.reader.farm import FARMReader


class ExtractiveQA:
    def __init__(self, model_path: str, document_store_mode: str='es', document_dir: str=None, top_k_retriever: int=10, top_k_reader: int=5, device: str='cpu'):
        from haystack.pipeline import ExtractiveQAPipeline
        
        self.device = device
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

            self.document_store = ElasticsearchDocumentStore()
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
    def __init__(self, model_path, top_k_retriever: int=10, device: str='cpu'):
        """
        A virtual assistant with question and answering type as backend of natural language understanding and inference.
        Use Haystack as nlu backend 
        """
        from haystack.pipeline import FAQPipeline
        
        self.device = device
        
        if cfg.document_store_mode == 'es':
            from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
            from haystack.retriever.dense import EmbeddingRetriever

            self.document_store = ElasticsearchDocumentStore(**cfg.haystack.document_store)
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=model_path,
                use_gpu=True if self.device == 'cuda:0' else False
            )
        else:
            raise ValueError(f'Only "es" is supported currently: {cfg.document_store_mode}')
        
        self.pipe = FAQPipeline(retriever=self.retriever) 
        
        self.top_k_retriever = self.top_k_retriever
    
    def retrieve_top_k(self, text, k=1):
        predictions = self.pipe.run(
            query=text,
            top_k_retriever=self.top_k_retriever
        )
        retrieved = predictions['answers'][:k]  # temp
        return retrieved
