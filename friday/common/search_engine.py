import json

from elasticsearch import Elasticsearch, NotFoundError
from annoy import AnnoyIndex

# from friday.errors import GeneralFridayErrro

@DeprecationWarning
class SearchEngine(object):
    def query(self):
        pass


@DeprecationWarning
class ElasticSearchEngine(SearchEngine):
    def __init__(self, index_name, doc_type='_doc'):
        self.index_name = index_name
        self.doc_type = doc_type
        self.es = Elasticsearch()

    def _prepare_vector_search(self, vector, limit=10, skip=0):
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['question_vector']) + 1.0",
                    "params": {"query_vector": vector}
                }
            }
        }

        body = {
            "size": limit,
            "query": script_query,
            "min_score": 0.0,
        }
        return body

    def document_search(self, body={}, source=[]):
        """Search index with query body"""
        try:
            found_documents = self.es.search(
                index=self.index_name, doc_type=self.doc_type, body=body,
                _source=source, search_type='dfs_query_then_fetch'
            )
        except NotFoundError:
            raise GeneralFridayErrro
        else:
            return found_documents

    def vector_search(self, query, *args, **kwargs):
        body = self._prepare_vector_search(query, *args, **kwargs)
        return self.document_search(body)


@DeprecationWarning
class AnnoySearchEngine(SearchEngine):
    N_TREES = 5
    def __init__(self, dim, metric, docs, key='question_vector', ckpt=None):
        if isinstance(docs, str):
            with open(docs, 'r') as f:
                self.docs = json.load(f)
        else:
            self.docs = docs
        self.key = key
        self.tree = AnnoyIndex(f=dim, metric=metric)
        if metric == 'angular':
            self._distance_to_score = self._angular_distance_to_score
        else:
            raise Exception(f'Only "angular" is supported: {metric}')
        if ckpt:
            self.load_tree(ckpt)
        else:
            self._build_tree()

    def _angular_distance_to_score(self, distance):
        """
        range of score will be [-1, 1]
        """
        score = (1 - distance**2 ) + 1.0 
        return score

    def _build_tree(self):
        for idx, doc in enumerate(self.docs):
            self.tree.add_item(idx, doc[self.key])
        self.tree.build(self.N_TREES)

    def load_tree(self, ckpt):
        self.tree.load(ckpt)

    def save_tree(self, ckpt):
        self.tree.save(ckpt)

    def vector_search(self, vector, k=10):
        indices, distances = self.tree.get_nns_by_vector(vector, k, include_distances=True)
        results = []
        for idx, distance in zip(indices, distances):
            result = self.docs[idx].copy()
            
            del result[self.key]
            
            result['score'] = self._distance_to_score(distance)
            results.append(result)
        return results 

