import annoy

from friday.fulfillments.engines.base_fulfillment_engine import BaseFulfillmentEngine
from friday.data_classes import Fulfillment


class FAQFulfillmentEngine(BaseFulfillmentEngine):
    NAME = 'faq_fulfillment'
    
    def __init__(self, docs, key_name='question_vector'):
        super().__init__()

        self.docs = docs
        self.key_name = key_name

        embedding_dim = len(docs[0][key_name])
        self.engine = annoy.AnnoyIndex(embedding_dim, 'angular')

        for i, doc in enumerate(docs):
            # print(doc[self.key_name].shape)
            self.engine.add_item(i, doc[self.key_name])

        self.engine.build(10)

    def run(self, key, top_k=1):
        indexes, scores = self.engine.get_nns_by_vector(key, top_k, include_distances=True)

        data = []
        for index, score in zip(indexes, scores):
            doc = self.docs[index].copy()
            del doc[self.key_name]  # remove vector keys

            data.append({
                'score': score,
                **doc
            })
        return Fulfillment(
            name=self.name,
            data=data,
        )