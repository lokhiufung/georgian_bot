def load_example_agent(docs_filepath):
    import json

    from friday.agents.example_agent import ExampleAgent
    from friday.fulfillments.engines.faq_fulfillement_engine import FAQFulfillmentEngine
    from friday.fulfillments.routers.hard_router import HardFulfillmentRouter
    from friday.sensors.text_sensor import TextEmbeddingSensor


    text_embedding_sensor = TextEmbeddingSensor()

    with open(docs_filepath, 'r') as f:
        docs = json.load(f)  # docs

    for doc in docs:
        doc['question_vector'] = text_embedding_sensor.process(doc['question'])['embedding']

    fulfillment_router = HardFulfillmentRouter(
            keys=(0, 1),
            fulfillment_engines=(
                FAQFulfillmentEngine()
            )
        )
    
    agent = ExampleAgent(fulfillment_router, sensors=[text_embedding_sensor], confidence_threshold=0.5)

    return agent
