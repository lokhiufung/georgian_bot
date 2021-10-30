# factory methods for producing agents


def load_example_agent(docs_filepath: str):
    """Factory method for producing an example agent

    Example agent is a faq-based agent, which an agent interpret text input by an sentence encoder. A hard router (hard-coded keys) is used for routing to fulfillment engine. A fulfillemnt engine then performs similarity search to find the contextually closest question and return to the agentl.     
    A json file with the following format should be prepared:
    [
        {"question": "What is your name?", "answer": "My name is Friday"},
        {"question": "What do you do?", "answer": "I am an virtual assistant."},
        ...
    ]

    :param docs_filepath: [description]
    :type docs_filepath: str
    :return: [description]
    :rtype: [type]
    """

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
            fulfillment_engines=[
                FAQFulfillmentEngine(docs),
            ]
        )
    
    agent = ExampleAgent(fulfillment_router, sensors=[text_embedding_sensor], confidence_threshold=0.5)

    return agent
