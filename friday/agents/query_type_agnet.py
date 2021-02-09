from friday.agents.base_agent import CompositionalAgent
from friday.common.dialog_history import DialogHistory

# from assist.helpers.nlp import pipeline_utils
# from assist.helpers.nlp.text_pipeline import TextPipeline
# from assist.helpers.nlp.text_encoders import SentenceEncoder
# from assist.adaptors.similarity_adaptor import SimilarityAdapter


class Assist(CompositionalAgent):
    def __init__(self, module_config):
        self.threshold = module_config['threshold']
        # text_pipeline
        pipeline_config = module_config['text_pipeline']
        processors = []
        for cleaner in pipeline_config['cleaners']:
            processors.append(getattr(pipeline_utils, cleaner))
        for processor, kwargs in pipeline_config['processors'].items():
            processor = getattr(pipeline_utils, processor)(**kwargs)
            processors.append(processor)
        self.text_pipeline = TextPipeline(processors=processors)
        # encoder
        self.encoder = SentenceEncoder(**module_config['text_encoder'])
        # similarity adapter
        search_engine = AnnoySearchEngine(**module_config['search_engine'])
        self.similarity_adaptor = SimilarityAdapter(search_engine=search_engine)

        self.dialog_history = DialogHistory(**module_config['dialog_history'])
    
    # @classmethod
    # def from_yaml(cls, yaml_config):
    #     from ruamel.yaml import YAML

    #     yaml = YAML(typ="safe")
    #     with open(yaml_config, 'r') as f:
    #         config = yaml.load(f)
    #     return cls(config)

    def register_task(self, task):
        self.task = task
        self.task.tie_dialog_history(self.dialog_history)

    def text_infer(self, client_id, text):
        if hasattr(self, 'task'):
            # self.dialog.add_dialog(text, is_bot=False)
            # respond system powered by neural encoder and adapter
            self.dialog_history.add_dialog(text, is_bot=False)

            text_embedding = self.encoder.encode(
                sentence=self.text_pipeline(text)
                )[0]  # 2d array (n, dims)

            retrieved = self.similarity_adaptor(text_embedding)
            if retrieved.score > self.threshold:
                # if retrieved.action:
                task_response = self.task.execute(command=retrieved.action)
                # ...
            
                response = self.dialog_flow(input_text=text, intent=retrieved.intent, task_response=task_response, bot_response=retrieved)
                self.dialog_history.add_dialog(retrieved.answer, is_bot=True)
            else:
                response = self.dialog_flow(input_text=text, intent=None, task_response=None, bot_response=retrieved)
                self.dialog_history.add_dialog(Fallout.answer, is_bot=True)
            return response
        else:
            raise Exception('Must register task via register_task() first.')

    def dialog_flow(self, input_text, intent, task_response, bot_response):
        """"""
        if intent is not None:
            return {
                'input_text': input_text,
                'intent': intent,
                'answer': task_response.answer if task_response.answer else bot_response.answer,
                'score': bot_response.score,
                'action_payload': task_response.action_payload,
                'dialog_history': {
                    'dialog': self.dialog_history.get_dialog(1, include_bot=False)[0].text if len(self.dialog_history) > 0 else '',
                    'include_bot': False,
                    'n': 1,
                }
            }
        else:
            return {
                'score': bot_response.score,
                **{member.name: member.value for member in Fallout},
            }