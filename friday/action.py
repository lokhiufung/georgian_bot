import re
import collections

import requests

from friday.response.action_response import ActionResponse


__all__ = ['Action']


def to_py_variable(text):
    # type_ = str
    if re.match(r'^[+-]?[0-9]+\.[0-9]+$', text):
        # print(re.match(r'[0-9]?\.[0-9]?'))
        print('match float')
        type_ = float 
    elif re.match(r'^[+-]?[0-9]+$', text):
        print('match int')
        type_ = int
    else:
        type_ = str
        # raise ValueError('Only str, float, int are support types of args. Cannot parser {}'.format(text))
    return type_(text) 


    
class Action(object):
    """
    API caller for executing different actions
    """
    # DEFAULT_ACTION = '<DEFAULT>'
    # OUTPUT_TYPE = collections.namedtuple(
    #     'ResponseEntity',
    #     ['answer', 'action_payload']
    # )
    def __init__(self, agent):
        self.agent = agent
        # self.output_type = self.OUTPUT_TYPE
        # self.state_dict = dict()

    @staticmethod        
    def args_parser(text):
        text_args = text.split(',')
        kwargs = dict()
        args = []
        for text_arg in text_args:
            if '=' in text_arg:
                key, value = re.sub(r'[,\"\']', '', text_arg.strip()).split('=')
                kwargs = {key: to_py_variable(value), **kwargs}
            else:
                value = re.sub(r'[,\"\']', '', text_arg.strip())
                if value:
                    # the string must not be empty
                    args.append(to_py_variable(value))
        return args, kwargs

    def execute(self, command, *additional_args, **additional_kwargs):
        try:
            func = getattr(
                self,
                re.search(r'(?<!\()\w+(?![\)])', command).group(0)
            )
            args, kwargs = self.args_parser(
                re.search(r'(?<=\()(.*?)(?=\))', command).group(1)
            )
            if additional_args:
                args.extend(additional_args)

            if additional_kwargs:
                kwargs = {**kwargs, **additional_kwargs}
            return func(*args, **kwargs)
        except:
            raise Exception(f'Command error: {command}')
    
    def no_action(self, client_id):
        """
        Do no action just for consistency of the format of faq file
        """
        return ActionResponse(
            action_name='no_action',
            has_action_data=False
        )


# class MasterAction(Action):
#     def __init__(self):
#         super().__init__()
#         self.state_dict['current_domain'] = None

#     def chat(self):
#         return self.output_type(
#             answer='',
#             action_payload={}
#         )

#     def repeat_text(self, text):
#         return self.output_type(
#             answer=f"You have asked: {text}",
#             action_payload={}
#         )

#     def switch_domain(self, domain):
#         self.state_dict['current_domain'] = domain
#         return self.output_type(
#             answer=f'The current domain is now {domain}',
#             action_payload={}
#         )

#     def inform_domain(self):
#         domain = self.state_dict['current_domain']
#         if domain:
#             answer = f'The current domain is now {domain}'
#         else:
#             answer = 'You have not chosen any domain yet!'
#         return self.output_type(
#             answer=answer,
#             action_payload={}
#         )

#     def suggest_domain(self, *domains):
#         return self.output_type(
#             answer='You may ask questions in the following domains: ' + ', '.join(domains),
#             action_payload={}
#         )
