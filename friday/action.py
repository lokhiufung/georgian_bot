import re
import collections
from typing import Union, Optional, List, Any
from dataclasses import dataclass
import requests

from friday.response.action_response import ActionResponse


__all__ = ['Action', 'ActionRequest']


@dataclass
class ActionRequest:
    client_id: str
    text: str
    intent: Optional[str]=''
    command: Optional[str]=''
    nlu_data: Optional[Any]=None


    
def to_py_variable(text: str) -> Union[float, int, str]:
    """infer and convert the type of input text

    :param text: text
    :type text: str
    :return: casted variable
    :rtype: Union[float, int, str]
    """
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
    def __init__(self, agent):
        """API caller for executing different actions. All action of an friday agent must be declared in Action

        :param agent: agent object
        :type agent: [type]
        """
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

    def execute(self, command: str, *additional_args, **additional_kwargs):
        """execute command

        :param command: text of function call
        :type command: str
        :raises Exception: command error
        :return: action result
        :rtype: [type]
        """
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
    
    def execute_v2(self, action_request: ActionRequest):
        return self.execute(
            command=action_request.command,
            client_id=action_request.client_id,
            text=action_request.text,
            nlu_data=action_request.nlu_data,
        )

    def no_action(self, client_id: str) -> ActionResponse:
        """Do no action just for consistency of the format of faq file

        :param client_id: client_id
        :type client_id: str
        :return: no_action response
        :rtype: ActionResponse
        """
        return ActionResponse(
            action_name='no_action',
            has_action_data=False
        )

    def fallout_response(self, action_request: ActionRequest):
        """when agent.nlu() cannot extract an appropriate intent (e.g score under threshold)

        :param action_request: request
        :type action_request: ActionRequest
        :raises NotImplementedError: [description]
        """
        raise NotImplementedError

    def clarifying_response(self, action_request: ActionRequest):
        """when agent is confused with the intent extracted by agent.nlu()

        :param action_request: request object
        :type action_request: ActionRequest
        :raises NotImplementedError: [description]
        """
        raise NotImplementedError


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
