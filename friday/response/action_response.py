from typing import Union, Optional, Any

from dataclasses import dataclass


@dataclass
class ActionResponse:
    """All action reponse must be a subclass of the base class
    """
    action_name: str
    has_action_data: bool
    action_answer: Optional[str]=''
    action_data: Optional[dict]=None
    additional_answer: Optional[Any]=None
         

