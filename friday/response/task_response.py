from typing import Union, Optional

from dataclasses import dataclass


@dataclass
class TaskResponse:
    """All task reponse must be a subclass of the base class."""
    task_name: str
    task_data: Optional[dict]=None
         

