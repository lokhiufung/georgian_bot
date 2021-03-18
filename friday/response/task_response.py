from typing import Union, Optional

from dataclasses import dataclass


@dataclass
class BaseTaskResponse:
    """All task reponse must be a subclass of the base class."""
    action_name: str




