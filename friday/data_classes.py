from typing import Optional
from dataclasses import dataclass


@dataclass
class Fulfillment:
    name: str
    data: Optional[list]=None


@dataclass
class DialogOutput:
    reply: str
