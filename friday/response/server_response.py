from typing import Optional, List, Dict
from dataclasses import dataclass

from .agent_response import AgentResponse


@dataclass
class FridayServerResponse:
    """Base dataclass for any server response in friday-server"""
    client_id: Optional[str]
    time: float


@dataclass
class AgentServerResponse(AgentResponse, FridayServerResponse):
    """response from other agent through server"""


@dataclass
class ASRServerResponse(FridayServerResponse):
    """"""
    transcription: str


@dataclass
class TTSServerResponse(FridayServerResponse):
    """"""
    audio: str  # base64 encoded audio bytes, decoded with utf-8


@dataclass
class NLPQAServerResponse(FridayServerResponse):
    """"""
    answers: List[Dict]
    max_score: float
    size: int


# @dataclass
# class NLPFAQServerResponse(FridayServerResponse):
#     """"""
#     answers: List[Dict]
#     max_score: float
#     size: int

