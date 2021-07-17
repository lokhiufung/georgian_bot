from typing import Union, Optional, Any

from dataclasses import dataclass

from .action_response import ActionResponse


@dataclass
class AgentResponse:
    input_text: str
    is_fallout: bool
    has_action: bool
    text_answer: str
    is_voice: Optional[bool]=False
    voice_answer: Optional[bytes]=None
    action_response: Optional[ActionResponse]=None
    additional_answers: Optional[Any]=None

