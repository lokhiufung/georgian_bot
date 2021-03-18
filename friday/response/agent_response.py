from typing import Union, Optional

from dataclasses import dataclass

from .task_response import BaseTaskResponse


@dataclass
class AgentResponse:
    input_text: str
    is_fallout: bool
    is_voice: bool
    has_task: bool
    text_answer: str
    voice_answer: Optional[bytes]=None
    task_response: Optional[BaseTaskResponse]=None

