from dataclasses import dataclass


@dataclass
class ASRResponse:
    transcription: str
    time: float


@dataclass
class TTSResponse:
    time: float


class NLPQAResponse:
    time: 