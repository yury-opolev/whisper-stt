"""HTTP contract DTOs (the wire shape Cortex's RemoteSpeechToText depends on)."""
from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    text: str


class Token(BaseModel):
    text: str
    startMs: int
    endMs: int


class DetailedResponse(BaseModel):
    text: str
    tokens: list[Token] = []


class InfoResponse(BaseModel):
    version: str
    modelId: str
    device: str
    modelLoaded: bool = True
