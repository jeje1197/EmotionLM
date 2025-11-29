from pydantic import BaseModel
from typing import any, Literal
from enum import Enum

class MemoryType(Enum):
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"

class Memory(BaseModel):
    id: int
    type: MemoryType
    event: str
    cause: str
    date: str
    semantic_embedding: any
    emotional_embedding: any
