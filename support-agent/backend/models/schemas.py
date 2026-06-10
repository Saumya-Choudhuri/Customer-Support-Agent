from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ChatMessage(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    source: list[str] =[]
    escalated: bool = False
    memory_used: bool = False

class Memory(BaseModel):
    user_id: str
    fact: str
    created_at: datetime
    catagory: str

class IngestRequest(BaseModel):
    filepath: str