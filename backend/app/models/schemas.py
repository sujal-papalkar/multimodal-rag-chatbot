from typing import List, Optional, Literal
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    doc_ids: List[str] = []
    is_web_search_enabled: bool = False
    # EDGE CASE: optional so a brand-new chat can omit it — the backend
    # creates a fresh conversation automatically. Sending the SAME id back
    # on subsequent turns is what threads follow-up questions together.
    conversation_id: Optional[str] = None


class Citation(BaseModel):
    id: str
    doc_id: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[int] = None
    page_end: Optional[int] = None  # set when this chunk spans multiple pages (page..page_end)
    type: Literal["text", "web"]
    content: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


class PageImage(BaseModel):
    doc_id: Optional[str] = None
    page: int
    image: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    page_images: List[PageImage]
    # EDGE CASE: every response carries the exact conversation + message it
    # belongs to. The frontend must key citations off `message_id`, never
    # off "the most recent response", so re-renders / async races / retries
    # can never attach the wrong citation set to the wrong chat bubble.
    conversation_id: str = ""
    message_id: str = ""


class ConversationMessage(BaseModel):
    message_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: List[Citation] = []
    page_images: List[PageImage] = []
    created_at: float


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[ConversationMessage]
