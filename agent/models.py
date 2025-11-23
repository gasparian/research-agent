from datetime import datetime
from pydantic import BaseModel


class SearchItem(BaseModel):
    title: str
    link: str
    snippet: str
    published: str | None = None


class SearchResult(BaseModel):
    query: str
    results: list[SearchItem]
    retrieved_at: datetime
