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


class FetchResult(BaseModel):
    url: str
    status_code: int
    content_type: str | None
    title: str | None
    text: str
    html: str


class Source(BaseModel):
    title: str
    link: str
    note: str | None = None
