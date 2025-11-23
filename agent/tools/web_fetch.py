from typing import Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from langchain_core.tools import tool

from agent.models import FetchResult


MAX_TEXT_CHARS = 20000


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return text


def _extract_title(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    if not title_tag or not title_tag.string:
        return None
    return title_tag.string.strip()


@tool
def fetch_url(url: str, timeout: int = 10) -> FetchResult:
    """Fetch a URL and return cleaned page text.

    Returns:
    - url
    - status_code
    - content_type
    - title
    - plain text content (truncated)
    """
    try:
        resp = requests.get(url, timeout=timeout)
        content_type = resp.headers.get("Content-Type", "")

        html = resp.text if "text/html" in content_type.lower() else ""
        text = _extract_text(html) if html else ""
        title = _extract_title(html) if html else None

        return FetchResult(
            url=url,
            status_code=resp.status_code,
            content_type=content_type,
            title=title,
            text=text,
        )
    except Exception as e:
        return FetchResult(
            url=url,
            status_code=0,
            content_type=None,
            title=None,
            text=f"Fetch error: {e}",
        )
