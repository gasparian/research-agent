import time
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs, unquote

from langchain_core.tools import tool
from ddgs import DDGS

from agent.models import SearchItem, SearchResult


def _normalize_link(raw: str) -> str:
    if not raw:
        return raw

    parsed = urlparse(raw)

    if parsed.netloc == "duckduckgo.com" and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg")
        if uddg:
            return unquote(uddg[0])

    return raw


@tool
def search(query: str, site: str | None = None, days: int | None = None) -> SearchResult:
    """Use this tool to search the web using DuckDuckGo.

    - Use `site` to restrict results to a domain, e.g. "github.com" or "arxiv.org".
    - Use `days` to prefer recent results: 1≈daily, 7≈weekly, 30≈monthly, >30≈yearly.

    :param query: The search query.
    :param site: Optional domain restriction.
    :param days: Optional recency window in days.
    :return: SearchResult with structured results.
    """
    try:
        if site:
            query = f"site:{site} {query}"

        if days is None:
            timelimit = None
        elif days <= 1:
            timelimit = "d"
        elif days <= 7:
            timelimit = "w"
        elif days <= 31:
            timelimit = "m"
        else:
            timelimit = "y"

        ddgs = DDGS()
        raw_results = list(
            ddgs.text(
                query,
                max_results=5,
                timelimit=timelimit,
            )
        )
        time.sleep(0.2)  # to avoid rate limiting

        items: list[SearchItem] = []

        for result in raw_results:
            title = result.get("title", "No title")
            link_raw = result.get("href") or result.get("url") or ""
            link = _normalize_link(link_raw)
            snippet = result.get("body", "No description")
            published = result.get("date") or result.get("published")

            items.append(
                SearchItem(
                    title=title,
                    link=link,
                    snippet=snippet,
                    published=published,
                )
            )

        return SearchResult(
            query=query,
            results=items,
            retrieved_at=datetime.now(timezone.utc),
        )
    except Exception:
        return SearchResult(
            query=query,
            results=[],
            retrieved_at=datetime.now(timezone.utc),
        )
