from typing import List

from langchain_core.tools import tool

from agent.models import Source


@tool
def format_citations(items: List[Source]) -> str:
    """Format a list of sources into a markdown References section.

    Each source must have a title and a link. Optionally, a note can be included.
    """
    lines = []
    for i, src in enumerate(items, 1):
        title = src.title or src.link
        link = src.link
        note = src.note or ""
        if note:
            lines.append(f"{i}. [{title}]({link}) — {note}")
        else:
            lines.append(f"{i}. [{title}]({link})")
    return "\n".join(lines)