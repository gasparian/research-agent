import os

from langchain_core.tools import tool
from ddgs import DDGS


@tool
def think(thought: str) -> str:
    """Use the tool to think about something.
           This is perfect to start your workflow.
           It will not collect new information or take any actions, but just append the thought to the log and return the result.
           Use it when complex reasoning or some cache memory or a scratchpad is needed.


           :param thought: A thought to think about and log.
           :return: The full log of thoughts and the new thought.
    """
    return thought


@tool
def search(query: str) -> str:
    """Use this tool to search the web using DuckDuckGo for current information or facts not in your knowledge base.

    :param query: The search query to look up.
    :return: Search results as formatted text with titles, links, and snippets.
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=5)

        if not results:
            return "No results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            link = result.get('href', '')
            snippet = result.get('body', 'No description')
            formatted_results.append(f"{i}. {title}\n   {link}\n   {snippet}")

        print(f"Search results for query '{query}':\n" + "\n".join(formatted_results))
        print()

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"
