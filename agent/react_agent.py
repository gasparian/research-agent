import operator
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from agent.tools.think import think
from agent.tools.search import search
from agent.tools.web_fetch import fetch_url
from agent.tools.citations import format_citations
from agent.models import SearchResult, FetchResult
from agent.prompt import build_prompt


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    search_results: Annotated[list[dict], operator.add]
    fetched_pages: Annotated[list[dict], operator.add]


MAX_HISTORY_MESSAGES = 20


def build_graph():
    tools = [
        think,
        search,
        fetch_url,
        format_citations,
    ]

    base_llm = ChatOpenAI(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        base_url="https://foundation-models.api.cloud.ru/v1",
        temperature=0,
    )

    agent_model = base_llm.bind_tools(tools)

    prompt = build_prompt(tools)
    system_message = SystemMessage(content=prompt)

    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> AgentState:
        history = state["messages"]
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        messages = [system_message] + history
        response = agent_model.invoke(messages)
        return {"messages": [response]}

    def tools_node(state: AgentState) -> AgentState:
        result = tool_node.invoke(state)
        messages = result.get("messages", [])

        collected_search: list[dict] = []
        collected_pages: list[dict] = []

        for m in messages:
            if isinstance(m, ToolMessage) and m.name == "search":
                content = m.content
                if isinstance(content, dict):
                    try:
                        sr = SearchResult(**content)
                        collected_search.append(sr.model_dump())
                    except Exception:
                        pass
            if isinstance(m, ToolMessage) and m.name == "fetch_url":
                content = m.content
                if isinstance(content, dict):
                    fr = FetchResult(**content)
                    collected_pages.append(fr.model_dump())

        data: AgentState = {"messages": messages}
        if collected_search:
            data["search_results"] = collected_search
        if collected_pages:
            data["fetched_pages"] = collected_pages
        return data

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")

    app = graph.compile(checkpointer=MemorySaver())
    return app
