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

from agent.tools import think, search
from agent.models import SearchResult
from agent.prompt import build_prompt


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    search_results: Annotated[list[SearchResult], operator.add]


def build_graph():
    tools = [
        think,
        search,
    ]

    model = ChatOpenAI(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        base_url="https://foundation-models.api.cloud.ru/v1",
        temperature=0,
    ).bind_tools(tools)

    prompt = build_prompt(tools)
    system_message = SystemMessage(content=prompt)

    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> AgentState:
        messages = [system_message] + state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    def tools_node(state: AgentState) -> AgentState:
        result = tool_node.invoke(state)
        all_messages = result["messages"]
        new_messages = all_messages[len(state["messages"]) :]

        collected: list[SearchResult] = []

        for m in new_messages:
            if isinstance(m, ToolMessage) and m.name == "search":
                content = m.content
                if isinstance(content, dict):
                    try:
                        collected.append(SearchResult(**content))
                    except Exception:
                        pass

        data: AgentState = {"messages": new_messages}
        if collected:
            data["search_results"] = collected
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
