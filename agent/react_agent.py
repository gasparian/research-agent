from typing import Annotated

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
)

from agent.tools.think import think
from agent.tools.search import search
from agent.tools.citations import format_citations
from agent.tools.current_date import get_current_datetime
from agent.tools.web_fetch import fetch_url
from agent.models import SearchResult
from agent.prompt import build_prompt


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


MAX_HISTORY_MESSAGES = 20


def build_graph():
    tools = [
        think,
        search,
        format_citations,
        get_current_datetime,
        fetch_url,
    ]

    base_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    agent_model = base_llm.bind_tools(tools)

    prompt = build_prompt(tools)
    system_message = SystemMessage(content=prompt)

    def agent_node(state: AgentState) -> AgentState:
        history = state["messages"]
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        messages = [system_message] + history
        response = agent_model.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent", 
        should_continue, 
        {
            "tools": "tools", 
            END: END,
        }
    )
    graph.add_edge("tools", "agent")

    app = graph.compile(checkpointer=MemorySaver())
    return app
