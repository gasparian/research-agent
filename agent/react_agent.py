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
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.tools.think import think
from agent.tools.search import search
from agent.tools.citations import format_citations
from agent.tools.current_date import get_current_datetime
from agent.tools.web_fetch import fetch_url
from agent.prompt import build_prompt


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    clarify_count: Annotated[int, operator.add]

MAX_HISTORY_MESSAGES = 20
MAX_CLARIFY_CALLS = 2


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
    clarifier_model = base_llm

    prompt = build_prompt(tools)
    system_message = SystemMessage(content=prompt)

    def clarify_node(state: AgentState) -> AgentState:
        clarify_count = state.get("clarify_count", 0)
        if clarify_count >= MAX_CLARIFY_CALLS:
            return {}

        history = state.get("messages") or []
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        user_msg = None
        for m in reversed(history):
            if isinstance(m, HumanMessage):
                user_msg = m
                break
        if user_msg is None:
            return {}
        sys = SystemMessage(
            content=(
                "You decide if the user's request is clear enough to start research.\n"
                "- If it is clear and specific enough, answer exactly: CLEAR\n"
                "- If you need more information, answer exactly: ASK: <one clarifying question>\n"
                "Do not call tools. Do not add anything else."
            )
        )
        resp = clarifier_model.invoke([sys, user_msg])
        text = str(resp.content).strip()
        if text.upper().startswith("CLEAR"):
            return {}

        question = text
        if ":" in text:
            question = text.split(":", 1)[1].strip() or "Could you clarify your request?"

        return {
            "messages": [AIMessage(content=question)],
            "clarify_count": clarify_count + 1,
        }

    def agent_node(state: AgentState) -> AgentState:
        history = state["messages"]
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        messages = [system_message] + history
        response = agent_model.invoke(messages)
        return {"messages": [response]}

    def after_clarify(state: AgentState) -> str:
        messages = state.get("messages") or []
        if messages and isinstance(messages[-1], AIMessage):
            return "ask"
        return "proceed"

    def should_continue(state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("clarify", clarify_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("clarify")
    graph.add_conditional_edges(
        "clarify",
        after_clarify,
        {
            "ask": END,
            "proceed": "agent",
        },
    )
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
