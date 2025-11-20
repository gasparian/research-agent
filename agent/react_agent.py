from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agent.tools import think, search
from utils.prompt import build_prompt


def build_graph():
    tools = [
        think,
        search,
    ]
    agent = create_agent(
        ChatOpenAI(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct", 
            base_url="https://foundation-models.api.cloud.ru/v1", 
            temperature=0,
        ),
        tools,
        checkpointer=MemorySaver(),
        system_prompt=build_prompt(tools),
    )
    return agent
