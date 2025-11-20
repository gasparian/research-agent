from dotenv import load_dotenv
import asyncio

from langchain_core.messages import HumanMessage

from agent.react_agent import build_graph


async def main():
    question = "WHat's the current APPLE stock price?"

    graph = build_graph()
    result = await graph.ainvoke({"messages": [HumanMessage(content=question)]}, config={"configurable": {"thread_id": "1"}})
    answer = result["messages"][-1].content

    print()
    print("--------------")
    print(answer)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
