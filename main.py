from dotenv import load_dotenv
import asyncio

from langchain_core.messages import HumanMessage

from agent.react_agent import build_graph


async def main():
    graph = build_graph()

    config = {
        "configurable": {"thread_id": "research-1"},
        "recursion_limit": 100,
    }

    messages = []
    search_results = []

    print("Type an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            break

        messages.append(HumanMessage(content=user_input))

        state = {
            "messages": messages,
            "search_results": search_results,
        }

        result = await graph.ainvoke(state, config=config)

        messages = result["messages"]
        search_results = result.get("search_results", [])

        last = messages[-1]
        print("Assistant:", last.content)
        print()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
