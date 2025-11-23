from dotenv import load_dotenv
import asyncio

from langchain_core.messages import HumanMessage

from agent.react_agent import build_graph
from agent.tracing import AgentTracer, ConsoleSink


async def run_with_trace(graph, state, config, tracer: AgentTracer):
    result_state = None

    async for mode, chunk in graph.astream(
        state,
        config=config,
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            tracer.handle_update(chunk)
        elif mode == "values":
            result_state = chunk

    if result_state is None:
        raise RuntimeError("Graph finished without emitting a final state")

    return result_state


async def main():
    graph = build_graph()
    tracer = AgentTracer(ConsoleSink(show_payload=False))

    config = {
        "configurable": {"thread_id": "research-1"},
        "recursion_limit": 50,
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

        result = await run_with_trace(graph, state, config, tracer)

        messages = result["messages"]
        search_results = result.get("search_results", [])

        last = messages[-1]
        print("Assistant:", last.content)
        print()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
