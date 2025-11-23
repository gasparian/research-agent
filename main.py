from dotenv import load_dotenv
import asyncio

from langchain_core.messages import HumanMessage

from agent.react_agent import build_graph
from agent.tracing import AgentTracer, ConsoleSink


def print_sources(search_results: list[dict]) -> None:
    links: list[tuple[str, str]] = []

    for r in search_results:
        results = r.get("results") or []
        for item in results:
            title = item.get("title", "")
            link = item.get("link", "")
            if link:
                links.append((title, link))

    seen = set()
    deduped: list[tuple[str, str]] = []
    for title, link in links:
        if link in seen:
            continue
        seen.add(link)
        deduped.append((title, link))

    if not deduped:
        return

    print("Sources:")
    for i, (title, link) in enumerate(deduped, 1):
        print(f"{i}. {title} - {link}")
    print()


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

    print("Type an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            break

        state = {
            "messages": [HumanMessage(content=user_input)],
        }

        result = await run_with_trace(graph, state, config, tracer)

        messages = result["messages"]
        search_results = result.get("search_results", [])

        last = messages[-1]
        print("Assistant:", last.content)
        print()

        print_sources(search_results)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
