from typing import Any, Protocol


class TraceSink(Protocol):
    def on_step(self, node: str, kind: str, info: dict[str, Any]) -> None:
        ...


class ConsoleSink:
    def __init__(self, show_payload: bool = False) -> None:
        self.show_payload = show_payload

    def on_step(self, node: str, kind: str, info: dict[str, Any]) -> None:
        if kind == "node":
            if self.show_payload:
                keys = ", ".join(info.get("keys", []))
                print(f"[trace] {node}: state keys -> {keys}")
            else:
                print(f"[trace] {node}")
        elif kind == "tool_call":
            tool = info.get("tool", "?")
            args = info.get("args", {})
            short_args = str(args)
            if len(short_args) > 120:
                short_args = short_args[:117] + "..."
            print(f"[trace] {node}: call {tool}({short_args})")
        elif kind == "tool_result":
            tool = info.get("tool", "?")
            print(f"[trace] {node}: result from {tool}")
        else:
            print(f"[trace] {node}: {kind}")


class AgentTracer:
    def __init__(self, sink: TraceSink) -> None:
        self.sink = sink

    def handle_update(self, chunk: Any) -> None:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            _, data = chunk
        else:
            data = chunk

        if not isinstance(data, dict) or not data:
            return

        node, payload = next(iter(data.items()))
        if not isinstance(payload, dict):
            payload = {}

        self.sink.on_step(
            node=node,
            kind="node",
            info={"keys": list(payload.keys())},
        )

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return

        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = None
                args = None
                if isinstance(call, dict):
                    name = call.get("name")
                    args = call.get("args")
                else:
                    name = getattr(call, "name", None)
                    args = getattr(call, "args", None)
                if name:
                    self.sink.on_step(
                        node=node,
                        kind="tool_call",
                        info={"tool": name, "args": args or {}},
                    )

        for m in messages:
            name = getattr(m, "name", None)
            tool_call_id = getattr(m, "tool_call_id", None)
            if name and tool_call_id:
                self.sink.on_step(
                    node=node,
                    kind="tool_result",
                    info={"tool": name},
                )
