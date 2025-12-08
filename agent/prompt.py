from pathlib import Path

from langchain_core.prompts import PromptTemplate


def build_prompt(tools) -> str:
    template_text = Path("agent/react.prompt").read_text()
    template = PromptTemplate.from_template(template_text)

    tool_lines = []
    for t in tools:
        name = getattr(t, "name", t.__class__.__name__)
        desc = getattr(t, "description", "") or ""
        args_list = []
        args = getattr(t, "args", None)
        if isinstance(args, dict):
            args_list = list(args.keys())
        args_str = f" Args: {', '.join(args_list)}" if args_list else ""
        tool_lines.append(f"- {name}: {desc}{args_str}")

    tools_block = "\n".join(tool_lines)

    return str(template.invoke(input={"tools": tools_block}))
