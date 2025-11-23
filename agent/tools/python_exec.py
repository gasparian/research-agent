import ast
import math
from pydantic import BaseModel
from langchain_core.tools import tool


class ExecResult(BaseModel):
    expression: str
    result: str
    ok: bool


_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Attribute,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
)


def _is_safe_node(node: ast.AST) -> bool:
    if not isinstance(node, _ALLOWED_NODE_TYPES):
        return False

    if isinstance(node, ast.Name):
        return node.id == "math"

    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "math"

    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Attribute):
            return False
        if not isinstance(func.value, ast.Name) or func.value.id != "math":
            return False
        for arg in node.args:
            if not _is_safe_node(arg):
                return False
        for kw in node.keywords:
            if kw.value is not None and not _is_safe_node(kw.value):
                return False
        return True

    for child in ast.iter_child_nodes(node):
        if not _is_safe_node(child):
            return False

    return True


@tool
def python_exec(expression: str) -> ExecResult:
    """Safely evaluate a pure Python expression for calculations.

    Allowed:
    - arithmetic and boolean expressions
    - comparisons
    - calls to math.* functions (e.g. math.sqrt, math.log)

    Not allowed:
    - imports, assignments, attribute access outside math.*
    - filesystem, network, or OS operations
    """
    try:
        expr = expression.strip()
        if len(expr) > 500:
            return ExecResult(
                expression=expression,
                result="Expression too long",
                ok=False,
            )

        tree = ast.parse(expr, mode="eval")
        if not _is_safe_node(tree):
            return ExecResult(
                expression=expression,
                result="Expression contains unsupported or unsafe constructs",
                ok=False,
            )

        value = eval(
            compile(tree, "<python_exec>", "eval"),
            {"__builtins__": {}},
            {"math": math},
        )
        return ExecResult(
            expression=expression,
            result=str(value),
            ok=True,
        )
    except Exception as e:
        return ExecResult(
            expression=expression,
            result=f"Execution error: {e}",
            ok=False,
        )
