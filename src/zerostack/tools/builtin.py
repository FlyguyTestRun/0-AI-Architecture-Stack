"""Built in tools.

These are deliberately mundane. The point of the reference app is that the tool
plumbing works end to end, not that the tools are impressive. Real deployments add
GitHub, Slack and database tools through MCP.
"""

from __future__ import annotations

import ast
import operator
from datetime import UTC, datetime
from typing import Any

from zerostack.tools.base import Tool, ToolResult

_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Evaluate a arithmetic AST node.

    Parsing with ``ast`` and walking an explicit operator allowlist avoids ``eval``,
    which would let a prompt injected expression execute arbitrary code.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {type(node).__name__}")


# Long enough for any expression a person types, short enough that parsing it
# cannot exhaust the stack. Expression text can originate in a retrieved
# document, so its length is not something to trust.
MAX_EXPRESSION_LENGTH = 500


def calculate(expression: str) -> ToolResult:
    """Evaluate an arithmetic expression safely."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return ToolResult(
            ok=False,
            content="",
            error=f"expression exceeds {MAX_EXPRESSION_LENGTH} characters",
        )
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
    except (
        SyntaxError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        # Deeply nested input exhausts the stack during parsing or evaluation.
        # It is a bad input, not a crash, so it is reported like any other.
        RecursionError,
        MemoryError,
    ) as exc:
        return ToolResult(ok=False, content="", error=str(exc) or type(exc).__name__)
    return ToolResult(ok=True, content=str(value), data={"result": value})


def current_time(timezone_name: str = "UTC") -> ToolResult:
    """Return the current timestamp. Only UTC is supported without extra packages."""
    if timezone_name.upper() != "UTC":
        return ToolResult(ok=False, content="", error="only UTC is supported by this build")
    now = datetime.now(UTC).isoformat()
    return ToolResult(ok=True, content=now, data={"iso8601": now})


def build_builtin_tools() -> list[Tool]:
    """Return the tools that are always available."""
    return [
        Tool(
            name="calculate",
            description="Evaluate a arithmetic expression, for example '(120 * 3) / 4'.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
            handler=calculate,
        ),
        Tool(
            name="current_time",
            description="Return the current UTC timestamp in ISO 8601 format.",
            parameters={
                "type": "object",
                "properties": {"timezone_name": {"type": "string", "default": "UTC"}},
                "required": [],
            },
            handler=current_time,
        ),
    ]
