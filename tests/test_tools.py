"""Tests for the tool layer (layer 5)."""

from __future__ import annotations

import json

from zerostack.tools import build_registry
from zerostack.tools.base import Tool, ToolResult
from zerostack.tools.builtin import calculate, current_time
from zerostack.tools.mcp import discover_mcp_tools, load_mcp_config
from zerostack.tools.registry import ToolRegistry


class TestCalculate:
    def test_respects_operator_precedence(self):
        assert calculate("2 + 3 * 4").data["result"] == 14.0

    def test_respects_parentheses(self):
        assert calculate("(120 * 3) / 4").data["result"] == 90.0

    def test_handles_unary_minus(self):
        assert calculate("-5 + 10").data["result"] == 5.0

    def test_division_by_zero_is_an_error_not_a_crash(self):
        result = calculate("1 / 0")
        assert result.ok is False
        assert result.error

    def test_rejects_arbitrary_code(self):
        """The evaluator must never execute anything outside the operator allowlist."""
        for hostile in (
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "[].__class__",
            "1 if True else 2",
        ):
            result = calculate(hostile)
            assert result.ok is False, f"{hostile} was not rejected"

    def test_rejects_booleans(self):
        assert calculate("True + True").ok is False


class TestCurrentTime:
    def test_returns_iso_utc(self):
        result = current_time()
        assert result.ok is True
        assert "T" in result.data["iso8601"]

    def test_rejects_other_timezones(self):
        assert current_time("America/Chicago").ok is False


class TestRegistry:
    def test_builtin_tools_are_registered(self, tools):
        assert "calculate" in tools
        assert "current_time" in tools
        assert len(tools) == 2

    def test_unknown_tool_returns_an_error_result(self, tools):
        result = tools.call("does_not_exist")
        assert result.ok is False
        assert "unknown tool" in result.error

    def test_a_raising_handler_is_contained(self):
        def explode() -> ToolResult:
            raise RuntimeError("boom")

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="explode",
                description="always fails",
                parameters={"type": "object", "properties": {}},
                handler=explode,
            )
        )
        result = registry.call("explode")
        assert result.ok is False
        assert "RuntimeError" in result.error

    def test_schemas_are_json_serialisable(self, tools):
        json.dumps(tools.schemas())

    def test_build_registry_without_mcp(self):
        registry = build_registry(include_mcp=False)
        assert set(registry.names()) == {"calculate", "current_time"}


class TestMCPAdapter:
    def test_missing_config_yields_no_tools(self, tmp_path):
        assert load_mcp_config(tmp_path / "absent.json") == {}
        assert discover_mcp_tools(tmp_path / "absent.json") == []

    def test_malformed_config_is_tolerated(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_mcp_config(path) == {}

    def test_config_without_command_is_skipped(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"mcpServers": {"broken": {}}}), encoding="utf-8")
        assert load_mcp_config(path) == {"broken": {}}
        assert discover_mcp_tools(path) == []
