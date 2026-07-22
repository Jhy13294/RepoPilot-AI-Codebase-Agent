import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent

from app.cli import _build_read_only_registry
from app.mcp.server import (
    build_read_only_registry_for_mcp,
    create_server,
    dispatch_to_mcp,
    list_registry_tools,
    main,
)
from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry


def _tool_names(registry: ToolRegistry) -> list[str]:
    return [
        cast(str, cast(dict[str, Any], schema["function"])["name"])
        for schema in registry.to_llm_schema()
    ]


def _context(root: Path) -> ToolContext:
    return ToolContext(run_id="mcp-test", jail=PathJail(root))


def _decoded_text(content: list[TextContent]) -> dict[str, Any]:
    assert len(content) == 1
    value = json.loads(content[0].text)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_mcp_tools__mirror_read_only_registry_names_schemas_and_risk() -> None:
    registry = build_read_only_registry_for_mcp()
    cli_registry = _build_read_only_registry()

    advertised = list_registry_tools(registry)
    advertised_by_name = {tool.name: tool for tool in advertised}
    assert [tool.name for tool in advertised] == _tool_names(cli_registry)
    assert set(advertised_by_name) == {"get_file_tree", "read_file", "search_code"}

    for name, tool in advertised_by_name.items():
        spec = registry._tools[name][0]
        assert tool.inputSchema == spec.args_schema.model_json_schema()
        assert spec.risk_level != "high"


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    (
        ("read_file", {"path": "notes.txt"}, {"content": "alpha\nbeta"}),
        ("search_code", {"query": "beta"}, {"total_found": 1}),
        ("get_file_tree", {}, {"root": "."}),
    ),
)
def test_dispatch_to_mcp__returns_read_only_success_payloads(
    tmp_path: Path,
    name: str,
    arguments: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    (tmp_path / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    content, is_error = dispatch_to_mcp(
        build_read_only_registry_for_mcp(),
        name,
        arguments,
        _context(tmp_path),
    )

    assert is_error is False
    payload = _decoded_text(content)
    assert payload.items() >= expected.items()


def test_dispatch_to_mcp__returns_path_jail_error_without_leaking_content(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = "outside-secret-value"
    (tmp_path / "secret.txt").write_text(secret, encoding="utf-8")

    content, is_error = dispatch_to_mcp(
        build_read_only_registry_for_mcp(),
        "read_file",
        {"path": "../secret.txt"},
        _context(workspace),
    )

    assert is_error is True
    error = _decoded_text(content)
    assert error["type"] == ErrorType.PathJailError
    assert secret not in content[0].text


def test_dispatch_to_mcp__returns_unknown_tool_error_without_raising(tmp_path: Path) -> None:
    content, is_error = dispatch_to_mcp(
        build_read_only_registry_for_mcp(),
        "missing",
        {},
        _context(tmp_path),
    )

    assert is_error is True
    error = _decoded_text(content)
    assert error["type"] == ErrorType.InvalidArgsError
    assert "missing" in error["message"]


def test_mcp_server__supports_in_memory_list_and_read_file_round_trip(tmp_path: Path) -> None:
    (tmp_path / "round-trip.txt").write_text("through MCP\n", encoding="utf-8")
    server = create_server(PathJail(tmp_path))

    async def exercise_server() -> None:
        async with create_connected_server_and_client_session(
            server,
            raise_exceptions=True,
        ) as session:
            listed = await session.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "get_file_tree",
                "read_file",
                "search_code",
            }

            result = await session.call_tool("read_file", {"path": "round-trip.txt"})
            assert result.isError is False
            assert len(result.content) == 1
            text = result.content[0]
            assert isinstance(text, TextContent)
            payload = json.loads(text.text)
            assert payload["path"] == "round-trip.txt"
            assert payload["content"] == "through MCP"

    asyncio.run(exercise_server())


def test_mcp_main__fails_closed_for_non_directory_repo(tmp_path: Path) -> None:
    repo_file = tmp_path / "not-a-directory"
    repo_file.write_text("file\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must exist and be a directory"):
        main(["--repo", str(repo_file)])
