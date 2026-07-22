"""Read-only stdio MCP server backed by RepoPilot's shared tool registry."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import JsonValue

from app.tools.base import PathJail, ToolContext
from app.tools.registry import ToolRegistry


def build_read_only_registry_for_mcp() -> ToolRegistry:
    """Return the shared read-only registry used by RepoPilot's CLI."""
    from app.cli import _build_read_only_registry

    return _build_read_only_registry()


def list_registry_tools(registry: ToolRegistry) -> list[types.Tool]:
    """Convert the registry's LLM schemas into MCP tool declarations."""
    tools: list[types.Tool] = []
    for schema in registry.to_llm_schema():
        function = cast(dict[str, JsonValue], schema["function"])
        tools.append(
            types.Tool(
                name=cast(str, function["name"]),
                description=cast(str, function["description"]),
                inputSchema=cast(dict[str, Any], function["parameters"]),
            )
        )
    return tools


def dispatch_to_mcp(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, JsonValue],
    context: ToolContext,
) -> tuple[list[types.TextContent], bool]:
    """Dispatch a registry call and map its payload or error to MCP content."""
    result = registry.dispatch(name, arguments, context)
    if result.ok:
        payload = result.data.model_dump(mode="json") if result.data is not None else None
        return [_text_content(payload)], False

    error = result.error.model_dump(mode="json") if result.error is not None else None
    return [_text_content(error)], True


def create_server(jail: PathJail) -> Server:
    """Create the thin asynchronous MCP shell around the synchronous registry."""
    registry = build_read_only_registry_for_mcp()
    server = Server("repopilot")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return list_registry_tools(registry)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        context = ToolContext(run_id=f"mcp-{uuid4().hex[:12]}", jail=jail)
        content, is_error = dispatch_to_mcp(
            registry,
            name,
            cast(dict[str, JsonValue], arguments),
            context,
        )
        mcp_content: list[types.ContentBlock] = list(content)
        return types.CallToolResult(content=mcp_content, isError=is_error)

    return server


async def _run_stdio(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the read-only MCP server over stdio."""
    parser = argparse.ArgumentParser(description="Run RepoPilot's read-only stdio MCP server.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Repository root exposed through the read-only path jail.",
    )
    args = parser.parse_args(argv)
    jail = PathJail(args.repo)
    server = create_server(jail)
    asyncio.run(_run_stdio(server))


def _text_content(value: object) -> types.TextContent:
    return types.TextContent(
        type="text",
        text=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
    )


if __name__ == "__main__":
    main()
