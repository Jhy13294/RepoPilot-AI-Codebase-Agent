# Read-only MCP server

RepoPilot exposes `get_file_tree`, `read_file`, and `search_code` to MCP clients over stdio. The
server reuses the same read-only registry as the CLI. It does not expose write, shell, Git, test, or
other approval-gated tools, and every file path remains confined to the selected repository root.

Install the project and start the server with a repository path:

```console
repopilot-mcp --repo <path>
```

Claude Desktop can launch the server with this `mcpServers` entry:

```json
{
  "mcpServers": {
    "repopilot": {
      "command": "repopilot-mcp",
      "args": ["--repo", "<path>"]
    }
  }
}
```

The process fails closed at startup when `<path>` does not exist or is not a directory. Tool-level
argument, path-jail, and read failures are returned as typed MCP error results without terminating
the server. Schema-level violations, such as an extra argument, are rejected by the MCP framework's
`inputSchema` validation; the client receives that framework's plain-text error rather than a
RepoPilot JSON error envelope.
