# ZS-004: Tool access goes through MCP

Status: Accepted
Date: 2026-08-24

## Context

Projects built on this baseline will need GitHub, Slack, database and filesystem
access. Writing a bespoke integration per service per project is the largest
recurring cost in this kind of work, and none of it is differentiating.

## Decision

The tool layer registers built in tools and MCP discovered tools in one registry with
one shape. Servers are declared in `mcp.json`, using the conventional MCP client
configuration shape, so a server configured once is configured for every client that
reads it.

The orchestrator resolves tools by name from the registry. It has no idea whether a
tool is local or remote.

Tool parameters are JSON Schema, which is both what MCP servers publish and what
function calling APIs expect, so no translation layer is needed.

Tool selection in this baseline is rule based rather than model driven. That keeps the
tool path deterministic and testable, and it works with any provider including the
offline one. Native function calling is a change confined to `tool_node`.

## Consequences

Good:

- A new integration is a config entry, not a code change.
- Tool configuration is portable across every MCP capable client the team uses.
- Discovery failures are contained per server: one unreachable server does not stop
  the others registering.

Costs:

- MCP discovery needs the `mcp` package and a working server command. When either is
  missing the layer logs and registers nothing, which is quiet by design but can look
  like nothing happened. `zerostack doctor` shows the registered tool count so the
  state is visible.
- Rule based tool selection does not generalise. Projects needing open ended tool use
  will replace `tool_node` with model driven selection.
