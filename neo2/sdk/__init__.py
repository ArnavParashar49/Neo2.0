"""The SDK's implementation: import from :mod:`neo2` itself (its names are loaded on first use).

- ``config``: AgentConfig, MemoryConfig — what an agent is given, turned into LibrarySettings.
- ``agent``: AsyncAgent — requests, approvals, events, host tools and NEO's tools one by one.
- ``sync``: Agent — the same on a private event loop, for code that isn't async.
- ``approval``, ``results``: the approval and result types.
- ``hosttools``: host_tool — a host's own function as one of an agent's tools.
- ``tools_api``: ToolTurn and the export formats, for NEO's tools in a host's own model loop.
"""
