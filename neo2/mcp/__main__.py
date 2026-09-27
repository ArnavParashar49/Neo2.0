"""``python -m neo2.mcp``: NEO as an MCP server over stdio (see :mod:`neo2.mcp`)."""

import sys

from neo2.mcp import main

if __name__ == "__main__":
    sys.exit(main())
