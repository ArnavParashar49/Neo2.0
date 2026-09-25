"""Import every tool module so their @tool decorators register with the registry."""

from __future__ import annotations

import importlib
import sys

_MODULES = ["neo.tools.system", "neo.tools.web", "neo.tools.memory_tool", "neo.tools.browser"]
_MAC_MODULES = ["neo.tools.computer"]


def load_all() -> list[str]:
    loaded: list[str] = []
    mods = _MODULES + (_MAC_MODULES if sys.platform == "darwin" else [])
    for m in mods:
        try:
            importlib.import_module(m)
            loaded.append(m)
        except Exception as e:  # noqa: BLE001 — one broken tool module must not stop startup
            print(f"[tools] failed to load {m}: {e}")
    return loaded
