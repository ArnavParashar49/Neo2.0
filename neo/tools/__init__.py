"""Import every tool module so their @tool decorators register with the registry."""

from __future__ import annotations

import importlib
import sys

_MODULES = [
    "neo.tools.system", "neo.tools.web", "neo.tools.weather", "neo.tools.memory_tool", "neo.tools.browser",
    "neo.tools.browse",
]  # fmt: skip
_MAC_ONLY = ["neo.tools.media"]
_MAC_MODULES = ["neo.tools.computer", "neo.tools.selftest"]


def load_all() -> list[str]:
    loaded: list[str] = []
    mods = _MODULES + (_MAC_MODULES + _MAC_ONLY if sys.platform == "darwin" else [])
    for m in mods:
        try:
            importlib.import_module(m)
            loaded.append(m)
        except Exception as e:  # noqa: BLE001 — one broken tool module must not stop startup
            print(f"[tools] failed to load {m}: {e}")
    if "neo.tools.tasks" in sys.modules:  # each task kind's tools as its mode shows them
        try:
            sys.modules["neo.tools.tasks"].apply_modes()
        except Exception as e:  # noqa: BLE001 — the tools keep the surface they were registered with
            print(f"[tools] task modes not applied: {e}")
    return loaded
