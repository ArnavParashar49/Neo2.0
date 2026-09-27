"""Import the tool modules so their @tool decorators register with the process catalog.

NEO.app loads everything (:func:`load_all`). An agent embedded in another app is given a
selection instead — presets ("web", "browser", "mac_read", "mac", "memory", "all", "none"), fnmatch
globs ("browser_*") and exact names, mixed — which :func:`resolve` turns into tool names and
:func:`load` imports only the modules of. Off macOS the portable tools are web_search, web_fetch,
weather, clock and memory, plus NEO's own browser when Playwright is installed.
"""

from __future__ import annotations

import fnmatch
import importlib
import logging
import sys
from collections.abc import Iterable, Sequence

from neo.runtime import UnsupportedPlatform

log = logging.getLogger(__name__)

_MODULES = [
    "neo.tools.system", "neo.tools.web", "neo.tools.weather", "neo.tools.memory_tool", "neo.tools.browser",
    "neo.tools.browse",
]  # fmt: skip
_MAC_ONLY = ["neo.tools.media"]
_MAC_MODULES = ["neo.tools.computer"]


def load_all() -> list[str]:
    loaded: list[str] = []
    # The Mac tools first: that is the catalog order NEO.app has always had (the fast paths break
    # ties by it, and Gemini Live declares its tools in it).
    mods = (_MAC_MODULES if sys.platform == "darwin" else []) + _MODULES + (_MAC_ONLY if sys.platform == "darwin" else [])
    for m in mods:
        try:
            importlib.import_module(m)
            loaded.append(m)
        except Exception as e:  # noqa: BLE001 — one broken tool module must not stop startup
            log.warning(f"[tools] failed to load {m}: {e}")
    if "neo.tools.tasks" in sys.modules:  # each task kind's tools as its mode shows them
        try:
            sys.modules["neo.tools.tasks"].apply_modes()
        except Exception as e:  # noqa: BLE001 — the tools keep the surface they were registered with
            log.warning(f"[tools] task modes not applied: {e}")
    return loaded


# ---- presets, for embedded agents ----------------------------------------------------------------

_COMPUTER = (
    "ax_tree", "ax_find", "screenshot", "apps_running", "ax_press", "ax_set_value", "click", "drag", "scroll",
    "type_text", "dictate", "hotkey", "click_text", "open_app", "activate_app", "quit_app", "close",
    "applescript", "open_url", "page_read", "safari_read", "finder_reveal", "shell", "read_file", "write_file",
    "edit_file", "trash",
)  # fmt: skip
_TASKS = ("mail_send", "mail_unread", "calendar_today", "calendar_add", "notes_create", "reminder_add")
_BROWSER = (
    "browser_open", "browser_read", "browser_click", "browser_type", "browser_press", "browser_scroll",
    "browser_back", "browser_tabs", "browser_screenshot", "browser_handoff", "browser_close",
)  # fmt: skip

PRESETS: dict[str, tuple[str, ...]] = {
    "web": ("web_search", "web_fetch", "weather", "clock"),
    "browser": _BROWSER,
    "mac_read": ("ax_tree", "ax_find", "screenshot", "apps_running", "mail_unread", "calendar_today", "page_read", "read_file"),
    "mac": _COMPUTER + _TASKS + ("volume", "brightness", "sleep_display", "timer", "search_site"),
    "memory": ("memory",),
    "none": (),
}
PRESETS["all"] = tuple(dict.fromkeys(n for p in ("web", "browser", "mac_read", "mac", "memory") for n in PRESETS[p]))

# Which module registers each tool. The task tools load with the computer tools (computer/__init__
# imports neo.tools.tasks at the place they have in the catalog).
MODULES_FOR: dict[str, str] = {
    **dict.fromkeys(("web_search", "web_fetch", "search_site"), "neo.tools.web"),
    "weather": "neo.tools.weather",
    **dict.fromkeys(("clock", "volume", "brightness", "sleep_display", "timer"), "neo.tools.system"),
    "memory": "neo.tools.memory_tool",
    **dict.fromkeys(_BROWSER, "neo.tools.browser"),
    **dict.fromkeys(_COMPUTER + _TASKS, "neo.tools.computer"),
}

# Tools that only work on macOS (their `requires` tag says "macos").
MACOS_ONLY: frozenset[str] = frozenset(PRESETS["mac"]) | frozenset(PRESETS["mac_read"])

# Never given to an embedded agent in this version, whatever the selection says: play_music is
# always decided by neo.decide, whose choices, learning and duplicate-request memory are
# process-wide (NEO.app's), not kept per agent yet; browse keeps the job it runs, and the one it
# may pick up again, process-wide too (and drives NEO.app's own browser and chat card).
REFUSED: dict[str, str] = {
    "play_music": "play_music isn't available to embedded agents yet: which app plays what is learned "
    "process-wide (NEO.app's), not per agent",
    "browse": "browse isn't available to embedded agents yet: its jobs are NEO.app's (one browser, one "
    "card, one job to pick up again); give the agent the browser tools and it browses itself",
}


def resolve(
    selection: str | Sequence[str], exclude: Iterable[str] = (), platform: str | None = None
) -> list[str]:
    """The tool names a selection means, in the "all" preset's order (an agent's registry keeps
    the catalog's order whatever this is).

    `selection`: presets, fnmatch globs and exact names, mixed (a string is one of them). A glob
    matches within the "all" preset. `exclude`: names or globs to leave out. Asking for a macOS
    preset or tool by name elsewhere raises UnsupportedPlatform; globs and "all" just leave those
    out. An unknown name, or one embedded agents can't have (play_music), raises ValueError."""
    plat = platform or sys.platform
    items = [selection] if isinstance(selection, str) else list(selection)
    universe = PRESETS["all"]
    want: set[str] = set()
    for raw in items:
        item = str(raw).strip()
        explicit = False
        if item in PRESETS:
            names: Sequence[str] = PRESETS[item]
            explicit = item in ("mac", "mac_read")
        elif item in REFUSED:
            raise ValueError(REFUSED[item])
        elif any(ch in item for ch in "*?["):
            names = [n for n in universe if fnmatch.fnmatchcase(n, item)]
        elif item in universe:
            names, explicit = [item], True
        else:
            raise ValueError(f"unknown tool or preset {item!r} (presets: {', '.join(PRESETS)})")
        if plat != "darwin":
            mac = [n for n in names if n in MACOS_ONLY]
            if mac and explicit:
                raise UnsupportedPlatform(f"{item!r} needs macOS ({', '.join(mac[:4])}…); this is {plat}")
            names = [n for n in names if n not in MACOS_ONLY]
        want.update(names)
    drop = [str(x).strip() for x in exclude]
    return [n for n in universe if n in want and not any(fnmatch.fnmatchcase(n, x) for x in drop)]


def modules_for(names: Iterable[str]) -> list[str]:
    """The modules that register these tools, each once, in the order NEO.app loads them."""
    wanted = {MODULES_FOR[n] for n in names if n in MODULES_FOR}
    order = _MAC_MODULES + _MODULES + _MAC_ONLY
    return [m for m in order if m in wanted]


def load(selection: str | Sequence[str] = "all", exclude: Iterable[str] = (), platform: str | None = None) -> dict[str, str]:
    """Import just the modules a selection needs (never NEO.app's task-mode switch: an embedded
    agent's tools keep the surface they were registered with). Returns {module: error} for the
    modules that failed to import — a missing optional dependency, say."""
    errors: dict[str, str] = {}
    for m in modules_for(resolve(selection, exclude, platform)):
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001 — reported to the caller, who decides
            errors[m] = f"{type(e).__name__}: {e}"
    return errors
