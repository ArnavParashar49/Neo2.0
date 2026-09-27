"""Registers the computer-control tools. Import this module to make them available.

Where things are:

- ``gates.py`` — every gate and protection helper (the confirmation gate on clicks, keys and drops,
  NEO's own windows and protected apps, Accessibility, what reading the screen taints). FROZEN:
  the tools call it through the module at call time.
- ``tools_ax.py`` (ax_tree, ax_find, ax_press, ax_set_value), ``tools_pointer.py`` (screenshot,
  click, drag, scroll), ``tools_keys.py`` (type_text, dictate, hotkey), ``tools_label.py``
  (click_text), ``tools_apps.py`` (apps, pages, shell and files) — the tools; ``phrases.py`` — the
  words of their fast paths.
- ``ax``, ``input``, ``pointer``, ``screen``, ``apps``, ``focus``, ``shell``, ``wait`` — the primitives.
  ``input._post`` is the one door for input events and ``ax.perform_action`` / ``ax.set_attribute``
  the one door for Accessibility actions: the controls self-test's watchers hook both.

Every name those files define reads here too (``neo2.core.tools.computer.click``, ``._ask_first``), and
a write to one here — a test's monkeypatch — lands in the file that defines it, where the tools
look it up: patching ``computer._guard_front`` patches the gate every tool calls.
"""

from __future__ import annotations

import sys
import types

from neo2.core.security import policy  # noqa: F401  (the gate module, as it has always been reachable here)
from neo2.core.tools.computer import (  # noqa: F401
    _catalog,
    apps,
    ax,
    gates,
    phrases,
    pointer,
    screen,
    shell,
    tools_apps,
    tools_ax,
    tools_keys,
    tools_label,
    tools_pointer,
)
from neo2.core.tools.computer import input as inp  # noqa: F401

# The catalog's order, which NEO.app has always had (the fast paths break ties by it, the tool
# picker's keyword fallback too, and Gemini Live declares its tools in it). A new tool takes its
# place here; one that doesn't comes after the rest. Names not defined yet are skipped. menu_command
# comes before click_text: its fast paths claim only words that name a menu ("click New Window in the
# File menu"), which click_text's generic "click X" would otherwise take.
ORDER: tuple[str, ...] = (
    "ax_tree", "ax_find", "screenshot", "map_board", "look", "apps_running", "ax_press", "ax_set_value", "click", "drag",
    "scroll", "type_text", "dictate", "hotkey", "menu_command", "click_text", "open_app", "activate_app",
    "quit_app", "close", "applescript", "open_url", "page_read", "safari_read",
    # (the task tools, neo2/core/tools/tasks.py, come here)
    "finder_reveal", "shell", "read_file", "write_file", "edit_file", "trash",
)  # fmt: skip

_TASKS_AFTER = ORDER.index("safari_read") + 1
_catalog.register(ORDER[:_TASKS_AFTER])
# The task tools were registered right here before they moved: loading them at this point keeps
# the catalog's order.
from neo2.core.tools import tasks  # noqa: E402, F401

_catalog.register(ORDER[_TASKS_AFTER:])
_catalog.register(_catalog.pending())


# ---- one namespace over the files ---------------------------------------------------------------

_FILES = (_catalog, gates, tools_ax, tools_pointer, tools_keys, tools_label, tools_apps, phrases)


def _homes() -> dict[str, types.ModuleType]:
    """Each name a file defines (not what it imports) -> that file; the first file wins."""
    out: dict[str, types.ModuleType] = {"_test_window_pid": ax}  # (ax holds it: allow_test_window sets it)
    for m in _FILES:
        for name, v in vars(m).items():
            if name.startswith("__") or isinstance(v, types.ModuleType) or name in out:
                continue
            if callable(v) and getattr(v, "__module__", m.__name__) != m.__name__:
                continue  # imported from elsewhere (ToolContext, fence, _catalog's tool…)
            out[name] = m
    return out


_HOME = _homes()


class _Package(types.ModuleType):
    """Reads fall through to the file that defines a name; writes (a test's monkeypatch) land there."""

    def __getattr__(self, name: str):
        home = _HOME.get(name)
        if home is None:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        return getattr(home, name)

    def __setattr__(self, name: str, value) -> None:
        home = _HOME.get(name)
        if home is not None:
            setattr(home, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        home = _HOME.get(name)
        if home is not None:
            delattr(home, name)
        else:
            super().__delattr__(name)


sys.modules[__name__].__class__ = _Package
