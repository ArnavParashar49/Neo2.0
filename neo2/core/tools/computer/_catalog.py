"""How the computer tools join the catalog: in one fixed order, whichever file defines them.

The catalog's order is behaviour: the fast paths break ties by it, the tool picker's keyword
fallback too, and Gemini Live declares its tools in it. The tools are defined across several files
(tools_ax, tools_pointer, tools_keys, tools_label, tools_apps), so ``tool`` here only takes note of
each one; the package (``neo2/core/tools/computer/__init__.py``) registers them in its ``ORDER`` once
all are defined.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from neo2.core.agent.registry import tool as _register

_OBJ = {"type": "object", "properties": {}}

# name -> (description, parameters, options, handler), in definition order, until registered
_PENDING: dict[str, tuple[str, dict | None, dict[str, Any], Callable]] = {}


def _p(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


def tool(name: str, description: str, parameters: dict | None = None, **options: Any) -> Callable[[Callable], Callable]:
    """:func:`neo2.core.agent.registry.tool` for a computer tool: every one drives this Mac
    (``requires={"macos"}``), and it joins the catalog when the package registers it."""
    options.setdefault("requires", frozenset({"macos"}))

    def deco(fn: Callable) -> Callable:
        _PENDING[name] = (description, parameters, options, fn)
        return fn

    return deco


def register(names: Iterable[str]) -> None:
    """Register these pending tools, in this order (names not defined yet are skipped)."""
    for name in names:
        if name in _PENDING:
            description, parameters, options, fn = _PENDING.pop(name)
            _register(name, description, parameters, **options)(fn)


def pending() -> list[str]:
    """Tools defined but not registered yet."""
    return list(_PENDING)
