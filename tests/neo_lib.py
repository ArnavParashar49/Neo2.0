"""Helpers for tests that embed NEO agents (neo.runtime) — built directly, as the SDK will."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neo import runtime
from neo.agent.registry import catalog
from neo.config import LibrarySettings


def make_runtime(
    data_dir: Path, name: str = "agent", tools: Iterable[str] = (), *, library: bool = True, **settings: Any
) -> runtime.Runtime:
    """An agent with its own LibrarySettings (only `settings` passed in), a strict copy of these
    catalog tools, and its data dir created. Turns library mode on, as the SDK does."""
    from neo.tools import load_all

    if not runtime.library_mode():  # NEO.app's loader (task modes read the app's settings)
        load_all()
    s = LibrarySettings(data_dir=data_dir, **settings)
    s.ensure_dirs()
    if library:
        runtime.set_library_mode()
    return runtime.Runtime(name, s, catalog().subset(tools))
