"""What an embedded agent is given: :class:`AgentConfig` (and :class:`MemoryConfig`).

Only what the host passes: never the environment, a ``.env`` file or ``~/.neo`` — unless the host
opts in with :meth:`AgentConfig.from_env`. The agent turns it into NEO's LibrarySettings, which pin
what an embedded agent always runs with (confirmation on, task kinds on their legacy scripts,
Laya's learning off).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from neo.config import BrainName

__all__ = ["AgentConfig", "MemoryConfig"]

_GATE_TTL_S = 120.0  # neo.agent.confirm._TTL_S: an approval must come before the card lapses


@dataclass(frozen=True)
class MemoryConfig:
    """Long-term memory in a SQLite file of the host's choosing: a new (or empty) file, or one NEO
    made for an agent before — never a database of the host's own (NEO marks the one it makes)."""

    path: Path  # e.g. /srv/neo/u42/memory.sqlite3
    cloud: Literal["full", "profile", "off"] = "profile"  # how much of it rides along to the cloud model
    playbooks: bool = True  # recall and save the tool paths of past tasks
    summaries: bool = False  # summarise long conversations into memory


class AgentConfig(BaseModel):
    """One embedded agent's configuration. Frozen: build a new one to change it."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    name: str = "default"  # recorded on its audit lines
    data_dir: Path | None = None  # None: a private temp dir (0700), removed by aclose()
    tools: str | tuple[str, ...] = "web"  # presets, globs and names: ("web", "browser_*")
    exclude_tools: tuple[str, ...] = ()
    brain: BrainName = "gemini"
    fast_brain: BrainName = "groq"
    gemini_api_keys: tuple[SecretStr, ...] = ()
    groq_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    # "laya" (NEO.app's local model) is one process-wide model: not for embedded agents yet.
    reflex: Literal["off", "lite", "laya"] = "off"
    embeddings: Literal["off", "local"] = "off"  # "local" downloads a small embedding model
    memory: bool | MemoryConfig = False  # True: data_dir/memory.sqlite3 (needs data_dir)
    fast_paths: bool = True  # zero-LLM regex commands, limited to the agent's tools
    persona: str | None = None  # replaces the identity paragraph; the safety rules always stay
    instructions: str = ""  # appended to the system prompt
    include_username: bool = False
    home_location: str = ""
    browser_headless: bool = True
    protected_apps: tuple[str, ...] = ()  # NEO's input tools never click or type into these
    extra_secret_paths: tuple[Path, ...] = ()  # never read or written by NEO's tools (data_dir never is)
    models_dir: Path | None = None  # where a local embedding model is kept (default: the user cache dir)
    # (defaults as keywords: type checkers see only Field(default=...))
    max_steps: int = Field(default=24, ge=1)
    max_seconds: float = Field(default=180.0, gt=0)
    max_concurrent_tasks: int = Field(default=2, ge=1)
    approval_timeout_s: float = 110.0  # an inline approver's time to answer; below the gate's 120 s

    @model_validator(mode="before")
    @classmethod
    def _no_learning(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "learn" in data:
            raise ValueError(
                "learn isn't available to embedded agents: what NEO learns from requests is "
                "process-wide (NEO.app's), not kept per agent"
            )
        return data

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 100:
            raise ValueError("name must be 1 to 100 characters")
        return v

    @field_validator("tools", mode="before")
    @classmethod
    def _tools(cls, v: Any) -> Any:
        return v if isinstance(v, str) else tuple(v)

    @field_validator("data_dir", "models_dir", mode="before")
    @classmethod
    def _path(cls, v: Any) -> Any:
        return None if v in (None, "") else Path(os.path.expanduser(str(v)))

    @field_validator("approval_timeout_s")
    @classmethod
    def _timeout(cls, v: float) -> float:
        if not 0 < v < _GATE_TTL_S:
            raise ValueError(f"approval_timeout_s must be between 0 and {int(_GATE_TTL_S)} s (the card's lifetime)")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> AgentConfig:
        if self.reflex == "laya":
            raise ValueError(
                'reflex="laya" isn\'t available to embedded agents yet: Laya is one process-wide model '
                'that NEO.app trains on its user\'s requests. Use "off" or "lite"'
            )
        if self.reflex == "lite" and not self.gemini_api_keys:
            raise ValueError('reflex="lite" needs gemini_api_keys (it calls Gemini Flash-Lite itself)')
        if self.memory is True and self.data_dir is None:
            raise ValueError("memory=True keeps it in data_dir/memory.sqlite3: set data_dir, or pass MemoryConfig(path=...)")
        from neo.config import _refuse_neo_home

        for what, p in (
            ("data_dir", self.data_dir),
            ("models_dir", self.models_dir),
            ("memory path", self.memory.path if isinstance(self.memory, MemoryConfig) else None),
        ):
            if p is not None:
                _refuse_neo_home(Path(p), what)
        return self

    # ---- what NEO runs with ------------------------------------------------------------------
    @property
    def memory_config(self) -> MemoryConfig | None:
        if self.memory is False:
            return None
        if self.memory is True:
            assert self.data_dir is not None
            return MemoryConfig(path=Path(self.data_dir) / "memory.sqlite3")
        return self.memory

    def library_settings(self, data_dir: Path) -> Any:
        """NEO's settings for this agent, with `data_dir` its folder (neo.config.LibrarySettings)."""
        from neo.config import LibrarySettings

        keys = [k.get_secret_value().strip() for k in self.gemini_api_keys]
        keys = [k for k in keys if k]
        mem = self.memory_config
        values: dict[str, Any] = {
            "data_dir": data_dir,
            "brain": self.brain,
            "fast_brain": self.fast_brain,
            "gemini_api_key": keys[0] if keys else "",
            "gemini_api_keys": ",".join(keys[1:]),
            "groq_api_key": self.groq_api_key.get_secret_value() if self.groq_api_key else "",
            "anthropic_api_key": self.anthropic_api_key.get_secret_value() if self.anthropic_api_key else "",
            "reflex": self.reflex,
            "embeddings": self.embeddings,
            "memory_enabled": mem is not None,
            "playbooks": bool(mem and mem.playbooks),
            "summaries": bool(mem and mem.summaries),
            "cloud_memory": mem.cloud if mem else "off",
            "fast_paths": self.fast_paths,
            "persona": self.persona,
            "instructions": self.instructions,
            "include_username": self.include_username,
            "home_location": self.home_location,
            "browser_headless": self.browser_headless,
            "protected_apps": list(self.protected_apps),
            "extra_secret_paths": list(self.extra_secret_paths),
            "max_steps": self.max_steps,
            "max_seconds": self.max_seconds,
        }
        if mem is not None and Path(mem.path) != Path(data_dir) / "memory.sqlite3":
            values["memory_path"] = mem.path
        if self.embeddings == "local":
            values["models_path"] = self.models_dir or default_models_dir()
        elif self.models_dir is not None:
            values["models_path"] = self.models_dir
        return LibrarySettings(**values)

    # ---- explicit opt-in: the environment ----------------------------------------------------
    @classmethod
    def from_env(
        cls, *, prefix: str = "NEO_", env_files: Sequence[str | Path] = (), **overrides: Any
    ) -> AgentConfig:
        """Read keys (GEMINI_API_KEY, GEMINI_API_KEYS, GROQ_API_KEY, ANTHROPIC_API_KEY) and this
        config's fields (`prefix` + FIELD, e.g. NEO_BRAIN, NEO_TOOLS) from `env_files` (later ones
        win), then the process environment, then `overrides`. Only when asked for: an AgentConfig
        built directly reads none of it. Confirmation isn't a field — nothing here turns it off —
        and a file inside ~/.neo (NEO.app's own settings) is refused. NEO_DATA_DIR in the process
        environment is where NEO.app keeps its data, so it's refused as an agent's folder: name the
        agent's with another `prefix`, or pass data_dir= as an override."""
        from neo.config import _refuse_neo_home

        env: dict[str, str] = {}
        for f in env_files:
            p = Path(os.path.expanduser(str(f)))
            _refuse_neo_home(p, "env file")
            _refuse_neo_home(p.resolve(), "env file")  # a link to NEO.app's .env is that file
            env.update(_read_env_file(p))
        env.update(os.environ)
        values: dict[str, Any] = {}
        keys = [env.get("GEMINI_API_KEY", ""), *env.get("GEMINI_API_KEYS", "").split(",")]
        keys = list(dict.fromkeys(k.strip() for k in keys if k.strip()))
        if keys:
            values["gemini_api_keys"] = tuple(keys)
        for name in ("groq_api_key", "anthropic_api_key"):
            if env.get(name.upper(), "").strip():
                values[name] = env[name.upper()].strip()
        for name, field in cls.model_fields.items():
            if name in ("gemini_api_keys", "groq_api_key", "anthropic_api_key"):
                continue
            raw = env.get(f"{prefix}{name.upper()}")
            if raw is None or raw.strip() == "":
                continue
            values[name] = _env_value(raw, field.annotation)
        values.update(overrides)
        return cls(**values)


def default_models_dir() -> Path:
    """The user's cache folder for downloaded models (shared by every agent in the process)."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "neo2" / "models"


def _env_value(raw: str, annotation: Any) -> Any:
    """A tuple field: a JSON list or comma-separated words. Anything else as written (pydantic
    turns "true" into True, "12" into 12…)."""
    s = raw.strip()
    if "tuple" in str(annotation):
        if s.startswith("["):
            try:
                return tuple(json.loads(s))
            except ValueError:
                pass
        return tuple(p.strip() for p in s.split(",") if p.strip())
    return s


def _read_env_file(p: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    return {k: v for k, v in dotenv_values(p).items() if v is not None}
