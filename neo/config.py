"""NEO settings — environment-driven, with sane free/local-first defaults.

Everything lives under ``~/.neo`` (data dir) unless overridden. API keys come from the
environment or a ``.env`` file in the project root; nothing is written to Git.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Keys and settings: ~/.neo/.env (where NEO.app's installed runtime looks), and a checkout's own
# .env on top when running from source.
USER_ENV = Path.home() / ".neo" / ".env"

BrainName = Literal["gemini", "gemini_lite", "groq", "local", "claude"]
VoiceBackend = Literal["auto", "live", "local", "off"]
DecideMode = Literal["legacy", "shadow", "on", "off"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(USER_ENV, PROJECT_ROOT / ".env"),  # later files win
        env_file_encoding="utf-8",
        env_prefix="NEO_",
        extra="ignore",
    )

    # ---- keys (no prefix — read the conventional names) -------------------------------
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_api_keys: str = Field(
        default="", validation_alias="GEMINI_API_KEYS"
    )  # extra keys, comma-separated
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")

    # ---- brains ----------------------------------------------------------------------
    brain: BrainName = "gemini"  # primary agentic brain
    fast_brain: BrainName = "groq"  # chat-only turns; falls back to `brain` if no key
    offline_brain: BrainName = "local"
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_models: list[str] = ["gemini-3.7-flash", "gemini-3.5-flash"]
    gemini_lite_model: str = "gemini-3.5-flash-lite"
    gemini_live_model: str = "gemini-3.8-live"
    groq_model: str = "openai/gpt-oss-120b"
    local_model: str = "mlx-community/gemma-4-12B-it-4bit"
    claude_model: str = "claude-opus-5"

    # ---- reflex ----------------------------------------------------------------------
    reflex: Literal["laya", "lite", "off"] = "laya"
    # The Laya checkpoint under the reflex heads: multilingual also understands Hindi and Hinglish
    # and embeds ~2x faster; english is the older one (its heads must be retrained after a switch).
    reflex_encoder: Literal["multilingual", "english"] = "multilingual"
    laya_device: str = "mps"
    reflex_confidence_floor: float = 0.55  # below this, escalate to the lite model
    reflex_tool_floor: float = 0.85  # Laya may pick the tool itself above this confidence
    home_location: str = ""  # e.g. "Dubai" — where "the weather" means, if not remembered
    learn: bool = True  # NEO_LEARN=off: Laya stops recording requests to learn from
    # NEO_LEARN_TEACHER=local|cloud: while idle, have a model re-check the requests Laya was least
    # sure about (reflex/teacher.py) — the local model, or only the provider that already saw each one.
    learn_teacher: Literal["off", "local", "cloud"] = "off"
    # How much of NEO's memory rides along to the cloud models: full | profile | off
    cloud_memory: Literal["full", "profile", "off"] = "full"

    @field_validator("cloud_memory", mode="before")
    @classmethod
    def _cloud_memory(cls, v):
        """Forgiving about how it's typed ("Off", "none", " profile "): a typo in the settings
        file must never stop NEO from starting — and when unsure, share less."""
        s = str(v or "full").strip().lower()
        if s in ("full", "all", "on", "yes", "true"):
            return "full"
        if s in ("off", "none", "no", "false", "0", "nothing"):
            return "off"
        return "profile"

    @field_validator("learn_teacher", mode="before")
    @classmethod
    def _learn_teacher(cls, v):
        """Anything but exactly local or cloud is off: this sends requests to a model, so it only
        runs when asked for (and a typo must never stop NEO from starting)."""
        s = str(v or "off").strip().lower()
        return s if s in ("local", "cloud") else "off"

    @field_validator("reflex_encoder", mode="before")
    @classmethod
    def _reflex_encoder(cls, v):
        """A typo must not stop NEO from starting: anything but english is the default."""
        return "english" if str(v or "").strip().lower() in ("english", "en") else "multilingual"

    # ---- voice -----------------------------------------------------------------------
    voice: VoiceBackend = "auto"  # auto = Live if key + quota, else local cascade
    listen_timeout_s: float = 5.0  # close the voice session after this much silence from you
    # Voice commands the fast paths fully cover run the moment you stop talking, without
    # waiting for Gemini Live to pick the tool (questions still go to the model).
    live_command_lane: bool = True
    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    tts_voice: str = "af_heart"
    live_voice: str = "Charon"

    # ---- agent -----------------------------------------------------------------------
    max_steps: int = 24
    max_seconds: float = 180.0  # wall-clock budget for one agent run
    max_tool_result_chars: int = 6000
    confirm_destructive: bool = True

    # ---- task kinds (neo/decide) -------------------------------------------------------
    # How each kind runs (notes, reminders, calendar, mail…): "legacy" = today's scripts,
    # "shadow" = those, while the new path only logs what it would have chosen, "on" = the new
    # path, "off" = a new kind stays unavailable. A kind left out gets its release default.
    # NEO_DECIDE_MODES="add_event=on,read_calendar=shadow"
    decide_modes: Annotated[dict[str, DecideMode], NoDecode] = {}
    decide_kind_routing: bool = False  # Laya may name a task kind no fast path covered (after its eval)
    notes_dir: Path = Path.home() / "Documents" / "NEO Notes"  # a note no notes app could take
    # "play …": "web" plays it in the browser (on play_site) unless the user names an app ("on
    # Spotify"); "apps" lets NEO pick among the players on this Mac first. The user's choice.
    play_default: Literal["web", "apps"] = "web"
    play_site: str = "youtube"  # the site word the web prefers ("" = whichever video site the search finds)

    @field_validator("decide_modes", mode="before")
    @classmethod
    def _decide_modes(cls, v):
        """A dict, JSON, or "kind=mode, kind=mode". An unknown mode is dropped: the kind keeps
        its default, and a typo never stops NEO from starting."""
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = json.loads(v)
            except ValueError:
                v = {}
        elif isinstance(v, str):
            v = dict(p.split("=", 1) for p in v.replace(";", ",").split(",") if "=" in p)
        if not isinstance(v, dict):
            return {}
        pairs = ((str(k).strip().lower(), str(m).strip().lower()) for k, m in v.items())
        return {k: m for k, m in pairs if k and m in get_args(DecideMode)}

    # ---- server / ui -----------------------------------------------------------------
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765

    # ---- paths -----------------------------------------------------------------------
    data_dir: Path = Path.home() / ".neo"

    @property
    def gemini_keys(self) -> list[str]:
        """All Gemini keys in priority order (free-tier quota is per key, so more keys = more quota)."""
        out: list[str] = []
        for k in [self.gemini_api_key, *self.gemini_api_keys.split(",")]:
            k = k.strip()
            if k and k not in out:
                out.append(k)
        return out

    @field_validator("data_dir", "notes_dir", mode="before")
    @classmethod
    def _expand(cls, v):  # NEO_DATA_DIR=~/.neo must not create a literal "~" directory
        return Path(str(v)).expanduser().resolve()

    @property
    def audit_log(self) -> Path:
        return self.data_dir / "audit.log"

    @property
    def memory_db(self) -> Path:
        return self.data_dir / "memory.sqlite3"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)
        # Memory, transcripts' traces, keys and logs: this user's only.
        try:
            self.data_dir.chmod(0o700)
        except OSError:
            pass


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        s = Settings()
        _refuse_real_data_under_tests(s.data_dir)
        s.ensure_dirs()
        _settings = s
    return _settings


# The release default of each kind that has an evidence-chosen driver: "shadow" until the user,
# present, has checked the new path once (then NEO_DECIDE_MODES="<kind>=on"). Any other kind
# runs its legacy code.
DECIDE_DEFAULTS: dict[str, DecideMode] = dict.fromkeys(
    ("write_note", "add_reminder", "add_event", "read_calendar", "read_mail", "open_app", "web_search"),
    "shadow",
)


def decide_mode(kind: str) -> DecideMode:
    """How a task kind runs now: the user's `decide_modes` setting, else its release default."""
    return settings().decide_modes.get(kind, DECIDE_DEFAULTS.get(kind, "legacy"))


def _refuse_real_data_under_tests(data_dir: Path) -> None:
    """A test (or a scratch harness borrowing the suite's fixtures) that isn't isolated must fail
    loudly, never read, train on or erase the user's real memory. This happened once: a scratch
    conftest skipped the isolation fixture and an erase test wiped the real ~/.neo."""
    if "pytest" not in sys.modules and "PYTEST_CURRENT_TEST" not in os.environ:
        return
    real = (Path.home() / ".neo").resolve()
    if data_dir.resolve() == real or real in data_dir.resolve().parents:
        raise RuntimeError(f"refusing to use the real {real} under pytest: set NEO_DATA_DIR to a temporary directory")
