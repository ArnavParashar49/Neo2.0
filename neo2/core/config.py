"""NEO settings — environment-driven, with sane free/local-first defaults.

Everything lives under ``~/.neo`` (data dir) unless overridden. API keys come from the
environment or a ``.env`` file in the project root; nothing is written to Git.

An agent embedded in another app (neo2.core.runtime) has :class:`LibrarySettings` instead: only what
the host passes in — never the environment, ~/.neo/.env or ~/.neo itself — and
:func:`settings` returns them while that agent is bound.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, PydanticBaseSettingsSource, SettingsConfigDict

from neo2.core import effects, runtime

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # neo2/core/config.py → the checkout (or site-packages)
# Keys and settings: ~/.neo/.env (where NEO.app's installed runtime looks), and a checkout's own
# .env on top when running from source (a checkout has a pyproject.toml; installed as a package,
# PROJECT_ROOT is site-packages, whose .env is nobody's).
USER_ENV = Path.home() / ".neo" / ".env"
NOTES_DIR = Path.home() / "Documents" / "NEO Notes"  # where a note goes that no notes app could take


def _env_files(root: Path) -> tuple[Path, ...]:
    return (USER_ENV, root / ".env") if (root / "pyproject.toml").exists() else (USER_ENV,)


_ENV_FILES = _env_files(PROJECT_ROOT)

BrainName = Literal["gemini", "gemini_lite", "groq", "local", "claude", "groq_vision", "cerebras", "cerebras_vision"]
VoiceBackend = Literal["auto", "live", "local", "off"]
DecideMode = Literal["legacy", "shadow", "on", "off"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,  # later files win
        env_file_encoding="utf-8",
        env_prefix="NEO_",
        extra="ignore",
        hide_input_in_errors=True,  # a bad value's error never prints it (a key, say)
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):  # pydantic-settings calls it with these, by name
        """With effects off (a test, a scratch script run with NEO_NO_EFFECTS=1), never the .env
        files: ~/.neo/.env holds the user's real API keys."""
        if effects.why_off():
            return init_settings, env_settings, file_secret_settings
        return init_settings, env_settings, dotenv_settings, file_secret_settings

    # ---- keys (no prefix — read the conventional names) -------------------------------
    # repr=False: a logged or printed Settings never shows a key.
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY", repr=False)
    gemini_api_keys: str = Field(
        default="", validation_alias="GEMINI_API_KEYS", repr=False
    )  # extra keys, comma-separated
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY", repr=False)
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY", repr=False)
    # A paid Cerebras key (the user adds it themselves): its brains join the chains only with it
    # set and a daily budget above 0 (neo2.core.providers.cerebras_hooks.enabled).
    cerebras_api_key: str = Field(default="", validation_alias="CEREBRAS_API_KEY", repr=False)

    # ---- brains ----------------------------------------------------------------------
    brain: BrainName = "gemini"  # primary agentic brain
    fast_brain: BrainName = "groq"  # chat-only turns; falls back to `brain` if no key
    offline_brain: BrainName = "local"
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_models: list[str] = ["gemini-3.7-flash", "gemini-3.5-flash"]
    gemini_lite_model: str = "gemini-3.5-flash-lite"
    gemini_live_model: str = "gemini-3.8-live"
    groq_model: str = "openai/gpt-oss-120b"
    # Groq's model that sees pictures: the last seeing fallback (neo2.core.providers.EYES), on GROQ_API_KEY.
    # It is sent a screenshot only when no other brain that sees can answer. "" = none.
    groq_vision_model: str = "qwen/qwen3.8-27b"
    # The seeing fallbacks (Flash-Lite, Claude, Groq's vision model) for a picture the chain's own
    # brains can't answer. "auto": they join — except for a user who put the local brain first and
    # named no cloud brain that sees (their pictures stayed on the Mac: a fallback doesn't change
    # that on its own). "on": they join for that user too.
    cloud_eyes: Literal["auto", "on"] = "auto"
    local_model: str = "mlx-community/gemma-4-12B-it-4bit"
    claude_model: str = "claude-opus-5"
    # Cerebras (neo2.core.providers.cerebras): the fast text brain ahead of Groq, and a seeing one
    # ahead of Groq's vision model. Spend is usage × price per model ($ per million tokens in / out;
    # JSON entries override or add), against a daily budget: 80% is said, then it is skipped until
    # midnight. 0 turns Cerebras off.
    cerebras_model: str = "gpt-oss-120b"
    cerebras_vision_model: str = "qwen-3.8-27b"
    cerebras_daily_budget: float = 0.50
    cerebras_prices: Annotated[dict[str, Any] | str, NoDecode] = ""

    @field_validator("cerebras_daily_budget", mode="before")
    @classmethod
    def _cerebras_budget(cls, v):
        """A typo (or "$1") must not stop NEO from starting: an unreadable budget is the default."""
        try:
            f = float(str(v).strip().lstrip("$"))
        except (TypeError, ValueError):
            return 0.50
        return f if f == f and 0 <= f < float("inf") else 0.50

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
    # Apps NEO's input tools never click, press, drag, scroll or type into (NEO's own overlay
    # never, always): a host's own approval dialog, for one. An entry is the app's name as the
    # Dock shows it, its bundle id or its executable's name, any case: '["MyApp"]' or
    # '["com.example.myapp"]' in NEO_PROTECTED_APPS. The bundle id is the one that can't drift.
    protected_apps: list[str] = []
    fast_paths: bool = True  # zero-LLM regex commands (neo2.core.agent.fastpath)

    # ---- computer controls (neo2/core/tools/computer) -----------------------------------------
    # The thinking level a step on the screen starts at (then one up after a miss, back down after
    # a success — neo2.core.providers.effort). Today's is the agent's own ("high"); "low" answers in
    # ~1–2 s instead of ~10–24 s. It changes only after a measured A/B on the controls self-test's
    # window (≥10 runs per arm).
    screen_effort: Literal["minimal", "low", "medium", "high"] = "high"
    # Several actions per reply, and a run that ends on `finish` once NEO verified its last actions
    # (neo2.core.agent.batch): off until the user's A/B on the self-test window (DECISIONS 47).
    batch_actions: bool = False
    # The replay lane (neo2.core.agent.replay): a request a model did the same way twice in the same
    # app is done the third time with no model call. None: on in NEO.app, off where effects are off.
    replay_flows: bool | None = None
    # A first look at the requested app, taken before the first model call — only when that app
    # is already in front (or none is named and the front one is where NEO last worked), never a
    # protected app's, with a picture only when the screen is needed. Off: the model looks itself.
    prefetch_look: bool = True
    prefetch_look_s: float = 0.5  # the most that first look may add before the first model call
    # Laya's resolver is loaded at the wake word (the start of speech), not when the words are in:
    # its first answer drops from ~0.5 s to ~45 ms once the user has spoken for about a second.
    prewarm_resolver: bool = True
    shot_format: Literal["jpeg", "png"] = "jpeg"  # screenshots for a model: JPEG q85, or PNG
    # Laya's UI resolver ("click the start button", "start wala button dabao"): "shadow" logs
    # what it would press and acts only on an exact label; "live" lets every stage that passed
    # its eval act; "off" leaves UI commands to the model. The confirmation gate asks either way.
    resolver: Literal["live", "shadow", "off"] = "shadow"
    # Laya's pick counts only this sure (0.95 among more than 60 controls), and only agreeing with
    # MiniLM: 0.7 measured 93.5–94.4% right on the grounding eval, 0.9 98.4–98.7% (decision 40: ≥97%).
    resolver_laya_p: float = 0.9
    # The shadow stages allowed to act once the resolver is "live", each after passing its eval: a
    # comma list of alias, os_strings, meaning ("" = none: the exact-label stage alone acts).
    resolver_promoted: str = ""
    local_grounder: str = ""  # a local vision model for pointing at things ("" = none: the user picks one)
    ground_port: int = 8081  # where that model is served, on this Mac only
    sck_capture: bool = False  # ScreenCaptureKit one-shot captures instead of the screencapture tool
    sck_stream: bool = False  # ScreenCaptureKit live frames during a pixel task

    @field_validator("screen_effort", "shot_format", "resolver", mode="before")
    @classmethod
    def _controls_choice(cls, v, info):
        """A typo must not stop NEO from starting, and never turns a feature on: an unknown value
        is the default (today's behaviour)."""
        s = str(v or "").strip().lower()
        allowed = {"screen_effort": ("minimal", "low", "medium", "high"), "shot_format": ("jpeg", "png"),
                   "resolver": ("live", "shadow", "off")}[info.field_name]  # fmt: skip
        default = {"screen_effort": "high", "shot_format": "jpeg", "resolver": "shadow"}[info.field_name]
        s = {"jpg": "jpeg"}.get(s, s)
        return s if s in allowed else default

    @field_validator("cloud_eyes", mode="before")
    @classmethod
    def _cloud_eyes(cls, v):
        """Only a clear yes sends a local-first user's pictures to a cloud fallback: anything else
        (a typo too) is auto."""
        return "on" if str(v or "").strip().lower() in ("on", "true", "yes", "1", "always") else "auto"

    # ---- memory & prompt ---------------------------------------------------------------
    embeddings: Literal["local", "off"] = "local"  # off: keyword-only recall, no local model
    memory_enabled: bool = True  # long-term memory (memory.sqlite3) at all
    playbooks: bool = True  # recall and save the tool paths of past tasks (needs memory)
    summaries: bool = True  # session summaries (needs memory)
    persona: str | None = None  # replaces the identity paragraph of the prompt; the rules stay
    instructions: str = ""  # appended to the system prompt
    include_username: bool = True  # the macOS user name in the prompt's context line
    browser_headless: bool = False  # NEO's own browser window (visible: the user can step in)

    # ---- task kinds (neo2/core/decide) -------------------------------------------------------
    # How each kind runs (notes, reminders, calendar, mail…): "legacy" = today's scripts,
    # "shadow" = those, while the new path only logs what it would have chosen, "on" = the new
    # path, "off" = a new kind stays unavailable. A kind left out gets its release default.
    # NEO_DECIDE_MODES="add_event=on,read_calendar=shadow"
    decide_modes: Annotated[dict[str, DecideMode], NoDecode] = {}
    decide_kind_routing: bool = False  # Laya may name a task kind no fast path covered (after its eval)
    notes_dir: Path = NOTES_DIR
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

    # ---- NEO's own browser -------------------------------------------------------------
    # "chat": it runs headless and shows live in the chat's browser card, where the user can watch
    # it, take it over, or pop it out into its own window. "window": a visible Chromium window.
    browser_view: Literal["chat", "window"] = "chat"

    @field_validator("browser_view", mode="before")
    @classmethod
    def _browser_view(cls, v):
        """A typo must not stop NEO from starting: anything but a visible window is the chat."""
        s = str(v or "").strip().lower()
        return "window" if s in ("window", "headed", "visible", "separate") else "chat"

    # ---- server / ui -----------------------------------------------------------------
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765

    # ---- paths -----------------------------------------------------------------------
    data_dir: Path = Path.home() / ".neo"
    models_path: Path | None = None  # downloaded models (default: data_dir/models)
    memory_path: Path | None = None  # the memory database (default: data_dir/memory.sqlite3)
    extra_secret_paths: list[Path] = []  # never read or written by NEO's tools (data_dir never is)

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

    @field_validator("models_path", "memory_path", mode="before")
    @classmethod
    def _expand_opt(cls, v):
        return None if v in (None, "") else Path(str(v)).expanduser().resolve()

    @field_validator("extra_secret_paths", mode="before")
    @classmethod
    def _expand_all(cls, v):
        return [Path(str(p)).expanduser().resolve() for p in (v or [])]

    @property
    def audit_log(self) -> Path:
        return self.data_dir / "audit.log"

    @property
    def memory_db(self) -> Path:
        return self.memory_path or self.data_dir / "memory.sqlite3"

    @property
    def models_dir(self) -> Path:
        return self.models_path or self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)
        # Memory, transcripts' traces, keys and logs: this user's only.
        try:
            self.data_dir.chmod(0o700)
        except OSError:
            pass


class LibrarySettings(Settings):
    """The settings of an agent embedded in another app (neo2.core.runtime).

    Only what the host passes in: never the environment, ~/.neo/.env or a .env next to the
    package — a host process's NEO_CONFIRM_DESTRUCTIVE=false or its own keys must not reach an
    agent it didn't give them to. `data_dir` is required, and is the agent's own folder: never
    ~/.neo, inside it or holding it (however it's named), never a folder of other files, and never
    another live agent's (neo2.core.runtime). `models_path` and `memory_path` are never in ~/.neo either.
    Pinned whatever is passed: confirmation on; every task kind runs its legacy script (the
    evidence-chosen drivers, their dry runs and their learning are process-wide, NEO.app's);
    Laya's learning and its teacher off; notes that no notes app could take go under data_dir;
    NEO's browser in a window of its own (headless by default), never the chat's browser card.
    """

    # frozen: the pins can't be switched off later (settings().confirm_destructive = False)
    model_config = SettingsConfigDict(
        env_file=None, extra="forbid", populate_by_name=True, frozen=True, hide_input_in_errors=True
    )

    data_dir: Path  # required: an embedded agent never defaults to ~/.neo
    # Embedders get the quiet defaults: no local models, no memory, no voice, a headless browser.
    reflex: Literal["lite", "off"] = "off"  # "laya" is one process-wide model: NEO.app's
    # No brain but the ones the host gave: the local model server is one only when named here.
    # (Whatever answers on 127.0.0.1:8080 of a server is some other service.)
    offline_brain: BrainName | None = None
    # Nor a model the host didn't name: its Groq key sends no picture to Groq's vision model unless
    # it names one (a seeing fallback: neo2.core.providers.EYES).
    groq_vision_model: str = ""
    embeddings: Literal["local", "off"] = "off"
    memory_enabled: bool = False
    learn: bool = False
    voice: VoiceBackend = "off"
    browser_headless: bool = True
    browser_view: Literal["chat", "window"] = "window"  # no chat card: that's NEO.app's overlay
    include_username: bool = False
    # No early first look at the host's screen unless the host asks for it (neo2.core.agent.prefetch).
    prefetch_look: bool = False

    def __init__(self, **values: Any) -> None:
        try:
            super().__init__(**values)
        except ValidationError as e:
            # Nothing that was passed rides along in the error — not in str(), errors() or json():
            # a host's keys come through here, and error trackers keep all of them.
            details: list[Any] = [
                {"type": d["type"], "loc": d["loc"], "input": "<hidden>", **({"ctx": d["ctx"]} if "ctx" in d else {})}
                for d in e.errors(include_url=False, include_input=False)
            ]
            raise ValidationError.from_exception_data(e.title, details, hide_input=True) from None

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> LibrarySettings:
        """A copy is validated like a new one: its pins and folder checks hold too."""
        return type(self)(**{**{k: getattr(self, k) for k in type(self).model_fields}, **(update or {})})

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)

    def _pins(self) -> dict[str, Any]:
        """What an embedded agent always runs with, whatever was passed."""
        return {
            "confirm_destructive": True,
            "decide_modes": dict.fromkeys(sorted({*DECIDE_DEFAULTS, *self.decide_modes}), "legacy"),
            "decide_kind_routing": False,
            "learn": False,
            "learn_teacher": "off",
            "notes_dir": self.data_dir / "notes",
            "browser_view": "window",
        }

    @model_validator(mode="after")
    def _pinned(self) -> LibrarySettings:
        _refuse_neo_home(self.data_dir)
        if self.models_path is not None:
            _refuse_neo_home(self.models_path, "models_path")
        if self.memory_path is not None:
            _refuse_neo_home(self.memory_path, "memory_path")
        _refuse_real_data_without_effects(self.data_dir)
        _refuse_foreign_folder(self.data_dir)
        if self.memory_path is not None:
            _refuse_foreign_db(self.memory_path)
        for k, v in self._pins().items():
            object.__setattr__(self, k, v)  # frozen: set while it's being built
        if self.reflex == "lite" and not self.gemini_keys:
            raise ValueError('reflex="lite" needs a Gemini key (it calls Gemini Flash-Lite itself)')
        return self

    def check(self) -> None:
        """Raise unless these are still as validated (neo2.core.runtime, for an agent it's given):
        a copy made without validation (model_construct) or edited past the freeze is refused."""
        _refuse_neo_home(self.data_dir)
        if self.models_path is not None:
            _refuse_neo_home(self.models_path, "models_path")
        if self.memory_path is not None:
            _refuse_neo_home(self.memory_path, "memory_path")
        off = sorted(k for k, v in self._pins().items() if getattr(self, k) != v)
        if self.reflex not in ("lite", "off"):
            off.append("reflex")
        if off:
            raise ValueError(f"an embedded agent's settings must keep their pinned values: {', '.join(off)}")

    def ensure_dirs(self) -> None:
        """Create what's missing, private to this user. A folder the host already made keeps its
        own permissions; the models folder only exists when local embeddings are on. The data dir
        gets NEO's marker: from then on it is NEO's own folder, and taken again next time."""
        _mkdirs_private(self.data_dir)
        try:
            fd = os.open(self.data_dir / AGENT_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w") as f:
                f.write(_MARKER_TEXT)
        if self.embeddings == "local":
            _mkdirs_private(self.models_dir)


# An embedded agent's data dir holds this file once NEO has made the folder its own (ensure_dirs):
# a folder of someone else's files is never taken over (NEO keeps it secret, and an erase clears it).
AGENT_MARKER = ".neo-agent"
_MARKER_TEXT = "This folder holds an embedded NEO agent's data (memory, audit log, browser profile).\n"
_IGNORED = {".DS_Store"}  # what Finder leaves in any folder it showed


def _fold(parts: tuple[str, ...]) -> list[str]:
    # APFS ignores case and Unicode normalisation: ~/.NEO is ~/.neo.
    return [unicodedata.normalize("NFC", x).casefold() for x in parts]


def _file_id(p: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(p)
    except (OSError, ValueError):
        return None
    return st.st_dev, st.st_ino


def _ancestry(p: str | os.PathLike) -> list[tuple[Path, tuple[str, ...]]]:
    """`p` (absolute, ~ expanded) and each folder above it, each with the parts below it."""
    parts = Path(os.path.abspath(os.path.expanduser(str(p)))).parts
    return [(Path(*parts[:i]), parts[i:]) for i in range(len(parts), 0, -1)]


def overlaps(a: str | os.PathLike, b: str | os.PathLike) -> bool:
    """`a` and `b` are the same folder, or one lies inside the other — however either is named:
    case, Unicode normalisation, symlinks, or the /System/Volumes/Data firmlink. Compared by what
    is on disk (the deepest folder above both that exists), then by name below it."""
    below_b: dict[tuple[int, int], tuple[str, ...]] = {}
    for anc, rest in _ancestry(b):
        fid = _file_id(anc)
        if fid is not None:
            below_b.setdefault(fid, rest)
    for anc, rest in _ancestry(a):
        fid = _file_id(anc)
        if fid is not None and fid in below_b:
            ra, rb = _fold(rest), _fold(below_b[fid])
            n = min(len(ra), len(rb))
            return ra[:n] == rb[:n]
    ra, rb = _fold(Path(os.path.abspath(a)).parts), _fold(Path(os.path.abspath(b)).parts)
    n = min(len(ra), len(rb))
    return ra[:n] == rb[:n]


def _neo_homes() -> list[Path]:
    """NEO.app's own data: ~/.neo of $HOME, and of the login's home when a host moved $HOME."""
    homes = [Path.home()]
    try:
        import pwd

        homes.append(Path(pwd.getpwuid(os.getuid()).pw_dir))
    except (ImportError, KeyError, OSError):
        pass
    out: list[Path] = []
    for h in homes:
        if (p := h / ".neo") not in out:
            out.append(p)
    return out


def moved_app_data() -> list[Path]:
    """Where NEO_DATA_DIR and NEO_MEMORY_PATH in this process's environment put NEO.app's data (its
    folder, its memory database). Only the environment: an embedded agent never opens ~/.neo/.env
    to find out (a NEO_DATA_DIR set only there is NEO.app's own business)."""
    out: list[Path] = []
    for k in ("NEO_DATA_DIR", "NEO_MEMORY_PATH"):
        v = os.environ.get(k, "").strip()
        if v and (p := Path(os.path.abspath(os.path.expanduser(v)))) not in out:
            out.append(p)
    return out


def _in_neo_home(path: Path, *, moved: bool = True) -> Path | None:
    """The NEO.app data that `path` is, lies inside, or holds (the home folder, "/"), if any:
    ~/.neo, and (`moved`) wherever the environment moved it."""
    homes = _neo_homes() + (moved_app_data() if moved else [])
    return next((h for h in homes if overlaps(path, h)), None)


def _refuse_neo_home(path: Path, what: str = "data_dir") -> None:
    if (home := _in_neo_home(path)) is not None:
        raise ValueError(
            f"an embedded agent never uses NEO.app's own data ({home}): {what} {path} is it, lies "
            f"inside it or holds it — pick a folder of the agent's own"
        )


def _refuse_foreign_folder(data_dir: Path) -> None:
    """The data dir is missing, empty, or already NEO's (its marker): a folder of the host's or
    the user's own files never becomes an agent's (a secret to NEO's tools, cleared by an erase)."""
    try:
        names = {p.name for p in Path(data_dir).iterdir()} - _IGNORED
    except FileNotFoundError:
        return
    except OSError as e:  # a file, or a folder this user can't list
        raise ValueError(f"data_dir {data_dir} can't be the agent's folder ({e.strerror or e})") from None
    if names and AGENT_MARKER not in names:
        raise ValueError(
            f"data_dir {data_dir} already holds other files: give the agent its own folder (a new "
            "or empty one; NEO marks it as its own)"
        )


# The SQLite application id of a memory database NEO made for an embedded agent ("NEOA"): a
# database it didn't make — a host's own, NEO.app's — is never an agent's memory, nor erased.
AGENT_DB_ID = 0x4E454F41
_SQLITE_MAGIC = b"SQLite format 3\x00"
_NEO_TABLES = re.compile(r"memory|memory_vec|memory_fts(?:_[a-z]+)?|playbook|playbook_vec|sessions|sqlite_\w+")


def agent_memory_db(path: str | os.PathLike) -> bool:
    """`path` is a memory database NEO made for an embedded agent (neo2.core.memory.store.Store with
    agent=True): SQLite, marked with AGENT_DB_ID, holding NEO's tables and nothing else. Read
    without leaving a trace: the header first, and SQLite only for NEO's own file."""
    try:
        with open(path, "rb") as f:
            head = f.read(100)
    except OSError:  # missing, a folder, unreadable
        return False
    if len(head) < 100 or head[:16] != _SQLITE_MAGIC or int.from_bytes(head[68:72], "big") != AGENT_DB_ID:
        return False
    import sqlite3
    from urllib.parse import quote

    try:
        db = sqlite3.connect(f"file:{quote(os.path.abspath(path))}?mode=ro&immutable=1", uri=True)
        try:
            names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        finally:
            db.close()
    except sqlite3.Error:
        return False
    return all(_NEO_TABLES.fullmatch(n) for n in names)


def _refuse_foreign_db(path: Path) -> None:
    """An agent's memory_path is missing, empty, or a memory database NEO made for an agent: never
    a file of the host's own (NEO would add its tables to it, and an erase would delete it)."""
    try:
        empty = os.path.isfile(path) and os.path.getsize(path) == 0
    except OSError:
        empty = False
    if not os.path.lexists(path) or empty or agent_memory_db(path):
        return
    raise ValueError(
        f"memory path {path} is a file NEO didn't make for an agent's memory: give the agent a new "
        "(or empty) file — NEO marks the database it makes as its own"
    )


def _mkdirs_private(path: Path) -> None:
    missing: list[Path] = []
    p = Path(path)
    while not p.exists() and p != p.parent:
        missing.append(p)
        p = p.parent
    for d in reversed(missing):
        try:
            d.mkdir(mode=0o700)
        except FileExistsError:
            pass


_settings: Settings | None = None


def settings() -> Settings:
    """The bound agent's settings (neo2.core.runtime), else NEO.app's — built on first use."""
    rt = runtime.current_runtime()
    if rt is not None:
        return rt.settings
    runtime.check_unbound("settings()")
    global _settings
    if _settings is None:
        s = Settings()
        _refuse_real_data_without_effects(s.data_dir)
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


def _refuse_real_data_without_effects(data_dir: Path) -> None:
    """A test (or a scratch harness borrowing the suite's fixtures, or a scratch script run with
    NEO_NO_EFFECTS=1) that isn't isolated must fail loudly, never read, train on or erase the user's
    real memory. This happened once: a scratch conftest skipped the isolation fixture and an erase
    test wiped the real ~/.neo."""
    why = effects.why_off()
    if not why:
        return
    if (real := _in_neo_home(data_dir, moved=False)) is not None:  # NEO_DATA_DIR is the test's own
        raise RuntimeError(f"refusing to use the real {real} ({why}): set NEO_DATA_DIR to a temporary directory")
