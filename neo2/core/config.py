"""NEO settings — environment-driven, with sane free/local-first defaults.

Everything lives under ``~/.neo`` (data dir) unless overridden. Each setting is decided on its own,
first match wins: an environment variable; what was saved in NEO (the Brains & Keys window or
``python -m neo2 keys``: the keys in ~/.neo/api_keys.json, the brains in ~/.neo/brains.json —
neo2.core.keystore); the .env files (a checkout's own ``.env`` beats ``~/.neo/.env``); the default
here. Nothing is ever copied from one into another, and nothing is written to Git.

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
from typing import Annotated, Any, ClassVar, Literal, get_args

from pydantic import BeforeValidator, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, PydanticBaseSettingsSource, SettingsConfigDict

from neo2.core import brainref, effects, runtime
from neo2.core.brainref import CustomProvider

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # neo2/core/config.py → the checkout (or site-packages)
# Keys and settings: ~/.neo/.env (where NEO.app's installed runtime looks), and a checkout's own
# .env on top when running from source (a checkout has a pyproject.toml; installed as a package,
# PROJECT_ROOT is site-packages, whose .env is nobody's).
USER_ENV = Path.home() / ".neo" / ".env"
# What the user saved in NEO itself (neo2.core.keystore): read after the environment, before the
# .env files. None: not read at all (the suite's conftest; a path of a test's own is read even
# with effects off, the real ones never are).
KEY_STORE_PATH: Path | None = Path.home() / ".neo" / "api_keys.json"
SAVED_PATH: Path | None = Path.home() / ".neo" / "brains.json"
NOTES_DIR = Path.home() / "Documents" / "NEO Notes"  # where a note goes that no notes app could take


def _env_files(root: Path) -> tuple[Path, ...]:
    return (USER_ENV, root / ".env") if (root / "pyproject.toml").exists() else (USER_ENV,)


_ENV_FILES = _env_files(PROJECT_ROOT)

BrainName = Literal["gemini", "gemini_lite", "groq", "local", "claude", "groq_vision", "cerebras", "cerebras_vision"]
# A role's brain: "<brain>[@<model>]" (neo2.core.brainref) — any of BrainName, openai, openrouter or
# custom:<slug>, "anthropic" for claude and "google" for gemini. One NEO can't read is kept as typed
# (the chains skip it and the health list says so): never a reason not to start, never another brain.
BrainRef = Annotated[str, BeforeValidator(brainref.normalize)]
VoiceBackend = Literal["auto", "live", "local", "off"]
DecideMode = Literal["legacy", "shadow", "on", "off"]
# Where NEO's UI lives: NEO.app's own window, or the notch (NotchHub hosts the same page in its
# island and NEO.app keeps its window hidden).
UiHost = Literal["window", "notch"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,  # later files win
        env_file_encoding="utf-8",
        env_prefix="NEO_",
        extra="ignore",
        hide_input_in_errors=True,  # a bad value's error never prints it (a key, say)
    )

    # The environment's own template variables (an endpoint's XAI_API_KEY) are read: not by an
    # embedded agent's settings (LibrarySettings), which take only what the host passes.
    _reads_environ: ClassVar[bool] = True

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):  # pydantic-settings calls it with these, by name
        """The environment, then what was saved in NEO (its key store, then its saved brains), then
        the .env files. With effects off (a test, a scratch script run with NEO_NO_EFFECTS=1),
        never the .env files nor NEO's own real files: ~/.neo holds the user's real API keys (a
        test's own store path is read)."""
        saved = _store_sources(settings_cls)
        if effects.why_off():
            return init_settings, env_settings, *saved, file_secret_settings
        return init_settings, env_settings, *saved, dotenv_settings, file_secret_settings

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
    # Only used where the user assigned them a role (a key alone sends nothing anywhere new).
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY", repr=False)
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY", repr=False)
    # The keys of the endpoints the user added (custom_providers), by id ("custom:deepseek"): from
    # the key store, or an endpoint's own variable (its `env`, XAI_API_KEY) in the environment
    # (which wins) or a .env file (which doesn't).
    custom_keys: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict, repr=False)

    # ---- brains ----------------------------------------------------------------------
    # Each role is a brain reference (BrainRef). The defaults rebuild the chains NEO always had.
    brain: BrainRef = "gemini"  # Brain: agent tasks with tools
    fast_brain: BrainRef = "groq"  # Quick replies: chat-only turns; falls back to `brain` if no key
    offline_brain: BrainRef = "local"
    # Eyes: the first brain asked for any step with a screenshot or photo in it, and for camera
    # questions. "": as before (the brain if it sees, then the seeing fallbacks).
    eyes_brain: BrainRef = ""
    # Backups: None is Automatic (as before: the agent's backup is fast_brain); a list (at most 4,
    # NEO_BACKUP_BRAINS="claude,groq") gives agent = [brain, *backups, offline] and
    # fast = [fast_brain, brain, *backups, offline].
    backup_brains: Annotated[list[str] | None, NoDecode] = None
    # One-tool actions that missed the regex fast path. "": fast_brain's.
    light_brain: BrainRef = "gemini_lite"
    # The endpoints the user added (brains.json, or NEO_CUSTOM_PROVIDERS as JSON): custom:<slug>
    custom_providers: Annotated[list[CustomProvider], NoDecode] = []
    # Dollars a day per provider id ({"anthropic": 2.0}); a provider without one has no NEO cap.
    daily_budgets: Annotated[dict[str, float], NoDecode] = {}
    # What OpenRouter's live list said of the models the roles use — {"openrouter": {model: {"sees",
    # "tools", "price": [in, out] $/M tokens}}} — kept in brains.json by brains_set, so a restart
    # still knows the Eyes brain sees and what its cap should count (catalog.learned).
    known_models: Annotated[dict[str, dict[str, dict[str, Any]]], NoDecode] = {}
    openai_model: str = "gpt-6.1-sol"
    openrouter_model: str = ""  # (no default: the user picks one; its ids are vendor-prefixed)
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
    # that on its own). "on": they join for that user too. "off": never — only the brains the user
    # picked ever see a picture.
    cloud_eyes: Literal["auto", "on", "off"] = "auto"
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
        """"$1.25", " 0.5 " or 2. A cap it can't read ("off", -1, a typo) is 0: Cerebras off, never
        a reason not to start (neo2.core.providers.spend.budget)."""
        from neo2.core.providers.spend import budget

        return budget(v, 0.50)

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
    workflow_suggestions: bool = True  # local shortcut suggestions after repeated successful app/site setups
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
    # Listen for "Neo". On: the microphone stays open so the wake word can be heard (macOS shows its
    # orange dot all the time). Off: the microphone is closed while NEO is idle and opens only when NEO
    # is summoned by hand (the summon keys, the orb, push-to-talk), until NEO is idle again. Switched
    # while NEO runs ({"cmd": "set", "name": "wake_word"}) and kept in ~/.neo/.env; see set_wake_word.
    wake_word: bool = True
    # The wake gate (neo2.core.voice.wakegate): after a spoken "Neo", does the Mac's own voice detector
    # hear someone go on talking — wake_gate_speech_ms of voice (past the end of "Neo" itself) within
    # wake_gate_s of the wake, or of the session opening if connecting took longer? "shadow" (the
    # default) only measures: the audio goes up exactly as before, and each wake's verdict is logged
    # and counted (numbers only) for the quality report. "on" holds the audio after a spoken "Neo"
    # back until a voice confirms it — then sends all of it, in order — and closes a wake nobody
    # spoke after without any of the room leaving the Mac. "off": no gate. A wake by hand (the
    # summon keys, the orb, push-to-talk, a host) and a check-in are never gated.
    wake_gate: Literal["off", "shadow", "on"] = "shadow"
    wake_gate_s: float = 3.0
    wake_gate_speech_ms: int = 300
    # The microphone heals itself or says so (neo2.core.voice.audio): after the device watcher reopens a
    # microphone NEO is using, sound has to come in (5 frames within 2 s). If none does, it tries again
    # after 5, 10, 20, 40, then every 60 s, and on the 3rd silent try in a row (about half a minute) it
    # tells you once — NEO's page and one macOS notification — until sound comes back. A microphone
    # closed on purpose is never checked. Off: the watcher as before (a reopen that opened counts as
    # working, a stall is retried every 30 s, nothing is said).
    mic_escalate: bool = True
    # Gentle check-ins (opt-in): repeated coughs, sniffles, sneezes or a few seconds of crying make NEO
    # ask a short, kind question. Detection stays on-device (Apple's sound classifier hears only what
    # NEO's microphone captures while it is open anyway; it never opens or holds it, and nothing about
    # it is saved). With Gemini Live, the check-in line and ~5 s of reply audio go to Gemini. At most 3
    # a day, backing off per kind; only while NEO is idle and no other app plays sound. See
    # neo2/core/voice/wellbeing.py. Switched while NEO runs ({"cmd": "set", "name": "wellbeing_checkins"}).
    wellbeing_checkins: bool = False
    # Voice commands the fast paths fully cover run the moment you stop talking, without
    # waiting for Gemini Live to pick the tool (questions still go to the model).
    live_command_lane: bool = True
    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    tts_voice: str = "af_heart"
    live_voice: str = "Charon"

    @field_validator("wake_gate", mode="before")
    @classmethod
    def _wake_gate(cls, v):
        """A typo must not stop NEO from starting, and never turns the gate on: anything but off,
        shadow or on is shadow (it only measures)."""
        s = str(v or "").strip().lower()
        return s if s in ("off", "shadow", "on") else "shadow"

    @field_validator("wake_gate_s", "wake_gate_speech_ms", mode="before")
    @classmethod
    def _wake_gate_numbers(cls, v, info):
        """A number it can't read is the default, and one out of reason is brought back into it: a
        gate of no time at all, switched on, would drop every spoken "Neo"."""
        bounds = {"wake_gate_s": (1.0, 10.0, 3.0), "wake_gate_speech_ms": (100, 2000, 300)}
        lo, hi, default = bounds[info.field_name]
        try:
            x = float(str(v).strip())
        except (TypeError, ValueError):
            return default
        if x != x:  # NaN
            return default
        x = min(hi, max(lo, x))
        return int(x) if info.field_name == "wake_gate_speech_ms" else x

    # ---- agent -----------------------------------------------------------------------
    max_steps: int = 24
    # The stuck-call breaker (neo2.core.agent.failures): in one agent run, an action that failed twice
    # the same way gets a hint by kind of failure, and a third identical call isn't run ("skipped: …";
    # the run goes on). Any action that works clears the count. Off: neither — the loop as before.
    stuck_breaker: bool = True
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
        """Only a clear yes sends a local-first user's pictures to a cloud fallback, and only "off"
        (or "never") keeps them from every fallback: anything else (a typo too) is auto."""
        s = str(v or "").strip().lower()
        if s in ("on", "true", "yes", "1", "always"):
            return "on"
        return "off" if s in ("off", "never") else "auto"

    @field_validator("backup_brains", mode="before")
    @classmethod
    def _backup_brains(cls, v):
        return brainref.backups(v)

    @field_validator("custom_providers", mode="before")
    @classmethod
    def _custom_providers(cls, v):
        return brainref.custom_list(v)

    @field_validator("daily_budgets", mode="before")
    @classmethod
    def _daily_budgets(cls, v):
        return brainref.budgets(v)

    @field_validator("known_models", mode="before")
    @classmethod
    def _known_models(cls, v):
        return brainref.model_notes(v)

    @field_validator("custom_keys", mode="before")
    @classmethod
    def _custom_keys(cls, v):
        """{"custom:<slug>": key}, from a dict or its JSON; an entry that isn't one is dropped."""
        return _custom_keys(v)

    @model_validator(mode="before")
    @classmethod
    def _endpoint_variables(cls, data: Any) -> Any:
        """An endpoint's own variable is its key too, in R1's order: the environment beats a key
        saved in NEO, which beats a .env file's line. Only its template's own variable, and only at
        that template's host (CustomProvider.key_env: XAI_API_KEY for api.x.ai) — never another
        provider's key nor any other secret the environment holds. Only where the environment is
        read at all (never an embedded agent's settings)."""
        if not cls._reads_environ or not isinstance(data, dict) or not data.get("custom_providers"):
            return data
        named = [(cp, v) for cp in brainref.custom_list(data.get("custom_providers")) if (v := cp.key_env)]
        if not named:
            return data
        keys = _custom_keys(data.get("custom_keys"))
        mine = _store_aliases()
        for cp, var in named:
            from_env = os.environ.get(var, "").strip()
            # (a .env file's line NEO has no field for: passed on by name, lower-cased — never a name
            # NEO's own key store fills, which a template's variable never is)
            from_file = None if var in mine else data.get(var, data.get(var.lower()))
            if from_env:
                keys[cp.id] = from_env
            elif cp.id not in keys and isinstance(from_file, str) and from_file.strip():
                keys[cp.id] = from_file.strip()
        return {**data, "custom_keys": keys}

    # ---- memory & prompt ---------------------------------------------------------------
    embeddings: Literal["local", "off"] = "local"  # off: keyword-only recall, no local model
    memory_enabled: bool = True  # long-term memory (memory.sqlite3) at all
    playbooks: bool = True  # recall and save the tool paths of past tasks (needs memory)
    summaries: bool = True  # conversation summaries (needs memory)
    # The journal (neo2.core.memory.journal): every line you say or type and every line NEO says is
    # written to the memory database as it happens, so NEO can find what was said last week and
    # a voice session that closed (or a crash) loses nothing. Off: only what NEO is told to
    # remember is kept.
    memory_journal: bool = True
    memory_retention_days: int = 365  # journal lines older than this are deleted (facts and summaries stay); 0 = keep
    # Facts taken from what you say (neo2.core.memory.extract): "rules" reads only sentences that
    # state one in so many words, on this Mac; "rules+llm" also has the fast brain read the
    # journal while NEO is idle (never with NEO_CLOUD_MEMORY=off); "off" keeps only what NEO is
    # told to remember.
    memory_extract: Literal["rules+llm", "rules", "off"] = "rules+llm"
    memory_recall_k: int = 6  # how many search hits a request's memory block may carry
    # Gemini Live: after each sentence, tell the model what memory has about it before it
    # answers. Off by default: a note that arrives after the model began would cut its answer
    # short, and whether it arrives in time is only measured (logged) until it is switched on.
    memory_live_inject: bool = False

    @field_validator("memory_extract", mode="before")
    @classmethod
    def _memory_extract(cls, v):
        """A typo must not stop NEO from starting, and never sends more to a model than asked:
        an unknown value is "rules" (nothing leaves the Mac)."""
        s = str(v or "").strip().lower().replace(" ", "")
        if s in ("rules+llm", "llm", "all", "full", "on", "true", "yes"):
            return "rules+llm"
        if s in ("off", "none", "no", "false", "0"):
            return "off"
        return "rules"

    @field_validator("memory_retention_days", "memory_recall_k", mode="before")
    @classmethod
    def _memory_number(cls, v, info):
        """A number it can't read is the default (a typo never stops NEO from starting)."""
        default = {"memory_retention_days": 365, "memory_recall_k": 6}[info.field_name]
        try:
            n = int(str(v).strip())
        except (TypeError, ValueError):
            return default
        return max(0, min(n, 36500 if info.field_name == "memory_retention_days" else 12))

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

    # ---- the camera's eyes (neo2/core/vision) ------------------------------------------
    # A camera question's one still goes to Gemini, then Cerebras's vision model, for the answer
    # (the user's choice). Off: the answer comes from the on-device labels only.
    vision_cloud_ok: bool = True
    # The object detector for the camera card's boxes (neo2.core.vision.detect): "auto" is YOLOE-26s
    # (Core ML; scripts/get_yoloe.py puts its model in models_dir) when it's there, else D-FINE small;
    # "yoloe" only YOLOE; a repo id or a folder is D-FINE (looked for in models_dir first, downloaded
    # from Hugging Face only once the user says yes: vision_download_ok).
    vision_detector: str = "auto"
    vision_download_ok: bool = False
    # The words YOLOE may give a box: "everyday" (~230 things held up to a camera or kept on a desk,
    # people as "person": far fewer wrong words) or "all" (its whole open vocabulary, 4,585 words).
    vision_words: str = "everyday"

    # ---- server / ui -----------------------------------------------------------------
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765
    # Set by the host over the websocket ({"cmd": "ui_host"}), kept in ~/.neo/.env as NEO_UI_HOST so
    # NEO.app's shell reads it at launch; see set_ui_host.
    ui_host: UiHost = "window"

    # ---- paths -----------------------------------------------------------------------
    data_dir: Path = Path.home() / ".neo"
    models_path: Path | None = None  # downloaded models (default: data_dir/models)
    memory_path: Path | None = None  # the memory database (default: data_dir/memory.sqlite3)
    extra_secret_paths: list[Path] = []  # never read or written by NEO's tools (data_dir never is)

    # ---- the camera (neo2.core.tools.camera) -----------------------------------------
    # Only ever on the user's own request, with the chat's live preview card showing it. (Whether
    # its one still per question may go to the cloud brains is neo2.core.vision's vision_cloud_ok.)
    camera: bool = True  # False: NEO never turns the camera on
    camera_follow_s: float = 20.0  # it stays on this long after an answer, for "and now?"

    # ---- Search Screen (neo2.core.lens) ------------------------------------------------
    # Hold ⌥⌘ about a second and let go (or the menu's Search Screen…, or "search my screen"): a
    # glowing overlay over the display under the pointer; circle or drag over anything to search
    # it. The quick ⌥⌘ tap is unaffected. Only the area the user selects ever leaves the Mac, and
    # only to their own brains (the Eyes first) — or, when they click "Search with …", to that site
    # through their browser. NEO.app's shell reads `lens` and `lens_hold_ms` from the core's hello.
    lens: bool = True  # False: holding ⌥⌘ does nothing, the menu item is off, the voice says so
    lens_hold_ms: int = 1000  # how long ⌥⌘ is held: 600–3000 (under 1000 the quick tap's window is hold − 100)
    # "Search with …": google (Google Lens, the default) | bing | off | any other site word (a text
    # search there for the answer's query; no picture is uploaded)
    lens_search: str = "google"
    lens_ask: bool = True  # False: on-device results only, until the user presses Ask

    @field_validator("lens", "lens_ask", mode="before")
    @classmethod
    def _lens_switch(cls, v):
        """A typo must not stop NEO from starting: only a clear "off" turns these off."""
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() not in ("0", "false", "no", "off", "never", "n", "f")

    @field_validator("lens_hold_ms", mode="before")
    @classmethod
    def _lens_hold(cls, v):
        """600–3000 ms; anything NEO can't read is the default (1000)."""
        try:
            ms = int(float(str(v).strip().lower().removesuffix("ms").strip()))
        except (TypeError, ValueError):
            return 1000
        return max(600, min(3000, ms))

    @field_validator("lens_search", mode="before")
    @classmethod
    def _lens_search(cls, v):
        """A site word (google, bing, duckduckgo, a domain) or "off"; anything else is google."""
        s = str(v if v is not None else "").strip().lower()
        if s in ("", "on", "true", "yes", "default", "lens", "google lens"):
            return "google"
        if s in ("false", "no", "none", "never", "0"):
            return "off"
        return s if len(s) <= 40 and re.fullmatch(r"[a-z0-9.-]+", s) else "google"

    # ---- naming a song (neo2.core.tools.music) ----------------------------------------
    # "What song is this?": NEO listens to the room (the microphone) and to the Mac's own sound
    # (while something plays) for up to song_listen_s, only on the user's own request, with its card
    # in the chat. The audio stays in memory; only Shazam's fingerprint of it goes to Shazam.
    song_id: bool = True  # False: NEO never listens for a song
    song_listen_s: float = 12.0  # the longest it listens before "I couldn't recognise it"

    # ---- "this": what the user selected, copied or has in front (neo2.core.tools.context_now) ----
    # Read only when the user's own words ask for it ("summarize what I copied", "translate the
    # selected text"), never on a model's initiative, and nothing of it is kept. False: in_front
    # refuses, the Session never hands such a request to it, and Gemini Live isn't told about it.
    context_capture: bool = True
    # A web page's selection Accessibility doesn't give (Google Chrome, the Chromium browsers and
    # Electron apps keep it to themselves) is copied: ⌘C to that app alone, the copy's plain text read,
    # and the clipboard put back exactly as it was (every item, every type's bytes), whatever happens —
    # and put back over a write that comes a second late. Only on a page whose focused element has no
    # AXSelectedText (never a text view, a terminal or a password manager's list). Never while the
    # clipboard holds a private copy (a password manager's), another device's, more than 32 MB or
    # anything that can't be kept whole, never on a Mac that asks before apps read it, never in a
    # password field. False: only what Accessibility gives ("Nothing is selected" in Chrome).
    context_copy_fallback: bool = True

    # ---- the user's own Shortcuts as NEO verbs (neo2.core.tools.shortcuts) ----
    # "run my Send ETA shortcut" runs it; a run a model chose, or one after NEO read something untrusted,
    # asks first. False: shortcut_run is hidden from every model and refuses, and NEO never reads the list.
    shortcuts: bool = True
    # The first 30 names in shortcut_run's description, so a model knows them before the user says one.
    # That sends the names to whichever model gets the tool, a cloud one included: off unless switched on.
    shortcuts_names_in_prompt: bool = False

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
    _reads_environ: ClassVar[bool] = False  # only what the host passes

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
    # Nothing of a conversation is stored unless the host turns it on: no journal, and no facts
    # taken from what is said (with the journal on, "rules" stays on the machine; "rules+llm"
    # sends journal lines to the agent's fast brain while it is idle).
    memory_journal: bool = False
    memory_extract: Literal["rules+llm", "rules", "off"] = "off"
    learn: bool = False
    voice: VoiceBackend = "off"
    browser_headless: bool = True
    browser_view: Literal["chat", "window"] = "window"  # no chat card: that's NEO.app's overlay
    camera: bool = False  # the camera only for a host that turns it on (camera=True)
    song_id: bool = False  # nor does it listen for a song unless the host turns that on (song_id=True)
    lens: bool = False  # Search Screen is NEO.app's own overlay: an embedded agent has none
    include_username: bool = False
    # No early first look at the host's screen unless the host asks for it (neo2.core.agent.prefetch).
    prefetch_look: bool = False
    # No camera still to any cloud brain unless the host asks for it (neo2.core.vision.answer).
    vision_cloud_ok: bool = False

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
_NEO_TABLES = re.compile(
    r"(?:memory|sessions|turns)(?:_(?:fts|tri)(?:_[a-z]+)?|_vec|_vocab)?|playbook|playbook_vec|conversations|meta|sqlite_\w+"
)


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


def _custom_keys(v: Any) -> dict[str, str]:
    if isinstance(v, str):
        try:
            v = json.loads(v) if v.strip() else {}
        except ValueError:
            return {}
    if not isinstance(v, dict):
        return {}
    out: dict[str, str] = {}
    for k, key in v.items():
        r = brainref.parse(k)
        if r.custom and r.model is None and isinstance(key, str) and key.strip():
            out[r.text] = key.strip()
    return out


def _store_aliases() -> set[str]:
    """The names NEO's key store fills settings by (GEMINI_API_KEY…, GEMINI_API_KEYS)."""
    from neo2.core import keystore

    try:
        return {*keystore.key_aliases().values(), "GEMINI_API_KEYS"}
    except Exception:  # noqa: BLE001
        return {"GEMINI_API_KEYS"}


def _store_sources(settings_cls: type[BaseSettings]) -> tuple[PydanticBaseSettingsSource, ...]:
    """What the user saved in NEO, as settings sources (neo2.core.keystore): the key store, then the
    saved brains. A path that is None isn't read; the real ones never while effects are off."""
    from neo2.core import keystore

    out: list[PydanticBaseSettingsSource] = []
    if KEY_STORE_PATH is not None and not keystore.refused(KEY_STORE_PATH):
        out.append(keystore.KeyStoreSource(settings_cls, KEY_STORE_PATH))
    if SAVED_PATH is not None and not keystore.refused(SAVED_PATH):
        out.append(keystore.SavedSource(settings_cls, SAVED_PATH))
    return tuple(out)


def _env_lines(env_path: Path | None, what: str) -> tuple[Path, list[str]]:
    """The file a setting NEO writes itself goes to (~/.neo/.env, or a test's own), and its lines.
    No file yet is no lines; a file that is there but can't be read stops the write (it would
    otherwise be replaced by the one setting, and the user's keys with it)."""
    path = env_path if env_path is not None else USER_ENV
    if env_path is None and effects.why_off():
        raise RuntimeError(f"{what}: effects are off, and ~/.neo/.env is the user's")
    try:
        return path, path.read_text().splitlines()
    except FileNotFoundError:
        return path, []


def _write_env(path: Path, lines: list[str]) -> None:
    """The file whole or not at all (neo2.core.keystore.atomic_write: a temporary file renamed over
    it), mode 0600 — it holds the user's keys: theirs to read, nobody else's."""
    from neo2.core.keystore import atomic_write

    atomic_write(path, ("\n".join(lines) + "\n").encode("utf-8"))


def set_ui_host(host: str, env_path: Path | None = None) -> None:
    """Record where NEO's UI lives ("window" | "notch"): the NEO_UI_HOST line of ~/.neo/.env (one of
    the two settings NEO writes there itself; the rest of the file is the user's), and the live
    settings. With effects off the user's own file is never touched — a test passes a path of its own."""
    if host not in ("window", "notch"):
        raise ValueError(f"ui_host must be window or notch, not {host!r}")
    path, lines = _env_lines(env_path, "set_ui_host")
    line = f"NEO_UI_HOST={host}"
    kept = [ln for ln in lines if not ln.strip().startswith("NEO_UI_HOST=")]
    if kept and kept[-1].strip():
        kept.append("")
    kept.append(line)
    _write_env(path, kept)
    settings().ui_host = host  # type: ignore[assignment]  # the live settings follow the file


def set_wake_word(on: bool, env_path: Path | None = None) -> None:
    """Record whether NEO listens for "Neo": the NEO_WAKE_WORD line of ~/.neo/.env — replaced where
    it stands, or added at the end; every other line of the file stays exactly as the user wrote it —
    and the live settings. With effects off the user's own file is never touched (a test passes a
    path of its own). The running voice follows through its own set_wake_word (neo2.core.voice.audio)."""
    if not isinstance(on, bool):
        raise ValueError(f"wake_word must be true or false, not {on!r}")
    path, lines = _env_lines(env_path, "set_wake_word")
    line = f"NEO_WAKE_WORD={'true' if on else 'false'}"
    out: list[str] = []
    placed = False
    for ln in lines:
        if not ln.strip().startswith("NEO_WAKE_WORD="):
            out.append(ln)
        elif not placed:  # (a second such line, if the user left one, goes: the last one would win)
            out.append(line)
            placed = True
    if not placed:
        if out and out[-1].strip():
            out.append("")
        out.append(line)
    _write_env(path, out)
    settings().wake_word = on  # the live settings follow the file


def set_wellbeing_checkins(on: bool, env_path: Path | None = None) -> None:
    """Persist the opt-in local sound check-ins without changing any other ~/.neo/.env line."""
    if not isinstance(on, bool):
        raise ValueError(f"wellbeing_checkins must be true or false, not {on!r}")
    path, lines = _env_lines(env_path, "set_wellbeing_checkins")
    line = f"NEO_WELLBEING_CHECKINS={'true' if on else 'false'}"
    out: list[str] = []
    placed = False
    for ln in lines:
        if not ln.strip().startswith("NEO_WELLBEING_CHECKINS="):
            out.append(ln)
        elif not placed:
            out.append(line)
            placed = True
    if not placed:
        if out and out[-1].strip():
            out.append("")
        out.append(line)
    _write_env(path, out)
    settings().wellbeing_checkins = on


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
