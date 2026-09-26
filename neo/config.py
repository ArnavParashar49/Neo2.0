"""NEO settings — environment-driven, with sane free/local-first defaults.

Everything lives under ``~/.neo`` (data dir) unless overridden. API keys come from the
environment or a ``.env`` file in the project root; nothing is written to Git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BrainName = Literal["gemini", "gemini_lite", "groq", "local", "claude"]
VoiceBackend = Literal["auto", "live", "local", "off"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
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
    laya_device: str = "mps"
    reflex_confidence_floor: float = 0.55  # below this, escalate to the lite model
    reflex_tool_floor: float = 0.85  # Laya may pick the tool itself above this confidence
    home_location: str = ""  # e.g. "Dubai" — where "the weather" means, if not remembered
    learn: bool = True  # NEO_LEARN=off: Laya stops recording requests to learn from

    # ---- voice -----------------------------------------------------------------------
    voice: VoiceBackend = "auto"  # auto = Live if key + quota, else local cascade
    wake_word: str = "hey neo"
    listen_timeout_s: float = 5.0  # close the voice session after this much silence from you
    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    tts_voice: str = "af_heart"
    live_voice: str = "Charon"

    # ---- agent -----------------------------------------------------------------------
    max_steps: int = 24
    max_seconds: float = 180.0  # wall-clock budget for one agent run
    max_tool_result_chars: int = 6000
    confirm_destructive: bool = True

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

    @field_validator("data_dir", mode="before")
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


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
