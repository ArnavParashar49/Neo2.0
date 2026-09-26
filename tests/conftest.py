"""Test-wide defaults: no model downloads, no GPU, no network."""

import os

os.environ.setdefault("NEO_REFLEX", "off")  # never load Laya (1.7 GB) inside the test suite
os.environ.setdefault("NEO_VOICE", "off")

from pathlib import Path

import pytest

from neo.config import Settings

# A scratch conftest doing `from tests.conftest import *` must get the underscore fixtures too —
# without them a test runs against the user's real ~/.neo.
__all__ = ["_isolated_data_dir", "_never_the_real_data_dir", "_no_real_embeddings", "_fresh_learning_state"]

# Hermetic: no test may read the developer's real keys or settings (~/.neo/.env, ./.env).
Settings.model_config["env_file"] = None
for _k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_k, None)


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Every test gets its own ~/.neo: nothing a test does may touch the user's real data
    (memory, audit log, Laya's training examples, cooldowns)."""
    from neo import config

    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path / "neo_data"))
    monkeypatch.setattr(config, "_settings", None)
    yield
    monkeypatch.setattr(config, "_settings", None)


@pytest.fixture(autouse=True)
def _never_the_real_data_dir(_isolated_data_dir):
    """The last line of defence: nothing runs (erase, retrain, record…) against the real ~/.neo.
    Read from the environment, not settings(): building those would already touch the dir."""
    data_dir = os.environ.get("NEO_DATA_DIR", "")
    if not data_dir or Path(data_dir).expanduser().resolve() == (Path.home() / ".neo").resolve():
        pytest.exit("refusing to run: the data dir is the real ~/.neo", returncode=3)
    yield


@pytest.fixture(autouse=True)
def _no_real_embeddings(monkeypatch):
    """Keyword-only memory by default; tests that need vectors install their own fake embedder."""
    from neo.memory import embed

    monkeypatch.setattr(embed, "embed_texts", lambda texts: None)


@pytest.fixture(autouse=True)
def _fresh_learning_state():
    """Requests still settling in RAM, the correction memory and unsaved counters are
    process-wide: none of them may carry from one test into the next."""
    from neo.reflex import learn, memory, outcomes

    outcomes.forget()
    memory.forget()
    learn._counts.clear()
    yield
    outcomes.forget()
    memory.forget()
    learn._counts.clear()
