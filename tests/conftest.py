"""Test-wide defaults: no model downloads, no GPU, no network."""

import os

os.environ.setdefault("NEO_REFLEX", "off")  # never load Laya (1.7 GB) inside the test suite
os.environ.setdefault("NEO_VOICE", "off")

import pytest


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
def _no_real_embeddings(monkeypatch):
    """Keyword-only memory by default; tests that need vectors install their own fake embedder."""
    from neo.memory import embed

    monkeypatch.setattr(embed, "embed_texts", lambda texts: None)
