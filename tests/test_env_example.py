""".env.example, copied as it is (`cp .env.example .env`), sets no key at all: every key line reads as
empty (python-dotenv, as Settings reads it, takes a comment after "KEY=" on the same line as the
key). It names every key NEO reads, and every NEO_ setting it mentions is one NEO has."""

from __future__ import annotations

import re
from pathlib import Path

from dotenv import dotenv_values

from neo2.core import config, keystore
from neo2.core.brainref import TEMPLATE_KEYS

EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def test_copied_as_it_is_it_sets_no_key():
    values = dotenv_values(EXAMPLE)
    assert values and all(v == "" for v in values.values()), {k: bool(v) for k, v in values.items()}


def test_it_names_every_key_and_only_settings_neo_has():
    text = EXAMPLE.read_text()
    for name in (*keystore.key_aliases().values(), "GEMINI_API_KEYS", *(env for env, _host in TEMPLATE_KEYS.values())):
        assert name in text, name
    for name in set(re.findall(r"\bNEO_([A-Z_]+)=", text)):
        assert name.lower() in config.Settings.model_fields, name
