"""What the Brains & Keys review found, each pinned the right way round (they were the review's probes,
which passed while the defect was there).

Keys stay with their own provider: an endpoint can name only its template's own variable, and that
only at the template's own host; a saved endpoint's key never follows a base URL the window sends
(nor a form that isn't saved); a key pasted where a model id or a name goes is never a brain, a label
or anything every client hears; a key whose shape names another provider isn't sent to this one; an
endpoint whose address only *says* localhost is in the cloud; OpenAI's brain talks to OpenAI.

Chains and caps do what the window says: a daily cap changed reaches the brain at once; OpenRouter is
counted at the prices its list gives (kept across a restart, with what each model can do); a cap is
never saved for a key the environment shadows, nor for an endpoint NEO has no prices for; an endpoint
that can't call tools is never handed a step with tools; one on this Mac that wants a key is up; the
camera asks one brain per provider; Groq's and Cerebras's seeing models are seeing brains by reference
too; a refused Gemini key is a refused key. And the window's commands run one at a time, reloads are
coalesced, and old copies of the key file are reported.

NEO's files are this test's own (tmp_path); provider checks and clients are faked; every key is an
obvious fake; nothing goes over the network."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from neo2.core import brainref, brains, config, health, keyshape, keystore, logredact
from neo2.core import providers as prov
from neo2.core.brainref import CustomProvider
from neo2.core.providers import catalog, keycheck, spend
from neo2.core.providers.base import KeyRefused, Message, ToolSpec, Turn, Unavailable

ANTHROPIC = "sk-ant-TEST-0000-abcdefghijklmnop-x9Qa"
OPENROUTER = "sk-or-TEST-0000-aaaaaaaaaaaaaaaaaaaa"
ENDPOINT_KEY = "sk-TEST-endpoint-00000000000000000000"
_VARS = (
    "GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY",
    "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY", "GITHUB_TOKEN", "OPENAI_BASE_URL", "NEO_BRAIN",
    "NEO_FAST_BRAIN", "NEO_OFFLINE_BRAIN", "NEO_EYES_BRAIN", "NEO_BACKUP_BRAINS", "NEO_LIGHT_BRAIN", "NEO_CLOUD_EYES",
    "NEO_CUSTOM_PROVIDERS", "NEO_VOICE", "NEO_DAILY_BUDGETS", "NEO_OPENROUTER_MODEL", "NEO_KNOWN_MODELS",
)  # fmt: skip


@pytest.fixture
def files(monkeypatch, tmp_path):
    for k in _VARS:
        monkeypatch.delenv(k, raising=False)
    home = tmp_path / "neo-home"
    monkeypatch.setattr(config, "KEY_STORE_PATH", home / "api_keys.json")
    monkeypatch.setattr(config, "SAVED_PATH", home / "brains.json")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(prov, "_said_unavailable", set())
    monkeypatch.setattr(health, "_live", {})
    monkeypatch.setattr(brains, "_probes", {})
    monkeypatch.setattr(brains, "_started", None)
    monkeypatch.setattr(logredact, "_remembered", set())
    monkeypatch.setattr(catalog, "_LEARNED", {})
    monkeypatch.setattr(keycheck, "_cache", {})
    checked: list[tuple[str, str | None]] = []

    async def check(provider, key=None, endpoint=None):
        checked.append((provider, key))
        return keycheck.KeyCheck(True, "ok", 200, "The key works (the server).", 3)

    monkeypatch.setattr(keycheck, "check", check)
    return home, checked


def run(**cmd):
    return asyncio.run(brains.command(cmd, restart=lambda: False))


class _Sent:
    """keycheck's openai-SDK client, faked: which key went to which base URL."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, str]] = []
        outer = self

        class Models:
            def __init__(self, base, key):
                self.base, self.key = base, key

            async def list(self):
                outer.calls.append((self.base, self.key))

                class Page:
                    data: list = []

                return Page()

        class Client:
            def __init__(self, base, key):
                self.models = Models(base, key)

        monkeypatch.setattr(keycheck, "_openai_client", lambda base, key: Client(base, key))

    def keys_to(self, host: str) -> list[str]:
        return [k for b, k in self.calls if host in b]


_REAL_CHECK = keycheck.check  # (the fixture fakes it for the saves; these tests ask the real one)


def _real_check(monkeypatch):
    return _REAL_CHECK


# ---- an endpoint's `env`: only its template's own variable, at its template's own host ----------------


def test_an_endpoint_names_only_its_templates_own_variable():
    base = {"id": "custom:x", "label": "X", "base_url": "https://api.x.ai/v1", "model": "grok-4.7"}
    assert CustomProvider.model_validate({**base, "template": "xai", "env": "XAI_API_KEY"}).env == "XAI_API_KEY"
    assert CustomProvider.model_validate({**base, "template": "xai"}).env is None  # (none named: none)
    for template, env in (("other", "ANTHROPIC_API_KEY"), ("other", "XAI_API_KEY"), ("xai", "GITHUB_TOKEN"),
                          ("xai", "DEEPSEEK_API_KEY"), ("deepseek", "GEMINI_API_KEYS"), ("ollama", "OPENAI_API_KEY")):  # fmt: skip
        with pytest.raises(ValueError):
            CustomProvider.model_validate({**base, "template": template, "env": env})
    # (the templates' variables are none of the catalogue's own key names: the store never fills them)
    aliases = set(keystore.key_aliases().values()) | {"GEMINI_API_KEYS"}
    assert not aliases & set(brainref.TEMPLATE_KEYS[t][0] for t in brainref.TEMPLATE_KEYS)
    for t, (env, host) in brainref.TEMPLATE_KEYS.items():
        tpl = catalog.TEMPLATES[t]
        assert tpl.env == env and httpx.URL(tpl.base_url).host == host, t


def test_no_endpoint_ever_gets_another_providers_key(files, monkeypatch):
    home, _ = files
    keystore.FileKeyStore(home / "api_keys.json").put("anthropic", ANTHROPIC)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_TEST000000000000000000")
    keystore.SavedConfig(home / "brains.json").update({"custom_providers": [
        {"id": "custom:evil", "label": "x", "base_url": "https://attacker.example/v1", "model": "m", "env": "ANTHROPIC_API_KEY"},
        {"id": "custom:evil2", "label": "x", "base_url": "https://attacker.example/v1", "model": "m", "env": "GITHUB_TOKEN"},
    ]})  # fmt: skip
    s = config.settings()
    assert s.anthropic_api_key == ANTHROPIC
    assert s.custom_keys == {} and s.custom_providers == []  # (an endpoint NEO can't read isn't set up)


def test_custom_save_takes_the_variable_from_the_template_never_the_window(files, monkeypatch):
    sent = _Sent(monkeypatch)
    keystore.FileKeyStore(files[0] / "api_keys.json").put("anthropic", ANTHROPIC)
    t, d = run(cmd="custom_save", id="1", endpoint={"label": "Fast", "template": "other",
               "base_url": "https://proxy.attacker.example/v1", "model": "m", "env": "ANTHROPIC_API_KEY"})  # fmt: skip
    assert d["ok"] is True and d["provider"] == "custom:fast"
    (cp,) = config.settings().custom_providers
    assert cp.env is None and config.settings().custom_keys == {}
    _, d = run(cmd="brains_get", id="2")
    assert d["keys"]["custom:fast"]["set"] is False
    _, d = run(cmd="models_list", id="3", provider="custom:fast")
    assert d["ok"] is False and sent.calls == []  # no key to send: nothing sent at all
    _, d = run(cmd="custom_save", id="4", endpoint={"label": "Grok", "template": "xai", "base_url": "https://api.x.ai/v1",
               "model": "grok-4.7", "env": "GITHUB_TOKEN"})  # fmt: skip
    assert d["ok"] is True
    assert {c.id: c.env for c in config.settings().custom_providers}["custom:xai"] == "XAI_API_KEY"


def test_a_templates_variable_counts_only_at_the_templates_own_host(files, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-TEST-env-000000000000000000")
    at_home = {"id": "custom:xai", "label": "xAI", "template": "xai", "base_url": "https://api.x.ai/v1", "model": "grok-4.7",
               "env": "XAI_API_KEY"}  # fmt: skip
    moved = {**at_home, "id": "custom:proxy", "base_url": "https://proxy.example.com/v1"}
    keystore.SavedConfig(files[0] / "brains.json").update({"custom_providers": [at_home, moved]})
    s = config.settings()
    assert s.custom_keys == {"custom:xai": "xai-TEST-env-000000000000000000"}
    srcs = keystore.sources(environ={"XAI_API_KEY": "xai-TEST-env-000000000000000000"}, custom=s.custom_providers, files=[])
    assert srcs["custom:xai"]["source"] == "env" and srcs["custom:proxy"]["source"] == "none"


# ---- a saved endpoint's key never follows a base URL the window sends ----------------------------------


def _saved_deepseek(home):
    keystore.SavedConfig(home / "brains.json").update({"custom_providers": [
        {"id": "custom:deepseek", "label": "DeepSeek", "template": "other", "base_url": "https://api.deepseek.com",
         "model": "deepseek-chat"}]})  # fmt: skip
    keystore.FileKeyStore(home / "api_keys.json").put("custom:deepseek", ENDPOINT_KEY)


def test_a_saved_endpoints_key_never_goes_to_another_host(files, monkeypatch):
    _saved_deepseek(files[0])
    real = _real_check(monkeypatch)
    sent = _Sent(monkeypatch)
    ep = {"label": "x", "base_url": "https://attacker.example/v1", "model": "m"}
    r = asyncio.run(real("custom:deepseek", None, ep))
    assert r.ok is False and r.status == "error"
    r = asyncio.run(keycheck.models_list("custom", None, {**ep, "id": "custom:deepseek"}))
    assert r["ok"] is False
    assert sent.keys_to("attacker.example") == []
    # …while the saved endpoint, as saved (or sent back unchanged), is tested with its own key.
    asyncio.run(real("custom:deepseek"))
    asyncio.run(real("custom", None, {"id": "custom:deepseek", "label": "DeepSeek", "base_url": "https://api.deepseek.com/",
                                      "model": "deepseek-chat"}))  # fmt: skip
    assert sent.keys_to("api.deepseek.com") == [ENDPOINT_KEY, ENDPOINT_KEY]


def test_a_form_not_saved_yet_borrows_no_saved_key(files, monkeypatch):
    sent = _Sent(monkeypatch)
    _, d = run(cmd="custom_save", id="1", key=ENDPOINT_KEY,
               endpoint={"id": None, "label": "New", "template": "other", "base_url": "https://api.provider-one.example/v1", "model": "m"})  # fmt: skip
    assert d["ok"] is True
    real = _real_check(monkeypatch)
    sent = _Sent(monkeypatch)
    r = asyncio.run(real("custom", None, {"id": None, "label": "Other", "template": "other",
                                          "base_url": "https://another-host.example/v1", "model": "probe"}))  # fmt: skip
    assert r.ok is False and "no key" in r.message.lower()
    assert sent.calls == []
    form = keycheck._endpoint("custom", {"label": "x", "base_url": "https://h.example/v1", "model": "m"}, None)
    assert form.id == keycheck.FORM_ID and not brainref.parse(form.id).valid  # (an id no endpoint can be saved under)


def test_moving_a_saved_endpoint_to_another_host_needs_its_key_again(files, monkeypatch):
    home, _ = files
    _saved_deepseek(home)
    ep = {"id": "custom:deepseek", "label": "DeepSeek", "template": "other", "base_url": "https://elsewhere.example/v1", "model": "deepseek-chat"}
    _, d = run(cmd="custom_save", id="1", endpoint=ep)
    assert d == {"id": "1", "ok": False, "provider": "custom:deepseek", "error": "needs_key"}
    assert config.settings().custom_providers[0].base_url == "https://api.deepseek.com"  # (nothing changed)
    _, d = run(cmd="custom_save", id="2", endpoint={**ep, "base_url": "https://api.deepseek.com/v1"})  # same host: fine
    assert d["ok"] is True and config.settings().custom_keys == {"custom:deepseek": ENDPOINT_KEY}
    _, d = run(cmd="custom_save", id="3", endpoint=ep, key="sk-TEST-new-key-for-elsewhere-0000")
    assert d["ok"] is True and config.settings().custom_keys == {"custom:deepseek": "sk-TEST-new-key-for-elsewhere-0000"}
    _, d = run(cmd="custom_save", id="4", endpoint={**ep, "base_url": "http://localhost:1234/v1"})  # to this Mac, no key
    assert d["ok"] is True and config.settings().custom_keys == {}  # (the cloud key isn't sent to whatever listens here)


# ---- a key pasted where a model id or a name goes ------------------------------------------------------------


PASTED = "sk-or-v1-" + "ab12" * 16


def test_a_key_pasted_as_a_model_is_never_a_brain(files):
    for ref in (f"openrouter@{PASTED}", f"custom:x@{ANTHROPIC}", "openai@sk-" + "A1b2" * 10, "groq@" + "a1B2" * 9):
        assert brainref.parse(ref).valid is False, ref
        assert brainref.shown(brainref.parse(ref).text) is None
    for model in ("meta-llama/llama-4-maverick-17b-128e-instruct", "qwen3:14b", "deepseek/deepseek-v4-pro", "x-ai/grok-4",
                  "mlx-community/gemma-4-12B-it-4bit", "ft:gpt-6-luna:my-org:custom:A1b2C3d4", "xai-org/grok-1",
                  "claude-opus-5-5", "Qwen/Qwen2.5-Coder-32B-Instruct", "openai/gpt-oss-120b:free"):  # fmt: skip
        assert brainref.valid_model(model) and brainref.parse(f"openrouter@{model}").valid, model
    base = {"id": "custom:x", "label": "X", "base_url": "https://api.example.com/v1", "model": "m"}
    for bad in ({"model": PASTED}, {"label": ANTHROPIC}, {"label": f"my {OPENROUTER}"}):
        with pytest.raises(ValueError):
            CustomProvider.model_validate({**base, **bad})
    s = config.settings()
    object.__setattr__(s, "openrouter_api_key", OPENROUTER)
    _, d = run(cmd="brains_set", id="s", brain=f"openrouter@{PASTED}")
    assert d["ok"] is False and d["error"] == "invalid" and d.get("message", "").startswith("That looks like an API key")


def test_no_label_any_client_hears_carries_a_key(files):
    s = config.settings()
    object.__setattr__(s, "openrouter_api_key", OPENROUTER)
    object.__setattr__(s, "openrouter_model", PASTED)  # (a model setting typed wrong in a .env file)
    object.__setattr__(s, "brain", "openrouter")
    blob = json.dumps(brains.summary())
    assert PASTED[:12] not in blob and "ab12ab12" not in blob
    from neo2.core.agent.session import brain_label

    assert "ab12ab12" not in brain_label("openrouter", PASTED) and "ab12ab12" not in catalog.label_for("openrouter", PASTED)
    assert keyshape.scrub(f"OpenRouter · {PASTED}") == "OpenRouter · ••••"
    assert keyshape.scrub("Claude Opus 5.5") == "Claude Opus 5.5"


# ---- a key for another provider is not sent to this one --------------------------------------------------


def test_a_key_shaped_for_another_provider_isnt_sent(files):
    _, checked = files
    _, d = run(cmd="key_save", id="1", provider="openrouter", key=ANTHROPIC)
    assert d["ok"] is False and d["error"] == "wrong_provider" and "Anthropic" in d["message"] and checked == []
    _, d = run(cmd="key_save", id="1b", provider="gemini", key="AIzaTEST-gemini-0000-abcdefghijk", extra=[ANTHROPIC])
    assert d["ok"] is False and d["error"] == "wrong_provider" and checked == []  # (an extra key too)
    _, d = run(cmd="key_test", id="2", provider="openai", key=OPENROUTER)
    assert d["ok"] is False and d["status"] == "wrong_provider" and checked == []
    other = {"id": None, "label": "Proxy", "template": "other", "base_url": "https://api.example.com/v1", "model": "m"}
    _, d = run(cmd="custom_save", id="3", endpoint=other, key=ANTHROPIC)
    assert d["ok"] is False and d["error"] == "wrong_provider"
    _, d = run(cmd="models_list", id="4", provider="custom", endpoint=other, key=ANTHROPIC)
    assert d["ok"] is False and d["error"] == "wrong_provider"
    # Where it does belong: its own provider, its own host (Anthropic's OpenAI-compatible address), this Mac.
    assert run(cmd="key_save", id="5", provider="anthropic", key=ANTHROPIC)[1]["ok"] is True
    own = {**other, "label": "Claude compat", "base_url": "https://api.anthropic.com/v1"}
    assert run(cmd="custom_save", id="6", endpoint=own, key=ANTHROPIC)[1]["ok"] is True
    here = {**other, "label": "Local proxy", "base_url": "http://localhost:4000/v1"}
    assert run(cmd="custom_save", id="7", endpoint=here, key=ANTHROPIC)[1]["ok"] is True
    xai = {**other, "label": "Grok", "template": "xai", "base_url": "https://api.x.ai/v1"}
    assert run(cmd="custom_save", id="8", endpoint=xai, key="xai-TEST-0000-aaaaaaaaaaaaaaaaaaaa")[1]["ok"] is True
    assert run(cmd="key_save", id="9", provider="openai", key="sk-TEST-old-shape-0000000000000000000")[1]["ok"] is True
    assert ANTHROPIC not in json.dumps(d)


def test_a_cloud_endpoint_whose_url_says_localhost_is_in_the_cloud():
    from neo2.core.providers.openai_compat import OpenAICompatProvider, custom

    for url in ("https://localhost.attacker.example/v1", "https://api.attacker.example/127.0.0.1/v1"):
        cp = CustomProvider(id="custom:x", label="x", base_url=url, model="m")
        p = custom(cp, ENDPOINT_KEY)
        assert (cp.on_device, p.on_device, p._is_local) == (False, False, False), url
    for url in ("http://localhost:11434/v1", "http://127.0.0.1:1234/v1", "http://[::1]:8000/v1"):
        assert custom(CustomProvider(id="custom:x", label="x", base_url=url, model="m")).on_device, url
    assert not OpenAICompatProvider(name="h", model="m", base_url="https://localhost.example.com/v1", api_key="k").on_device


def test_openais_brain_talks_to_openai_whatever_openai_base_url_says(files, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://somewhere-else.example/v1")
    s = config.settings()
    object.__setattr__(s, "openai_api_key", "sk-proj-TEST-0000-aaaaaaaaaaaaaaaa")
    from neo2.core.providers.openai_responses import openai

    p = openai("gpt-6-luna")
    assert str(p._client.base_url).rstrip("/") == "https://api.openai.com/v1"


# ---- caps ------------------------------------------------------------------------------------------------------


def test_a_daily_cap_changed_in_the_window_reaches_the_built_brain(files):
    _, d = run(cmd="key_save", id="a", provider="anthropic", key=ANTHROPIC)
    assert d["ok"] and d["daily_budget"] == 2.0
    p = prov.make("claude@claude-opus-5-5")
    assert p._cap.ledger.daily_budget == 2.0
    prov._demoted["claude@claude-opus-5-5"] = 9e18  # (today's budget was spent: resting till midnight)
    assert run(cmd="brains_set", id="b", daily_budgets={"anthropic": 10.0})[1]["ok"] is True
    q = prov.make("claude@claude-opus-5-5")
    assert q is not p and q._cap.ledger.daily_budget == 10.0 and "claude@claude-opus-5-5" not in prov._demoted
    assert run(cmd="brains_set", id="c", daily_budgets={})[1]["ok"] is True
    assert prov.make("claude@claude-opus-5-5")._cap is None
    prov._cache["cerebras"] = object()
    assert run(cmd="brains_set", id="d", daily_budgets={"cerebras": 1.0})[1]["ok"] is True
    assert "cerebras" not in prov._cache  # (Cerebras's own budget, by its entry)


def _openrouter_list(monkeypatch):
    async def models(key):
        assert key == OPENROUTER
        return [
            {"id": "deepseek/deepseek-v4-pro", "name": "DeepSeek V4 Pro", "supported_parameters": ["tools"],
             "architecture": {"input_modalities": ["text"]}, "pricing": {"prompt": "0.0000003", "completion": "0.0000012"}},
            {"id": "qwen/qwen3-vl-235b", "name": "Qwen3 VL", "supported_parameters": ["tools"],
             "architecture": {"input_modalities": ["text", "image"]}, "pricing": {"prompt": "0.0000002", "completion": "0.0000008"}},
        ]  # fmt: skip

    monkeypatch.setattr(keycheck, "_openrouter_models", models)


def _restart(monkeypatch):
    monkeypatch.setattr(catalog, "_LEARNED", {})
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(keycheck, "_cache", {})
    monkeypatch.setattr(config, "_settings", None)


def test_openrouter_is_counted_at_its_listed_prices_and_keeps_what_its_models_do(files, monkeypatch):
    _openrouter_list(monkeypatch)
    _, d = run(cmd="key_save", id="o", provider="openrouter", key=OPENROUTER)
    assert d["ok"] and d["daily_budget"] == 2.0  # (a paid key saved in the window: capped)
    assert run(cmd="models_list", id="m", provider="openrouter")[1]["ok"] is True
    _, d = run(cmd="brains_set", id="s", brain="openrouter@deepseek/deepseek-v4-pro", eyes_brain="openrouter@qwen/qwen3-vl-235b")
    assert d["ok"] is True
    saved = json.loads(config.SAVED_PATH.read_text())["known_models"]["openrouter"]
    assert saved["qwen/qwen3-vl-235b"] == {"sees": True, "tools": True, "price": [0.2, 0.8]}
    _restart(monkeypatch)  # NEO restarted: what the list said is in brains.json, not in memory
    assert catalog.describe("openrouter@qwen/qwen3-vl-235b").sees is True
    assert [p.name for p in prov.brain("agent")._lead_brains()] == ["openrouter@qwen/qwen3-vl-235b"]
    assert brains.summary()["eyes"]["ok"] is True
    cap = prov.make("openrouter@deepseek/deepseek-v4-pro")._cap
    assert cap.ledger.price("deepseek/deepseek-v4-pro") == (0.3, 1.2)
    steps = 0
    while not cap.ledger.over() and steps < 1000:
        cap.add("deepseek/deepseek-v4-pro", 20_000, 1_000)  # one agent step (~$0.007)
        steps += 1
    assert steps > 250  # $2 of real prices, not 8 steps at the dearest one
    assert cap.ledger.price("some/model-it-never-listed")[1] >= 50.0  # (one it never listed: the dearest)


def test_relearning_a_model_rebuilds_its_brain(files, monkeypatch):
    s = config.settings()
    object.__setattr__(s, "openrouter_api_key", OPENROUTER)
    old = prov.make("openrouter@qwen/qwen3-vl-235b")
    assert old.supports_vision is False  # (nothing known of it: the vendor-prefix rule)
    _openrouter_list(monkeypatch)
    assert run(cmd="models_list", id="m", provider="openrouter")[1]["ok"] is True
    new = prov.make("openrouter@qwen/qwen3-vl-235b")
    assert new is not old and new.supports_vision is True


def test_no_cap_is_saved_for_a_key_the_environment_shadows_nor_for_an_endpoint_without_prices(files, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TEST-env-0000-zzzzzzzzzzzzzz-envk")
    monkeypatch.setattr(config, "_settings", None)
    _, d = run(cmd="key_save", id="a", provider="anthropic", key=ANTHROPIC)
    assert d["ok"] and d["shadowed_by"] == "env" and d["daily_budget"] is None
    assert config.settings().daily_budgets == {} and prov.make("claude")._cap is None  # (decision 4: env keys uncapped)
    ds = {"id": None, "label": "DeepSeek", "template": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-v4-pro"}
    assert run(cmd="custom_save", id="c", endpoint=ds)[1]["ok"] is True
    _, d = run(cmd="key_save", id="k", provider="custom:deepseek", key=ENDPOINT_KEY)
    assert d["ok"] and d["daily_budget"] is None and "custom:deepseek" not in config.settings().daily_budgets
    _, detail = run(cmd="brains_get", id="g")
    assert detail["daily_budgets"]["prefill"]["custom:deepseek"] == 2.0  # (offered in Spending, saved only there)


# ---- chains ----------------------------------------------------------------------------------------------------


class _Brain:
    def __init__(self, name, *, tools=True, vision=True):
        self.name, self.model = name, name
        self.supports_vision, self.supports_tools = vision, tools
        self.asked: list[bool] = []

    def cooling_until(self):
        return 0.0

    async def complete(self, messages, *, tools=None, **kw):
        self.asked.append(bool(tools))
        return Turn(text=self.name)

    async def stream(self, messages, **kw):
        yield self.name


def test_an_endpoint_that_cant_call_tools_never_gets_a_step_with_tools(files, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTEST-gemini-0000-abcdefghijk")  # (a key: not "no brain at all")
    monkeypatch.setattr(config, "_settings", None)
    made = {"custom:chatonly": _Brain("custom:chatonly", tools=False), "gemini": _Brain("gemini")}
    monkeypatch.setattr(prov, "make", lambda n: made[n] if n in made else (_ for _ in ()).throw(Unavailable(n)))
    tool = ToolSpec("open_app", "Open an app", {"type": "object", "properties": {}})
    c = prov.Chain(["custom:chatonly", "gemini"])
    assert asyncio.run(c.complete([Message.user("open safari")], tools=[tool])).text == "gemini"
    assert made["custom:chatonly"].asked == [] and "custom:chatonly" not in prov._demoted  # (skipped, not failed)
    assert asyncio.run(c.complete([Message.user("hi")])).text == "custom:chatonly"  # a chat turn: it answers
    with pytest.raises(prov.AllBrainsFailed) as e:
        asyncio.run(prov.Chain(["custom:chatonly"]).complete([Message.user("open safari")], tools=[tool]))
    from neo2.core.providers.base import failure_words

    assert failure_words("custom:chatonly", e.value.failures[0][1]).endswith("can't use tools")


def test_backups_must_call_tools(files, monkeypatch):
    cp = {"id": "custom:chatonly", "label": "Chat only", "template": "other", "base_url": "https://api.example.com/v1",
          "model": "m1", "tools": False}  # fmt: skip
    monkeypatch.setenv("NEO_CUSTOM_PROVIDERS", json.dumps([cp]))
    monkeypatch.setattr(config, "_settings", None)
    object.__setattr__(config.settings(), "custom_keys", {"custom:chatonly": ENDPOINT_KEY})
    _, d = run(cmd="brains_set", id="b", backup_brains=["custom:chatonly"])
    assert d["ok"] is False and (d["error"], d["field"]) == ("no_tools", "backup_brains")
    assert run(cmd="brains_set", id="q", fast_brain="custom:chatonly")[1]["ok"] is True  # (Quick replies: fine)


def test_an_endpoint_on_this_mac_that_wants_a_key_is_up(files, monkeypatch):
    from neo2.core.providers.openai_compat import custom

    async def fake_get(self, url, **kw):  # LM Studio with its auth on: /v1/models without a key is 401
        return httpx.Response(401, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    cp = CustomProvider(id="custom:lmstudio", label="LM Studio", template="lmstudio", base_url="http://localhost:1234/v1", model="qwen3-8b")
    asyncio.run(custom(cp, "lm-TEST-key-000000000000")._preflight())  # (no Unavailable: it answered)


def test_the_camera_asks_one_brain_per_provider_eyes_first(files):
    s = config.settings()
    assert prov.camera_order(s) == ["gemini", "cerebras_vision"]  # today's
    for f, v in (("fast_brain", "claude@claude-haiku-5-5"), ("eyes_brain", "claude@claude-opus-5-5"), ("brain", "claude@claude-opus-5-5")):
        object.__setattr__(s, f, v)
    assert prov.camera_order(s) == ["claude@claude-opus-5-5", "gemini", "cerebras_vision"]  # one Claude, not two
    for f, v in (("fast_brain", "gemini_lite"), ("eyes_brain", "gemini"), ("brain", "gemini")):
        object.__setattr__(s, f, v)
    assert prov.camera_order(s) == ["gemini", "cerebras_vision"]  # the Gemini quick path: today's camera


def test_groq_and_cerebras_seeing_models_see_by_reference_too(files, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_TEST-groq-0000-aaaaaaaaaaaaaaaa")
    monkeypatch.setattr(config, "_settings", None)
    ref = "groq@qwen/qwen3.8-27b"
    i = catalog.describe(ref)
    assert (i.sees, i.eyes_only) == (True, True)
    assert catalog.describe("cerebras@qwen-3.8-27b").sees is True
    assert catalog.describe("groq_vision@openai/gpt-oss-120b").sees is False
    assert catalog.describe("groq").sees is False and catalog.describe("groq_vision").sees is True  # (as before)
    assert run(cmd="brains_set", id="e", eyes_brain=ref)[1]["ok"] is True
    p = prov.make(ref)
    assert p.supports_vision is True and p.name == ref and p.per_minute is True
    assert prov.make("groq@openai/gpt-oss-120b").supports_vision is False
    from neo2.core.providers import cerebras_hooks

    assert cerebras_hooks._groq_text("groq") and not cerebras_hooks._groq_text(ref)


def test_cloud_eyes_off_in_a_env_file_or_the_environment_is_off(files, monkeypatch):
    monkeypatch.setenv("NEO_CLOUD_EYES", "off")
    monkeypatch.setattr(config, "_settings", None)
    assert config.settings().cloud_eyes == "off"  # (before Brains & Keys "off" read as auto: the README says so)
    for v, want in (("never", "off"), ("on", "on"), ("no", "auto"), ("", "auto")):
        monkeypatch.setenv("NEO_CLOUD_EYES", v)
        assert config.Settings().cloud_eyes == want, v


# ---- the window's commands: one at a time; reloads coalesced; old copies of the key file ---------------------


def test_the_windows_commands_never_lose_each_others_changes(files):
    openai_key = "sk-proj-TEST-0000-aaaaaaaaaaaaaaaa"

    async def both():
        return await asyncio.gather(
            brains.command({"cmd": "key_save", "id": "a", "provider": "anthropic", "key": ANTHROPIC}),
            brains.command({"cmd": "key_save", "id": "b", "provider": "openai", "key": openai_key}),
            brains.command({"cmd": "brains_set", "id": "c", "cloud_eyes": "off"}),
        )

    got = asyncio.run(both())
    assert all(d["ok"] for _, d in got)
    saved = json.loads(config.SAVED_PATH.read_text())
    assert saved["cloud_eyes"] == "off" and saved["daily_budgets"] == {"anthropic": 2.0, "openai": 2.0}
    s = config.settings()
    assert s.cloud_eyes == "off" and (s.anthropic_api_key, s.openai_api_key) == (ANTHROPIC, openai_key)
    assert s.daily_budgets == {"anthropic": 2.0, "openai": 2.0}


def test_reloads_are_coalesced_and_each_sees_the_files_as_they_were_when_asked(files, monkeypatch):
    applied: list[int] = []
    real = brains.apply

    async def counting(*, tell=True):
        applied.append(1)
        await asyncio.sleep(0.01)
        return await real(tell=False)

    monkeypatch.setattr(brains, "apply", counting)
    monkeypatch.setattr(brains, "RELOAD_GAP_S", 0.0)

    async def many():
        return await asyncio.gather(*(brains.reload() for _ in range(20)))

    got = asyncio.run(many())
    assert all(isinstance(g, dict) for g in got) and 1 <= len(applied) <= 2


def test_old_copies_of_the_key_file_are_reported(files):
    home, _ = files
    home.mkdir(parents=True, exist_ok=True)
    (home / "api_keys.json").write_text("{ not json")
    _, d = run(cmd="brains_get", id="1")
    assert d["store"]["ok"] is False and d["store"]["old_copies"] == 0
    assert run(cmd="key_save", id="2", provider="anthropic", key=ANTHROPIC)[1]["ok"] is True  # (set aside, then saved)
    _, d = run(cmd="brains_get", id="3")
    assert d["store"]["ok"] is True and d["store"]["old_copies"] == 1
    assert keystore.old_copies(home / "api_keys.json") == 1


def test_a_gemini_key_refused_everywhere_is_a_refused_key(monkeypatch):
    g = pytest.importorskip("neo2.core.providers.gemini")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    p = g.GeminiProvider(model="gemini-3.8-flash")
    import time

    for ki in range(len(p._clients)):
        p._bad_keys[p._kid(ki)] = time.time() + 3600
    e = p._all_blocked_error()
    assert isinstance(e, KeyRefused) and e.kind == "key"
