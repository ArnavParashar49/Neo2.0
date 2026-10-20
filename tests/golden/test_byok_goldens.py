"""Today's setups, pinned before "Brains & Keys" (bring your own key) changes anything: the "zero
action" proof. Captured at a41960b, before keystore.py, catalog.py or the new brain roles existed,
and kept byte-identical from then on.

* chains.json: brain(purpose)'s order (names, seeing fallbacks, which brains get built, which are
  missing) for agent, fast, light, vision and offline, across keys x NEO_BRAIN/NEO_FAST_BRAIN x
  NEO_CLOUD_EYES.
* settings.json: what Settings() resolves from a ~/.neo/.env stand-in, a checkout .env and the
  environment (only the fields that existed then: a field added later doesn't count).
* claude_kwargs.json: what ClaudeProvider sends for claude-opus-5 (a fake client takes it).
* hello.json: the keys of the websocket hello each kind of client gets, and the shell's. Rewritten
  once, on purpose, when the hello gained "brains" (Brains & Keys' summary): only that key was added;
  and once more when the shell's hello gained "lens" (Search Screen's switch and hold time): only that.

Every key here is an obvious fake. Nothing reads ~/.neo, the real .env files or the network (the
websocket server is this test's own, on a free port of this process).

NEO_GOLDEN_WRITE=1 rewrites the files instead of comparing (only ever at the capture).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
from pathlib import Path
from typing import Any

import pytest

from neo2.core import config, effects
from neo2.core import providers as prov
from neo2.core.providers.base import ImagePart, Message, ToolCall, ToolResult, ToolSpec

HERE = Path(__file__).parent
WRITE = os.environ.get("NEO_GOLDEN_WRITE", "") == "1"

# Fakes, never real keys.
GEMINI = "AIzaTEST-gemini-0000"
GROQ = "gsk_TEST-groq-0000"
ANTHROPIC = "sk-ant-TEST-0000"
CEREBRAS = "csk-TEST-cerebras-0000"

_KEYS = ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY")


def _compare(name: str, got: Any) -> None:
    path = HERE / name
    # One line per case: a change shows as the case that changed.
    rows = (json.dumps(k) + ": " + json.dumps(got[k], sort_keys=True, ensure_ascii=False, separators=(",", ":")) for k in sorted(got))
    text = "{\n" + ",\n".join(rows) + "\n}\n"
    if WRITE:
        path.write_text(text)
        return
    want = json.loads(path.read_text())
    assert json.loads(text) == want, f"{name} changed: today's setups no longer resolve as they did"


def _clean_env(monkeypatch, tmp_path: Path) -> None:
    """No NEO_ setting or key of the developer's shell: only what each case sets (and the suite's
    own data folders, in tmp_path)."""
    for k in list(os.environ):
        if k.startswith("NEO_") or k in _KEYS:
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path / "neo_data"))
    monkeypatch.setenv("NEO_NOTES_DIR", str(tmp_path / "notes"))
    monkeypatch.setenv("NEO_REFLEX", "off")
    monkeypatch.setenv("NEO_VOICE", "off")
    monkeypatch.setenv("NEO_PREFETCH_LOOK", "0")


# ---- (a) the chains ----------------------------------------------------------------------------

KEY_SETS: dict[str, dict[str, str]] = {
    "none": {},
    "gemini": {"GEMINI_API_KEY": GEMINI},
    "gemini+groq": {"GEMINI_API_KEY": GEMINI, "GROQ_API_KEY": GROQ},
    "gemini+groq+cerebras": {"GEMINI_API_KEY": GEMINI, "GROQ_API_KEY": GROQ, "CEREBRAS_API_KEY": CEREBRAS},
    "gemini+groq+anthropic": {"GEMINI_API_KEY": GEMINI, "GROQ_API_KEY": GROQ, "ANTHROPIC_API_KEY": ANTHROPIC},
    "gemini+groq+cerebras+anthropic": {
        "GEMINI_API_KEY": GEMINI,
        "GROQ_API_KEY": GROQ,
        "CEREBRAS_API_KEY": CEREBRAS,
        "ANTHROPIC_API_KEY": ANTHROPIC,
    },
    "anthropic": {"ANTHROPIC_API_KEY": ANTHROPIC},
    "groq": {"GROQ_API_KEY": GROQ},
    "cerebras": {"CEREBRAS_API_KEY": CEREBRAS},
}
# (NEO_BRAIN, NEO_FAST_BRAIN): None leaves it at its default
BRAINS: list[tuple[str | None, str | None]] = [
    (None, None),
    ("gemini", None),
    ("claude", None),
    ("groq", None),
    ("local", None),
    ("cerebras", None),
    ("cerebras_vision", None),
    ("claude", "claude"),
    ("local", "local"),
    ("gemini", "local"),
    ("local", "gemini"),
    ("cerebras", "local"),
    ("gemini", "gemini_lite"),
]
CLOUD_EYES = (None, "on")
PURPOSES = ("agent", "fast", "light", "vision", "offline")


def _limit(lim: Any) -> list[Any] | None:
    return None if lim is None else [lim.long_edge, lim.pixels, lim.bytes]


def _chains() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in PURPOSES:
        c = prov.brain(p)  # type: ignore[arg-type]
        built = c._providers()
        eyes = c._eye_brains()
        out[p] = {
            "names": [str(n) for n in c.names],
            "eyes": list(c.eyes),
            "sticky": c.sticky,
            "built": [[q.name, str(getattr(q, "model", "")), bool(getattr(q, "supports_vision", False))] for q in built],
            "missing": [n for n, _ in c._missing],
            "eye_brains": [[q.name, str(getattr(q, "model", ""))] for q in eyes],
        }
    out["image_limit"] = {"vision": _limit(prov.image_limit("vision")), "agent": _limit(prov.image_limit("agent"))}
    return out


def test_brain_chains_are_todays(monkeypatch, tmp_path):
    got: dict[str, Any] = {}
    for keys_name, keys in KEY_SETS.items():
        for brain, fast in BRAINS:
            for eyes in CLOUD_EYES:
                _clean_env(monkeypatch, tmp_path)
                for k, v in keys.items():
                    monkeypatch.setenv(k, v)
                if brain is not None:
                    monkeypatch.setenv("NEO_BRAIN", brain)
                if fast is not None:
                    monkeypatch.setenv("NEO_FAST_BRAIN", fast)
                if eyes is not None:
                    monkeypatch.setenv("NEO_CLOUD_EYES", eyes)
                monkeypatch.setattr(config, "_settings", None)
                monkeypatch.setattr(prov, "_cache", {})
                monkeypatch.setattr(prov, "_demoted", {})
                monkeypatch.setattr(prov, "_said_unavailable", set())
                case = f"keys={keys_name} brain={brain or '-'} fast={fast or '-'} cloud_eyes={eyes or '-'}"
                got[case] = _chains()
    _compare("chains.json", got)


# ---- (b) settings from the .env files and the environment -------------------------------------

# Every field Settings had at the capture (a41960b): a field added later is not compared.
TODAY_FIELDS = (
    "anthropic_api_key batch_actions brain browser_headless browser_view camera camera_follow_s "
    "cerebras_api_key cerebras_daily_budget cerebras_model cerebras_prices cerebras_vision_model "
    "claude_model cloud_eyes cloud_memory confirm_destructive context_capture context_copy_fallback "
    "data_dir decide_kind_routing decide_modes embeddings extra_secret_paths fast_brain fast_paths "
    "gemini_api_key gemini_api_keys gemini_fallback_models gemini_lite_model gemini_live_model "
    "gemini_model ground_port groq_api_key groq_model groq_vision_model home_location include_username "
    "instructions laya_device learn learn_teacher listen_timeout_s live_command_lane live_voice "
    "local_grounder local_model max_seconds max_steps max_tool_result_chars memory_enabled "
    "memory_extract memory_journal memory_live_inject memory_path memory_recall_k "
    "memory_retention_days mic_escalate models_path notes_dir offline_brain persona play_default "
    "play_site playbooks prefetch_look prefetch_look_s prewarm_resolver protected_apps reflex "
    "reflex_confidence_floor reflex_encoder reflex_tool_floor replay_flows resolver resolver_laya_p "
    "resolver_promoted sck_capture sck_stream screen_effort shortcuts shortcuts_names_in_prompt "
    "shot_format song_id song_listen_s stt_model stuck_breaker summaries tts_model tts_voice ui_host "
    "vision_cloud_ok vision_detector vision_download_ok vision_words voice wake_gate wake_gate_s "
    "wake_gate_speech_ms wake_word wellbeing_checkins workflow_suggestions ws_host ws_port"
).split()

# (a ~/.neo/.env stand-in, a checkout .env, the environment)
SETTINGS_CASES: dict[str, tuple[str | None, str | None, dict[str, str]]] = {
    "nothing anywhere": (None, None, {}),
    "a gemini key in ~/.neo/.env": (f"GEMINI_API_KEY={GEMINI}\n", None, {}),
    "the app's template, filled in": (
        "# NEO settings\n"
        f"GEMINI_API_KEY={GEMINI}\nGROQ_API_KEY={GROQ}\nCEREBRAS_API_KEY=\n"
        "NEO_CEREBRAS_DAILY_BUDGET=0.50\nHOME_LOCATION=\nNEO_CLOUD_MEMORY=full\nNEO_LEARN=true\n",
        None,
        {},
    ),
    "claude as the brain": (
        f"GEMINI_API_KEY={GEMINI}\nANTHROPIC_API_KEY={ANTHROPIC}\nNEO_BRAIN=claude\nNEO_FAST_BRAIN=claude\n",
        None,
        {},
    ),
    "a checkout .env beats ~/.neo/.env": (
        f"GEMINI_API_KEY=AIzaTEST-user-1111\nGROQ_API_KEY={GROQ}\nNEO_BRAIN=gemini\nNEO_MAX_STEPS=10\n",
        f"GEMINI_API_KEY={GEMINI}\nNEO_BRAIN=claude\nANTHROPIC_API_KEY={ANTHROPIC}\n",
        {},
    ),
    "the environment beats both files": (
        "GEMINI_API_KEY=AIzaTEST-user-1111\nNEO_BRAIN=groq\n",
        "GEMINI_API_KEY=AIzaTEST-checkout-2222\nNEO_FAST_BRAIN=local\n",
        {"GEMINI_API_KEY": GEMINI, "NEO_BRAIN": "local", "NEO_OFFLINE_BRAIN": "groq"},
    ),
    "extra gemini keys, quotes, export and comments": (
        "export GEMINI_API_KEY='AIzaTEST-quoted-3333'\n"
        'GEMINI_API_KEYS=" AIzaTEST-x-1 ,AIzaTEST-x-2,,AIzaTEST-quoted-3333"\n'
        "# a comment\nGROQ_API_KEY=gsk_TEST-groq-4444 # trailing\n",
        None,
        {},
    ),
    "cerebras with a budget and its prices": (
        f"CEREBRAS_API_KEY={CEREBRAS}\nNEO_CEREBRAS_DAILY_BUDGET=$1.25\n"
        'NEO_CEREBRAS_PRICES={"gpt-oss-120b": [0.3, 0.7]}\nNEO_BRAIN=cerebras\n',
        None,
        {},
    ),
    "typos never stop NEO from starting": (
        "NEO_CLOUD_EYES=always\nNEO_CLOUD_MEMORY=nothing\nNEO_LEARN_TEACHER=sometimes\nNEO_REFLEX_ENCODER=EN\n"
        "NEO_SCREEN_EFFORT=extreme\nNEO_SHOT_FORMAT=jpg\nNEO_RESOLVER=maybe\nNEO_MEMORY_EXTRACT=llm\n"
        "NEO_MEMORY_RETENTION_DAYS=lots\nNEO_WAKE_GATE=ON\nNEO_WAKE_GATE_S=99\nNEO_BROWSER_VIEW=headed\n"
        "NEO_CEREBRAS_DAILY_BUDGET=off\nNEO_DECIDE_MODES=add_event=on,read_mail=bogus\n",
        None,
        {},
    ),
    "models and voice": (
        "NEO_CLAUDE_MODEL=claude-sonnet-5-5\nNEO_GEMINI_MODEL=gemini-3.7-flash\n"
        'NEO_GEMINI_FALLBACK_MODELS=["gemini-3.5-flash"]\nNEO_GROQ_VISION_MODEL=\nNEO_VOICE=local\n'
        "NEO_WAKE_WORD=false\nNEO_UI_HOST=notch\nNEO_LOCAL_MODEL=mlx-community/other-4bit\n",
        None,
        {"NEO_VOICE": "live"},
    ),
}


def _plain(v: Any, tmp: str) -> Any:
    if isinstance(v, Path):
        v = str(v)
    if isinstance(v, str):
        return v.replace(tmp, "<tmp>").replace(str(Path.home()), "<home>")
    if isinstance(v, dict):
        return {str(k): _plain(x, tmp) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_plain(x, tmp) for x in v]
    return v


def test_settings_resolve_as_today(monkeypatch, tmp_path):
    got: dict[str, Any] = {}
    tmp = str(tmp_path.resolve())
    for case, (user, checkout, environ) in SETTINGS_CASES.items():
        _clean_env(monkeypatch, tmp_path)
        for k, v in environ.items():
            monkeypatch.setenv(k, v)
        d = tmp_path / case.replace(" ", "_").replace("/", "_").replace("~", "home")
        d.mkdir()
        files: list[Path] = []
        for name, text in (("user.env", user), ("checkout.env", checkout)):
            p = d / name
            if text is not None:
                p.write_text(text)
            files.append(p)  # (a file that isn't there is skipped, as ~/.neo/.env is before it's made)
        token = effects._as_app.set(True)  # read the files as NEO.app does (under the suite, never)
        try:
            s = config.Settings(_env_file=tuple(files))
        finally:
            effects._as_app.reset(token)
        dumped = s.model_dump()
        row = {f: _plain(dumped[f], tmp) for f in TODAY_FIELDS}
        row["(gemini_keys)"] = s.gemini_keys
        got[case] = row
    _compare("settings.json", got)


def test_settings_fields_listed_are_todays():
    """The compared fields all still exist (none was renamed away)."""
    assert set(TODAY_FIELDS) <= set(config.Settings.model_fields)


# ---- (c) what ClaudeProvider sends for claude-opus-5 ------------------------------------------


class _Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not (exclude_none and v is None)}


class _Usage:
    input_tokens = 120
    output_tokens = 34


class _Resp:
    def __init__(self, content: list[_Block], stop_reason: str = "end_turn") -> None:
        self.content, self.stop_reason, self.usage = content, stop_reason, _Usage()


class _Stream:
    def __init__(self) -> None:
        async def gen():
            for t in ("Hel", "lo"):
                yield t

        self.text_stream = gen()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAnthropic:
    """Takes what ClaudeProvider would send and answers from a script."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        me = self

        class _Messages:
            async def create(self, **kw):
                me.sent.append({"call": "create", **kw})
                return _Resp(
                    [
                        _Block(type="thinking", thinking="looked", signature="sig-1"),
                        _Block(type="text", text="Done."),
                        _Block(type="tool_use", id="toolu_1", name="open_app", input={"name": "Notes"}),
                    ]
                )

            def stream(self, **kw):
                me.sent.append({"call": "stream", **kw})
                return _Stream()

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


def _json(v: Any) -> Any:
    import anthropic

    if isinstance(v, anthropic.NotGiven):
        return "<NOT_GIVEN>"
    if isinstance(v, dict):
        return {str(k): _json(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_json(x) for x in v]
    if isinstance(v, bytes):
        return v.decode()
    return v


PNG = b"\x89PNG\r\n\x1a\nfake"
TOOLS = [
    ToolSpec("open_app", "Open an app by its name.", {"type": "object", "properties": {"name": {"type": "string"}}}),
    ToolSpec("screenshot", "Look at the screen.", {"type": "object", "properties": {}}),
]


def _conversation() -> list[Message]:
    return [
        Message.user("open notes", [ImagePart(PNG, "image/png")]),
        Message.assistant(
            "",
            [ToolCall("toolu_0", "screenshot", {})],
            raw=[{"type": "thinking", "thinking": "", "signature": "sig-0"}, {"type": "tool_use", "id": "toolu_0", "name": "screenshot", "input": {}}],
        ),
        Message.tool([ToolResult("toolu_0", "screenshot", "the desktop", images=[ImagePart(PNG, "image/png")])]),
        Message.assistant("Opening it.", [ToolCall("toolu_x", "open_app", {"name": "Notes"})]),
        Message.tool([ToolResult("toolu_x", "open_app", "", ok=False)]),
    ]


def test_claude_opus_5_requests_are_todays(monkeypatch, tmp_path):
    from neo2.core.providers.claude import ClaudeProvider

    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC)
    monkeypatch.setattr(config, "_settings", None)

    async def run() -> dict[str, Any]:
        out: dict[str, Any] = {}
        hosts = _FakeAnthropic()
        p = ClaudeProvider(client=hosts)  # a host's own client: the release's model
        out["host model"] = p.model
        for effort in ("low", "medium", "high"):
            await p.complete([Message.user("hi")], effort=effort)  # type: ignore[arg-type]
        turn = await p.complete(_conversation(), system="You are NEO.", tools=TOOLS, effort="high", max_tokens=32000)
        out["turn"] = {
            "text": turn.text,
            "thinking": turn.thinking,
            "tool_calls": [[c.id, c.name, c.args, c.source] for c in turn.tool_calls],
            "raw": turn.raw,
            "usage": [turn.usage.input_tokens, turn.usage.output_tokens],
            "offered": turn.offered,
        }
        out["streamed"] = "".join([t async for t in p.stream([Message.user("tell me a joke")], system="Be brief.")])
        out["host requests"] = _json(hosts.sent)
        mine = ClaudeProvider()  # NEO.app's: key and model from settings
        mine._client = _FakeAnthropic()
        out["app model"] = mine.model
        await mine.complete([Message.user("hi")])
        out["app requests"] = _json(mine._client.sent)
        return out

    _compare("claude_kwargs.json", asyncio.run(run()))


# ---- (d) the hello's keys --------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_hello_keys_are_todays(monkeypatch, tmp_path):
    import contextlib

    import websockets

    from neo2.core.server import shell_link
    from neo2.core.server import ws as ws_mod

    _clean_env(monkeypatch, tmp_path)
    port = _free_port()
    monkeypatch.setenv("NEO_WS_PORT", str(port))
    monkeypatch.setattr(config, "_settings", None)

    async def on_command(cmd):
        pass

    async def hello(origin: str | None, role: str | None) -> dict[str, Any]:
        q = f"/?token={ws_mod.ws_token()}" + (f"&role={role}" if role else "")
        headers = {"Origin": origin} if origin else None
        async with websockets.connect(f"ws://127.0.0.1:{port}{q}", additional_headers=headers, open_timeout=3) as c:
            while True:
                msg = json.loads(await asyncio.wait_for(c.recv(), 3))
                if msg["type"] == "hello":
                    return msg

    def keys(msg: dict[str, Any]) -> dict[str, Any]:
        data = msg["data"]
        return {
            "top": sorted(msg),
            "data": sorted(data),
            "shell": sorted(data.get("shell") or {}),
        }

    async def run() -> dict[str, Any]:
        srv = ws_mod.UIServer(on_command)
        await srv.start()
        out: dict[str, Any] = {}
        try:
            out["app"] = keys(await hello("tauri://localhost", None))
            out["page"] = keys(await hello("neo://app", None))
            out["host"] = keys(await hello(None, "host"))
            out["tool"] = keys(await hello(None, None))
            ours, theirs = socket.socketpair()
            os.set_inheritable(theirs.fileno(), True)
            monkeypatch.setenv(shell_link.ENV, str(theirs.detach()))
            r, w = await asyncio.open_connection(sock=ours)
            srv.shell_task = shell_link.start(srv, shell_link.inherited())
            first = json.loads(await asyncio.wait_for(r.readline(), 3))
            out["shell link"] = sorted(first)
            w.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(srv.shell_task, 3)
        finally:
            if srv._server:
                srv._server.close()
        return out

    _compare("hello.json", asyncio.run(run()))


@pytest.mark.parametrize("name", ["chains.json", "settings.json", "claude_kwargs.json", "hello.json"])
def test_no_golden_holds_a_real_looking_key(name):
    """The captured files hold only this test's fakes."""
    text = (HERE / name).read_text()
    for m in re.finditer(r"(?:sk-ant-|AIza|gsk_|csk-)[^\"\s,]*", text):
        assert "TEST" in m.group(0), f"{name}: a key that isn't a fake"
