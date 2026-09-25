"""Regressions from the sequential Notes test: AX element visibility/ids, brain fallback speed,
and the Live silence window."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

# ---- AX ----------------------------------------------------------------------------------------


def _ax():
    return pytest.importorskip("neo.tools.computer.ax")


def test_ax_find_keeps_ids_from_the_tree_the_model_holds(monkeypatch):
    ax = _ax()
    e = ax.Element("e14", "AXTextArea", "", "", 0, 0, 10, 10)
    monkeypatch.setattr(ax, "_last_tree", {"e14": e})
    monkeypatch.setattr(ax, "_last_tree_ts", time.time() - 10)  # 10 s old: still the same UI
    monkeypatch.setattr(ax, "_last_tree_app", "notes")
    walked = []
    monkeypatch.setattr(ax, "snapshot", lambda app=None, **kw: walked.append(app))
    assert "e14" in ax.find("textarea", app="Notes")  # role matches too, and no re-walk…
    assert walked == []
    assert ax.get("e14") is e  # …so the id the model already has still resolves
    ax.find("x", app="Safari")  # a different app is a different tree
    assert walked == ["Safari"]


def test_ax_find_rewalks_a_stale_tree(monkeypatch):
    ax = _ax()
    monkeypatch.setattr(ax, "_last_tree", {"e1": ax.Element("e1", "AXButton", "OK", "", 0, 0, 1, 1)})
    monkeypatch.setattr(ax, "_last_tree_ts", time.time() - 60)
    monkeypatch.setattr(ax, "_last_tree_app", "notes")
    walked = []
    monkeypatch.setattr(ax, "snapshot", lambda app=None, **kw: walked.append(app))
    assert "No element" in ax.find("zzz")
    assert walked == [None]


def test_pid_for_app_prefers_the_real_app_over_a_helper(monkeypatch):
    ax = _ax()

    class A:
        def __init__(self, name, pid, policy):
            self._n, self._p, self._pol = name, pid, policy

        def localizedName(self):
            return self._n

        def processIdentifier(self):
            return self._p

        def activationPolicy(self):
            return self._pol

    apps = [A("LinkedNotesUIService", 1, 1), A("Notes", 2, 0), A("NotesHelper", 3, 0)]

    class WS:
        @staticmethod
        def sharedWorkspace():
            return WS()

        def runningApplications(self):
            return apps

    monkeypatch.setattr(ax, "NSWorkspace", WS)  # the objc class itself can't be patched
    assert ax.pid_for_app("Notes") == 2
    assert ax.pid_for_app("notes helper".replace(" ", "")) == 3
    assert ax.pid_for_app("Safari") is None


def test_empty_text_inputs_are_listed():
    ax = _ax()
    assert "AXTextArea" in ax._ALWAYS_SHOW and "AXTextField" in ax._ALWAYS_SHOW


# ---- brain fallback ----------------------------------------------------------------------------


def test_chain_demotes_a_failed_brain_for_a_while(monkeypatch):
    import neo.providers as prov
    from neo.providers.base import ProviderError, Turn

    calls: list[str] = []

    class P:
        supports_vision = False
        supports_tools = True

        def __init__(self, name, ok):
            self.name, self.ok, self.model = name, ok, ""

        async def complete(self, messages, **kw):
            calls.append(self.name)
            if not self.ok:
                raise ProviderError(f"{self.name} down")
            return Turn(text=self.name)

        async def stream(self, messages, **kw):
            yield self.name

    providers = {"gemini": P("gemini", False), "groq": P("groq", True)}
    monkeypatch.setattr(prov, "make", lambda n: providers[n])
    monkeypatch.setattr(prov, "_demoted", {})

    async def run():
        a = await prov.Chain(["gemini", "groq"]).complete([])
        b = await prov.Chain(["gemini", "groq"]).complete([])  # a fresh chain, same request order
        c = await prov.Chain(["gemini", "groq"], sticky=False).complete([])
        return a, b, c

    a, b, c = asyncio.run(run())
    assert a.text == b.text == c.text == "groq"
    # 1st: gemini fails then groq; 2nd: groq straight away; 3rd (vision-style): original order kept
    assert calls == ["gemini", "groq", "groq", "gemini", "groq"]
    prov._demoted["gemini"] = time.time() - 1  # demotion expired
    asyncio.run(prov.Chain(["gemini", "groq"]).complete([]))
    assert calls[-2:] == ["gemini", "groq"]


def test_gemini_cooldowns_persist_and_slow_5xx_is_not_retried(monkeypatch, tmp_path):
    gem = pytest.importorskip("neo.providers.gemini")
    from google.genai import errors as gerrors

    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_API_KEYS", "k2")
    p = gem.GeminiProvider(model="gemini-3.8-flash")
    assert len(p._clients) == 2 and p.models[0] == "gemini-3.8-flash"

    attempts: list[tuple[str, int]] = []

    def err(code):
        return gerrors.APIError(code, {"error": {"message": "x", "status": "s"}})

    def fake(prov):
        async def fn(client, model, **kw):
            ki = prov._clients.index(client)
            attempts.append((model, ki))
            if model == "gemini-3.8-flash" and ki == 0:
                raise err(429)  # out of quota
            if model == "gemini-3.8-flash":
                raise err(503)  # overloaded
            return "ok"

        return fn

    monkeypatch.setattr(gem, "_SLOW_FAIL_S", -1.0)  # every 5xx counts as "slow" → no retry
    monkeypatch.setattr(gem, "_BACKOFF_S", (0.0,))

    async def quiet(text):
        pass

    monkeypatch.setattr(gem, "_note", quiet)
    assert asyncio.run(p._call(fake(p))) == "ok"
    # key 1 → 429, key 2 → one 503 (no retry), then the next model succeeds on its first key
    assert attempts == [("gemini-3.8-flash", 0), ("gemini-3.8-flash", 1), ("gemini-3.7-flash", 0)]
    saved = json.loads((tmp_path / gem._COOLDOWN_FILE).read_text())
    assert "gemini-3.8-flash|0" in saved and "gemini-3.8-flash|-1" in saved
    # a new provider (a restart) starts with the same knowledge and skips the dead pairs
    p2 = gem.GeminiProvider(model="gemini-3.8-flash")
    attempts.clear()
    assert asyncio.run(p2._call(fake(p2))) == "ok"
    assert attempts == [("gemini-3.7-flash", 0)]


# ---- Live silence window -----------------------------------------------------------------------


def test_live_quiet_since_counts_typed_text_and_model_turns():
    live = pytest.importorskip("neo.voice.live")

    class Spk:
        last_stop = 0.0

    v = live.LiveVoice.__new__(live.LiveVoice)
    v.spk = Spk()
    v._session_open = 100.0
    v._last_user_speech = 0.0
    v._last_turn_end = 0.0
    assert v._quiet_since() == 100.0
    v._last_turn_end = 130.0  # the model finished a turn / got a tool result: not silence
    assert v._quiet_since() == 130.0
    v._last_user_speech = 140.0  # typed text bumps this too (send_text)
    assert v._quiet_since() == 140.0


# ---- agent loop: stale observations -------------------------------------------------------------


def test_older_observations_are_stubbed_before_each_model_call():
    from neo.agent.loop import _supersede_observations
    from neo.providers.base import ImagePart, Message, ToolResult

    big = "e1 Window\n" * 200
    msgs = [
        Message.user("old history"),
        Message.tool([ToolResult("0", "ax_tree", big)]),  # history: untouched (start=2)
        Message.user("goal"),
        Message.tool([ToolResult("1", "ax_tree", big), ToolResult("2", "read_file", "short")]),
        Message.tool([ToolResult("3", "screenshot", "img", images=[ImagePart(b"x")])]),
        Message.tool([ToolResult("4", "ax_tree", big)]),
        Message.tool([ToolResult("5", "screenshot", "img2", images=[ImagePart(b"y")])]),
    ]
    _supersede_observations(msgs, start=2)
    assert msgs[1].tool_results[0].content == big  # outside this run
    assert "trimmed" in msgs[3].tool_results[0].content and len(msgs[3].tool_results[0].content) < 300
    assert msgs[3].tool_results[1].content == "short"  # only one read_file: kept
    assert msgs[4].tool_results[0].images == [] and "trimmed" in msgs[4].tool_results[0].content
    assert msgs[5].tool_results[0].content == big  # newest ax_tree verbatim
    assert msgs[6].tool_results[0].images  # newest screenshot keeps its image
