"""Embedded, NEO prints nothing: its lines go to the `neo` logger. NEO.app still shows them on stdout."""

from __future__ import annotations

import asyncio
import logging

import pytest
from neo_lib import make_runtime, scripted

from neo2.core import runtime
from neo2.core.agent import confirm
from neo2.core.agent.registry import RegisteredTool
from neo2.core.agent.session import Session
from neo2.core.providers import BrainPool, Chain
from neo2.core.providers.base import ProviderError, ToolCall, Turn

_OBJ = {"type": "object", "properties": {}}


class _Down:
    name, model, supports_vision, supports_tools = "down", "", False, True

    async def complete(self, messages, **kw):
        raise ProviderError("down for maintenance")

    async def stream(self, messages, **kw):
        raise ProviderError("down for maintenance")
        yield ""


@pytest.fixture
def neo_stdout():
    """NEO.app's logging setup, removed again afterwards (it's process-wide). Installed by the
    test itself: capsys swaps its stream between the setup and call phases."""
    import neo2.core

    lg = logging.getLogger("neo")
    before, level = list(lg.handlers), lg.level
    yield neo2.core.log_to_stdout
    lg.handlers[:] = before
    lg.setLevel(level)


def _agent(tmp_path):
    ran: list = []
    rt = make_runtime(tmp_path / "q", "q")

    def pay(a, c):
        gate = confirm.gated("zz_pay", a, "pay the invoice")
        if isinstance(gate, str):
            return gate
        ran.append(a)
        return "paid"

    def broken(a, c):
        return "Error: the tool broke"

    rt.registry.register(RegisteredTool("zz_pay", "pay", _OBJ, pay, risk="destructive"))
    rt.registry.register(RegisteredTool("zz_broken", "breaks", _OBJ, broken, fast_path=[(r"^\s*break it\s*$", {})]))
    return rt, ran


def test_an_agent_turn_and_its_approval_print_nothing(tmp_path, capsys, caplog):
    rt, ran = _agent(tmp_path)
    brain = scripted(rt, [Turn(tool_calls=[ToolCall("1", "zz_pay", {})]), "Paid.", "Handled."])
    rt.brains = BrainPool(
        overrides={p: [_Down(), brain] for p in ("agent", "fast", "vision", "light", "offline")}
    )  # the first brain fails every time: NEO says so, to its logger

    async def go():
        with runtime.bind(rt.scope()):
            sess = Session()
            r = await sess.handle("pay the invoice")
            aid = confirm.peek().action_id
            r2 = await sess.answer(True, action_id=aid)
            r3 = await sess.handle("break it")  # a fast path fails: handed to the agent, logged
            return r, r2, r3

    with caplog.at_level(logging.INFO, logger="neo"):
        r, r2, r3 = asyncio.run(go())
    assert (r.route, r2.text, r3.text, ran) == ("confirm", "Paid.", "Handled.", [{}])
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    names = {rec.name for rec in caplog.records}
    assert {"neo.providers", "neo.agent.session"} <= names  # logger names: NEO's kept theirs  neo2-codemod: keep
    texts = [rec.getMessage() for rec in caplog.records]
    assert any(t.startswith("[brain] down failed (ProviderError: down for maintenance)") for t in texts)
    assert "  ~ zz_broken failed; handing to the agent" in texts


def test_neo_apps_lines_still_reach_stdout_with_the_same_text(neo_stdout, capsys, monkeypatch):
    from neo2.core import providers as prov

    ok = type("Ok", (), {"name": "ok", "model": "", "supports_vision": False, "supports_tools": True})()

    async def complete(messages, **kw):
        return Turn(text="fine")

    ok.complete = complete
    monkeypatch.setattr(prov, "_demoted", {})
    neo_stdout()
    asyncio.run(Chain([_Down(), ok]).complete([]))
    out = capsys.readouterr().out
    assert out == "[brain] down failed (ProviderError: down for maintenance); trying next\n"


def test_neo_apps_setup_leaves_other_libraries_alone(neo_stdout, capsys):
    neo_stdout()
    neo_stdout()  # twice (main() and app.run()): still one handler
    logging.getLogger("httpx").info("HTTP Request: GET https://example.com/?token=abc")
    logging.getLogger("neo.x").info("[x] a line")
    out = capsys.readouterr().out
    assert out == "[x] a line\n"  # an httpx request line (with its URL) never lands in core.log


def test_importing_neo_configures_nothing_but_a_null_handler():
    import neo2.core

    lg = logging.getLogger("neo")
    assert any(isinstance(h, logging.NullHandler) for h in lg.handlers)
    assert not logging.getLogger().handlers or all(
        type(h).__module__.startswith("_pytest") for h in logging.getLogger().handlers
    )
    assert callable(neo2.core.log_to_stdout)


@pytest.mark.parametrize("module", ["generate", "finetune", "hf_data", "dataset"])
def test_the_training_commands_still_print_neos_lines(module):
    """`python -m neo2.core.reflex.<module>` used to print the providers' status lines ("[gemini] …
    quota exhausted; trying next key"); now they are log records, which a command line shows
    only with NEO's stdout logging set up — first thing in its __main__ block."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parents[1] / "neo2" / "core" / "reflex" / f"{module}.py").read_text())
    mains = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
    assert mains, module
    first = next(st for st in mains[0].body if not isinstance(st, ast.Import | ast.ImportFrom))
    assert isinstance(first, ast.Expr) and ast.unparse(first.value).endswith("log_to_stdout()"), ast.unparse(first)


@pytest.fixture
def silent_safari(monkeypatch):
    """`close` with Safari running and its tab script failing: every Mac call faked."""
    pytest.importorskip("AppKit")
    from types import SimpleNamespace

    from neo2.core.decide import close, inventory, media

    safari = SimpleNamespace(name="Safari", bundle_id="com.apple.Safari", handles=("https",), running=True)
    monkeypatch.setattr(close, "_running_apps", lambda: [safari])
    monkeypatch.setattr(close, "_is_browser", lambda app: True)
    monkeypatch.setattr(close, "_tab_codes", lambda app: close._TabCodes("cwin", "bTab", "pURL", "pnam", "coreclos"))
    monkeypatch.setattr(inventory, "named_in", lambda what, apps: None)

    async def gate(app):
        return None

    async def osa(script, args, timeout):
        return 1, "", "execution error: Safari got an error (-1712)"

    monkeypatch.setattr(media, "_control_gate", gate)
    monkeypatch.setattr(media, "_osascript", osa)


def test_an_agents_close_prints_nothing(tmp_path, capsys, caplog, silent_safari):
    from neo2.core.agent.registry import ToolContext, registry

    rt = make_runtime(tmp_path / "c", "c", ["close"])
    with caplog.at_level(logging.INFO, logger="neo"), runtime.bind(rt.scope()):
        out = asyncio.run(registry().invoke("close", {"target": "youtube"}, ToolContext()))
    assert out.text == "Error: nothing called 'youtube' is open"
    assert capsys.readouterr().out == ""
    assert any(r.getMessage().startswith("[close] Safari didn't answer (execution error") for r in caplog.records)


def test_neo_apps_close_line_still_reaches_stdout(neo_stdout, capsys, silent_safari):
    from neo2.core.decide import close

    neo_stdout()
    asyncio.run(close.close_target("youtube"))
    assert capsys.readouterr().out == "[close] Safari didn't answer (execution error: Safari got an error (-1712))\n"
