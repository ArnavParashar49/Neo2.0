"""NEO's MCP server, driven by the mcp package's own in-memory client, in both protocol eras:
"legacy" (the initialize handshake; elicitation is a server-to-client request) and "auto" (the
2026-07-28 per-request protocol; elicitation rides an InputRequiredResult and the client's retry).

Only fake brains and host tools run here: nothing reaches a real model, app, dialog or website.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

pytest.importorskip("mcp")

import mcp_types as types  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402

import neo2  # noqa: E402
from neo2.mcp import approvers, server  # noqa: E402
from neo2.mcp.server import TASK_TOOL, NeoMCP, ServerOptions  # noqa: E402
from neo2.testing import ScriptedBrain, ToolCall, Turn  # noqa: E402

PAY = Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "mallory", "amount": 1000})])
ERAS = ["legacy", "auto"]
TRUSTED = types.Implementation(name="trusted-app", version="1")
STRANGER = types.Implementation(name="some-app", version="1")


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    """No test here may start a process (the dialog's osascript above all)."""

    async def refuse(*a, **kw):
        raise AssertionError(f"a test tried to start a process: {a[:1]}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", refuse)


def _wire(calls: list):
    @neo2.host_tool(risk="destructive", confirm=lambda a: f"wire {a['amount']} to {a['to']}")
    async def wire_money(to: str, amount: float) -> str:
        """Send money."""
        calls.append((to, amount))
        return f"wired {amount} to {to}"

    return wire_money


def _neo(calls: list, *, turns=(PAY, "Sent."), audit: list | None = None, tools="none", **kw) -> NeoMCP:
    cfg = kw.pop("config", None) or neo2.AgentConfig(tools=tools, fast_paths=False)
    return NeoMCP(
        cfg,
        provider=ScriptedBrain(list(turns)),
        host_tools=[_wire(calls)],
        on_audit=audit.append if audit is not None else None,
        **kw,
    )


def _answer(action: str = "accept", content: dict | None = None, seen: list | None = None):
    async def cb(ctx, params):
        if seen is not None:
            seen.append(params)
        return types.ElicitResult(action=action, content=content)

    return cb


YES = {"approve": True}


def _status(r: types.CallToolResult) -> str:
    return r.structured_content["status"]


def _text(r: types.CallToolResult) -> str:
    return "".join(c.text for c in r.content if c.type == "text")


def _ids(audit: list) -> set[str]:
    """The action ids the audit saw (at least one: an empty set would prove nothing)."""
    ids = {a["action"] for a in audit if a.get("action")}
    assert ids and all(len(i) >= 8 for i in ids)
    return ids


# ---- tools/list --------------------------------------------------------------------------------


@pytest.mark.parametrize("era", ERAS)
async def test_list_tools_follows_expose(era):
    calls: list = []
    seen = {}
    for expose in ("task", "tools", "both"):
        async with _neo(calls, tools="web", expose=expose) as neo, Client(neo.server, mode=era) as c:
            seen[expose] = [t.name for t in (await c.list_tools()).tools]
    web = {"web_search", "web_fetch", "weather", "clock"}  # (in the catalog's order, whatever it is)
    assert seen["task"] == [TASK_TOOL]
    assert sorted(seen["tools"]) == sorted({*web, "wire_money"})
    assert seen["both"][0] == TASK_TOOL and seen["both"][1:] == seen["tools"]
    for names in seen.values():
        assert len(names) == len(set(names))
        assert not {"approve", "confirm", "deny", "cancel", "more_tools"} & set(names)
        assert not any("approv" in n or "confirm" in n for n in names)


async def test_annotations_come_from_the_tools_own_fields(tmp_path):
    calls: list = []
    cfg = neo2.AgentConfig(tools=("web", "memory"), memory=True, data_dir=tmp_path / "agent", fast_paths=False)
    async with _neo(calls, config=cfg, expose="both") as neo, Client(neo.server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    memory = tools["memory"].annotations
    assert memory.read_only_hint is False and memory.destructive_hint is False  # parallel_safe, but it writes
    assert memory.open_world_hint is False
    fetch = tools["web_fetch"].annotations
    assert fetch.read_only_hint is True and fetch.open_world_hint is True and fetch.destructive_hint is False
    assert tools["clock"].annotations.read_only_hint is True
    wire = tools["wire_money"].annotations
    assert wire.destructive_hint is True and wire.read_only_hint is False and wire.open_world_hint is True
    task = tools[TASK_TOOL].annotations
    assert task.destructive_hint is True and task.read_only_hint is False and task.open_world_hint is True
    assert tools[TASK_TOOL].input_schema["required"] == ["goal"]
    assert set(tools[TASK_TOOL].input_schema["properties"]) == {"goal", "context", "conversation"}


@pytest.mark.skipif(sys.platform != "darwin", reason="NEO's shell tool is macOS-only")
async def test_shell_is_listed_as_destructive():
    async with _neo([], tools=("shell", "read_file"), expose="tools") as neo, Client(neo.server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert tools["shell"].annotations.destructive_hint is True
    assert tools["shell"].annotations.read_only_hint is False
    assert tools["read_file"].annotations.read_only_hint is True
    assert tools["read_file"].annotations.destructive_hint is False


async def test_an_unknown_or_unlisted_tool_is_an_error_and_runs_nothing():
    calls: list = []
    async with _neo(calls, expose="task") as neo, Client(neo.server) as c:
        for name in ("wire_money", "approve", "confirm", "more_tools", "shell"):
            with pytest.raises(MCPError, match="Unknown tool"):
                await c.call_tool(name, {"to": "mallory", "amount": 1})
    async with _neo(calls, expose="tools") as neo, Client(neo.server) as c:
        with pytest.raises(MCPError, match="Unknown tool"):
            await c.call_tool(TASK_TOOL, {"goal": "pay mallory"})
    assert calls == []


# ---- --approve deny (the default) --------------------------------------------------------------


@pytest.mark.parametrize("era", ERAS)
async def test_a_gated_task_with_deny_runs_nothing(era):
    calls: list = []
    audit: list = []
    async with _neo(calls, audit=audit) as neo, Client(neo.server, mode=era) as c:
        assert neo.approve == "deny"
        r = await c.call_tool(TASK_TOOL, {"goal": "pay mallory 1000"})
    assert calls == []
    assert r.is_error is False
    assert r.structured_content == {"status": "denied", "summary": "wire 1000.0 to mallory"}
    assert "can't ask for it" in _text(r) and "Nothing was done" in _text(r)
    assert [a["status"] for a in audit] == ["staged", "cancelled: user cancelled"]
    assert audit[-1]["via"] == {"channel": "none"}
    assert not any(i in _text(r) or i in str(r.structured_content) for i in _ids(audit))


@pytest.mark.parametrize("era", ERAS)
async def test_a_gated_tool_call_with_deny_runs_nothing(era):
    calls: list = []
    audit: list = []
    async with _neo(calls, audit=audit, expose="tools") as neo, Client(neo.server, mode=era) as c:
        r = await c.call_tool("wire_money", {"to": "mallory", "amount": 5})
    assert calls == []
    assert r.structured_content == {"status": "denied", "summary": "wire 5.0 to mallory"}
    assert audit[-1]["via"] == {"channel": "none"} and audit[-1]["status"].startswith("cancelled")
    assert not any(i in _text(r) for i in _ids(audit))


async def test_a_task_with_nothing_gated_just_answers():
    calls: list = []
    async with _neo(calls, turns=("Paris.",)) as neo, Client(neo.server) as c:
        r = await c.call_tool(TASK_TOOL, {"goal": "capital of France?", "context": "geography quiz"})
    assert _status(r) == "done" and _text(r) == "Paris." and r.is_error is False
    assert "summary" not in r.structured_content


async def test_a_bad_argument_is_an_error_the_model_can_fix():
    async with _neo([]) as neo, Client(neo.server) as c:
        for args in ({}, {"goal": "  "}, {"goal": 3}, {"goal": "x", "conversation": "a b"}, {"goal": "x", "context": 1}):
            r = await c.call_tool(TASK_TOOL, args)
            assert r.is_error is True and _status(r) == "error"


# ---- --approve elicit --------------------------------------------------------------------------


@pytest.mark.parametrize("era", ERAS)
async def test_elicit_is_never_sent_to_a_client_off_the_allowlist(era):
    calls: list = []
    audit: list = []
    asked: list = []
    neo = _neo(calls, audit=audit, approve="elicit", trust_clients=["trusted-app"])
    async with neo, Client(neo.server, mode=era, client_info=STRANGER, elicitation_callback=_answer(content=YES, seen=asked)) as c:
        r = await c.call_tool(TASK_TOOL, {"goal": "pay mallory"})
    assert asked == [] and calls == []
    assert r.structured_content == {"status": "denied", "summary": "wire 1000.0 to mallory"}
    assert audit[-1]["status"].startswith("cancelled")
    assert audit[-1]["via"]["channel"] == "mcp-elicit" and "trust-client" in audit[-1]["via"]["note"]


@pytest.mark.parametrize("era", ERAS)
async def test_elicit_accept_with_approve_true_runs_it_once(era):
    calls: list = []
    audit: list = []
    asked: list = []
    neo = _neo(calls, audit=audit, approve="elicit", trust_clients=["trusted-app"])
    async with neo, Client(neo.server, mode=era, client_info=TRUSTED, elicitation_callback=_answer(content=YES, seen=asked)) as c:
        r = await c.call_tool(TASK_TOOL, {"goal": "pay mallory"})
    assert calls == [("mallory", 1000.0)]
    assert _status(r) == "done" and _text(r) == "Sent." and r.is_error is False
    assert len(asked) == 1
    msg = asked[0].message
    assert "wire 1000.0 to mallory" in msg and '"amount": 1000.0' in msg  # the summary and the exact detail
    assert "the task itself (another model wrote it)" in msg  # source="model": says so
    assert not any(i in msg for i in _ids(audit))
    assert asked[0].requested_schema["properties"]["approve"]["type"] == "boolean"
    confirmed = [a for a in audit if a["status"] == "confirmed"]
    assert len(confirmed) == 1
    assert confirmed[0]["via"] == {
        "channel": "mcp-elicit",
        "approver": "client:trusted-app",
        "note": "elicitation: trusts the client",
    }


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize(
    ("action", "content"),
    [
        ("decline", None),
        ("cancel", None),
        ("accept", {"approve": False}),
        ("accept", {"approve": "true"}),
        ("accept", {"approve": 1}),
        ("accept", {}),
        ("accept", None),
    ],
)
async def test_elicit_anything_but_an_accepted_true_is_denied(era, action, content):
    calls: list = []
    audit: list = []
    neo = _neo(calls, audit=audit, approve="elicit", trust_clients=["trusted-app"])
    async with neo, Client(neo.server, mode=era, client_info=TRUSTED, elicitation_callback=_answer(action, content)) as c:
        r = await c.call_tool(TASK_TOOL, {"goal": "pay mallory"})
    assert calls == []
    assert r.structured_content == {"status": "denied", "summary": "wire 1000.0 to mallory"}
    assert "was not given" in _text(r)
    assert audit[-1]["status"].startswith("cancelled") and audit[-1]["via"]["note"] == "not approved"


@pytest.mark.parametrize("era", ERAS)
async def test_elicit_a_trusted_client_that_cant_ask_its_user_is_denied(era):
    calls: list = []
    neo = _neo(calls, approve="elicit", trust_clients=["trusted-app"])
    async with neo, Client(neo.server, mode=era, client_info=TRUSTED) as c:  # no elicitation callback
        r = await c.call_tool(TASK_TOOL, {"goal": "pay mallory"})
    assert calls == [] and _status(r) == "denied"


@pytest.mark.parametrize("era", ERAS)
async def test_elicit_approves_a_single_tool_call(era):
    calls: list = []
    neo = _neo(calls, approve="elicit", trust_clients=["trusted-app"], expose="tools")
    async with neo, Client(neo.server, mode=era, client_info=TRUSTED, elicitation_callback=_answer(content=YES)) as c:
        r = await c.call_tool("wire_money", {"to": "bob", "amount": 3})
    assert calls == [("bob", 3.0)]
    assert _status(r) == "done" and "wired 3.0 to bob" in _text(r)
    assert _text(r).startswith('<untrusted source="wire_money">')  # a host tool's output stays fenced


@pytest.mark.parametrize("era", ERAS)
async def test_elicit_asks_again_when_the_approved_run_stages_again(era):
    calls: list = []
    asked: list = []
    pay2 = Turn(tool_calls=[ToolCall("2", "wire_money", {"to": "bob", "amount": 7})])
    neo = _neo(calls, turns=(PAY, pay2, "Both sent."), approve="elicit", trust_clients=["trusted-app"])
    async with neo, Client(neo.server, mode=era, client_info=TRUSTED, elicitation_callback=_answer(content=YES, seen=asked)) as c:
        r = await c.call_tool(TASK_TOOL, {"goal": "pay mallory and bob"})
    assert calls == [("mallory", 1000.0), ("bob", 7.0)]
    assert len(asked) == 2 and "wire 7.0 to bob" in asked[1].message
    assert _status(r) == "done" and _text(r) == "Both sent."


async def test_elicit_a_forged_or_reused_request_state_runs_nothing():
    calls: list = []
    neo = _neo(calls, approve="elicit", trust_clients=["trusted-app"])
    async with neo, Client(neo.server, client_info=TRUSTED, elicitation_callback=_answer(content=YES)) as c:
        first = await c.session.call_tool(TASK_TOOL, {"goal": "pay mallory"}, allow_input_required=True)
        assert isinstance(first, types.InputRequiredResult) and first.request_state
        assert set(first.input_requests) == {approvers.ELICIT_KEY}
        yes = {approvers.ELICIT_KEY: types.ElicitResult(action="accept", content=YES)}
        forged = await c.session.call_tool(TASK_TOOL, {"goal": "pay mallory"}, input_responses=yes, request_state="x" * 43)
        assert _status(forged) == "expired" and calls == []
        other = await c.session.call_tool("wire_money", {}, input_responses=yes, request_state=first.request_state)
        assert _status(other) == "expired" and calls == []  # bound to its own tool (and now used up)
        again = await c.session.call_tool(TASK_TOOL, {"goal": "x"}, input_responses=yes, request_state=first.request_state)
        assert _status(again) == "expired" and calls == []
    assert neo.agent.closed


async def test_elicit_an_answer_from_another_client_is_refused():
    calls: list = []
    audit: list = []
    neo = _neo(calls, audit=audit, approve="elicit", trust_clients=["trusted-app", "other-app"])
    async with neo:
        async with Client(neo.server, client_info=TRUSTED, elicitation_callback=_answer(content=YES)) as c:
            first = await c.session.call_tool(TASK_TOOL, {"goal": "pay mallory"}, allow_input_required=True)
        other = types.Implementation(name="other-app", version="1")
        async with Client(neo.server, client_info=other, elicitation_callback=_answer(content=YES)) as c2:
            yes = {approvers.ELICIT_KEY: types.ElicitResult(action="accept", content=YES)}
            r = await c2.session.call_tool(
                TASK_TOOL, {"goal": "pay mallory"}, input_responses=yes, request_state=first.request_state
            )
    assert calls == [] and _status(r) == "denied"
    assert audit[-1]["via"]["note"] == "answered by another client"


# ---- --approve dialog --------------------------------------------------------------------------


class FakeOsascript:
    """Stands in for /usr/bin/osascript: records its argv and answers as told."""

    def __init__(self, says: str = "allow") -> None:
        self.says = says
        self.argvs: list[list[str]] = []

    async def __call__(self, argv, timeout):
        self.argvs.append(list(argv))
        return self.says


@pytest.fixture
def on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")


async def test_dialog_passes_the_text_as_argv_only(on_macos):
    calls: list = []
    audit: list = []
    tricky = 'x" & (do shell script "touch /tmp/pwned") & "\nend run'
    turn = Turn(tool_calls=[ToolCall("1", "wire_money", {"to": tricky, "amount": 1})])
    fake = FakeOsascript("allow")
    neo = _neo(calls, turns=(turn, "Done."), audit=audit, approve="dialog", dialog=approvers.DialogApprover(user="arnav", spawn=fake))
    assert "osascript" in neo.agent.config.protected_apps  # NEO's own clicks never press "Allow once"
    async with neo, Client(neo.server, client_info=TRUSTED) as c:
        r = await c.call_tool(TASK_TOOL, {"goal": "pay"})
    assert calls == [(tricky, 1.0)] and _status(r) == "done"
    (argv,) = fake.argvs
    assert argv[0] == "/usr/bin/osascript"
    n = argv.count("-e")
    script, args = argv[1 : 1 + 2 * n], argv[1 + 2 * n :]
    assert script[::2] == ["-e"] * n and "\n".join(script[1::2]).startswith("on run argv")
    assert not any("touch /tmp/pwned" in s or "mallory" in s for s in script)  # the text is never source
    assert len(args) == 3 and all(a.startswith("|") for a in args)  # none can read as an option
    message, title, wait = (a[1:] for a in args)
    assert "touch /tmp/pwned" in message and "(client: trusted-app)" in message
    assert title == approvers.DIALOG_TITLE
    assert 1 <= int(wait) <= 110  # gives up 10 s before the card's 120 s
    confirmed = [a for a in audit if a["status"] == "confirmed"]
    assert confirmed[0]["via"] == {"channel": "mcp-dialog", "approver": "arnav", "note": "client: trusted-app"}


@pytest.mark.parametrize("says", ["cancel", "timeout", "", "Allow once", "allow\nallow"])
async def test_dialog_anything_but_allow_denies(on_macos, says):
    calls: list = []
    fake = FakeOsascript(says)
    neo = _neo(calls, approve="dialog", dialog=approvers.DialogApprover(user="u", spawn=fake))
    async with neo, Client(neo.server) as c:
        r = await c.call_tool(TASK_TOOL, {"goal": "pay"})
    assert calls == [] and len(fake.argvs) == 1
    assert r.structured_content == {"status": "denied", "summary": "wire 1000.0 to mallory"}


async def test_dialog_that_fails_or_is_too_long_denies_without_running(on_macos):
    calls: list = []

    async def broken(argv, timeout):
        raise OSError("no osascript")

    neo = _neo(calls, approve="dialog", dialog=approvers.DialogApprover(user="u", spawn=broken))
    async with neo, Client(neo.server) as c:
        assert _status(await c.call_tool(TASK_TOOL, {"goal": "pay"})) == "denied"
    long = Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "m" * 5000, "amount": 1})])
    fake = FakeOsascript("allow")
    neo = _neo(calls, turns=(long, "x"), approve="dialog", dialog=approvers.DialogApprover(user="u", spawn=fake))
    async with neo, Client(neo.server) as c:
        assert _status(await c.call_tool(TASK_TOOL, {"goal": "pay"})) == "denied"
    assert calls == [] and fake.argvs == []  # never half-shown


async def test_dialog_spawns_osascript_with_argv_and_no_shell(on_macos, monkeypatch):
    """The real spawn path, with the process faked: argv goes to exec as given (no shell)."""
    seen: dict = {}

    class Proc:
        returncode = None

        async def communicate(self):
            self.returncode = 0
            return b"allow\n", b""

    async def exec_(*argv, **kw):
        seen["argv"], seen["kw"] = list(argv), kw
        return Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_)
    argv = approvers.dialog_argv('say "hi"; rm -rf ~', "T", 30)
    assert await approvers._spawn(argv, 5) == "allow"
    assert seen["argv"] == argv and "shell" not in seen["kw"]
    assert seen["kw"]["env"]["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert set(seen["kw"]["env"]) <= {"PATH", "HOME", "USER", "LOGNAME", "LANG", "TMPDIR"}


def test_dialog_needs_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(neo2.UnsupportedPlatform):
        NeoMCP(neo2.AgentConfig(tools="none"), provider=ScriptedBrain([]), approve="dialog")


def test_approval_text_shows_hidden_characters():
    req = neo2.ApprovalRequest(
        action_id="abc",
        tool="t",
        summary="send ‮reversed‬",
        detail="line1\x00\x1b[2Jline2\r\nx",
        params={},
        after_reading=("caller", "web_fetch"),
        conversation="c",
        request_id="r",
        staged_at=0.0,
        expires_at=120.0,
    )
    t = approvers.approval_text(req, client="c‮1")
    assert "‮" not in t and "<U+202E>" in t and "\x00" not in t and "\x1b" not in t
    assert "<0x00>" in t and "<0x1B>" in t and "<0x0D>" in t and "abc" not in t
    assert "web_fetch" in t


# ---- no brain, options -------------------------------------------------------------------------


async def test_without_a_brain_neo_task_says_so_and_tools_still_work():
    calls: list = []
    neo = NeoMCP(neo2.AgentConfig(tools="web", fast_paths=False), expose="both", host_tools=[_wire(calls)])
    assert neo.has_brain is False
    async with neo, Client(neo.server) as c:
        names = [t.name for t in (await c.list_tools()).tools]
        assert TASK_TOOL in names and "clock" in names
        r = await c.call_tool(TASK_TOOL, {"goal": "what time is it"})
        assert r.is_error is True and "--env" in _text(r)
        r = await c.call_tool("wire_money", {"to": "x", "amount": 1})
        assert _status(r) == "denied"
    assert calls == []


def test_options_and_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("NEO_MCP_TOOLS", "web,memory")
    monkeypatch.setenv("NEO_TOOLS", "mac")  # NEO.app's own settings are never the server's
    cfg = server.agent_config(ServerOptions())
    assert cfg.tools == "web" and cfg.gemini_api_keys == () and cfg.name == "mcp"
    cfg = server.agent_config(ServerOptions(env=True))
    assert cfg.tools == ("web", "memory") and len(cfg.gemini_api_keys) == 1
    cfg = server.agent_config(ServerOptions(env=True, tools="none"))
    assert cfg.tools == "none"  # the command line wins
    with pytest.raises(ValueError, match="trust-client"):
        NeoMCP(neo2.AgentConfig(tools="none"), provider=ScriptedBrain([]), trust_clients=["x"])
    with pytest.raises(ValueError):
        NeoMCP(neo2.AgentConfig(tools="none"), provider=ScriptedBrain([]), approve="yes")  # type: ignore[arg-type]


def test_the_command_line():
    from neo2.mcp import _parser, _tools_arg

    a = _parser().parse_args([])
    assert (a.tools, a.expose, a.approve, a.trust_client, a.env) == (None, "task", "deny", [], False)
    a = _parser().parse_args(
        ["--tools", "web,browser_*", "--expose", "both", "--approve", "elicit", "--trust-client", "A", "--trust-client", "B"]
    )
    assert _tools_arg(a.tools) == ("web", "browser_*") and a.trust_client == ["A", "B"]
    assert _tools_arg("none") == "none" and _tools_arg(" , ") == "none"
    with pytest.raises(SystemExit):
        _parser().parse_args(["--approve", "yes"])


async def test_old_conversations_are_dropped(monkeypatch):
    monkeypatch.setattr(server, "MAX_CONVERSATIONS", 2)
    async with _neo([], turns=("a", "b", "c", "d")) as neo, Client(neo.server) as c:
        for conv in ("one", "two", "three", "four"):
            r = await c.call_tool(TASK_TOOL, {"goal": "hi", "conversation": conv})
            assert _status(r) == "done"
        kept = set(neo.agent._rt.scopes)
    assert {"task:three", "task:four"} <= kept and not {"task:one", "task:two"} & kept
