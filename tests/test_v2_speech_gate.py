"""Voice: NEO speaks when asked something, never to narrate a command; commands the fast paths
fully cover run locally the moment the sentence ends (the command lane)."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS

import pytest

from neo.agent.registry import Registry, ToolOutput
from neo.events import EventBus


class FakeSpk:
    def __init__(self):
        self.played, self.playing, self.last_stop = [], threading.Event(), 0.0

    def play_pcm16(self, data):
        self.played.append(data)

    def interrupt(self):
        pass

    async def wait_done(self):
        return None


class FakeLive:
    def __init__(self):
        self.responses = []
        self.texts = []

    async def send_tool_response(self, function_responses):
        self.responses.extend(function_responses)

    async def send_client_content(self, turns):
        self.texts.append(turns)


def sc(**kw):
    base = dict(
        interrupted=False, input_transcription=None, interim_input_transcription=None,
        output_transcription=None, model_turn=None, turn_complete=False,
    )
    base.update(kw)
    return NS(server_content=NS(**base), tool_call=None, tool_call_cancellation=None, go_away=None)


def heard(text, finished=True):
    return sc(input_transcription=NS(text=text, finished=finished))


def audio(n=1):
    return sc(model_turn=NS(parts=[NS(inline_data=NS(data=b"\x00\x00" * 2400)) for _ in range(n)]))


def said(text):
    return sc(output_transcription=NS(text=text))


def call(name, args, cid="c1"):
    return NS(server_content=None, tool_call=NS(function_calls=[NS(id=cid, name=name, args=args)]),
              tool_call_cancellation=None, go_away=None)


DONE = sc(turn_complete=True)


@pytest.fixture
def voice(monkeypatch):
    from neo import config, events
    from neo.tools import load_all
    from neo.voice import live as live_mod

    load_all()
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", EventBus())
    ran = []

    async def fake_invoke(self, name, args, ctx):
        ran.append((name, dict(args)))
        return ToolOutput(f"did {name}", ok=True)

    monkeypatch.setattr(Registry, "invoke", fake_invoke)
    lane = []

    class Session:
        memory_context = ""

        async def command(self, text, steps):
            from neo.agent.session import Reply

            lane.append((text, [t for t, _ in steps]))
            return Reply("did it", "quick", silent=True)

        async def handle(self, text, **kw):
            from neo.agent.session import Reply

            return Reply("agent did it", "agent", silent=True)

    v = live_mod.LiveVoice(lambda t: None, Session())
    v.spk = FakeSpk()
    v._live = FakeLive()
    v.ran, v.lane = ran, lane
    return v


async def feed(v, *msgs, settle=0.05):
    ub, mb = [], []
    for m in msgs:
        await v._on_message(m, ub, mb, v._live)
        await asyncio.sleep(0)
    await asyncio.sleep(settle)
    return ub, mb


def test_a_command_is_done_silently_even_if_the_model_narrates(voice, monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_LIVE_COMMAND_LANE", "false")  # the model path on its own
    monkeypatch.setattr(config, "_settings", None)
    asyncio.run(feed(voice, heard("open notes"), said("Sure, opening Notes"), audio(), call("open_app", {"target": "Notes"}), audio(), DONE, settle=0.2))
    assert voice.spk.played == []  # "Sure, opening Notes" never reached the speaker
    assert ("open_app", {"target": "Notes"}) in voice.ran
    assert not [line for line in voice._dialog if line[1] == "NEO"]  # and isn't remembered as said


def test_a_question_is_answered_out_loud(voice):
    asyncio.run(feed(voice, heard("what's the capital of France?"), said("Paris."), audio(), DONE))
    assert len(voice.spk.played) == 1 and voice._dialog[-1][1:] == ("NEO", "Paris.")


def test_a_plain_reply_to_small_talk_still_plays(voice):
    asyncio.run(feed(voice, heard("thanks neo"), said("Any time."), audio(), DONE))
    assert len(voice.spk.played) == 1  # held a moment, then played: no tool call came


def test_the_command_lane_runs_the_sentence_itself_and_skips_the_models_calls(voice):
    async def run():
        await feed(voice, heard("open notes and type hello world"))
        await feed(voice, audio(), call("open_app", {"target": "Notes"}, "m1"), call("type_text", {"text": "Hello world."}, "m2"), audio(), DONE)

    asyncio.run(run())
    assert voice.lane == [("open notes and type hello world", ["open_app", "type_text"])]
    assert voice.ran == []  # the model's own calls never ran: nothing is done twice
    skipped = [r for r in voice._live.responses if "already did" in r.response["result"]]
    assert {r.id for r in skipped} == {"m1", "m2"} and voice.spk.played == []


def test_questions_and_references_go_to_the_model(voice):
    asyncio.run(feed(voice, heard("what time is it"), DONE))
    asyncio.run(feed(voice, heard("search for it on amazon"), DONE))
    asyncio.run(feed(voice, heard("check my email"), DONE))
    assert voice.lane == []


def test_search_on_a_site_is_a_command_not_a_question(voice):
    asyncio.run(feed(voice, heard("search cat videos on youtube"), said("Searching YouTube"), audio(), DONE))
    assert voice.lane and voice.lane[0][1] == ["search_site"] and voice.spk.played == []


def test_a_failed_step_is_spoken_but_a_finished_step_is_not(voice, monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_LIVE_COMMAND_LANE", "false")
    monkeypatch.setattr(config, "_settings", None)

    async def failing(self, name, args, ctx):
        return ToolOutput("Error: couldn't open Notez", ok=False)

    async def run():
        await feed(voice, heard("open notez"), call("open_app", {"target": "Notez"}), DONE, settle=0.2)
        await feed(voice, said("I couldn't find Notez."), audio(), DONE)  # the model's reply to the failure

    monkeypatch.setattr(Registry, "invoke", failing)
    asyncio.run(run())
    assert len(voice.spk.played) == 1  # the failure is heard


def test_multi_step_progress_is_never_narrated(voice, monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_LIVE_COMMAND_LANE", "false")
    monkeypatch.setattr(config, "_settings", None)

    async def run():
        await feed(voice, heard("type github.com and press enter"), call("type_text", {"text": "github.com"}), DONE, settle=0.2)
        # the model's turn after the first step: it narrates and makes the next call
        await feed(voice, said("Typed it, now pressing enter"), audio(), call("hotkey", {"keys": "return"}, "c2"), DONE, settle=0.2)
        await feed(voice, said("All done!"), audio(), DONE)

    asyncio.run(run())
    assert [n for n, _ in voice.ran] == ["type_text", "hotkey"] and voice.spk.played == []


# ---- review of 66b1288 ----------------------------------------------------------------------------


def test_a_command_said_twice_runs_twice(voice):
    async def run():
        await feed(voice, heard("scroll down"), audio(), call("scroll", {"direction": "down"}, "m1"), DONE)
        await feed(voice, heard("scroll down"), audio(), call("scroll", {"direction": "down"}, "m2"), DONE)

    asyncio.run(run())
    assert [t for t, _ in voice.lane] == ["scroll down", "scroll down"] and voice.ran == []


def test_a_payload_that_swallowed_an_instruction_goes_to_the_model(voice):
    asyncio.run(feed(voice, heard("type see you soon and send"), call("type_text", {"text": "see you soon"}, "m1"),
                     call("hotkey", {"keys": "return"}, "m2"), DONE, settle=0.2))
    assert voice.lane == [] and [n for n, _ in voice.ran] == ["type_text", "hotkey"]


def test_answers_are_spoken_however_the_question_is_phrased(voice):
    for said_text in ("weather in delhi tomorrow", "kal delhi ka mausam kaisa rahega", "time in tokyo"):
        voice.spk.played.clear()
        asyncio.run(feed(voice, heard(said_text), said("Let me check"), audio(), call("weather", {"place": "Delhi"}),
                         settle=0.2))
        # the blocking result is back: the model answers in the same turn
        asyncio.run(feed(voice, said("It'll be 31 degrees and sunny."), audio(2), DONE))
        # The answer is heard (a sentence recognised as a question also keeps its "let me check").
        assert len(voice.spk.played) >= 2, said_text


def test_remembering_something_stays_quiet(voice):
    asyncio.run(feed(voice, heard("remember that my car is on level 3"), call("memory", {"action": "remember", "text": "car on level 3"}), settle=0.2))
    asyncio.run(feed(voice, said("Got it, I'll remember that."), audio(), DONE))
    assert voice.spk.played == []


def test_a_model_call_before_the_transcript_keeps_the_lane_out(voice):
    asyncio.run(feed(voice, call("open_app", {"target": "Notes"}, "m1"), heard("open notes"), DONE, settle=0.2))
    assert voice.lane == [] and [n for n, _ in voice.ran] == ["open_app"]  # once, by the model


def test_the_lane_is_tracked_so_stop_can_cancel_it(voice):
    started = asyncio.Event()

    async def slow(text, steps):
        started.set()
        await asyncio.sleep(10)

    voice._agent_session.command = slow

    async def run():
        await feed(voice, heard("open notes"))
        await started.wait()
        assert any(k.startswith("lane:") for k in voice._tool_tasks)
        await voice.interrupt()
        await asyncio.sleep(0.05)
        assert not any(k.startswith("lane:") for k in voice._tool_tasks)

    asyncio.run(run())


def test_a_typed_question_after_a_quiet_step_is_answered(voice, monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_LIVE_COMMAND_LANE", "false")
    monkeypatch.setattr(config, "_settings", None)

    async def run():
        await feed(voice, heard("type github.com and press enter"), call("type_text", {"text": "github.com"}), DONE, settle=0.2)
        await feed(voice, said("typed it"), audio(), DONE)  # the sequencing reply: muted
        await voice.send_text("what's the capital of France?")
        await feed(voice, said("Paris."), audio(), DONE)

    asyncio.run(run())
    assert len(voice.spk.played) == 1  # only "Paris."


def test_an_approved_action_via_the_models_call_stays_quiet(voice):
    from neo.agent import confirm
    from neo.agent.session import Reply

    async def answer(yes, **kw):
        confirm.cancel("t")
        return Reply("Clicked 'Send'", "confirm", silent=True)

    voice._agent_session.answer = answer

    async def run():
        confirm.stage("click_text", {"label": "Send"}, "click “Send” in Mail")
        p = confirm.peek()
        voice._note_question("NEEDS_CONFIRM: click “Send” in Mail — Should I go ahead?")
        voice._voiced_action = (p.action_id, "Should I click Send in Mail?", time.time())
        await asyncio.sleep(0.01)
        await feed(voice, heard("yes"), call("click_text", {"label": "Send"}, "m9"), settle=0.2)
        await feed(voice, said("Done, I clicked Send."), audio(), DONE)

    import time

    asyncio.run(run())
    assert voice.spk.played == []
