"""Voice: NEO speaks when asked something, never to narrate a command; commands the fast paths
fully cover run locally the moment the sentence ends (the command lane)."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS

import numpy as np
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


_TONE = (np.sin(np.arange(2400) * 2 * np.pi * 220 / 24000) * 6000).astype(np.int16).tobytes()  # 0.1 s of voice


def audio(n=1):
    """n chunks of the model speaking (0.1 s each)."""
    return sc(model_turn=NS(parts=[NS(inline_data=NS(data=_TONE)) for _ in range(n)]))


def silence(n=1):
    """n chunks of the near-silent audio the model sends when told to say nothing."""
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


def test_the_same_open_again_is_a_retry_but_the_same_keystroke_is_not(voice, monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_LIVE_COMMAND_LANE", "false")
    monkeypatch.setattr(config, "_settings", None)
    url = {"url": "web.whatsapp.com"}

    async def run():
        await feed(voice, heard("open whatsapp on google"))
        for i in range(5):  # the model re-issued the same call five times
            await feed(voice, call("open_url", dict(url), f"o{i}"))
        await feed(voice, call("type_text", {"text": "hi"}, "t1"), call("type_text", {"text": "hi"}, "t2"), DONE, settle=0.2)

    asyncio.run(run())
    assert voice.ran.count(("open_url", url)) == 1  # one tab, not five
    assert voice.ran.count(("type_text", {"text": "hi"})) == 2  # typing twice is what was asked
    retries = [r for r in voice._live.responses if "same call as" in r.response["result"]]
    assert len(retries) == 4


# ---- after a command, NEO goes straight back to listening ---------------------------------------


def _states(monkeypatch):
    from neo.events import bus

    seen = []
    orig = bus().publish

    async def pub(kind, **data):
        if kind == "state":
            seen.append(data["state"])
        return await orig(kind, **data)

    monkeypatch.setattr(bus(), "publish", pub)
    return seen


def test_a_silent_reply_to_a_command_never_shows_speaking(voice, monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_LIVE_COMMAND_LANE", "false")
    monkeypatch.setattr(config, "_settings", None)
    seen = _states(monkeypatch)
    asyncio.run(feed(voice, heard("Can you open YouTube?"), silence(20), call("open_url", {"url": "www.youtube.com"}), silence(30), DONE, settle=0.3))
    assert ("open_url", {"url": "www.youtube.com"}) in voice.ran
    assert "speaking" not in seen and voice.spk.played == [] and not voice._speaking
    assert seen[-1] == "listening"


def test_a_polite_or_addressed_command_runs_on_the_fast_lane(voice):
    asyncio.run(feed(voice, heard("Neo, can you open YouTube?"), call("open_url", {"url": "youtube.com"}, "m1"), DONE))
    asyncio.run(feed(voice, heard("Hey Neo, open Notes, please."), DONE))
    assert [t for t, _ in voice.lane] == ["Neo, can you open YouTube?", "Hey Neo, open Notes, please."]
    assert voice.ran == []  # the model's own call was skipped: NEO already did it
    assert voice.spk.played == []


def test_a_real_answer_after_leading_silence_is_still_spoken(voice):
    asyncio.run(feed(voice, heard("what's the capital of France?"), silence(10), said("Paris."), audio(), DONE))
    assert len(voice.spk.played) == 1  # the silence was skipped, the voice played


@pytest.mark.parametrize(
    "text,asks",
    [
        ("Can you open YouTube?", False),
        ("Neo, could you please play 505?", False),
        ("Could you send a message to Sam?", False),
        ("Is it raining?", True),
        ("Can you tell me the weather?", True),
        ("Neo, what's the time?", True),
        ("kya time hai", True),
        ("open notes and tell me how many notes I have", True),
    ],
)
def test_a_polite_frame_does_not_make_a_command_a_question(text, asks):
    from neo.voice.live import _asks

    assert _asks(text) is asks


@pytest.mark.parametrize(
    "text,multi",
    [("Neo, open YouTube, please.", False), ("Okay, open YouTube", False), ("open notes and type hi", True)],
)
def test_a_vocative_comma_is_not_a_second_step(text, multi):
    from neo.tools import load_all
    from neo.voice.live import _multi_step

    load_all()
    assert _multi_step(text) is multi


# ---- review of c6848e0 ----------------------------------------------------------------------------


def _lane_off(monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_LIVE_COMMAND_LANE", "false")
    monkeypatch.setattr(config, "_settings", None)


def test_a_blip_then_silence_never_opens_the_hold(voice, monkeypatch):
    _lane_off(monkeypatch)
    seen = _states(monkeypatch)

    async def run():
        # "Mm." and then the model's usual near-silence while it gets round to the call
        await feed(voice, heard("open YouTube"), audio(1), silence(12))
        assert voice._gate == "hold"  # 0.1 s of voice: nowhere near a spoken reply
        await feed(voice, call("open_url", {"url": "youtube.com"}), silence(30), DONE, settle=0.6)

    asyncio.run(run())
    assert ("open_url", {"url": "youtube.com"}) in voice.ran
    assert "speaking" not in seen and voice.spk.played == [] and seen[-1] == "listening"


def test_a_real_reply_keeps_its_pauses(voice):
    asyncio.run(feed(voice, heard("thanks neo"), audio(6), silence(10), audio(6), DONE))
    assert len(voice.spk.played) == 22  # 1.2 s of voice opened the hold; the pause in it plays too


@pytest.mark.parametrize(
    "text,steps",
    [
        # the last word may be manners or the payload's own: the model hears which, the lane doesn't
        ("play Pray for Me", None),
        ("play Please Please Please", None),
        ("play Jaane Tu Ya Jaane Na", None),
        ("neo play bohemian rhapsody please", None),
        ("type can you send me the file please", None),
        ("type thanks yaar", None),
        ("write down thanks for me", None),
        ("type please", None),
        ("search YouTube for songs for me", None),
        ("play Despacito", [("play_music", {"query": "Despacito"})]),
        # a polite word after an app name is no part of the name
        ("open YouTube please", [("open_app", {"target": "YouTube"})]),
        ("Neo, open Notes, please.", [("open_app", {"target": "Notes"})]),
        ("close Spotify yaar", [("quit_app", {"name": "Spotify"})]),
    ],
)
def test_the_lane_never_cuts_a_payload_short(text, steps):
    from neo.tools import load_all
    from neo.voice.live import _command_steps

    load_all()
    assert _command_steps(text) == steps


def test_a_title_ending_in_a_polite_word_plays_whole(voice):
    asyncio.run(feed(voice, heard("play Pray for Me"), call("play_music", {"query": "Pray for Me"}, "m1"), DONE, settle=0.2))
    assert voice.lane == [] and voice.ran == [("play_music", {"query": "Pray for Me"})]


@pytest.mark.parametrize(
    "text",
    ["close this tab and please", "open notes and please", "open notes and for me", "open notes and, please?"],
)
def test_a_sentence_ending_in_and_please_is_not_over(text):
    from neo.tools import load_all
    from neo.voice.live import _command_steps

    load_all()
    assert _command_steps(text) is None and _command_steps(text, settled=True) is None


@pytest.mark.parametrize(
    "text",
    [
        "Can you open files?",
        "Can you open PDFs?",
        "can you launch games?",
        "Can you close tabs?",
        "Can you kill time?",
        "can you open YouTube later?",
        "Can you play songs?",
    ],
)
def test_a_capability_question_is_left_to_the_model(text):
    from neo.tools import load_all
    from neo.voice.live import _command_steps

    load_all()
    assert _command_steps(text) is None


@pytest.mark.parametrize(
    "text,tool",
    [
        ("Neo, can you open YouTube?", "open_app"),
        ("Could you open github.com?", "open_app"),
        ("Can you open VS Code?", "open_app"),
        ("Neo, could you please play 505?", "play_music"),
        ("Can you play Shape of You?", "play_music"),
        ("Can you close this tab?", "hotkey"),
    ],
)
def test_a_polite_command_that_names_its_target_still_runs_on_the_lane(text, tool):
    from neo.tools import load_all
    from neo.voice.live import _command_steps

    load_all()
    assert [t for t, _ in _command_steps(text) or []] == [tool]


def test_a_capability_question_is_answered_out_loud(voice):
    asyncio.run(feed(voice, heard("Can you open files?"), said("Yes, I can open files."), audio(15), DONE, settle=0.2))
    assert voice.lane == [] and len(voice.spk.played) == 15


@pytest.mark.parametrize("text", ["Neil, close this tab", "Zara open YouTube", "Nio open Notes"])
def test_words_to_someone_else_are_left_to_the_model(text):
    from neo.tools import load_all
    from neo.voice.live import _command_steps

    load_all()
    assert _command_steps(text) is None


@pytest.mark.parametrize(
    "text,asks",
    [
        ("Can you hear me?", True),
        ("Could you repeat that?", True),
        ("Can you speak Hindi?", True),
        ("Can you see my screen?", True),
        ("Could you help me?", True),
        ("Will you be my friend?", True),
        ("Can you see if my wifi is on?", True),
        ("could you see whether my build passed?", True),
        ("can you open Safari and check if my order shipped?", True),
        ("Could you turn on do not disturb?", False),
        ("Can you remind me to call mom at 5?", False),
        ("zara volume badhao?", False),
        # commands the lane leaves to the model are still commands
        ("search youtube for cats please", False),
        ("search YouTube for songs for me please", False),
        ("type hello please", False),
        ("Can you open files?", False),  # held: the model answers or acts, and the gate follows
    ],
)
def test_a_polite_question_is_still_a_question(text, asks):
    from neo.tools import load_all
    from neo.voice.live import _asks

    load_all()
    assert _asks(text) is asks


def test_a_polite_yes_no_question_is_answered_at_once(voice):
    asyncio.run(feed(voice, heard("Can you hear me?"), said("Yes, loud and clear."), audio(5)))
    assert voice._gate == "open" and len(voice.spk.played) == 5  # not held back until the turn ends


def test_a_polite_question_the_agent_answers_is_spoken(voice, monkeypatch):
    from google.genai import types as gt

    _lane_off(monkeypatch)
    asyncio.run(feed(voice, heard("Can you see if my wifi is on?"), call("agent_task", {"goal": "check wifi"}, "a1"), settle=0.2))
    r = next(x for x in voice._live.responses if x.id == "a1")
    assert r.scheduling == gt.FunctionResponseScheduling.WHEN_IDLE  # the model is prompted to answer


def test_a_memory_write_goes_straight_back_to_listening(voice, monkeypatch):
    from neo.events import bus

    seen = _states(monkeypatch)

    async def run():
        await feed(voice, heard("remember that my car is on level 3"), call("memory", {"action": "remember", "text": "car on level 3"}), settle=0.1)
        await feed(voice, said("Got it, I'll remember that."), audio(15), settle=0.1)  # muted, still streaming
        return bus().state.value

    during = asyncio.run(run())
    assert voice.spk.played == [] and "thinking" not in seen and during == "listening"


def test_steps_of_a_command_go_straight_back_to_listening(voice, monkeypatch):
    from neo.events import bus

    _lane_off(monkeypatch)
    seen = _states(monkeypatch)

    async def run():
        await feed(voice, heard("open notes and type hi"), call("open_app", {"target": "Notes"}, "c1"), silence(3), DONE, settle=0.1)
        await feed(voice, call("type_text", {"text": "hi"}, "c2"), silence(10), settle=0.1)
        return bus().state.value

    during = asyncio.run(run())
    assert [n for n, _ in voice.ran] == ["open_app", "type_text"]
    assert "thinking" not in seen and during == "listening" and voice.spk.played == []


def test_an_older_settle_never_cuts_a_newer_thinking_short(voice, monkeypatch):
    from neo.events import bus
    from neo.voice import live as live_mod

    monkeypatch.setattr(live_mod, "_THINK_SETTLE_S", 0.1)

    async def run():
        await voice._think()
        await asyncio.sleep(0.06)
        await voice._think()  # another result the model will answer out loud
        await asyncio.sleep(0.06)  # the first one's settle is due now: the second still stands
        mid = bus().state.value
        await asyncio.sleep(0.1)
        return mid, bus().state.value

    assert asyncio.run(run()) == ("thinking", "listening")


def test_a_can_you_the_tool_itself_accepts_still_runs_on_the_lane():
    from neo.tools import load_all
    from neo.voice.live import _command_steps

    load_all()
    assert _command_steps("Can you type hello?") == [("type_text", {"text": "hello"})]  # as before c6848e0
