"""Brain effort: how hard each step thinks, which brains may see a picture, the run's time left,
picture limits per brain, and the wait after opening an app.

Hermetic: every brain is a fake (nothing is sent anywhere), clocks are fake where time matters,
and the settle wait reads a fake Mac."""

from __future__ import annotations

import asyncio
import io
import time
from types import SimpleNamespace

import pytest

import neo2.core.providers as prov
from neo2.core.providers import budget
from neo2.core.providers.base import (
    ImagePart,
    Message,
    NoEyes,
    OutOfTime,
    Overloaded,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    ToolCall,
    ToolResult,
    Turn,
    Unavailable,
)
from neo2.core.providers.effort import Effort, for_brain, start_level, trouble
from neo2.core.reflex.schema import Decision

# ---- fakes ------------------------------------------------------------------------------------------


class Brain:
    """A brain that answers (or fails) at once, and keeps what it was asked."""

    def __init__(self, name, vision=True, *, error=None, answer="ok", efforts=None, delay=0.0, cool=0.0):
        self.name, self.model = name, name
        self.supports_vision, self.supports_tools = vision, True
        self.error, self.answer, self.delay, self.cool = error, answer, delay, cool
        self.asked: list[str] = []  # the effort of each ask
        if efforts is not None:
            self.efforts = efforts

    def cooling_until(self):
        return self.cool

    async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
        self.asked.append(effort)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return Turn(text=self.answer, sees=self.supports_vision)

    async def stream(self, messages, **kw):
        yield self.answer


def _brains(monkeypatch, **brains):
    monkeypatch.setattr(
        prov,
        "make",
        lambda n: brains[n] if n in brains else (_ for _ in ()).throw(prov.ProviderError(f"no {n}")),
    )
    monkeypatch.setattr(prov, "_demoted", {})
    return brains


PNG = ImagePart(b"\x89PNG fake", "image/png")


def _picture_step() -> list[Message]:
    return [
        Message.user("play e4"),
        Message.assistant("", [ToolCall("c1", "screenshot", {})]),
        Message.tool([ToolResult("c1", "screenshot", "Screenshot s1 attached.", images=[PNG])]),
    ]


def _d(intent="agent_task", conf=0.9, margin=None) -> Decision:
    return Decision(intent, conf, False, 0.0, True, 0.9, margin=margin)


def _step(name, args, content, ok=True, effect=None, cid=None) -> list[Message]:
    cid = cid or f"c{abs(hash((name, str(args), content))) % 10_000}"
    return [
        Message.assistant("", [ToolCall(cid, name, args)]),
        Message.tool([ToolResult(cid, name, content, ok, effect=effect)]),
    ]


# ---- where a run's thinking starts --------------------------------------------------------------


def test_the_default_keeps_every_run_at_the_level_it_had():
    """screen_effort defaults to 'high' until the user's A/B: nothing changes before then."""
    assert start_level(_d(), screen=True) == "high"  # a screen task (agent_task): high, as before
    assert start_level(_d("quick_action"), screen=True) == "medium"  # a non-agent screen request: medium
    assert start_level(_d("quick_action"), screen=False, light=True) == "medium"  # the light lane
    assert start_level(_d(), screen=False) == "high"  # an agent task off the screen
    assert start_level(_d("question"), screen=False) == "medium"
    assert start_level(None, screen=False) == "medium"


def test_a_low_screen_effort_moves_only_screen_steps_and_the_light_lane():
    assert start_level(_d(), screen=True, knob="low") == "low"
    assert start_level(_d("quick_action"), screen=False, light=True, knob="low") == "low"
    assert start_level(_d(), screen=False, knob="low") == "high"  # research, code: still high
    assert start_level(_d("question"), screen=False, knob="low") == "medium"


def test_when_laya_is_unsure_a_screen_run_starts_one_level_higher_but_never_past_todays():
    unsure = _d(margin=0.05)
    assert start_level(unsure, screen=True, knob="low") == "medium"
    assert start_level(_d(conf=0.3), screen=True, knob="low") == "medium"  # not confident
    assert start_level(unsure, screen=True, knob="high") == "high"  # never above 'high'
    assert start_level(_d("quick_action", margin=0.05), screen=True, knob="medium") == "medium"  # nor today's


def test_a_typo_in_the_setting_is_todays_level(monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_SCREEN_EFFORT", "lwo")
    monkeypatch.setattr(config, "_settings", None)
    assert config.settings().screen_effort == "high"
    monkeypatch.setenv("NEO_SCREEN_EFFORT", " Low ")
    monkeypatch.setattr(config, "_settings", None)
    assert config.settings().screen_effort == "low" and start_level(_d(), screen=True) == "low"


# ---- each step's level --------------------------------------------------------------------------


def test_one_level_up_after_a_miss_and_back_down_after_an_action_that_worked():
    e = Effort("low")
    msgs = [Message.user("move the pawn to e4")]
    assert e.at(msgs) == "low"
    msgs += _step(
        "click",
        {"id": "e12"},
        "Clicked. Nothing visibly changed there within 1s — it may have missed",
        ok=False,
    )
    assert e.at(msgs) == "medium"  # a miss: think more
    msgs += _step("click", {"id": "e13"}, "Clicked. Changed: the screen changed around the point.")
    assert e.at(msgs) == "low"  # it worked: back down
    msgs += _step("click", {"id": "e14"}, "Clicked.")
    assert e.at(msgs) == "low"  # never below where it started


def test_the_miss_signals_each_step_it_up_and_high_is_the_ceiling():
    no_change = SimpleNamespace(changed=False, sheet="")
    sheet = SimpleNamespace(changed=True, sheet="“Start a New Game?” (buttons: Start, Cancel)")
    cases = [
        _step("hotkey", {"keys": "cmd+z"}, "Error: couldn't press it", ok=False),
        _step("click", {"id": "e1"}, "Clicked.", effect=no_change),  # the Effect says nothing changed
        _step("hotkey", {"keys": "cmd+n"}, "Pressed cmd+n.", effect=sheet),  # a sheet opened
    ]
    for step in cases:
        assert Effort("low").at([Message.user("go"), *step]) == "medium"
    e, msgs = Effort("low"), [Message.user("go")]
    for step in cases * 2:
        msgs += step
    assert e.at(msgs) == "high"  # capped


def test_the_same_action_twice_steps_up_even_when_it_said_ok():
    msgs = [Message.user("press start")]
    msgs += _step("click", {"id": "e7"}, "Clicked.", cid="a")
    msgs += _step("click", {"id": "e7"}, "Clicked.", cid="b")
    assert Effort("low").at(msgs) == "medium"


def test_looks_and_questions_to_the_user_are_neither_a_miss_nor_a_success():
    msgs = [Message.user("go")]
    msgs += _step("click", {"id": "e1"}, "Error: no such element", ok=False)
    msgs += _step("ax_tree", {}, "window “Board” …")  # only looked
    msgs += _step("click", {"id": "e9"}, "NEEDS_CONFIRM: press “Resign”?", ok=False)  # asked the user
    assert Effort("low").at(msgs) == "medium"
    assert trouble([ToolResult("x", "screenshot", "Screenshot s2 attached.")]) is None


def test_a_run_carried_on_after_a_confirmation_keeps_its_level_and_a_new_request_starts_fresh():
    msgs = [Message.user("resign the game")]
    msgs += _step("click", {"id": "e1"}, "Error: it didn't take", ok=False)
    msgs.append(Message.user("[User confirmed (request: “resign the game”). Go on.]"))
    assert Effort("low").at(msgs) == "medium"
    msgs.append(Message.user("now open notes"))  # the user's own words: a new request
    assert Effort("low").at(msgs) == "low"


def test_minimal_goes_only_to_a_brain_that_takes_it():
    assert for_brain(Brain("groq", False), "minimal") == "low"
    assert (
        for_brain(Brain("gemini", efforts=frozenset({"minimal", "low", "medium", "high"})), "minimal")
        == "minimal"
    )
    assert for_brain(Brain("x", efforts=frozenset({"medium", "high"})), "low") == "medium"
    assert for_brain(Brain("x"), "high") == "high"


def test_the_chain_asks_each_step_at_the_policys_level(monkeypatch):
    b = _brains(monkeypatch, gemini=Brain("gemini"))
    chain = prov.Chain(["gemini"], effort=Effort("low"))
    msgs = [Message.user("click start")]
    asyncio.run(chain.complete(msgs, effort="high"))  # the policy's level, not the one passed
    msgs += _step("click", {"id": "e1"}, "Error: not found", ok=False)
    asyncio.run(chain.complete(msgs))
    msgs += _step("click", {"id": "e2"}, "Clicked.")
    asyncio.run(chain.complete(msgs))
    assert b["gemini"].asked == ["low", "medium", "low"] and chain.levels == ["low", "medium", "low"]
    assert all(type(x) is str for x in b["gemini"].asked)  # a plain level, whatever the brain
    plain = prov.Chain(["gemini"])
    asyncio.run(plain.complete([Message.user("hi")], effort="high"))
    assert b["gemini"].asked[-1] == "high"  # no policy: the caller's level, as before


# ---- Gemini: minimal where the model takes it, a picture's media resolution ---------------------------


def _err(code: int, message: str):
    from google.genai import errors as gerrors

    return gerrors.APIError(code, {"error": {"code": code, "message": message, "status": "INVALID_ARGUMENT"}})


def _reply(text="done"):
    from google.genai import types as gt

    return gt.GenerateContentResponse(
        candidates=[gt.Candidate(content=gt.Content(role="model", parts=[gt.Part.from_text(text=text)]))]
    )


@pytest.fixture
def gem(monkeypatch):
    g = pytest.importorskip("neo2.core.providers.gemini")

    async def quiet(text):
        pass

    monkeypatch.setattr(g, "_note", quiet)
    p = g.GeminiProvider(model="gemini-3.7-flash", api_key="test-key", fallback_models=[])
    sent: list = []
    script: list = []

    async def generate_content(*, model, contents, config):
        sent.append((model, contents, config))
        r = script.pop(0) if script else _reply()
        if isinstance(r, Exception):
            raise r
        return r

    p._clients = [
        SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    ]
    return SimpleNamespace(g=g, p=p, sent=sent, script=script)


def test_gemini_asks_at_low_from_then_on_when_a_model_refuses_minimal(gem):
    from google.genai import types as gt

    gem.script.append(_err(400, "Thinking level MINIMAL is not supported for this model."))
    turn = asyncio.run(gem.p.complete([Message.user("hi")], effort="minimal"))
    levels = [c.thinking_config.thinking_level for _, _, c in gem.sent]
    assert turn.text == "done" and levels == [gt.ThinkingLevel.MINIMAL, gt.ThinkingLevel.LOW]
    asyncio.run(gem.p.complete([Message.user("again")], effort="minimal"))
    assert gem.sent[-1][2].thinking_config.thinking_level == gt.ThinkingLevel.LOW  # learned: no second 400
    assert len(gem.sent) == 3


def test_gemini_minimal_where_the_model_takes_it(gem):
    from google.genai import types as gt

    asyncio.run(gem.p.complete([Message.user("hi")], effort="minimal"))
    assert gem.sent[-1][2].thinking_config.thinking_level == gt.ThinkingLevel.MINIMAL


def test_another_400_is_not_mistaken_for_the_thinking_level(gem):
    from neo2.core.providers.base import BadRequest

    gem.script.append(_err(400, "Request contains an invalid argument."))
    with pytest.raises(BadRequest):
        asyncio.run(gem.p.complete([Message.user("hi")], effort="low"))
    assert len(gem.sent) == 1


def test_a_pictures_detail_sets_its_media_resolution(gem):
    from google.genai import types as gt

    g = gem.g
    msgs = [
        Message.user("look", images=[ImagePart(b"\x89PNG a", detail="medium")]),
        Message.assistant("", [ToolCall("c1", "screenshot", {})]),
        Message.tool(
            [
                ToolResult(
                    "c1",
                    "screenshot",
                    "s1",
                    images=[ImagePart(b"\x89PNG b", detail="low"), ImagePart(b"\x89PNG c")],
                )
            ]
        ),
    ]
    contents = g._to_contents(msgs, "gemini-3.8-flash")
    pics = [p for c in contents for p in c.parts if p.inline_data is not None]
    levels = [p.media_resolution.level if p.media_resolution else None for p in pics]
    L = gt.PartMediaResolutionLevel
    assert levels == [L.MEDIA_RESOLUTION_MEDIUM, L.MEDIA_RESOLUTION_LOW, None]  # auto: the model's default
    assert all(p.media_resolution is None for c in g._to_contents(msgs, "m", detail=False) for p in c.parts)


def test_a_model_that_refuses_a_pictures_resolution_gets_it_without(gem):
    gem.script.append(_err(400, "media_resolution is not supported for this model"))
    msgs = [Message.user("look", images=[ImagePart(b"\x89PNG a", detail="high")])]
    assert asyncio.run(gem.p.complete(msgs)).text == "done"
    first, second = (c for _, c, _ in gem.sent)
    assert first[0].parts[1].media_resolution is not None and second[0].parts[1].media_resolution is None


def test_gemini_says_why_it_is_cooling(gem):
    """What the chain records for a Gemini it skips inside a run: its quota and when it resets, or
    its overload — the words it would have failed with had it been asked."""
    g, p = gem.g, gem.p
    assert p.cooling_error() is None  # nothing cooling: nothing to say
    tomorrow = time.time() + 36_000
    p._exhausted[("gemini-3.7-flash", p._kid(0))] = tomorrow
    e = p.cooling_error()
    assert isinstance(e, QuotaExhausted) and e.until == tomorrow
    p._exhausted.clear()
    p._exhausted[("gemini-3.7-flash", g._ANY_KEY)] = time.time() + 90  # the model overloaded
    assert isinstance(p.cooling_error(), Overloaded)


# ---- which brains may see a picture ---------------------------------------------------------------


def test_the_vision_chain_order(monkeypatch):
    from neo2.core import config

    monkeypatch.setattr(config, "_settings", None)
    # Groq's vision model is only a seeing fallback there: a step with no picture (an AX-only step)
    # goes to the text brains and never spends its 8K tokens a minute; a picture step reaches it last.
    assert prov.brain("vision").names == ["gemini", "gemini_lite", "claude", "groq", "local"]
    assert prov.brain("vision").eyes == ["groq_vision"]
    assert prov.brain("agent").eyes == ["gemini_lite", "claude", "groq_vision"]
    assert prov.brain("offline").eyes == []  # offline means this Mac: no cloud eyes


def test_a_screenshot_step_goes_to_flash_lite_when_flash_is_overloaded_never_to_groqs_text_model(monkeypatch):
    b = _brains(
        monkeypatch,
        gemini=Brain("gemini", error=Overloaded("busy")),
        groq=Brain("groq", False, answer="blind"),
        gemini_lite=Brain("gemini_lite", answer="lite saw it"),
        groq_vision=Brain("groq_vision", answer="groq saw it"),
    )
    chain = prov.brain("agent")
    turn = asyncio.run(chain.complete(_picture_step()))
    assert turn.text == "lite saw it" and b["groq"].asked == [] and chain.last_used == "gemini_lite"
    b["gemini_lite"].error = Overloaded("busy too")
    assert asyncio.run(chain.complete(_picture_step())).text == "groq saw it"  # Groq's vision model, last
    assert b["groq"].asked == []


def test_a_step_with_no_picture_on_the_vision_chain_goes_to_groqs_text_model_not_its_vision_one(monkeypatch):
    b = _brains(
        monkeypatch,
        gemini=Brain("gemini", error=Overloaded("busy")),
        gemini_lite=Brain("gemini_lite", error=Overloaded("busy")),
        groq_vision=Brain("groq_vision", answer="qwen"),
        groq=Brain("groq", False, answer="gpt-oss"),
    )
    chain = prov.brain("vision")
    assert asyncio.run(chain.complete([Message.user("list the windows")])).text == "gpt-oss"
    assert b["groq_vision"].asked == []  # its per-minute budget is kept for the steps with a picture
    assert asyncio.run(chain.complete(_picture_step())).text == "qwen"  # …which still reach it
    assert len(b["groq"].asked) == 1  # and never Groq's text model


def test_every_brain_that_sees_busy_stops_the_step_and_says_what_still_works(monkeypatch):
    b = _brains(
        monkeypatch,
        gemini=Brain("gemini", error=Overloaded("busy")),
        gemini_lite=Brain("gemini_lite", error=Overloaded("busy")),
        groq_vision=Brain("groq_vision", error=RateLimited("429", retry_after=None)),
        groq=Brain("groq", False, answer="blind"),
    )
    with pytest.raises(NoEyes) as stop:
        asyncio.run(prov.brain("vision").complete(_picture_step()))
    said = stop.value.spoken()
    assert b["groq"].asked == [] and "can see the screen is busy" in said and "by their names" in said
    assert "Gemini Lite is overloaded" in said and "Groq can't see" not in said  # why the seeing ones failed
    assert any(e.kind == "blind" for _, e in stop.value.failures)  # (the log still has who was left out)


def _settings_env(monkeypatch, **env):
    from neo2.core import config

    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "NEO_BRAIN", "NEO_FAST_BRAIN",
              "NEO_GROQ_VISION_MODEL", "NEO_CLOUD_EYES"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)


def test_when_no_brain_set_up_can_see_the_stop_says_that_not_busy(monkeypatch):
    """Only Groq's text model (its vision model turned off): nothing that sees was asked, so nothing
    is 'busy' — the user hears that none can see, and what would fix it."""
    _settings_env(monkeypatch, GROQ_API_KEY="g", NEO_GROQ_VISION_MODEL="")
    b = _brains(monkeypatch, groq=Brain("groq", False))
    with pytest.raises(NoEyes) as stop:
        asyncio.run(prov.brain("agent").complete(_picture_step()))
    said = stop.value.spoken()
    assert b["groq"].asked == [] and "busy" not in said
    assert "None of the models set up for me can see the screen" in said and "Gemini key" in said
    # A photo the user sent is "the picture", not the screen.
    with pytest.raises(NoEyes) as stop:
        asyncio.run(prov.brain("agent").complete([Message.user("what's in this photo?", images=[PNG])]))
    assert "can see the picture" in stop.value.spoken()


def test_a_blind_brains_refused_coordinates_make_the_next_step_a_pixel_step(monkeypatch):
    """Every brain that sees is down from the first step: the text-only brain may take the step with
    no picture — but once it tried coordinates and was refused, the task needs eyes: it isn't asked
    again to repeat that (the logged run's three blind clicks), the run stops and says why."""
    b = _brains(monkeypatch, gemini=Brain("gemini", error=Overloaded("busy")), groq=Brain("groq", False, answer="x"))
    chain = prov.Chain(["gemini", "groq"], sticky=False)
    msgs = [Message.user("move e2 to e4")]
    asyncio.run(chain.complete(msgs))
    assert len(b["groq"].asked) == 1  # no picture yet: an AX-only step is its to take
    msgs += _step("click", {"x": 612, "y": 540}, "Error: you can't see the screen — this model has no vision", ok=False)
    with pytest.raises(NoEyes):
        asyncio.run(chain.complete(msgs))
    assert len(b["groq"].asked) == 1


def test_a_step_with_no_picture_still_goes_to_a_text_only_brain(monkeypatch):
    b = _brains(
        monkeypatch, gemini=Brain("gemini", error=Overloaded("busy")), groq=Brain("groq", False, answer="ax")
    )
    assert (
        asyncio.run(
            prov.Chain(["gemini", "groq"], sticky=False).complete([Message.user("list the windows")])
        ).text
        == "ax"
    )
    assert b["groq"].asked == ["medium"]


def test_a_local_brain_the_user_put_first_still_keeps_the_screen_on_the_mac(monkeypatch):
    here = Brain("local", False, answer="local")
    here.on_device = True
    b = _brains(
        monkeypatch, local=here, groq=Brain("groq", False), gemini_lite=Brain("gemini_lite", answer="cloud")
    )
    assert (
        asyncio.run(prov.Chain(["local", "groq"], eyes=prov.EYES).complete(_picture_step())).text == "local"
    )
    assert b["gemini_lite"].asked == []
    # …but one the user put last is no guard: a picture skips it for the brains that see
    last = Brain("local", False, answer="local")
    last.on_device = True
    _brains(monkeypatch, gemini=Brain("gemini", error=Overloaded("x")), groq=Brain("groq", False), local=last,
            gemini_lite=Brain("gemini_lite", answer="cloud"))  # fmt: skip
    assert (
        asyncio.run(prov.Chain(["gemini", "groq", "local"], eyes=prov.EYES).complete(_picture_step())).text
        == "cloud"
    )
    assert last.asked == []


def test_a_local_first_user_keeps_pictures_off_the_cloud_fallbacks_unless_they_opt_in(monkeypatch):
    """NEO_BRAIN=local with only a Groq key: before the seeing fallbacks, a screenshot never left the
    Mac on that chain (Groq's text model gets none). A fallback doesn't change that on its own: the
    step stops (nothing set up for them sees), and Groq's vision model is never sent the picture."""
    _settings_env(monkeypatch, GROQ_API_KEY="g", NEO_BRAIN="local")
    assert prov.brain("agent").eyes == [] and prov.brain("vision").eyes == []
    here = Brain("local", False, error=Unavailable("local model server not running"))
    here.on_device = True
    b = _brains(monkeypatch, local=here, groq=Brain("groq", False), groq_vision=Brain("groq_vision", answer="cloud"))
    with pytest.raises(NoEyes) as stop:
        asyncio.run(prov.brain("agent").complete(_picture_step()))
    assert b["groq_vision"].asked == [] and "None of the models set up" in stop.value.spoken()
    # Opting in (NEO_CLOUD_EYES=on), or naming a cloud brain that sees, lets them look.
    _settings_env(monkeypatch, GROQ_API_KEY="g", NEO_BRAIN="local", NEO_CLOUD_EYES="on")
    assert asyncio.run(prov.brain("agent").complete(_picture_step())).text == "cloud"
    _settings_env(monkeypatch, GROQ_API_KEY="g", NEO_BRAIN="local", NEO_FAST_BRAIN="gemini")
    assert "groq_vision" in prov.brain("agent").eyes
    _settings_env(monkeypatch, GROQ_API_KEY="g")  # a cloud brain first (the default): as decided (43)
    assert "groq_vision" in prov.brain("agent").eyes and prov.brain("vision").eyes == ["groq_vision"]


def test_an_embedded_agent_sends_no_picture_to_groqs_vision_model_unless_its_host_names_one(tmp_path):
    from neo2.core.config import LibrarySettings

    assert LibrarySettings(data_dir=tmp_path / "a").groq_vision_model == ""
    assert LibrarySettings(data_dir=tmp_path / "b", groq_vision_model="qwen/x").groq_vision_model == "qwen/x"


def test_the_input_tools_refuse_a_blind_brain_in_the_words_the_chain_looks_for():
    """A refused blind click makes the task a pixel task only while the refusal holds BLIND_REFUSAL:
    a rewording of pointer.blind() must fail here, not silently hand the next steps back to the blind
    brain."""
    from neo2.core.providers.base import BLIND_REFUSAL

    pointer = pytest.importorskip("neo2.core.tools.computer.pointer")
    for tool in ("click", "drag", "scroll", "move", "right_click", "double_click"):
        assert BLIND_REFUSAL in pointer.blind(tool), tool


def test_groq_vision_is_its_own_seeing_brain_from_the_groq_key(monkeypatch):
    from neo2.core import config

    monkeypatch.setattr(config, "_settings", None)
    with pytest.raises(prov.ProviderError):
        prov._build("groq_vision")  # no key: not set up
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setattr(config, "_settings", None)
    p = prov._build("groq_vision")
    assert (p.name, p.model, p.supports_vision, p.per_minute) == (
        "groq_vision",
        "qwen/qwen3.8-27b",
        True,
        True,
    )
    assert p.token_budget == 8000 and not p.on_device


# ---- the run's time left --------------------------------------------------------------------------


def test_a_rate_limit_wait_past_the_runs_time_left_stops_the_run_now(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    _brains(monkeypatch, groq=Brain("groq", False, error=RateLimited("429", retry_after=40.0)))
    monkeypatch.setattr(prov.asyncio, "sleep", fake_sleep)
    with budget.run(lambda: 30.0, limit_s=180.0), pytest.raises(OutOfTime) as stop:
        asyncio.run(prov.Chain(["groq"]).complete([Message.user("hi")]))
    assert slept == [] and "time limit (180s)" in stop.value.spoken()
    # with the time for it, it waits as before
    with budget.run(lambda: 120.0), pytest.raises(prov.ProviderError):
        asyncio.run(prov.Chain(["groq"]).complete([Message.user("hi")]))
    assert slept and slept[0] >= 40.0


def test_outside_a_run_the_chain_waits_as_before(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    b = _brains(monkeypatch, groq=Brain("groq", False, error=RateLimited("429", retry_after=40.0)))
    monkeypatch.setattr(prov.asyncio, "sleep", fake_sleep)
    with pytest.raises(prov.ProviderError):
        asyncio.run(prov.Chain(["groq"]).complete([Message.user("hi")]))
    assert slept and len(b["groq"].asked) == 2


def test_a_call_still_out_when_the_time_is_up_is_given_up_on():
    slow = Brain("groq_vision", delay=5.0)

    async def go():
        with budget.run(lambda: 0.05, limit_s=180.0):
            return await prov.Chain([slow], sticky=False).complete([Message.user("hi")])

    t0 = time.monotonic()
    with pytest.raises(OutOfTime):
        asyncio.run(go())
    assert time.monotonic() - t0 < 1.0


def test_a_spent_quota_inside_a_run_is_said_as_the_quota_not_as_the_time_limit(monkeypatch):
    """Gemini's keys are spent until tomorrow, 170 s of the run are left and Groq fails: the user hears
    the quota and when it resets, as outside a run — the time limit isn't what stopped it."""
    _settings_env(monkeypatch, GEMINI_API_KEY="k")
    tomorrow = time.time() + 36_000
    gem = Brain("gemini", error=QuotaExhausted("spent", until=tomorrow), cool=tomorrow)
    _brains(monkeypatch, gemini=gem, groq=Brain("groq", False, error=ProviderError("groq: 500 internal")))
    with budget.run(lambda: 170.0, limit_s=180.0), pytest.raises(prov.AllBrainsFailed) as stop:
        asyncio.run(prov.Chain(["gemini", "groq"]).complete([Message.user("hi")]))
    said = stop.value.spoken()
    assert not isinstance(stop.value, OutOfTime) and "time limit" not in said
    assert "Gemini is out of free quota until" in said and "Groq failed" in said
    assert gem.asked == []  # (still not asked: its wait ends past the time left)
    # A brain that knows why it's cooling says so in its own words (Gemini: overloaded).
    busy = Brain("gemini", cool=time.time() + 90)
    busy.cooling_error = lambda: Overloaded("every model is overloaded", retry_after=90.0)
    _brains(monkeypatch, gemini=busy, groq=Brain("groq", False, error=ProviderError("groq: 500 internal")))
    with budget.run(lambda: 60.0, limit_s=180.0), pytest.raises(prov.AllBrainsFailed) as stop:
        asyncio.run(prov.Chain(["gemini", "groq"]).complete([Message.user("hi")]))
    assert not isinstance(stop.value, OutOfTime) and "Gemini is overloaded" in stop.value.spoken()


def test_a_pixel_step_with_the_seeing_brains_cooling_inside_a_run_stops_as_no_eyes(monkeypatch):
    """DECISIONS 39: a pixel step no brain that sees could answer stops with that reason (and what
    still works), in a run as outside one — not as a time limit with 170 s left."""
    tomorrow = time.time() + 36_000
    b = _brains(
        monkeypatch,
        gemini=Brain("gemini", error=Overloaded("x"), cool=tomorrow),
        gemini_lite=Brain("gemini_lite", error=Overloaded("busy")),
        groq_vision=Brain("groq_vision", error=ProviderError("groq_vision: 404 model not found")),
        groq=Brain("groq", False),
    )
    with budget.run(lambda: 170.0, limit_s=180.0), pytest.raises(NoEyes) as stop:
        asyncio.run(prov.brain("agent").complete(_picture_step()))
    assert b["gemini"].asked == [] and "by their names" in stop.value.spoken()


def test_with_no_time_left_to_answer_the_stop_is_the_time_limit(monkeypatch):
    """When the time really is up (less left than a brain needs to answer), that is the reason."""
    _brains(monkeypatch, gemini=Brain("gemini", cool=time.time() + 90),
            groq=Brain("groq", False, error=ProviderError("groq: 500 internal")))  # fmt: skip
    with budget.run(lambda: 2.0, limit_s=180.0), pytest.raises(OutOfTime) as stop:
        asyncio.run(prov.Chain(["gemini", "groq"]).complete([Message.user("hi")]))
    assert "time limit (180s)" in stop.value.spoken()
    # …but a picture step no brain set up could see stops for that, time or not.
    _settings_env(monkeypatch, GROQ_API_KEY="g")
    _brains(monkeypatch, groq=Brain("groq", False))
    with budget.run(lambda: 2.0, limit_s=180.0), pytest.raises(NoEyes):
        asyncio.run(prov.Chain(["groq"], eyes=prov.EYES).complete(_picture_step()))


def test_the_time_limit_stop_names_only_the_brains_that_were_asked(monkeypatch):
    _brains(
        monkeypatch,
        gemini=Brain("gemini", error=Overloaded("busy")),
        gemini_lite=Brain("gemini_lite", error=Overloaded("busy")),
        groq_vision=Brain("groq_vision", error=RateLimited("429", retry_after=50.0)),
        groq=Brain("groq", False),
    )
    with budget.run(lambda: 20.0, limit_s=180.0), pytest.raises(OutOfTime) as stop:
        asyncio.run(prov.brain("agent").complete(_picture_step()))
    said = stop.value.spoken()
    assert said.startswith("I had to stop: Gemini is overloaded") and "can't see pictures" not in said
    assert "Groq's vision model is rate-limited" in said and "time limit (180s)" in said


def test_a_brain_whose_wait_ends_after_the_run_is_not_asked(monkeypatch):
    spent = Brain("groq_vision", cool=time.time() + 50)
    b = _brains(monkeypatch, gemini=Brain("gemini", error=Overloaded("busy")), groq_vision=spent)
    with budget.run(lambda: 20.0), pytest.raises(OutOfTime):
        asyncio.run(prov.Chain(["gemini", "groq_vision"], sticky=False).complete(_picture_step()))
    assert spent.asked == [] and b["gemini"].asked


def test_no_time_left_asks_nobody(monkeypatch):
    b = _brains(monkeypatch, gemini=Brain("gemini"))
    with budget.run(lambda: 0.0), pytest.raises(OutOfTime):
        asyncio.run(prov.Chain(["gemini"]).complete([Message.user("hi")]))
    assert b["gemini"].asked == []


def test_the_budget_clock_leaves_out_the_time_the_user_held_the_run():
    now, held = [100.0], [0.0]
    with budget.seconds(180.0, paused=lambda: held[0], clock=lambda: now[0]):
        now[0] += 100.0
        held[0] += 60.0  # a minute of it the user was driving NEO's browser
        assert budget.left() == pytest.approx(140.0) and budget.limit() == 180.0
    assert budget.left() is None


def test_the_loops_budget_message_for_a_stop_is_the_errors_own():
    """The loop speaks AllBrainsFailed.spoken() (loop._model_trouble): both stops are ones."""
    from neo2.core.agent.loop import _model_trouble

    said = _model_trouble(NoEyes([("gemini", Overloaded("x"))]), [])
    assert said.startswith("The model that can see the screen is busy")
    assert "time limit" in _model_trouble(OutOfTime([], limit_s=180.0), [])


# ---- picture limits -------------------------------------------------------------------------------


def _keys(monkeypatch, **env):
    from neo2.core import config

    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "NEO_CLAUDE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})


def test_image_limit_is_the_strictest_of_the_brains_that_may_see(monkeypatch):
    _keys(monkeypatch, GEMINI_API_KEY="g")
    assert prov.image_limit() is None  # Gemini alone: no limit (it bills by resolution instead)
    _keys(monkeypatch, GEMINI_API_KEY="g", GROQ_API_KEY="q")
    lim = prov.image_limit()
    assert lim is not None and (lim.long_edge, lim.pixels, lim.bytes) == (None, 33_177_600, 3 * 1024 * 1024)
    _keys(monkeypatch, GEMINI_API_KEY="g", GROQ_API_KEY="q", ANTHROPIC_API_KEY="a")  # claude-opus-5
    lim = prov.image_limit()
    assert (lim.long_edge, lim.pixels, lim.bytes) == (2576, 3_750_000, 3 * 1024 * 1024)
    _keys(monkeypatch, ANTHROPIC_API_KEY="a", NEO_CLAUDE_MODEL="claude-sonnet-4-5")
    assert prov.image_limit().long_edge == 1568
    assert prov.image_limit("agent").long_edge == 1568  # an agent chain's seeing fallbacks count too


def test_a_hosts_own_brain_gives_its_own_limit(monkeypatch):
    from neo2.core.providers import BrainPool, ImageLimit

    host = Brain("hosts", True)
    host.image_limit = ImageLimit(1024, None, None)
    blind = Brain("text", False)
    blind.image_limit = ImageLimit(10, 10, 10)  # a brain that never sees a picture doesn't count
    monkeypatch.setattr(prov, "_app_pool", BrainPool(overrides={"vision": [host, blind]}))
    assert prov.image_limit() == ImageLimit(1024, None, None)


def test_a_brain_that_says_its_request_size_caps_one_picture_at_it():
    from neo2.core.providers import ImageLimit, limit_of

    b = Brain("cerebras_vision", True)
    b.max_request_bytes = 10 * 1024 * 1024  # base64 pictures included
    assert limit_of(b) == ImageLimit(bytes=10 * 1024 * 1024 * 3 // 4)
    b.name = "groq_vision"  # a known kind too: the stricter of each
    assert limit_of(b) == ImageLimit(None, 33_177_600, 3 * 1024 * 1024)


def test_image_limit_scale_and_tighter():
    from neo2.core.providers import ImageLimit

    assert ImageLimit(1568).box(3024, 1964) == (1568, int(1964 * 1568 / 3024))
    assert ImageLimit(1568).box(800, 600) == (800, 600)
    assert ImageLimit(1568).scale(3024, 1964) == pytest.approx(1568 / 3024)
    assert ImageLimit(None, 1_150_000).scale(3024, 1964) == pytest.approx((1_150_000 / (3024 * 1964)) ** 0.5)
    assert ImageLimit(4000).scale(800, 600) == 1.0
    assert ImageLimit(2576, None, 5).tighter(ImageLimit(1568, 9, None)) == ImageLimit(1568, 9, 5)


# ---- what a picture costs each brain ----------------------------------------------------------------


def _png(w, h) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (30, 60, 90)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_size_reads_the_header():
    from PIL import Image

    from neo2.core.providers.fit import image_size

    for fmt in ("PNG", "JPEG", "GIF", "WEBP"):
        buf = io.BytesIO()
        Image.new("RGB", (321, 123)).save(buf, format=fmt)
        assert image_size(buf.getvalue()) == (321, 123), fmt
    assert image_size(b"not a picture") is None


def test_a_picture_costs_what_its_brain_bills():
    from neo2.core.providers.fit import image_tokens

    shot = ImagePart(_png(3024, 1964))
    assert image_tokens(ImagePart(b"x", detail="low"), "gemini") == 280
    assert image_tokens(ImagePart(b"x", detail="medium"), "gemini_lite") == 560
    assert image_tokens(shot, "gemini") == 1120  # auto: Gemini 3's default for a picture
    assert image_tokens(shot, "groq_vision") == 2048
    assert image_tokens(shot, "claude") == pytest.approx(3_750_000 / 750, rel=0.01)  # scaled to its limit
    assert image_tokens(ImagePart(_png(750, 100)), "claude") == 100
    assert image_tokens(shot, "groq", vision=False) == 8  # a text-only brain gets a note
    assert image_tokens(shot, "someone") == 1300


def test_the_effect_survives_normalizing_and_trimming():
    from neo2.core.providers.fit import fit
    from neo2.core.providers.history import normalize

    eff = SimpleNamespace(changed=False)
    msgs = [
        Message.user("go"),
        Message.assistant("", [ToolCall("a", "click", {"id": "e1"})]),
        Message.tool(
            [ToolResult("a", "clicked", "Clicked. " * 800, False, images=[PNG], effect=eff)]
        ),  # (named off)
        Message.assistant("", [ToolCall("b", "ax_tree", {})]),
        Message.tool([ToolResult("b", "ax_tree", "tree " * 3000)]),
        Message.assistant("", [ToolCall("c", "ax_tree", {})]),
        Message.tool([ToolResult("c", "ax_tree", "tree " * 3000)]),
    ]
    fixed = normalize(msgs)[2].tool_results[0]
    assert fixed.name == "click" and fixed.effect is eff
    f = fit(msgs, system="s", budget=3000, max_tokens=512, vision=True, name="groq_vision")
    r = f.messages[2].tool_results[0]
    assert r.effect is eff and not r.images and len(r.content) < len("Clicked. " * 800)


# ---- the wait after opening an app ------------------------------------------------------------------


class Mac:
    """A fake Mac on a fake clock: the app is up from `front_at`, its field from `field_at`."""

    def __init__(self, front_at=0.0, field_at=None, fail=False):
        self.t, self.front_at, self.field_at, self.fail = 0.0, front_at, field_at, fail
        self.looks = 0

    def clock(self):
        return self.t

    async def sleep(self, s):
        self.t += s

    def probe(self, app, typing):
        from neo2.core.agent.settle import Ready

        self.looks += 1
        self.t += 0.002  # a look costs ~2 ms of Accessibility reads
        if self.fail:
            raise RuntimeError("no Accessibility")
        up = self.t >= self.front_at
        return Ready(up, up, self.field_at is not None and self.t >= self.field_at)


def _wait(monkeypatch, mac, **kw):
    from neo2.core.agent import settle

    monkeypatch.setattr(settle, "probe", mac.probe)
    return asyncio.run(settle.app_ready(clock=mac.clock, sleep=mac.sleep, **kw))


def test_an_app_already_up_is_ready_in_one_poll_not_the_old_fixed_time(monkeypatch):
    waited = _wait(monkeypatch, Mac(front_at=0.0))
    assert waited < 0.1  # was 0.8 s every time


def test_a_cold_launch_is_waited_for_past_the_old_fixed_time(monkeypatch):
    waited = _wait(monkeypatch, Mac(front_at=1.3))
    assert 1.3 <= waited < 1.45  # the old 0.8 s sent the first keys into the void


def test_typing_waits_for_a_field_that_takes_text_or_the_old_time(monkeypatch):
    assert 0.4 <= _wait(monkeypatch, Mac(front_at=0.1, field_at=0.4), typing=True) < 0.5
    # a field Accessibility never names: a focused window is enough once the old 0.8 s has passed
    assert 0.8 <= _wait(monkeypatch, Mac(front_at=0.1, field_at=None), typing=True) < 0.9


def test_an_app_in_front_with_no_window_goes_ahead_after_the_old_fixed_time(monkeypatch):
    """An app whose windows are all closed (activate_app on it, a window that never takes focus):
    in front is enough once the old 0.8 s has passed — never the 2 s cap."""
    from neo2.core.agent.settle import Ready

    class Windowless(Mac):
        def probe(self, app, typing):
            r = super().probe(app, typing)
            return Ready(r.front, False, False)

    assert 0.8 <= _wait(monkeypatch, Windowless(front_at=0.0)) < 0.9
    assert 0.8 <= _wait(monkeypatch, Windowless(front_at=0.0), typing=True) < 0.9
    assert 1.3 <= _wait(monkeypatch, Windowless(front_at=1.3)) < 1.45  # still waits for it to come forward


def test_the_wait_never_passes_its_cap_and_a_mac_that_cant_answer_gets_the_old_wait(monkeypatch):
    assert _wait(monkeypatch, Mac(front_at=99.0)) == pytest.approx(2.0, abs=0.06)
    assert _wait(monkeypatch, Mac(fail=True)) == pytest.approx(0.8, abs=0.01)


# ---- through the real agent loop (the wiring the integrator adds, emulated) ---------------------------


def _board_registry(miss_second_click: bool):
    from neo2.core.agent.registry import RegisteredTool, Registry, ToolOutput

    reg, clicks = Registry(), [0]

    async def ax_tree(a, ctx):
        return ToolOutput("window “Board”: the board area has no accessible elements")

    async def screenshot(a, ctx):
        return ToolOutput("Screenshot s1 attached.", images=[PNG], fence=False)

    async def click(a, ctx):
        clicks[0] += 1
        if miss_second_click and clicks[0] == 2:
            return ToolOutput("Clicked. Nothing visibly changed there within 1s — it may have missed", ok=False)
        return ToolOutput("Clicked. Changed: the screen changed around the point.")

    reg.register(RegisteredTool("ax_tree", "look", {"type": "object"}, ax_tree, parallel_safe=True))
    reg.register(RegisteredTool("screenshot", "look", {"type": "object"}, screenshot))
    reg.register(RegisteredTool("click", "click", {"type": "object"}, click))
    return reg


class Scripted(Brain):
    def __init__(self, name, calls, **kw):
        super().__init__(name, **kw)
        self.turns = [Turn(tool_calls=[ToolCall(f"t{i}", n, a)]) for i, (n, a) in enumerate(calls)] + [Turn(text="Done.")]

    async def complete(self, messages, **kw):
        await super().complete(messages, **kw)
        return self.turns[min(sum(m.role == "assistant" for m in messages), len(self.turns) - 1)]


_MOVE = [("ax_tree", {}), ("screenshot", {}), ("click", {"shot": "s1", "x": 1, "y": 2}), ("screenshot", {}),
         ("click", {"shot": "s1", "x": 3, "y": 4}), ("screenshot", {}), ("click", {"shot": "s1", "x": 3, "y": 5})]  # fmt: skip


def test_in_the_loop_a_low_run_steps_up_after_the_miss_and_back_down(monkeypatch):
    from neo2.core.agent.loop import run_agent
    from neo2.core.events import EventBus

    g = Scripted("gemini", _MOVE)
    _brains(monkeypatch, gemini=g)
    reg = _board_registry(miss_second_click=True)
    chain = prov.brain("vision", effort=Effort(start_level(_d(), screen=True, knob="low")))
    res = asyncio.run(run_agent("move e2 to e4", provider=chain, system="", reg=reg, ev=EventBus(), effort="high",
                                tools=[t.spec() for t in reg._tools.values()]))  # fmt: skip
    assert res.stopped == "done"
    # the miss (5th step's click) → medium until a click works → low again, and for the loop's [verify] turn
    assert g.asked == ["low", "low", "low", "low", "low", "medium", "medium", "low", "low"]
    # the default setting: every step at 'high', exactly as before
    g2 = Scripted("gemini", _MOVE)
    _brains(monkeypatch, gemini=g2)
    chain = prov.brain("vision", effort=Effort(start_level(_d(), screen=True)))
    asyncio.run(run_agent("move e2 to e4", provider=chain, system="", reg=_board_registry(True), ev=EventBus(),
                          effort="high", tools=[t.spec() for t in reg._tools.values()]))  # fmt: skip
    assert set(g2.asked) == {"high"} and len(g2.asked) == 9


def test_in_the_loop_a_bound_budget_ends_the_run_inside_its_limit(monkeypatch):
    """Only a rate-limited seeing brain left (a 50 s wait before every step): the chain used to sit
    out a wait that ended past the run's limit; bound to the run's clock, it stops first."""
    from neo2.core.agent.loop import run_agent
    from neo2.core.events import EventBus

    now = [0.0]
    real_sleep = asyncio.sleep

    async def fake_sleep(s, *a, **k):
        now[0] += float(s)
        await real_sleep(0)

    class Throttled(Scripted):
        fresh = True

        async def complete(self, messages, **kw):
            self.fresh = not self.fresh
            if not self.fresh:
                raise RateLimited("429 tokens per minute", retry_after=50.0)
            now[0] += 5.0
            return await super().complete(messages, **kw)

    gv = Throttled("groq_vision", _MOVE)
    _brains(monkeypatch, groq_vision=gv)
    monkeypatch.setattr(prov.asyncio, "sleep", fake_sleep)
    chain = prov.Chain(["groq_vision"], sticky=False)

    class Bound:  # run_agent's wiring: each step asked within the run's time left
        def __getattr__(self, k):
            return getattr(chain, k)

        async def complete(self, *a, **kw):
            with budget.seconds(180.0 - now[0], clock=lambda: now[0]):
                return await chain.complete(*a, **kw)

    reg = _board_registry(False)
    res = asyncio.run(run_agent("move e2 to e4", provider=Bound(), system="", reg=reg, ev=EventBus(),
                                tools=[t.spec() for t in reg._tools.values()]))  # fmt: skip
    assert now[0] <= 180.0 and res.stopped == "error" and "time limit" in res.text
