"""Targeted generation: contrastive pairs, abstracted confusions, two-model agreement, the cap."""

from __future__ import annotations

import json
import re

import pytest

from neo.providers.base import Turn
from neo.reflex import generate as g


class Model:
    """A fake LLM: generation prompts get `gen(target key, n)`, labelling prompts `label(text)` per line."""

    def __init__(self, gen=None, label=None):
        self.gen, self.label, self.prompts = gen, label, []

    async def complete(self, messages, *, system="", tools=None, effort="low", max_tokens=4000):
        prompt = messages[0].text
        self.prompts.append(system + "\n" + prompt)
        if system.startswith("You write training data"):
            key = re.search(r"clearly of this kind:\n- (\S+):", prompt).group(1)
            n = int(re.search(r"Generate (\d+)", prompt).group(1))
            return Turn(text=json.dumps(self.gen(key, n)))
        texts = [line.split(". ", 1)[1] for line in prompt.splitlines()[1:]]
        return Turn(text=json.dumps([self.label(t) for t in texts]))


def _by_prefix(text):
    return text.split(" #")[0]  # "chat #3" was meant as chat


def _counter():
    n = iter(range(100_000))
    return lambda key, k: [f"{key} #{next(n)}" for _ in range(k)]


async def _no_sleep(_):
    return None


def _models(a: Model, b: Model):
    return [("gemini", a, 0.0), ("groq", b, 0.0)]


# ---- classes and pairs ------------------------------------------------------------------------


def test_focus_names_intents_tools_kinds_and_the_destructive_head(monkeypatch):
    monkeypatch.setattr(g, "_tool_description", lambda name: f"does {name}")
    p = g.parse_focus("chat:quick_action")
    assert (p.a.key, p.b.key, p.a.intent) == ("chat", "quick_action", "chat")
    p = g.parse_focus("tool:open_app:tool:play_music")
    assert (p.a.key, p.b.key) == ("tool:open_app", "tool:play_music")
    assert p.a.intent == "quick_action" and p.a.fields == {"tool": "open_app"}
    assert g.parse_focus("tool:open_app:play_music").b.key == "tool:play_music"  # the prefix carries over
    p = g.parse_focus("kind:send_message:send_email")
    assert p.a.intent is None and p.a.fields == {"kind": "send_message"}  # the models state the route
    p = g.parse_focus("destructive:yes:no")
    assert p.a.fields == {"destructive": True} and p.b.fields == {"destructive": False}
    assert p.a.intent == p.b.intent == "agent_task"
    assert g.parse_focus("chat:tool:web_search").b.key == "tool:web_search"
    for bad in ("chat", "chat:chat", "bogus:quick_action", "kind:no_such_kind:other", "destructive:maybe:no"):
        with pytest.raises(ValueError):
            g.parse_focus(bad)


def test_confusion_summary_top_pairs_merge_both_directions(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_tool_description", lambda name: f"does {name}")
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"pairs": [
        ["chat", "quick_action", 482], ["chat", "agent_task", 391], ["quick_action", "chat", 100],
        ["stop", "quick_action", 5], ["chat", "nonsense", 900], ["chat", "chat", 999],
    ]}))  # fmt: skip
    pairs = g.read_confusion(path, top=2)
    assert [({p.a.key, p.b.key}, p.n) for p in pairs] == [({"chat", "quick_action"}, 582), ({"chat", "agent_task"}, 391)]
    path.write_text(json.dumps({"head": "tool", "pairs": [["open_app", "web_search", 12, ["open the docs page"]]]}))
    (p,) = g.read_confusion(path)
    assert (p.a.key, p.b.key) == ("tool:open_app", "tool:web_search")
    assert p.examples == {"tool:open_app": ["open the docs page"]}
    path.write_text(json.dumps({"destructive": {"pairs": [[True, False, 7]]}, "intent": {"pairs": [["stop", "chat", 3]]}}))
    assert [(p.a.key, p.b.key, p.n) for p in g.read_confusion(path)] == [
        ("destructive:yes", "destructive:no", 7), ("stop", "chat", 3)
    ]  # fmt: skip


def test_corrected_requests_become_examples_of_the_pair_they_fell_between(monkeypatch):
    monkeypatch.setattr(g, "_tool_description", lambda name: f"does {name}")
    chat_qa, tools = g.parse_focus("chat:quick_action"), g.parse_focus("tool:open_app:web_search")
    g.add_corrections([chat_qa, tools], [
        {"text": "what's playing", "intent": "chat", "wrong": ["quick_action"]},
        {"text": "turn it up", "intent": "quick_action", "wrong": ["chat"]},
        {"text": "open the wiki", "intent": "quick_action", "tool": "open_app", "wrong": ["web_search"]},
        {"text": "how are you", "intent": "chat", "wrong": []},  # not a correction
        {"text": "write the report", "intent": "agent_task", "wrong": ["chat"]},  # another pair
    ])  # fmt: skip
    assert chat_qa.examples == {"chat": ["what's playing"], "quick_action": ["turn it up"]}
    assert tools.examples == {"tool:open_app": ["open the wiki"]}


def test_a_prompt_shows_only_the_shape_of_a_request_never_its_words():
    public = [{"text": t} for t in (
        "remind me what the plan said", "what did the email say about lunch", "remind me what the doctor said",
        "tell me what she said about the trip", "remind me about the meeting", "what about the report",
    )]  # fmt: skip
    mine = "remind me what Zanzibarella said about the 4417 invoice"
    vocab = g.public_vocab([*public, {"text": mine}, {"text": mine}, {"text": mine}], {mine.lower()})
    assert "zanzibarella" not in vocab and "invoice" not in vocab and "remind" in vocab
    assert g.abstract(mine, vocab) == "remind me what {x} said about the {n} {x}"
    assert g.patterns([mine, mine, "zanzibarella invoice 4417 now", "remind me about the"], vocab) == [
        "remind me what {x} said about the {n} {x}"
    ]  # mostly placeholders says nothing; no placeholder at all would be the request itself
    many = [{"text": f"remind me about thing{i}"} for i in range(1000)]
    assert "remind" in g.public_vocab(many, set()) and "thing7" not in g.public_vocab(many, set())
    assert g.public_vocab([*many, *({"text": "call alex"} for _ in range(4))], set()) >= {"remind", "about"}
    assert "alex" not in g.public_vocab([*many, *({"text": "call alex"} for _ in range(4))], set())  # 4 of 1004 < 0.5%


# ---- the cycle ------------------------------------------------------------------------------------


async def test_rows_are_kept_only_when_the_other_model_agrees():
    def gen(key, n):
        return [f"{key} #1", f"{key} #2", f"{key} #maybe", "x"]  # "x" is too short to keep

    def label(text):
        return "agent_task" if text.endswith("#maybe") else _by_prefix(text)

    a, b = Model(gen, label), Model(gen, label)
    written = []
    rows = await g.focus([g.parse_focus("chat:quick_action")], _models(a, b), set(), cap=10, write=written.append, sleep=_no_sleep)
    assert rows == written
    assert {r["text"] for r in rows} == {"chat #1", "chat #2", "quick_action #1", "quick_action #2"}
    for r in rows:
        assert r["intent"] == _by_prefix(r["text"]) and r["focus"] == "chat|quick_action" and r["lang"] == "en"
        gen_name, lab_name = r["source"].split(">")
        assert gen_name != lab_name  # never labelled by the model that wrote it


async def test_a_cycle_never_writes_more_than_the_cap():
    gen = _counter()
    a, b = Model(gen, _by_prefix), Model(gen, _by_prefix)
    pairs = [g.parse_focus("chat:quick_action"), g.parse_focus("stop:quick_action")]
    rows = await g.focus(pairs, _models(a, b), set(), cap=10_000, sleep=_no_sleep)
    assert len(rows) == g.CYCLE_CAP == 300
    assert len({r["text"] for r in rows}) == 300
    by_side = {}
    for r in rows:
        by_side[(r["focus"], r["intent"])] = by_side.get((r["focus"], r["intent"]), 0) + 1
    assert set(by_side.values()) == {75}  # both sides of both pairs, evenly


async def test_known_and_repeated_sentences_are_not_generated_again():
    a = Model(lambda key, n: [f"{key} #1", f"{key} #1", "chat #old"], _by_prefix)
    rows = await g.focus([g.parse_focus("chat:stop")], _models(a, a), set(), seen={"chat #old"}, sleep=_no_sleep)
    assert sorted(r["text"] for r in rows) == ["chat #1", "stop #1"]


async def test_prompts_carry_the_pair_and_the_shapes_of_its_confusions_but_no_user_words():
    gen = _counter()
    a, b = Model(gen, _by_prefix), Model(gen, _by_prefix)
    pair = g.parse_focus("chat:quick_action")
    pair.examples = {"chat": ["remind me what Zanzibarella said about lunch"]}
    vocab = {"remind", "me", "what", "said", "about"}
    await g.focus([pair], _models(a, b), vocab, cap=4, sleep=_no_sleep)
    gen_prompts = [p for p in a.prompts + b.prompts if p.startswith("You write")]
    assert gen_prompts and all("quick_action" in p and "chat" in p for p in gen_prompts)
    assert any("remind me what {x} said about {x}" in p for p in gen_prompts)
    assert not any("zanzibarella" in p.lower() or "lunch" in p for p in a.prompts + b.prompts)


async def test_task_kinds_keep_a_row_only_when_both_models_give_the_same_route():
    def gen(key, n):
        return [[f"{key} #1", "agent_task"], [f"{key} #2", "agent_task"], [f"{key} #3", "quick_action"]]

    def label(text):
        return [_by_prefix(text), "quick_action" if text.endswith("#2") else "agent_task" if text.endswith("#1") else "quick_action"]

    a = Model(gen, label)
    rows = await g.focus([g.parse_focus("kind:send_message:send_email")], _models(a, a), set(), cap=10, sleep=_no_sleep)
    got = {(r["text"], r["intent"], r["kind"]) for r in rows}
    assert got == {
        ("kind:send_message #1", "agent_task", "send_message"), ("kind:send_message #3", "quick_action", "send_message"),
        ("kind:send_email #1", "agent_task", "send_email"), ("kind:send_email #3", "quick_action", "send_email"),
    }  # fmt: skip
    label_sys = [p for p in a.prompts if p.startswith("Classify")]
    assert label_sys and "[class, route]" in label_sys[0]


async def test_hinglish_destructive_tasks():
    def label(text):
        return [_by_prefix(text), "quick_action" if text.endswith("#9") else "agent_task"]

    a = Model(lambda key, n: [f"{key} #1", f"{key} #9"], label)
    pair = g.parse_focus(g.HINGLISH_WEAK[2])
    rows = await g.focus([pair], _models(a, a), set(), cap=10, lang="hinglish", sleep=_no_sleep)
    # "#9" was labelled a quick action: not a task, so not a row of either side
    assert {(r["text"], r["destructive"], r["intent"], r["lang"]) for r in rows} == {
        ("destructive:yes #1", True, "agent_task", "hinglish"), ("destructive:no #1", False, "agent_task", "hinglish"),
    }  # fmt: skip
    assert all("Hinglish only" in p for p in a.prompts if p.startswith("You write"))
    assert [g.parse_focus(s).a.key for s in g.HINGLISH_WEAK] == ["stop", "chat", "destructive:yes"]


async def test_a_labeller_that_skips_an_item_labels_nothing_in_that_batch(monkeypatch):
    monkeypatch.setattr(g, "_tool_description", lambda name: f"does {name}")

    class Skips(Model):
        async def complete(self, messages, *, system="", **kw):
            turn = await super().complete(messages, system=system, **kw)
            if not system.startswith("You write"):
                turn.text = json.dumps(json.loads(turn.text)[1:])  # one label short
            return turn

    a = Skips(lambda key, n: [f"{key} #1", f"{key} #2"], _by_prefix)
    assert await g.focus([g.parse_focus("chat:stop")], _models(a, a), set(), cap=4, sleep=_no_sleep) == []
    b = Model(lambda key, n: [f"{key} #1"], lambda text: [text.split(" #")[0].split(":", 1)[-1], "quick_action"])
    rows = await g.focus([g.parse_focus("tool:open_app:web_search")], _models(b, b), set(), cap=4, sleep=_no_sleep)
    assert {r["tool"] for r in rows} == {"open_app", "web_search"}  # "open_app" alone names tool:open_app


async def test_one_model_cannot_agree_with_itself():
    with pytest.raises(ValueError):
        await g.focus([g.parse_focus("chat:stop")], [("gemini", Model(), 0.0)], set())


def test_labels_stay_aligned_with_their_utterances():
    assert g._items(["chat", 3, ["tool:x", "quick_action"], None, "stop"]) == [
        ("chat", None), ("", None), ("tool:x", "quick_action"), ("", None), ("stop", None)
    ]  # fmt: skip


async def test_cli_focus_run_writes_tagged_rows_and_keeps_user_words_private(monkeypatch):
    from neo.config import settings
    from neo.reflex import learn

    public = [{"text": t, "intent": "chat", "source": "template"} for t in (
        "what did you say about it", "what did the plan say", "did you say that", "what about it",
        "tell me about it", "is it about the plan",
    )]  # fmt: skip
    learn._path(learn.EXTRA).write_text(
        json.dumps({"text": "what did Zanzibarella say about it", "intent": "chat", "wrong": ["quick_action"], "ts": 9e9}) + "\n"
    )
    gen = _counter()
    a, b = Model(gen, _by_prefix), Model(gen, _by_prefix)
    monkeypatch.setattr(g, "_models", lambda: _models(a, b))
    monkeypatch.setattr(g, "build_templates", lambda: public)
    monkeypatch.setattr("sys.argv", ["generate", "--focus", "chat:quick_action", "--max-rows", "6"])
    await g.main()
    rows = [json.loads(line) for line in (settings().data_dir / "reflex_generated.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and {r["focus"] for r in rows} == {"chat|quick_action"}
    assert any("what did {x} say about it" in p for p in a.prompts + b.prompts)
    assert not any("zanzibarella" in p.lower() for p in a.prompts + b.prompts)


def test_errors_never_show_an_api_key(monkeypatch):
    from neo import config

    monkeypatch.setenv("GROQ_API_KEY", "gsk_live_secret_123")
    monkeypatch.setattr(config, "_settings", None)
    msg = g.safe_error(RuntimeError("401 invalid key gsk_live_secret_123 for model"))
    assert "gsk_live_secret_123" not in msg and "***" in msg
