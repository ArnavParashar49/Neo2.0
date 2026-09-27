"""The resolver (neo.tools.computer.resolve) and its phrase table (phrases.py): which on-screen control
a request in English, Hinglish or Hindi means, which stages may act, and the safety rules on top of
the unchanged gate. Hermetic: fake candidates, a fake string index, fake MiniLM and Laya."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from neo.tools.computer import phrases
from neo.tools.computer import resolve as rs

C = rs.Candidate


def cands(*labels: str, role: str = "AXButton", modal: tuple[str, ...] = ()) -> list[rs.Candidate]:
    return [C(lab, role, modal=lab in modal, box=(0.0, 40.0 * i, 80.0, 20.0), id=f"e{i}") for i, lab in enumerate(labels)]


class Index:
    """A fake OS string index: English label → Hindi forms, and back."""

    def __init__(self, pairs: dict[str, list[str]], ready: bool = True) -> None:
        self.f = pairs
        self.r: dict[str, list[str]] = {}
        for en, his in pairs.items():
            for h in his:
                self.r.setdefault(h, []).append(en)
        self._ready = ready

    def ready(self) -> bool:
        return self._ready

    def forms(self, label: str) -> list[str]:
        return self.f.get(label, [])

    def sources(self, label: str) -> list[str]:
        return self.r.get(label, [])


def fake_embed(table: dict[str, list[float]]):
    """Unit vectors from a table (unknown texts: a vector orthogonal to everything)."""
    dim = len(next(iter(table.values()))) + 1
    calls: list[list[str]] = []

    def embed(texts):
        calls.append(list(texts))
        out = []
        for t in texts:
            v = np.array(table.get(t, [0.0] * (dim - 1)) + [0.0 if t in table else 1.0], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    embed.calls = calls  # type: ignore[attr-defined]
    return embed


def fake_choose(pick_label: str | None, p: float):
    seen: list[tuple[str, list[str]]] = []

    def choose(request, options, wait_s):
        seen.append((request, list(options)))
        if pick_label is None or pick_label not in options:
            return None, 0.0
        return options.index(pick_label), p

    choose.seen = seen  # type: ignore[attr-defined]
    return choose


@pytest.fixture
def live_meaning(monkeypatch):
    """resolver = live with the meaning stage promoted."""
    monkeypatch.setattr(rs, "mode", lambda: "live")
    monkeypatch.setattr(rs, "live_stages", lambda: frozenset({"lexical", "meaning", "os_strings"}))


# ---- the frame stripper ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, core",
    [
        ("click on the Start button", "Start"),
        ("please tap the Done button now", "Done"),
        ("can you click reply all", "reply all"),
        ('click the "Sign in" link', "Sign in"),
        ("press the send button please", "send"),
        ("new game pe click karo", "new game"),
        ("new game wala button dabao", "new game"),
        ("start wala button dabao", "start"),
        ("Print… pe tap karo", "Print…"),
        ("save karo", "save"),
        ("नया गेम वाला बटन दबाओ", "नया गेम"),
        ("भेजें पर क्लिक करो", "भेजें"),
        ("सेव करो", "सेव"),
        ("सहेजें चुनो", "सहेजें"),
    ],
)
def test_the_frame_stripper_finds_the_name_in_three_languages(text, core):
    assert phrases.strip_frame(text) == core


@pytest.mark.parametrize(
    "text, core",
    [
        ("click Start", "Start"),
        ("hit play", "play"),  # a press without a control noun
        ("press start", "start"),
        ("new game pe click karo", "new game"),
        ("bravo dabao", "bravo"),
        ("नया गेम दबाओ", "नया गेम"),
        ("भेजें पर क्लिक करो", "भेजें"),
        # not pointing: a key, a question, two things, conversation, a bare "X karo"
        ("press enter", None),
        ("hit cmd+s", None),
        ("press the escape key", None),
        ("select all", None),
        ("what does the share button do?", None),
        ("kya ye button kaam karta hai", None),
        ("ये बटन क्या करता है", None),
        ("is button pe click karo", None),  # "this button": no name (and not a question: "is" is Hindi's "this")
        ("is mail ko forward karo", None),
        ("click reply and send", None),
        ("save karo", None),
        ("pizza order karo", None),
        ("click it", None),
    ],
)
def test_pointing_is_strict(text, core):
    assert phrases.pointing(text) == core


def test_content_words_drop_hindi_verb_endings_and_fold_the_nukta():
    assert phrases.content_words("नया गेम शुरू करो") == {"नया", "गेम", "शुरू"}
    assert phrases.content_words("फ़ाइल खोलो") == phrases.content_words("फाइल")
    assert phrases.content_words("click the start button") == {"start"}


def test_script_and_plain_latin():
    assert phrases.script("नया गेम") == "devanagari" and phrases.script("naya game") == "latin"
    assert phrases.latin_only("Reply All") and not phrases.latin_only("भेजें") and not phrases.latin_only("Löschen")


# ---- the lexical stage (live) ----------------------------------------------------------------------
def test_an_exact_label_resolves_lexically():
    res = rs.resolve("click on the Start button", cands("Start", "Cancel", "New Game"), log_it=False)
    assert isinstance(res, rs.Resolved) and res.label == "Start" and res.stage == "lexical" and res.p == 1.0


@pytest.mark.parametrize("text", ["new game pe click karo", "new game wala button dabao", "click new game", "tap the new game button"])
def test_hinglish_and_english_frames_resolve_the_same_control(text):
    res = rs.resolve(text, cands("Start", "New Game", "Resign"), log_it=False)
    assert isinstance(res, rs.Resolved) and res.label == "New Game"


def test_a_whole_word_of_exactly_one_label():
    res = rs.resolve("click archive", cands("Archive this thread", "Reply", "Forward"), log_it=False)
    assert isinstance(res, rs.Resolved) and res.label == "Archive this thread"
    amb = rs.resolve("click reply", cands("Reply to sender", "Reply to all", "Archive"), log_it=False)
    assert isinstance(amb, rs.Ambiguous) and amb.reason == "several"


def test_the_same_label_on_two_controls_asks_but_a_dialog_wins():
    amb = rs.resolve("click Start", cands("Start", "Start", "Cancel"), log_it=False)
    assert isinstance(amb, rs.Ambiguous) and amb.reason == "same_label" and len(amb.shortlist) == 2
    els = cands("Start", "Cancel", "Start", modal=("Cancel",))
    els[0] = C("Start", "AXButton", modal=True, box=(0, 0, 80, 20), id="sheet-start")
    res = rs.resolve("press start", els, log_it=False)
    assert isinstance(res, rs.Resolved) and res.candidate.id == "sheet-start"


def test_a_button_and_its_own_text_are_one_control():
    els = [C("Save", "AXButton", box=(10, 10, 80, 30), id="b"), C("Save", "AXStaticText", box=(20, 15, 40, 20), id="t")]
    res = rs.resolve("click save", els, log_it=False)
    assert isinstance(res, rs.Resolved) and res.candidate.id == "b"


def test_a_control_beats_a_heading_of_the_same_name_elsewhere():
    els = [C("Settings", "AXStaticText", box=(0, 0, 200, 30), id="h"), C("Settings", "AXButton", box=(0, 300, 80, 20), id="b")]
    res = rs.resolve("click settings", els, log_it=False)
    assert isinstance(res, rs.Resolved) and res.candidate.id == "b"


def test_a_few_words_inside_a_text_are_no_controls_name():
    els = [C("Welcome back", "AXStaticText", box=(0, 0, 200, 20)), C("पीछे", "AXButton", box=(0, 40, 80, 20))]
    assert isinstance(rs.resolve("tap back", els, index=False, log_it=False), rs.Ambiguous)
    rows = [C("Inbox — 3 unread", "AXRow", box=(0, 0, 200, 20)), C("Sent", "AXRow", box=(0, 40, 200, 20))]
    res = rs.resolve("click inbox", rows, log_it=False)
    assert isinstance(res, rs.Resolved) and res.label.startswith("Inbox")  # a row or a cell still counts
    heading = rs.resolve("click welcome back", els, index=False, log_it=False)
    assert isinstance(heading, rs.Resolved)  # the whole text, exactly: as today


@pytest.mark.parametrize("text", ["click it", "click the", "tap on", "click"])
def test_vague_words_are_never_a_label(text):
    assert isinstance(rs.resolve(text, cands("It", "The", "On"), log_it=False), rs.Ambiguous)


def test_no_candidates_is_nothing():
    amb = rs.resolve("click start", [], log_it=False)
    assert isinstance(amb, rs.Ambiguous) and amb.reason == "nothing" and not amb


# ---- candidates ----------------------------------------------------------------------------------
class El:
    def __init__(self, role, title="", value="", *, x=0.0, y=0.0, w=80.0, h=20.0, enabled=True, subrole="", eid="e1", desc=""):
        self.role, self.title, self.value, self.x, self.y, self.w, self.h = role, title, value, x, y, w, h
        self.enabled, self.subrole, self.id, self.desc = enabled, subrole, eid, desc


def test_candidates_keep_only_what_can_be_pressed_and_what_the_caller_allows():
    els = [
        El("AXButton", "Start", eid="ok"),
        El("AXButton", "Disabled", enabled=False),
        El("AXButton", "Zero", w=0),
        El("AXButton", "Far away", x=5000, y=5000),
        El("AXTextField", "Password", subrole="AXSecureTextField"),
        El("AXButton", "", eid="empty"),
        El("AXButton", "Protected", eid="prot"),
        El("AXWindow", "A window"),
    ]
    out = rs.candidates(els, frame=(0, 0, 1000, 800), skip=lambda e: e.id == "prot")
    assert [c.label for c in out] == ["Start"]


def test_a_skip_check_that_raises_keeps_the_element_out():
    def boom(e):
        raise RuntimeError("can't tell")

    assert rs.candidates([El("AXButton", "Start")], skip=boom) == []


def test_a_fields_label_is_its_title_never_its_value():
    field_ = El("AXTextField", "", value="my secret draft", desc="Search")
    got = rs.candidates([field_], want="field")
    assert [c.label for c in got] == ["Search"]
    assert rs.candidates([El("AXTextField", "", value="typed text")], want="field") == []


def test_modal_from_ids_or_the_elements_flag():
    a, b = El("AXButton", "Start", eid="s1"), El("AXButton", "Start", eid="s2", y=300)
    assert [c.modal for c in rs.candidates([a, b], modal={"s2"})] == [False, True]
    b.sheet = True
    assert [c.modal for c in rs.candidates([a, b])] == [False, True]


# ---- safety ---------------------------------------------------------------------------------------
def test_a_destructive_request_on_a_harmless_label_abstains(live_meaning):
    els = cands("Archive", "Reply", "Flag")
    emb = fake_embed({"trash it": [1, 0, 0], "Archive": [0.9, 0.1, 0], "Reply": [0, 1, 0], "Flag": [0, 0, 1]})
    res = rs.resolve("delete it", els, embed=emb, choose=fake_choose("Archive", 0.95), log_it=False)
    assert isinstance(res, rs.Ambiguous)


def test_a_destructive_request_on_a_commit_label_resolves_and_says_so():
    res = rs.resolve("delete kar do", cands("Delete", "Archive"), log_it=False)
    assert isinstance(res, rs.Resolved) and res.label == "Delete" and res.commits


def test_a_lexical_match_that_mismatches_the_request_abstains():
    # the reflex's destructive head says this request destroys something; the label doesn't commit
    amb = rs.resolve("click draft", cands("Draft", "Inbox"), destructive=True, log_it=False)
    assert isinstance(amb, rs.Ambiguous) and amb.reason == "destructive_mismatch"
    assert [c.label for c in amb.shortlist][0] == "Draft"
    ok = rs.resolve("click draft", cands("Draft", "Inbox"), destructive=False, log_it=False)
    assert isinstance(ok, rs.Resolved)


def test_a_hindi_label_without_the_os_index_fails_closed():
    res = rs.resolve("भेजें दबाओ", cands("भेजें", "रद्द करें"), index=False, log_it=False)
    assert isinstance(res, rs.Resolved) and res.ask == "unreadable_label"


def test_a_hindi_label_with_its_english_source_is_read_by_the_gate():
    idx = Index({"Send": ["भेजें"], "Cancel": ["रद्द करें"]})
    send = rs.resolve("भेजें दबाओ", cands("भेजें", "रद्द करें"), index=idx, log_it=False)
    assert isinstance(send, rs.Resolved) and send.commits and send.ask == "commits"  # the gate's regex can't read it
    cancel = rs.resolve("रद्द करें दबाओ", cands("भेजें", "रद्द करें"), index=idx, log_it=False)
    assert isinstance(cancel, rs.Resolved) and not cancel.commits and cancel.ask == ""


def test_an_index_that_isnt_ready_counts_as_none():
    idx = Index({"Send": ["भेजें"]}, ready=False)
    res = rs.resolve("भेजें दबाओ", cands("भेजें"), index=idx, log_it=False)
    assert isinstance(res, rs.Resolved) and res.ask == "unreadable_label"


@pytest.mark.parametrize("label", ["Forward", "Restart", "Shut Down…", "Force Quit", "Quit", "Don't Save", "Revert"])
def test_labels_with_consequences_the_gate_regex_misses_are_asked(label):
    res = rs.resolve(f"click {label}", cands(label, "Other"), log_it=False)
    assert isinstance(res, rs.Resolved) and res.commits and res.ask == "commits"


def test_the_gate_still_decides_the_labels_it_reads():
    res = rs.resolve("click send", cands("Send", "Cancel"), log_it=False)
    assert isinstance(res, rs.Resolved) and res.commits and res.ask == ""  # the frozen gate asks by itself


# ---- the OS strings stage (shadow) ----------------------------------------------------------------------
def test_a_devanagari_request_matches_a_labels_os_hindi_form(live_meaning):
    idx = Index({"New Game": ["नया गेम"], "Resign": ["इस्तीफ़ा दें"]})
    res = rs.resolve("नया गेम शुरू करो", cands("New Game", "Resign", "Hint"), index=idx, log_it=False)
    assert isinstance(res, rs.Resolved) and res.stage == "os_strings" and res.label == "New Game"


def test_a_latin_request_on_a_hindi_window_goes_through_the_english_source(live_meaning):
    idx = Index({"New Game": ["नया गेम"], "Hint": ["संकेत"]})
    res = rs.resolve("click new game", cands("नया गेम", "संकेत"), index=idx, log_it=False)
    assert isinstance(res, rs.Resolved) and res.stage == "os_strings" and res.label == "नया गेम"


def test_in_shadow_the_os_stage_only_logs():
    idx = Index({"New Game": ["नया गेम"]})
    res = rs.resolve("नया गेम शुरू करो", cands("New Game", "Hint"), index=idx, log_it=False)
    assert isinstance(res, rs.Ambiguous) and res.shadow["picks"] == {"os_strings": "New Game"}


def test_a_devanagari_request_without_the_index_hands_off():
    res = rs.resolve("नया गेम शुरू करो", cands("New Game", "Hint"), index=False, log_it=False)
    assert isinstance(res, rs.Ambiguous) and res.shadow["os_strings"] == "no_index"


# ---- the meaning stage (MiniLM + Laya) ---------------------------------------------------------------------
TABLE = {
    "start the new game": [1.0, 0.1, 0.0],
    "New Game": [0.95, 0.15, 0.0],
    "Start": [0.6, 0.6, 0.2],
    "Resign": [0.0, 0.2, 1.0],
    "Hint": [0.0, 1.0, 0.0],
}


def test_meaning_acts_only_when_laya_agrees_with_minilm_and_is_sure(live_meaning):
    els = cands("Start", "New Game", "Resign", "Hint")
    ok = rs.resolve("start the new game", els, embed=fake_embed(TABLE), choose=fake_choose("New Game", 0.9), index=False, log_it=False)
    assert isinstance(ok, rs.Resolved) and ok.stage == "meaning" and ok.label == "New Game" and ok.p == 0.9
    for p in (0.6, 0.8):  # 0.7 measured 93.5% precise: below the bar
        unsure = rs.resolve("start the new game", els, embed=fake_embed(TABLE), choose=fake_choose("New Game", p), index=False, log_it=False)
        assert isinstance(unsure, rs.Ambiguous) and unsure.reason == "unsure"
    other = rs.resolve("start the new game", els, embed=fake_embed(TABLE), choose=fake_choose("Start", 0.99), index=False, log_it=False)
    assert isinstance(other, rs.Ambiguous)
    assert [c.label for c in other.shortlist][0] == "New Game"  # MiniLM's order, for the model's multiple choice


def test_a_crowded_window_needs_a_surer_laya(live_meaning):
    filler = [f"Row {i}" for i in range(70)]
    els = cands("New Game", "Start", *filler)
    res = rs.resolve("start the new game", els, embed=fake_embed(TABLE), choose=fake_choose("New Game", 0.93), index=False, log_it=False)
    assert isinstance(res, rs.Ambiguous)
    res = rs.resolve("start the new game", els, embed=fake_embed(TABLE), choose=fake_choose("New Game", 0.96), index=False, log_it=False)
    assert isinstance(res, rs.Resolved)


def test_minilm_alone_never_acts_even_when_live(live_meaning):
    els = cands("New Game", "Resign", "Hint")
    res = rs.resolve("start the new game", els, embed=fake_embed(TABLE), choose=fake_choose(None, 0.0), index=False, log_it=False)
    assert isinstance(res, rs.Ambiguous)
    assert res.shadow["picks"].get("minilm_only") == "New Game"  # logged, for the shadow numbers


def test_by_default_meaning_is_shadow_only():
    els = cands("Start", "New Game", "Resign", "Hint")
    res = rs.resolve("start the new game", els, embed=fake_embed(TABLE), choose=fake_choose("New Game", 0.99), index=False, log_it=False)
    assert rs.mode() == "shadow" and rs.live_stages() == {"lexical"}
    assert isinstance(res, rs.Ambiguous) and res.shadow["picks"] == {"meaning": "New Game"}


def test_only_the_users_words_get_the_by_meaning_stages(live_meaning):
    emb = fake_embed(TABLE)
    ch = fake_choose("New Game", 0.99)
    for origin in ("live", "agent", "approval"):
        res = rs.resolve("start the new game", cands("Start", "New Game"), origin=origin, embed=emb, choose=ch, index=Index({"New Game": ["नया गेम"]}), log_it=False)
        assert isinstance(res, rs.Ambiguous)
    assert emb.calls == [] and ch.seen == []
    lex = rs.resolve("click new game", cands("Start", "New Game"), origin="agent", log_it=False)
    assert isinstance(lex, rs.Resolved) and lex.stage == "lexical"


def test_off_mode_is_the_lexical_stage_alone(monkeypatch):
    monkeypatch.setattr(rs, "mode", lambda: "off")
    emb = fake_embed(TABLE)
    res = rs.resolve("start the new game", cands("Start", "New Game"), embed=emb, choose=fake_choose("New Game", 0.99))
    assert isinstance(res, rs.Ambiguous) and res.reason == "off" and emb.calls == []
    from neo.decide import shadow

    assert not shadow.path().exists()


def test_label_vectors_are_cached_in_ram_only(live_meaning):
    emb = fake_embed(TABLE)
    els = cands("Start", "New Game", "Resign", "Hint")
    for _ in range(3):
        rs.resolve("start the new game", els, embed=emb, choose=fake_choose("New Game", 0.9), index=False, log_it=False)
    labels_embedded = [t for call in emb.calls for t in call if t in ("Start", "New Game", "Resign", "Hint")]
    assert sorted(labels_embedded) == ["Hint", "New Game", "Resign", "Start"]  # once each


def test_a_cold_or_busy_laya_is_never_waited_for(monkeypatch):
    from neo.reflex import zeroshot

    monkeypatch.setattr(zeroshot, "loaded", lambda: False)

    def ask(*a, **k):
        raise AssertionError("a cold Laya must not be asked (it would start a 1.7 GB load)")

    monkeypatch.setattr(zeroshot, "ask", ask)
    assert rs._default_choose("start the new game", ["New Game", "Start"], 1.5) == (None, 0.0)
    assert rs.choose("the board", ["a", "b"]) == (None, 0.0)


def test_a_loaded_laya_is_asked_with_the_measured_prompt(monkeypatch):
    from neo.reflex import zeroshot

    got = {}

    def ask(state, questions, wait_s, stale=None):
        got.update(state=state, q=questions, wait=wait_s)
        return {"answers": {"el": {"probabilities": {"Start": 0.2, "New Game": 0.8}}}}

    monkeypatch.setattr(zeroshot, "loaded", lambda: True)
    monkeypatch.setattr(zeroshot, "ask", ask)
    assert rs._default_choose("start the new game", ["Start", "New Game"], 0.2) == (1, 0.8)
    assert got["state"] == {"request": "start the new game"} and got["wait"] == 0.2
    assert got["q"]["el"]["criteria"]["New Game"] == "the on-screen control labelled “New Game”"


def test_the_models_embedder_is_used_only_when_already_loaded(monkeypatch):
    from neo.memory import embed

    monkeypatch.setattr(embed, "ready", lambda: False)
    assert rs._default_embed() is None


# ---- what a handler calls ----------------------------------------------------------------------------
class Ctx:
    def __init__(self, origin="user", **extras):
        self.origin, self.extras = origin, extras

    def get(self, k, default=None):
        return self.extras.get(k, default)


def test_ground_takes_the_lanes_utterance_or_the_label_and_the_callers_origin():
    import asyncio

    els = [El("AXButton", "New Game", eid="a"), El("AXButton", "Start", eid="b", y=40), El("AXButton", "Hidden", eid="c", y=80)]
    res = asyncio.run(rs.ground({"utterance": "new game pe click karo", "label": "new game"}, Ctx(), els, skip=lambda e: e.id == "c"))
    assert isinstance(res, rs.Resolved) and res.element is els[0]
    res = asyncio.run(rs.ground({"label": "Hidden"}, Ctx("agent"), els, skip=lambda e: e.id == "c"))
    assert isinstance(res, rs.Ambiguous)  # the caller's protected check keeps it out
    res = asyncio.run(rs.ground({"label": "Start"}, Ctx("user", destructive=True), els))
    assert isinstance(res, rs.Ambiguous) and res.reason == "destructive_mismatch"  # the reflex's head counts
    assert rs.utterance_of({"label": " Start "}) == "Start"


def test_a_signature_carries_chrome_labels_and_the_bundle_only():
    res = rs.resolve("click new game", cands("New Game"), log_it=False)
    sig = rs.signature(res, bundle="org.example.board", changed=True, expect={"sheet": ["Start"]})
    assert sig == {"kind": "press", "role": "AXButton", "label": "New Game", "path": [], "bundle": "org.example.board",
                   "changed": True, "expect": {"sheet": ["Start"]}, "alias": False}  # fmt: skip
    row = rs.resolve("click priya", [C("Priya Sharma", "AXRow", box=(0, 0, 9, 9))], log_it=False)
    assert rs.signature(row, bundle="b", changed=True)["label"] == ""


def test_handoff_choices_are_ids_labels_and_roles():
    amb = rs.Ambiguous([C("Start", "AXButton", modal=True, box=(1, 2, 3, 4), id="e1")], "same_label")
    assert rs.handoff_choices(amb) == [{"id": "e1", "label": "Start", "role": "AXButton", "modal": True}]


# ---- the handoff ---------------------------------------------------------------------------------------
def test_the_handoff_is_a_multiple_choice_of_ids_and_quoted_labels_with_no_coordinates():
    els = [C("Start", "AXButton", modal=True, box=(412.0, 305.0, 80.0, 24.0), id="e12"), C("Start", "AXButton", box=(40.0, 600.0, 80.0, 24.0), id="e40")]
    amb = rs.Ambiguous(els, "same_label")
    text = rs.handoff_text(amb, "click start")
    assert "[e12]" in text and "“Start”" in text and "in the open dialog" in text
    assert not re.search(r"\d+\s*[,x×]\s*\d+|\b(?:x|y)\s*[=:]|box_2d|412|305|600", text)
    assert "ax_tree" in rs.handoff_text(rs.Ambiguous([], "no_match"))


def test_at_most_six_choices():
    els = cands(*[f"Reply {i}" for i in range(12)])
    amb = rs.resolve("click reply", els, log_it=False)
    assert isinstance(amb, rs.Ambiguous) and len(amb.shortlist) <= rs.SHORTLIST


# ---- matches() and choose() for the eyes' verification ----------------------------------------------------
def test_matches_scores_words_on_screen_against_the_request():
    assert rs.matches("start a new game", "Start a New Game?", index=False, embed=False) == 1.0
    assert rs.matches("new game pe click karo", "New Game", index=False, embed=False) == 1.0
    assert rs.matches("open the settings", "Untitled — Notes", index=False, embed=False) == 0.0
    assert 0.0 < rs.matches("start the new game", "New Game", index=False, embed=False) < 1.0
    idx = Index({"New Game": ["नया गेम"]})
    assert rs.matches("नया गेम शुरू करो", "New Game", index=idx, embed=False) >= 0.6


# ---- the shadow log -------------------------------------------------------------------------------------
def _log_lines():
    from neo.decide import shadow

    p = shadow.path()
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_each_resolution_is_logged_with_chrome_labels_only(live_meaning):
    rs.resolve("click new game", cands("New Game", "Start"))
    rs.resolve("click priya sharma", [C("Priya Sharma", "AXRow", box=(0, 0, 100, 20)), C("Other", "AXRow", box=(0, 40, 100, 20))])
    lines = [x for x in _log_lines() if x["kind"] == "ui_resolve"]
    assert len(lines) == 2
    assert lines[0]["acted"] == "lexical" and lines[0]["label"] == "New Game" and lines[0]["lexical_label"] == "New Game"
    assert lines[1]["acted"] == "lexical" and "label" not in lines[1] and "lexical_label" not in lines[1]
    assert "Priya" not in json.dumps(lines)


def test_outcomes_say_which_stage_was_right_and_feed_the_shadow_report():
    from neo.reflex import evaluate

    emb = fake_embed(TABLE)
    res = rs.resolve("start the new game", cands("Start", "New Game", "Resign", "Hint"), embed=emb, choose=fake_choose("New Game", 0.95), index=False)
    assert isinstance(res, rs.Ambiguous)
    assert rs.outcome(res.rid, "New Game", verified=True) == {"meaning_right": True}
    assert rs.outcome(res.rid, "New Game", verified=True) is None  # once
    rep = evaluate.shadow_report()
    assert rep["meaning"] == {"n": 1, "precision": 1.0} and rep["ship meaning"] is False  # far from SHIP_N answers


def test_nothing_is_logged_in_library_mode(monkeypatch):
    from neo import runtime

    monkeypatch.setattr(runtime, "library_bound", lambda: True)
    rs.resolve("click new game", cands("New Game"))
    assert _log_lines() == []


def test_the_privacy_erase_forgets_vectors_and_recent_picks(live_meaning):
    emb = fake_embed(TABLE)
    res = rs.resolve("start the new game", cands("Start", "New Game"), embed=emb, choose=fake_choose("New Game", 0.2), index=False)
    rs.forget()
    assert rs.outcome(res.rid, "New Game", True) is None and rs._vectors._d == {}


# ---- settings -------------------------------------------------------------------------------------------
def test_promotion_needs_live_mode_and_the_stage_listed(monkeypatch):
    from neo import config

    class S:
        resolver = "live"
        resolver_promoted = "meaning, os_strings, bogus"

    monkeypatch.setattr(config, "settings", lambda: S)
    assert rs.live_stages() == {"lexical", "meaning", "os_strings"}
    S.resolver = "shadow"
    assert rs.live_stages() == {"lexical"}
    S.resolver = "nonsense"
    assert rs.mode() == "shadow"


# ---- the benchmark -----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def default_report():
    from neo.reflex import evaluate

    return evaluate.ground_eval()


def test_the_live_stage_clears_the_ship_bar_on_the_benchmark(default_report):
    rep = default_report
    assert rep["rows"] >= 900
    assert rep["all"]["precision"] >= 0.97 and rep["all"]["acted"] >= 1000
    assert rep["script latin"]["precision"] >= 0.95 and rep["script devanagari"]["precision"] >= 0.95
    assert rep["ship"]["lexical"] is True


def test_every_commit_label_acted_on_reaches_the_gate(default_report):
    assert default_report["commit labels acted on"] > 50
    assert default_report["commit labels not flagged for the gate"] == 0


def test_without_the_index_every_hindi_label_is_asked_first():
    from neo.reflex import evaluate

    head, rows = evaluate.load_ground()
    hindi_ui = [r for r in rows if r["type"] == "hindi_ui_hi"]
    rep = evaluate.ground_eval(rows=hindi_ui, index=False)
    assert rep["all"]["acted"] > 0 and rep["asked first (label the gate can't read)"] == rep["all"]["acted"]


def test_the_lexical_stage_is_fast(default_report):
    lat = default_report["latency ms"]
    assert lat["200"]["p90"] < 25.0 and lat["12"]["p50"] < 5.0  # measured ~0.6 / ~0.1 ms


def test_the_benchmark_file_is_self_contained():
    from neo.reflex import evaluate

    head, rows = evaluate.load_ground()
    assert head["os"]["New Game"] == ["नया गेम"]
    assert {r["script"] for r in rows} == {"latin", "devanagari"}
    assert {r["source"] for r in rows} >= {"generated", "os", "hand", "selftest"}
    for r in rows:
        for size, _cs, gold in evaluate.ground_windows(head, r):
            assert gold, (r["id"], size)  # the right control is on every screen


# ---- no hardcoding ------------------------------------------------------------------------------------------
# App names that are no English word anywhere, and ones that are (checked as capitalised names in
# string constants only: "notes" is also a variable).
_APP_NAMES = re.compile(
    r"\b(?:chess|safari|slack|spotify|chrome|xcode|figma|discord|whatsapp|textedit|keynote)\b|\bcom\.[a-z]+\.[a-z]+", re.I
)
_APP_WORDS = re.compile(r"\b(?:Mail|Notes|Calendar|Messages|Music|Photos|Preview|Pages|Numbers|Terminal|Finder|Zoom)\b")


@pytest.mark.parametrize("mod", ["neo/tools/computer/resolve.py", "neo/tools/computer/phrases.py", "neo/agent/fastpath.py"])
def test_no_app_names_or_bundle_ids_in_the_resolvers_code(mod):
    import ast

    src = (Path(__file__).resolve().parents[1] / mod).read_text()
    tree = ast.parse(src)
    doc_nodes = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_nodes]
    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)] + [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert [s for s in strings + names if _APP_NAMES.search(s)] == []
    assert [s for s in strings if _APP_WORDS.search(s)] == []
