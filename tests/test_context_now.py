"""in_front (neo2.core.tools.context_now): "this" — what the user selected, copied or has in front of
them, read only when their own words ask for it, fenced as untrusted content, and never kept.

Accessibility, the pasteboard, the window server and Finder are all fakes here: nothing reads this
Mac's screen or clipboard (the real readers are refused while effects are off: neo2/core/effects.py).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from neo2.core.agent.registry import LIVE_CALL, USERS_WORDS, ToolContext, catalog, registry
from neo2.core.reflex.schema import Decision
from neo2.core.tools.context_now import PRIVATE_TYPES

ASKED = "summarize what I copied"
STRING = "public.utf8-plain-text"
CONCEALED = "org.nspasteboard.ConcealedType"


# ---- fakes ----------------------------------------------------------------------------------------


class Item:
    """An NSPasteboardItem: its types, in order, each with its bytes (each read of them noted). A type
    whose bytes are None is one its app doesn't give (a lazy provider that failed, a Universal
    Clipboard fetch that timed out, a paste prompt answered "Don't Allow")."""

    def __init__(self, pairs, looked=None):
        self.pairs, self.looked = dict(pairs), looked

    def types(self):
        return list(self.pairs)

    def dataForType_(self, t):
        if self.looked is not None:
            self.looked.append(t)
        return self.pairs.get(t)


class Pasteboard:
    """NSPasteboard as in_front asks it: its types first, its contents only then. Its items (every
    type's bytes) and change count are what a copy keeps and puts back. The change count moves as a
    Mac's does: on clearing only (clearContents returns it) — never on writeObjects_."""

    def __init__(self, types=(), text=None, behavior=0, items=None):
        if items is None:
            items = [{t: (text or "").encode() if t == STRING else b"" for t in types}] if types else []
        self.items = [dict(i) for i in items]
        self.behavior, self.count = behavior, 1
        self.read: list[str] = []  # strings read
        self.looked: list[str] = []  # types whose bytes were read (kept, to put back)
        self.writes = 0  # NEO's own writes (putting it back): each clearing and each writeObjects_
        self.refuse_writes = 0  # this many writeObjects_ calls answer False, writing nothing

    def types(self):
        out: list[str] = []
        for i in self.items:
            out += [t for t in i if t not in out]
        return out

    def accessBehavior(self):
        return self.behavior

    def changeCount(self):
        return self.count

    def pasteboardItems(self):
        return [Item(i, self.looked) for i in self.items]

    def stringForType_(self, t):
        self.read.append(t)
        return next((i[t].decode() for i in self.items if t in i), None)

    def clearContents(self):
        self.items, self.count, self.writes = [], self.count + 1, self.writes + 1
        return self.count

    def writeObjects_(self, objs):
        self.writes += 1
        if self.refuse_writes:
            self.refuse_writes -= 1
            return False
        self.items += [{t: bytes(o.dataForType_(t)) for t in o.types()} for o in objs]
        return True

    def app_copies(self, items):
        """Another app's write (its ⌘C): it clears the clipboard, then writes its copy (not NEO's)."""
        self.items, self.count = [dict(i) for i in items], self.count + 1

    def snapshot(self):
        """Every item, every type in its order, every byte."""
        return [list(i.items()) for i in self.items]


def _el(role, sel=None, sub="", marker="", parent=None):
    """An Accessibility element: sel None = it has no AXSelectedText at all (a web area, a group in it,
    a table); "" = it has one, and nothing is selected in it."""
    return SimpleNamespace(role=role, sub=sub, sel=sel, marker=marker, parent=parent)


class World:
    """The apps, the focused element and the pasteboard the tool sees."""

    def __init__(self):
        self.front = ("Notes", 600)
        self.windows = [("Notes", 600)]
        self.regular = {600, 700}
        self.neo = {400}
        self.protected: set[str] = set()
        self.focused: object | None = _el("AXTextArea", sel="")
        self.window = ("Groceries", "file:///Users/someone/Documents/Lists/Groceries%20list.txt")
        self.web = False
        self.pb = Pasteboard(["public.utf8-plain-text"], "Meeting moved to 4 pm on Thursday.")
        self.files: list[str] = []
        self.image: tuple[int, int] | None = None
        self.finder = ""
        self.pressed: list[int] = []  # ⌘C, by the pid it went to
        self.copies: list[dict] | None = None  # what the app copies for ⌘C (None: nothing)


@pytest.fixture
def world(monkeypatch):
    from neo2.core.tools import context_now as cn
    from neo2.core.tools import load_all
    from neo2.core.tools.computer import apps, ax, gates

    load_all()
    w = World()
    monkeypatch.setattr(cn, "_front", lambda: w.front)
    monkeypatch.setattr(cn, "_window_owners", lambda: list(w.windows))
    monkeypatch.setattr(ax, "regular", lambda pid: pid in w.regular)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid in w.neo)
    monkeypatch.setattr(gates, "_protected", lambda owner, pid=0: owner in w.protected)
    monkeypatch.setattr(cn, "_focused_in", lambda pid: w.focused)
    # The elements are fakes (_el): ax.py's readers answer for them, so the real _roles and
    # _selected_text decide (a password field; a page's selection Accessibility keeps to itself).
    real_attrs, real_attr = ax._attrs, ax._attr

    def attrs(el, names):
        if not isinstance(el, SimpleNamespace):
            return real_attrs(el, names)
        have = {"AXRole": el.role, "AXSubrole": el.sub, "AXSelectedText": el.sel}
        return {n: have.get(n) for n in names}

    def attr(el, name, default=None):
        if not isinstance(el, SimpleNamespace):
            return real_attr(el, name, default)
        return {"AXRole": el.role, "AXParent": el.parent}.get(name, default)

    monkeypatch.setattr(ax, "_attrs", attrs)
    monkeypatch.setattr(ax, "_attr", attr)
    monkeypatch.setattr(cn, "_marker_text", lambda el: el.marker)
    monkeypatch.setattr(cn, "_window_facts", lambda pid: w.window)
    monkeypatch.setattr(cn, "_shows_web", lambda pid: w.web)
    monkeypatch.setattr(cn, "_pasteboard", lambda: w.pb)
    monkeypatch.setattr(cn, "_file_names", lambda pb: list(w.files))
    monkeypatch.setattr(cn, "_image_size", lambda pb: w.image)

    def press(pid):
        w.pressed.append(pid)
        if w.copies is not None:
            w.pb.app_copies(w.copies)

    monkeypatch.setattr(cn, "_press_copy", press)
    monkeypatch.setattr(cn, "_item", lambda pairs: Item(pairs))
    monkeypatch.setattr(cn, "COPY_WAIT_S", 0.05)  # (an app that copies nothing: a short wait)
    monkeypatch.setattr(cn, "LATE_WATCH_S", 0.05)  # (and a short watch for a late copy after it)
    w.scripts = []

    async def osa(script, timeout=30):
        w.scripts.append(script)
        return w.finder

    monkeypatch.setattr(apps, "osascript", osa)
    yield w
    _settled()  # (no copy's late watch runs on into the next test)


def _settled():
    """A copy's late watch is over: it holds the clipboard lock until then."""
    from neo2.core.tools import context_now as cn

    assert cn._CLIPBOARD.acquire(timeout=5)
    cn._CLIPBOARD.release()


def _call(what="auto", words=ASKED, origin="agent", user_text=None, extras=None):
    if extras is None:  # what NEO itself marks: the Session the user's words, live.py its direct call
        extras = (
            {USERS_WORDS: words}
            if origin in ("agent", "user")
            else {LIVE_CALL: True}
            if origin == "live"
            else {}
        )
    ctx = ToolContext(user_text=words if user_text is None else user_text, origin=origin, extras=extras)
    out = asyncio.run(registry().invoke("in_front", {"what": what}, ctx))
    return out, ctx


# ---- the origin rule: the user's own words, or nothing ---------------------------------------------


def test_the_users_words_read_the_clipboard_fenced_and_tainting(world):
    out, ctx = _call()
    assert out.ok and "Meeting moved to 4 pm on Thursday." in out.text
    assert ctx.taint == ["in_front"]  # what was read counts: a send or a link next asks first
    t = registry().get("in_front")
    assert t.untrusted and t.parallel_safe and "macos" in t.requires
    # (not read-only: a selection Accessibility doesn't give is copied with ⌘C — tests/test_ctl_seams.py)
    assert t.read_only is False


@pytest.mark.parametrize("asked", ["what's the weather", "open notes", "what is this?", "ye kya hai"])
def test_a_live_call_without_the_users_words_is_refused(world, asked):
    out, ctx = _call(origin="live", words=asked)
    assert not out.ok and "only when you ask" in out.text
    # (it says the words didn't read as being about the selection — never that the user didn't ask)
    assert "didn't read as one about your selection or clipboard" in out.text
    assert world.pb.read == [] and ctx.taint == []


@pytest.mark.parametrize(
    "asked", ["what does this say", "read this to me", "what's this error", "ये क्या लिखा है"]
)
def test_a_live_call_for_a_question_about_what_is_in_front_reads_it_never_the_clipboard(world, asked):
    world.focused.sel = "Unexpected token in JSON at position 4"
    out, _ = _call(what="clipboard", origin="live", words=asked)
    assert not out.ok and world.pb.read == []  # the model asked for the clipboard: never on these words
    out, ctx = _call(what="auto", origin="live", words=asked)
    assert out.ok and "Unexpected token" in out.text and ctx.taint == ["in_front"]
    assert world.pb.read == []


def test_a_live_call_with_the_users_words_runs(world):
    out, _ = _call(origin="live", words="Neo, translate what I just copied into Hindi")
    assert out.ok and "Meeting moved" in out.text


def test_a_live_call_whose_only_words_are_its_own_arguments_is_refused(world):
    # live.py hands a direct tool `asked or str(args)`: with no sentence, the call's own arguments.
    out, _ = _call(
        origin="live", words="", user_text=str({"what": "clipboard", "why": "summarize what I copied"})
    )
    assert not out.ok and world.pb.read == []


def test_a_request_live_hands_the_session_in_its_models_words_never_reads(world):
    # Session.handle(goal, origin="live"): the words are the model's, even when they name the clipboard.
    # Only live.py's own direct call (marked LIVE_CALL) carries the user's sentence.
    out, _ = _call(origin="live", words=ASKED, extras={})
    assert not out.ok and "only when you ask" in out.text and world.pb.read == []


def test_an_agents_call_needs_the_users_own_words_marked_by_the_session(world):
    out, _ = _call(origin="agent", words=ASKED, extras={})  # a model's goal, a Live relay: no USERS_WORDS
    assert not out.ok and "only when you ask" in out.text
    out, _ = _call(origin="agent", words=ASKED, extras={USERS_WORDS: "book a table for two"})
    assert not out.ok  # the user's words don't ask for it
    assert world.pb.read == []


def test_an_approval_replay_never_reads(world):
    out, _ = _call(origin="approval", words=ASKED, extras={USERS_WORDS: ASKED})
    assert not out.ok and world.pb.read == []


def test_switched_off_it_refuses(world, monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "0")
    monkeypatch.setattr(config, "_settings", None)
    out, _ = _call()
    assert not out.ok and "off in NEO's settings" in out.text and world.pb.read == []


def test_an_embedded_agent_never_reads(world, monkeypatch):
    from neo2.core import runtime

    monkeypatch.setattr(runtime, "library_mode", lambda: True)
    out, _ = _call()
    assert not out.ok and world.pb.read == []


# ---- the clipboard -----------------------------------------------------------------------------------


# The markers apps put on a private copy: nspasteboard.org's three, and the older and vendor ones
# clipboard tools still honour (1Password's, KeeWeb's, TypeIt4Me's, Butler's, a generator's).
MARKERS = sorted(PRIVATE_TYPES)


def test_the_private_markers_are_the_ones_clipboard_tools_honour():
    assert {
        "org.nspasteboard.ConcealedType",
        "org.nspasteboard.TransientType",
        "org.nspasteboard.AutoGeneratedType",
        "de.petermaurer.TransientPasteboardType",
        "com.agilebits.onepassword",
        "net.antelle.keeweb",
        "com.typeit4me.clipping",
        "Pasteboard generator type",
    } <= PRIVATE_TYPES


@pytest.mark.parametrize("marker", MARKERS)
def test_a_private_copy_is_refused_unread(world, marker):
    world.pb = Pasteboard(["public.utf8-plain-text", marker], "hunter2")
    out, ctx = _call()
    assert not out.ok and "hunter2" not in out.text and world.pb.read == [] and ctx.taint == []


def test_a_secret_looking_copy_is_refused(world):
    world.pb = Pasteboard(
        ["public.utf8-plain-text"], "my api key is sk-proj-abcdefghijklmnopqrstuvwxyz012345"
    )
    out, ctx = _call()
    assert not out.ok and "sk-proj" not in out.text and ctx.taint == []


def test_a_mac_that_never_lets_apps_read_the_clipboard_is_honoured(world):
    from neo2.core.tools import context_now as cn

    world.pb = Pasteboard(["public.utf8-plain-text"], "a note", behavior=cn._always_deny())
    out, _ = _call()
    assert not out.ok and "never" in out.text and world.pb.read == []


def test_copied_files_are_their_names_only(world):
    world.pb = Pasteboard(["public.file-url", "public.utf8-plain-text"], "/Users/someone/Secret Plans/a.pdf")
    world.files = ["a.pdf", "b.txt"]
    out, _ = _call()
    assert out.ok and "a.pdf" in out.text and "b.txt" in out.text and "Secret Plans" not in out.text


def test_a_copied_picture_is_its_size(world):
    world.pb = Pasteboard(["public.png"])
    world.image = (1200, 800)
    out, _ = _call()
    assert out.ok and "image" in out.text and "1200×800" in out.text


def test_an_empty_clipboard(world):
    world.pb = Pasteboard([])
    out, ctx = _call()
    assert not out.ok and "empty" in out.text and ctx.taint == []


def test_the_clipboard_only_on_the_users_word_for_it(world):
    # The user said "this"; a model asking for the clipboard gets told, and nothing is read.
    world.focused.sel = "Bonjour tout le monde"
    out, _ = _call(what="clipboard", words="translate this")
    assert not out.ok and "clipboard" in out.text and world.pb.read == []


def test_the_text_is_capped(world):
    world.pb = Pasteboard(["public.utf8-plain-text"], "word " * 3000)
    out, _ = _call()
    assert out.ok and len(out.text) <= 4000


# ---- the selection ------------------------------------------------------------------------------------


def test_the_selection_in_the_users_app(world):
    world.focused.sel = "The quick brown fox."
    out, _ = _call(words="translate the selected text")
    assert out.ok and "The quick brown fox." in out.text and "Notes" in out.text
    assert world.pb.read == []  # the selection was asked for, not the clipboard


@pytest.mark.parametrize("sel", ["hunter2", None])
@pytest.mark.parametrize("role, sub", [("AXSecureTextField", ""), ("AXTextField", "AXSecureTextField")])
def test_a_password_field_is_never_read(world, role, sub, sel):
    # A web page's password field too, with nothing Accessibility gives: refused before anything is copied.
    world.focused = _el(role, sel=sel, sub=sub, parent=_el("AXWebArea"))
    world.copies = [{STRING: b"hunter2"}]
    before = world.pb.snapshot()
    out, ctx = _call(words="explain the selected text")
    assert not out.ok and "password field" in out.text and "hunter2" not in out.text and ctx.taint == []
    assert world.pressed == [] and world.pb.looked == [] and world.pb.writes == 0
    assert world.pb.snapshot() == before


@pytest.mark.parametrize("sel", ["private words", None])
def test_a_protected_app_is_never_read(world, sel):
    world.protected = {"Notes"}
    world.focused = _el("AXGroup", sel=sel, parent=_el("AXWebArea"))  # (None: a page's, copied elsewhere)
    world.copies = [{STRING: b"private words"}]
    out, _ = _call(words="translate the selected text")
    assert not out.ok and "private words" not in out.text
    assert world.pressed == [] and world.pb.looked == []


@pytest.mark.parametrize("front", [("NEO", 400), ("NotchHub", 500)])
def test_with_neo_or_an_accessory_panel_in_front_the_users_app_is_read(world, front):
    """Typed in NEO's window, or in a notch app's panel (an accessory app, not a regular one): the
    selection is the one in the app the user works in behind it."""
    world.front = front
    world.windows = [front, ("Pages", 700), ("Notes", 600)]
    world.focused.sel = "Paragraph two."
    out, _ = _call(words="proofread the selected text")
    assert (
        out.ok and "Pages" in out.text and "NotchHub" not in out.text and "NEO" not in out.text.split(":")[0]
    )


def test_no_app_of_the_users_is_said(world):
    world.front, world.windows = ("NEO", 400), [("NEO", 400)]
    out, _ = _call(words="translate the selected text")
    assert not out.ok and "which app" in out.text


def test_a_select_all_is_capped_and_a_secret_at_the_cut_is_still_left_out(world):
    # The secret's value lands just before the 4,000-character cut.
    world.focused.sel = "word " * 740 + "the password is Tr0ub4dor&3 " + "more " * 4000
    out, _ = _call(words="summarize the selected text")
    assert out.ok and len(out.text) <= 4000 and "Tr0ub4dor" not in out.text and "[secret removed]" in out.text


def test_a_secret_in_the_selection_is_left_out(world):
    world.focused.sel = "Login for the wiki: the password is Tr0ub4dor&3, ask Sam."
    out, _ = _call(words="summarize the selected text")
    assert out.ok and "Tr0ub4dor" not in out.text and "ask Sam" in out.text


# ---- the selection Accessibility doesn't give: copied, and the clipboard put back -------------------

CHROME = ("Google Chrome", 700)
# What the user had copied: two items, several types each (one private to its app, raw bytes).
KEPT = [
    {
        STRING: b"Meeting moved to 4 pm.",
        "public.html": b"<b>Meeting</b> moved to 4 pm.",
        "com.example.app-private": b"\x00\x01\xfe\xff\x00",
    },
    {"public.file-url": b"file:///Users/someone/a.pdf", STRING: b"a.pdf"},
]


def _chrome(world, copied="Highlighted on the page."):
    """Chrome in front, as a probe of the real one found it with text highlighted: a group in its web
    area focused, no AXSelectedText on either, and an empty text-marker string."""
    world.front, world.windows = CHROME, [CHROME]
    world.focused = _el("AXGroup", parent=_el("AXWebArea"))
    world.copies = None if copied is None else [{STRING: copied.encode(), "public.html": b"<p>x</p>"}]


def test_a_selection_accessibility_doesnt_give_is_copied_and_the_clipboard_put_back_byte_for_byte(world):
    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    _chrome(world)
    out, ctx = _call(what="selection", words="translate the selected text")
    assert out.ok and "Selected in Google Chrome" in out.text and "Highlighted on the page." in out.text
    assert "Meeting moved" not in out.text and ctx.taint == ["in_front"]  # the copy, never what was there
    assert world.pressed == [700]
    assert world.pb.snapshot() == before  # every item, every type in its order, every byte
    out, _ = _call(what="auto", words="explain this")  # "this": the same
    assert out.ok and "Highlighted on the page." in out.text and world.pb.snapshot() == before


def test_an_empty_clipboard_is_empty_again_after_a_copy(world):
    world.pb = Pasteboard([])
    _chrome(world)
    out, _ = _call(words="explain the selected text")
    assert out.ok and "Highlighted on the page." in out.text
    assert world.pb.items == [] and world.pb.types() == [] and world.pressed == [700]


def test_what_was_copied_is_handled_like_a_selection(world):
    world.pb = Pasteboard(items=KEPT)
    _chrome(world, copied="Login for the wiki: the password is Tr0ub4dor&3, ask Sam. " + "more " * 3000)
    out, _ = _call(words="summarize the selected text")
    assert out.ok and "Tr0ub4dor" not in out.text and "ask Sam" in out.text and len(out.text) <= 4000
    t = registry().get("in_front")
    assert t.untrusted and out.fence  # fenced as data for the model


@pytest.mark.parametrize("front", [("NEO", 400), ("NotchHub", 500)])
def test_the_copy_goes_to_the_users_app_alone_never_neo(world, front):
    _chrome(world)
    world.front, world.windows = front, [front, CHROME]
    out, _ = _call(words="translate the selected text")
    assert out.ok and "Highlighted on the page." in out.text and world.pressed == [700]


@pytest.mark.parametrize("marker", MARKERS)
def test_a_private_clipboard_is_never_copied_over(world, marker):
    # A password manager's copy is never touched — not even kept and put back: no key is pressed.
    world.pb = Pasteboard(items=[{STRING: b"note"}, {STRING: b"hunter2", marker: b""}])
    before = world.pb.snapshot()
    _chrome(world)
    out, ctx = _call(what="selection", words="translate the selected text")
    assert not out.ok and "private" in out.text and "hunter2" not in out.text and ctx.taint == []
    assert world.pressed == [] and world.pb.looked == [] and world.pb.read == [] and world.pb.writes == 0
    out, _ = _call(what="auto", words="explain this")  # "this": the window, saying why
    assert out.ok and "private" in out.text and "Google Chrome" in out.text and world.pressed == []
    assert world.pb.snapshot() == before


def test_a_clipboard_too_big_to_put_back_is_never_copied_over(world, monkeypatch):
    from neo2.core.tools import context_now as cn

    assert cn.CLIPBOARD_CAP_BYTES == 32 * 1024 * 1024
    monkeypatch.setattr(cn, "CLIPBOARD_CAP_BYTES", 16)
    world.pb = Pasteboard(items=[{STRING: b"x" * 10, "public.tiff": b"\x00" * 10}])
    before = world.pb.snapshot()
    _chrome(world)
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "too big" in out.text and world.pressed == [] and world.pb.writes == 0
    assert world.pb.snapshot() == before


def test_a_mac_that_never_lets_apps_read_the_clipboard_is_never_copied_over(world):
    from neo2.core.tools import context_now as cn

    world.pb = Pasteboard([STRING], "a note", behavior=cn._always_deny())
    _chrome(world)
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "never lets NEO read the clipboard" in out.text
    assert world.pressed == [] and world.pb.looked == [] and world.pb.writes == 0


def test_a_copy_marked_private_is_never_read(world):
    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    _chrome(world)
    world.copies = [{STRING: b"correct horse battery staple", CONCEALED: b""}]
    out, ctx = _call(words="translate the selected text")
    assert not out.ok and "private" in out.text and "correct horse" not in out.text and ctx.taint == []
    assert world.pressed == [700] and world.pb.read == []
    assert world.pb.snapshot() == before


def test_when_the_app_copies_nothing_it_is_nothing_selected_and_the_clipboard_as_it_was(world):
    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    _chrome(world, copied=None)
    out, ctx = _call(what="selection", words="translate the selected text")
    assert not out.ok and "Nothing is selected in Google Chrome" in out.text and ctx.taint == []
    assert world.pressed == [700] and world.pb.read == []
    # Nothing changed it: it is left alone — what it holds is exactly what was kept.
    _settled()
    assert world.pb.snapshot() == before and world.pb.writes == 0


def test_an_app_that_empties_the_clipboard_and_copies_nothing_gets_it_put_back(world):
    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    _chrome(world)
    world.copies = []
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "Nothing is selected in Google Chrome" in out.text
    _settled()
    assert world.pb.snapshot() == before and world.pb.writes == 2  # cleared, then written back


def test_an_error_midway_still_puts_the_clipboard_back(world, monkeypatch):
    from neo2.core.tools import context_now as cn
    from neo2.core.tools.computer import ax

    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    _chrome(world)

    def broken(t):
        raise RuntimeError("the pasteboard went away")

    monkeypatch.setattr(world.pb, "stringForType_", broken)
    with pytest.raises(RuntimeError):
        cn.copy_selection(700)
    assert world.pb.snapshot() == before
    out, ctx = _call(words="translate the selected text")
    assert not out.ok and "Highlighted" not in out.text and ctx.taint == []
    assert world.pb.snapshot() == before

    # The key's way refused after the app copied (its key-up, say): put back all the same.
    world.pb = Pasteboard(items=KEPT)

    def refused(pid):
        world.pb.app_copies([{STRING: b"Highlighted on the page."}])
        raise ax.OffLimits("that's NEO's own window")

    monkeypatch.setattr(cn, "_press_copy", refused)
    with pytest.raises(ax.OffLimits):
        cn.copy_selection(700)
    assert world.pb.snapshot() == before


def test_switched_off_a_selection_accessibility_doesnt_give_is_nothing_selected(world, monkeypatch):
    from neo2.core import config

    assert config.settings().context_copy_fallback is True  # on by default
    monkeypatch.setenv("NEO_CONTEXT_COPY_FALLBACK", "0")
    monkeypatch.setattr(config, "_settings", None)
    world.pb = Pasteboard(items=KEPT)
    _chrome(world)
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and out.text == "Nothing is selected in Google Chrome."
    assert world.pressed == [] and world.pb.looked == [] and world.pb.writes == 0


def test_what_accessibility_gives_wins_and_nothing_is_copied(world):
    world.pb = Pasteboard(items=KEPT)
    _chrome(world)
    world.focused.sel = "From Accessibility."
    out, _ = _call(words="translate the selected text")
    assert out.ok and "From Accessibility." in out.text and "Highlighted" not in out.text
    assert world.pressed == [] and world.pb.looked == [] and world.pb.writes == 0


def test_with_no_focused_element_nothing_is_copied(world):
    # (no element to check for a password field: never a ⌘C into what can't be seen)
    _chrome(world)
    world.focused = None
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "Nothing is selected" in out.text and world.pressed == []


def _deep(levels):
    """A text node ``levels`` groups below a web area."""
    node = _el("AXWebArea")
    for _ in range(levels):
        node = _el("AXGroup", parent=node)
    return _el("AXStaticText", parent=node)


@pytest.mark.parametrize(
    "focused",
    [
        _el("AXWebArea"),  # the web area itself
        _el("AXGroup", parent=_el("AXWebArea")),  # a group in it (the user's Chrome)
        _deep(3),
    ],
)
def test_only_a_pages_selection_accessibility_keeps_to_itself_is_copied(world, focused):
    world.pb = Pasteboard(items=KEPT)
    _chrome(world)
    world.focused = focused
    out, _ = _call(what="selection", words="translate the selected text")
    assert out.ok and "Highlighted on the page." in out.text and world.pressed == [700]


@pytest.mark.parametrize(
    "focused",
    [
        _el("AXTextArea", sel=""),  # a text view (a terminal, an editor): nothing selected, it says
        _el("AXTextArea"),  # a text view not answering
        _el("AXTextField", sel="", parent=_el("AXWebArea")),  # a page's field: nothing selected in it
        _el("AXTable"),  # a password manager's list: its ⌘C copies the password
        _el("AXRow", parent=_el("AXOutline")),
        _el("AXGroup", parent=_el("AXSplitGroup")),  # no web area: a native app's
        _deep(8),  # a web area too far up to be the page it is on
    ],
)
def test_nothing_is_copied_where_accessibility_answers_or_isnt_a_page(world, focused):
    # Elsewhere ⌘C may copy something else (a password manager's entry, an editor's line, a terminal's
    # interrupt): only a web page whose selection Accessibility keeps to itself is copied.
    world.pb = Pasteboard(items=KEPT)
    _chrome(world)
    world.focused = focused
    world.copies = [{STRING: b"Tr0ub4dor&3"}]
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and out.text == "Nothing is selected in Google Chrome."
    assert world.pressed == [] and world.pb.looked == [] and world.pb.writes == 0


@pytest.mark.parametrize(
    "items",
    [
        [{STRING: b"keep me", "public.tiff": None}, {"com.example.lazy": None}],  # bytes that don't come
        [{"com.example.lazy": None}],  # its only item: putting it back would wipe the clipboard
        [{STRING: b"note"}, {}],  # an item with no type to put back
    ],
)
def test_a_clipboard_that_cant_be_kept_whole_is_never_copied_over(world, items):
    # A promise its app doesn't keep (a Universal Clipboard fetch that failed, a hung provider, a paste
    # prompt answered "Don't Allow"): kept short, it would be put back short — so nothing is pressed.
    world.pb = Pasteboard(items=items)
    before = world.pb.snapshot()
    _chrome(world)
    out, ctx = _call(what="selection", words="translate the selected text")
    assert not out.ok and "couldn't be kept" in out.text and ctx.taint == []
    assert world.pressed == [] and world.pb.read == [] and world.pb.writes == 0
    assert world.pb.snapshot() == before


def test_a_copy_from_another_device_is_never_copied_over(world):
    # (Universal Clipboard: its bytes are fetched from the phone on demand — never just to keep them)
    world.pb = Pasteboard(items=[{STRING: b"from my phone", "com.apple.is-remote-clipboard": b""}])
    _chrome(world)
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "another device" in out.text
    assert world.pressed == [] and world.pb.looked == [] and world.pb.writes == 0


def test_a_mac_set_to_ask_before_apps_read_the_clipboard_is_never_copied_over(world):
    # (keeping it would ask the user about a clipboard they didn't ask about)
    from neo2.core.tools import context_now as cn

    world.pb = Pasteboard([STRING], "a note", behavior=cn._asks())
    _chrome(world)
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "asks before" in out.text
    assert world.pressed == [] and world.pb.looked == [] and world.pb.writes == 0
    out, _ = _call(what="clipboard", words="summarize what I copied")  # (asked for: the Mac asks them)
    assert out.ok and "a note" in out.text


def test_a_clipboard_slow_to_keep_is_never_copied_over(world, monkeypatch):
    from neo2.core.tools import context_now as cn

    monkeypatch.setattr(cn, "_KEEP_S", -1.0)  # (every type takes longer than keeping it may)
    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    _chrome(world)
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "too long" in out.text and world.pressed == [] and world.pb.writes == 0
    assert world.pb.snapshot() == before


@pytest.mark.parametrize("front", [("NEO", 400), ("NotchHub", 500)])
def test_an_app_behind_that_copies_nothing_is_said_plainly(world, front):
    # Typed in NEO's window, Chrome isn't the active app: its ⌘C may go nowhere. "Nothing is selected"
    # would be a guess — NEO says what it knows.
    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    _chrome(world, copied=None)
    world.front, world.windows = front, [front, CHROME]
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "Google Chrome" in out.text and "isn't the app in front" in out.text
    assert world.pressed == [700]
    out, _ = _call(what="auto", words="explain this")  # "this": the window, saying why
    assert out.ok and "isn't the app in front" in out.text and "In front: Google Chrome" in out.text
    _settled()
    assert world.pb.snapshot() == before and world.pb.writes == 0


def _later(world, delay, items):
    """An app's write ``delay`` seconds from now (a slow copy, a web page's own second write)."""
    import threading

    t = threading.Timer(delay, world.pb.app_copies, ([dict(i) for i in items],))
    t.start()
    return t


def test_a_copy_made_late_is_put_back_too(world, monkeypatch):
    # The app copies after NEO stopped waiting: "nothing selected", and its copy is written over too —
    # by the late watch, the reply not kept waiting for it.
    from neo2.core.tools import context_now as cn

    monkeypatch.setattr(cn, "LATE_WATCH_S", 0.6)
    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    _chrome(world, copied=None)
    late = []

    def press(pid):
        world.pressed.append(pid)
        late.append(_later(world, 0.12, [{STRING: b"late selection"}]))

    monkeypatch.setattr(cn, "_press_copy", press)
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "Nothing is selected in Google Chrome" in out.text
    late[0].join(5)
    _settled()
    assert world.pb.snapshot() == before and world.pb.writes == 2


def test_a_second_write_after_the_copy_is_put_back_too(world, monkeypatch):
    # A web page's own copy handler writes again a moment later (navigator.clipboard.write).
    from neo2.core.tools import context_now as cn

    monkeypatch.setattr(cn, "LATE_WATCH_S", 0.6)
    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    _chrome(world)
    late = []

    def press(pid):
        world.pressed.append(pid)
        world.pb.app_copies([{STRING: b"Highlighted on the page."}])
        late.append(_later(world, 0.1, [{STRING: b"Highlighted (async)", "public.html": b"<p>x</p>"}]))

    monkeypatch.setattr(cn, "_press_copy", press)
    out, _ = _call(what="selection", words="translate the selected text")
    assert out.ok and "Highlighted on the page." in out.text
    late[0].join(5)
    _settled()
    assert world.pb.snapshot() == before and world.pb.writes == 4  # put back, then once more


def test_a_private_copy_made_meanwhile_is_never_written_over(world, monkeypatch):
    # (the user's password manager copied in that second: not the app's copy to undo)
    from neo2.core.tools import context_now as cn

    monkeypatch.setattr(cn, "LATE_WATCH_S", 0.6)
    world.pb = Pasteboard(items=KEPT)
    _chrome(world)
    late = []

    def press(pid):
        world.pb.app_copies([{STRING: b"Highlighted on the page."}])
        late.append(_later(world, 0.1, [{STRING: b"hunter2", CONCEALED: b""}]))

    monkeypatch.setattr(cn, "_press_copy", press)
    out, _ = _call(what="selection", words="translate the selected text")
    assert out.ok and "hunter2" not in out.text
    late[0].join(5)
    _settled()
    assert world.pb.snapshot() == [[(STRING, b"hunter2"), (CONCEALED, b"")]] and world.pb.writes == 2


def test_a_clipboard_tool_that_rewrites_every_copy_is_no_tug_of_war(world, monkeypatch):
    from neo2.core.tools import context_now as cn

    monkeypatch.setattr(cn, "LATE_WATCH_S", 0.3)
    world.pb = Pasteboard(items=KEPT)
    _chrome(world)
    write = world.pb.writeObjects_

    def rewritten(objs):  # (a plain-text tool rewrites whatever lands on the clipboard)
        ok = write(objs)
        world.pb.app_copies([{STRING: b"plain"}])
        return ok

    monkeypatch.setattr(world.pb, "writeObjects_", rewritten)
    out, _ = _call(what="selection", words="translate the selected text")
    assert out.ok and "Highlighted on the page." in out.text
    _settled()
    assert world.pb.writes == 2 * (1 + cn._LATE_PUTS)  # put back, then at most _LATE_PUTS more times


def test_a_clipboard_that_cant_be_written_back_is_said(world, monkeypatch):
    from neo2.core.tools import context_now as cn

    # Refused once: cleared and written again — and it is as it was.
    world.pb = Pasteboard(items=KEPT)
    before = world.pb.snapshot()
    world.pb.refuse_writes = 1
    _chrome(world)
    out, _ = _call(what="selection", words="translate the selected text")
    assert out.ok and "Highlighted on the page." in out.text
    assert world.pb.snapshot() == before and world.pb.writes == 4
    # Refused twice: the user is told (never a quiet success with their clipboard gone).
    world.pb = Pasteboard(items=KEPT)
    world.pb.refuse_writes = 2
    out, ctx = _call(what="selection", words="translate the selected text")
    assert not out.ok and "couldn't put your clipboard back" in out.text and "Highlighted" not in out.text
    assert ctx.taint == []
    # An item that can't be made again: nothing is cleared first, and the user is told.
    world.pb = Pasteboard(items=KEPT)

    def broken(pairs):
        raise ValueError("no such type")

    monkeypatch.setattr(cn, "_item", broken)
    out, _ = _call(what="selection", words="translate the selected text")
    assert not out.ok and "couldn't put your clipboard back" in out.text and world.pb.writes == 0


def test_a_clipboard_read_waits_while_a_copy_has_borrowed_it(world, monkeypatch):
    # (in_front may run beside itself: a read mid-copy would read the copy, a second copy would keep it)
    import threading

    from neo2.core.tools import context_now as cn

    got: list[str] = []
    with cn._CLIPBOARD:
        t = threading.Thread(target=lambda: got.append(cn.read_clipboard()))
        t.start()
        t.join(0.1)
        assert t.is_alive() and got == [] and world.pb.read == []
    t.join(5)
    assert got and "Meeting moved" in got[0]
    seen: list[bool] = []
    press = cn._press_copy
    monkeypatch.setattr(cn, "_press_copy", lambda pid: seen.append(cn._CLIPBOARD.locked()) or press(pid))
    monkeypatch.setattr(cn, "LATE_WATCH_S", 0.5)
    _chrome(world)
    assert cn.copy_selection(700) == "Highlighted on the page."
    # Returned at once, the clipboard still held: its late watch has it (in a thread of its own).
    assert seen == [True] and cn._CLIPBOARD.locked()
    got.clear()
    t = threading.Thread(target=lambda: got.append(cn.read_clipboard()))
    t.start()
    t.join(0.1)
    assert t.is_alive() and got == []
    t.join(5)
    assert got and "Meeting moved" in got[0] and not cn._CLIPBOARD.locked()


def test_a_look_that_cant_have_the_clipboard_says_so(world, monkeypatch):
    # (a copy whose app hangs keeps it: the next look waits a while, then says so — never parks forever)
    from neo2.core.tools import context_now as cn

    monkeypatch.setattr(cn, "_LOCK_WAIT_S", 0.05)
    _chrome(world)
    with cn._CLIPBOARD:
        out, _ = _call()
        assert not out.ok and "busy" in out.text and world.pb.read == []
        out, _ = _call(what="selection", words="translate the selected text")
        assert not out.ok and "busy" in out.text and world.pressed == [] and world.pb.looked == []


def test_the_copy_key_is_cmd_c_for_that_app_alone_through_the_input_door(monkeypatch):
    """The real ⌘C, with only Quartz's posting faked: two events (down, up), each to that pid, C with
    ⌘ alone — and never to NEO's own processes or a protected app."""
    import Quartz

    from neo2.core.tools import context_now as cn
    from neo2.core.tools.computer import ax, gates

    posted: list[tuple[int, int, int]] = []

    def post(pid, ev):
        code = Quartz.CGEventGetIntegerValueField(ev, Quartz.kCGKeyboardEventKeycode)
        posted.append((pid, code, Quartz.CGEventGetFlags(ev)))

    monkeypatch.setattr(Quartz, "CGEventPostToPid", post)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == 400)
    monkeypatch.setattr(gates, "_protected", lambda owner, pid=0: pid == 500)
    cn._press_copy(700)
    others = Quartz.kCGEventFlagMaskShift | Quartz.kCGEventFlagMaskAlternate | Quartz.kCGEventFlagMaskControl
    assert [p for p, _, _ in posted] == [700, 700]
    assert all(
        code == 8 and flags & Quartz.kCGEventFlagMaskCommand and not flags & others
        for _, code, flags in posted
    )
    for pid in (400, 500):
        with pytest.raises(ax.OffLimits):
            cn._press_copy(pid)
    assert len(posted) == 2


def test_an_item_is_put_back_with_every_types_bytes_in_order():
    # The real NSPasteboardItem, in memory only (never written to any pasteboard).
    from neo2.core.tools import context_now as cn

    pairs = tuple((t, d) for t, d in KEPT[0].items())
    item = cn._item(pairs)
    assert [str(t) for t in item.types()] == list(KEPT[0])
    assert [bytes(item.dataForType_(t)) for t in KEPT[0]] == list(KEPT[0].values())


# ---- auto: the user's word first, then the selection, then the window ------------------------------


def test_auto_an_explicit_word_wins(world):
    world.focused.sel = "selected words"
    out, _ = _call(what="auto", words="summarize what I copied")
    assert "Meeting moved" in out.text and "selected words" not in out.text
    # The model's guess says the selection: the user's own word for the clipboard wins.
    out, _ = _call(what="selection", words="summarize what I copied")
    assert "Meeting moved" in out.text


def test_auto_this_is_the_selection_when_there_is_one(world):
    world.focused.sel = "selected words"
    out, _ = _call(what="auto", words="translate this")
    assert out.ok and "selected words" in out.text and world.pb.read == []


def test_auto_with_nothing_selected_is_the_window(world):
    out, _ = _call(what="auto", words="explain this")
    assert out.ok and "Groceries" in out.text and "Groceries list.txt" in out.text
    assert "Documents" not in out.text and "/Users" not in out.text  # the document's name, never its folders
    assert world.pb.read == []


def test_nothing_selected_when_the_selection_was_asked_for(world):
    out, ctx = _call(what="selection", words="translate the selected text")
    assert not out.ok and "Nothing is selected" in out.text and ctx.taint == []


def test_the_window_of_a_browser_points_at_page_read(world):
    world.web = True
    world.window = ("Deals — Today", "https://shop.example/deals?token=abc")
    out, _ = _call(what="window", words="explain this")
    assert out.ok and "page_read" in out.text and "token" not in out.text


# ---- Finder's selection --------------------------------------------------------------------------------


def test_the_finder_selection_is_names(world):
    world.finder = "report.pdf\nbudget.xlsx"
    out, _ = _call(what="auto", words="summarize these files")
    assert out.ok and "report.pdf" in out.text and "budget.xlsx" in out.text
    assert world.pb.read == []


def test_the_finder_script_only_reads():
    from neo2.core.security.policy import script_facts, script_is_read_only
    from neo2.core.tools import context_now as cn

    assert script_is_read_only(cn.FINDER_SCRIPT)
    assert script_facts(cn.FINDER_SCRIPT).app == "Finder"


def test_finder_errors_are_said_plainly(world):
    world.finder = "Error: Not authorized to send Apple events to Finder. (-1743)"
    out, _ = _call(what="files", words="what are the selected files")
    assert not out.ok and "Finder" in out.text and "-1743" not in out.text
    world.finder = ""
    out, _ = _call(what="files", words="what are the selected files")
    assert not out.ok and "Nothing is selected in Finder" in out.text


# ---- the real readers: refused in tests, never reading this Mac ------------------------------------------


def test_the_real_readers_are_effects():
    from neo2.core import effects
    from neo2.core.tools import context_now as cn

    for reader in (
        cn._front,
        cn._window_owners,
        lambda: cn._focused_in(1),
        lambda: cn._window_facts(1),
        cn._pasteboard,
        lambda: cn._press_copy(1),  # ⌘C: refused before any event is made
        lambda: cn.copy_selection(1),  # the clipboard: refused before it is kept
    ):
        with pytest.raises(effects.EffectRefused):
            reader()


def test_the_selected_text_falls_back_to_a_web_pages_text_markers(monkeypatch):
    """A page's selection (Safari, Chromium, Electron) has no AXSelectedText on the focused element:
    its text-marker range, on it or the web area it is in, has the words. When that is empty too, the
    page keeps its selection to itself (Chrome): the one place to copy it."""
    from neo2.core.tools import context_now as cn
    from neo2.core.tools.computer import ax

    area = _el("AXWebArea", marker="Selected on the page.")
    leaf = _el("AXStaticText", parent=area)
    monkeypatch.setattr(ax, "_attrs", lambda el, names: {"AXSelectedText": el.sel, "AXRole": el.role})
    monkeypatch.setattr(
        ax,
        "_attr",
        lambda el, name, default=None: {"AXRole": el.role, "AXParent": el.parent}.get(name, default),
    )
    monkeypatch.setattr(cn, "_marker_text", lambda el: el.marker)
    assert cn._selected_text(leaf) == ("Selected on the page.", False)
    assert cn._selected_text(_el("AXTextArea", sel="Plain selection.")) == ("Plain selection.", False)
    assert cn._selected_text(_el("AXStaticText", parent=_el("AXWebArea"))) == ("", True)  # Chrome's
    assert cn._selected_text(_el("AXTextField", sel="", parent=_el("AXWebArea"))) == ("", False)
    assert cn._selected_text(_el("AXTextArea", sel="")) == ("", False)
    assert cn._selected_text(_el("AXTable")) == ("", False)


# ---- the catalog --------------------------------------------------------------------------------------


def test_the_catalog_has_it_for_neo_app_only():
    from neo2.core import tools
    from neo2.core.tools import REFUSED, load_all, resolve

    load_all()
    assert catalog().get("in_front") is not None
    assert "neo2.core.tools.context_now" in tools._MAC_ONLY
    assert "in_front" in REFUSED and "in_front" not in resolve("*", platform="darwin")
    with pytest.raises(ValueError, match="NEO.app's user"):
        resolve(("web", "in_front"))


# ---- the Session: the user's own words reach a model that has the tool -------------------------------------


class _Reflex:
    def __init__(self, intent="chat"):
        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    async def decide(self, text):
        return self.d


@pytest.fixture
def session_env(world, monkeypatch):
    from neo2.core import events
    from neo2.core.agent import session as sess_mod
    from neo2.core.providers.base import ToolCall, Turn

    monkeypatch.setattr(events, "_bus", events.EventBus())

    class Chain:
        name = "fake"
        supports_vision = False
        supports_tools = True
        streamed = 0
        calls: list = []

        async def stream(self, messages, *, system="", max_tokens=2048):
            Chain.streamed += 1
            yield "a chat reply"

        async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
            Chain.calls.append((list(messages), [t.name for t in tools or []]))
            if len(Chain.calls) == 1 and "in_front" in Chain.calls[0][1]:
                return Turn(tool_calls=[ToolCall("c1", "in_front", {"what": "auto"})])
            return Turn(text="Your meeting moved to Thursday at 4.")

    Chain.calls = []
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": Chain())
    return Chain


def test_the_session_hands_the_users_words_to_a_model_with_in_front_even_when_laya_says_chat(session_env):
    from neo2.core.agent.session import Session

    r = asyncio.run(Session(reflex=_Reflex("chat")).handle(ASKED, from_user=True))
    assert r.route == "agent" and session_env.streamed == 0
    first_tools = session_env.calls[0][1]
    assert "in_front" in first_tools and "page_read" in first_tools
    # The tool ran for the user's words; the model got what it read fenced, and the request is tainted.
    second = session_env.calls[1][0]
    read = [res for m in second for res in m.tool_results if res.name == "in_front"]
    assert read and read[0].content.startswith('<untrusted source="in_front">')
    assert "Meeting moved" in read[0].content and r.taint == ("in_front",)


def test_the_session_hook_also_takes_laya_s_one_tool_lane(session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("quick_action"))
    s.reflex.d = Decision(
        "quick_action", 0.95, False, 0.0, False, 0.0, "fake", tool="calendar_add", tool_p=0.99
    )
    r = asyncio.run(s.handle("add this to my calendar", from_user=True))
    assert r.route == "agent" and "in_front" in session_env.calls[0][1]


@pytest.mark.parametrize(
    "text",
    ["what does this say", "read this to me", "what's this error", "ये कॉपी किया हुआ पढ़ो", "इसे ट्रांसलेट करो"],
)
def test_a_question_about_what_is_in_front_and_hindi_reach_in_front(session_env, text):
    from neo2.core.agent.session import Session

    r = asyncio.run(Session(reflex=_Reflex("chat")).handle(text, from_user=True))
    assert r.route == "agent" and "in_front" in session_env.calls[0][1] and session_env.streamed == 0


@pytest.mark.parametrize(
    "text",
    [
        "tell me a joke",
        "explain quantum physics",
        # the conversation's "this", a statement, and text the sentence carries itself: as before P8
        "explain this answer",
        "explain this in simpler words",
        "summarize this conversation",
        "how can I improve this plan",
        "the selection was great",
        "what I copied yesterday was wrong",
        "translate this to French: hello how are you",
        "add this to my to-do list: buy milk",
    ],
)
def test_other_words_take_their_old_route(session_env, text):
    from neo2.core.agent.session import Session

    r = asyncio.run(Session(reflex=_Reflex("chat")).handle(text, from_user=True))
    assert r.route == "chat" and session_env.streamed == 1 and session_env.calls == []


@pytest.mark.parametrize(
    "said, step",
    [
        ("search what I copied on youtube", "search_site"),
        ("search the selected text on amazon", "search_site"),
        ("play what I copied", "play_music"),
        ("open what I copied", "open_app"),
        ("google what I copied", "web_search"),
    ],
)
def test_a_fast_path_never_runs_on_a_name_for_what_is_in_front(world, monkeypatch, said, step):
    # The query would be the words "what I copied": the text is on the clipboard, so no fast path takes
    # the sentence (the Session's hook, or Gemini Live's model, reads it first with in_front).
    from neo2.core import config
    from neo2.core.agent import fastpath
    from neo2.core.voice import live as live_mod

    assert fastpath.plan(said) is None and live_mod._command_steps(said) is None
    assert fastpath.plan("search cat videos on youtube") is not None  # (a real query still runs)
    monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "0")  # switched off: the words run as they always did
    monkeypatch.setattr(config, "_settings", None)
    plan = fastpath.plan(said)
    assert plan is not None and plan[0][0] == step


@pytest.mark.parametrize("intent", ["quick_action", "agent_task"])
def test_the_session_reads_what_was_copied_before_searching_for_it(session_env, intent):
    from neo2.core.agent.session import Session

    r = asyncio.run(Session(reflex=_Reflex(intent)).handle("search for what I copied", from_user=True))
    assert r.route == "agent" and "in_front" in session_env.calls[0][1]


def test_a_models_goal_never_takes_the_hook(session_env):
    from neo2.core.agent.session import Session

    r = asyncio.run(Session(reflex=_Reflex("chat")).handle(ASKED, origin="live"))
    assert r.route == "chat" and session_env.calls == []


def test_switched_off_the_hook_is_skipped(session_env, monkeypatch):
    from neo2.core import config
    from neo2.core.agent.session import Session

    monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "false")
    monkeypatch.setattr(config, "_settings", None)
    r = asyncio.run(Session(reflex=_Reflex("chat")).handle(ASKED, from_user=True))
    assert r.route == "chat" and session_env.calls == []


def test_a_routine_definition_keeps_its_lane(session_env):
    from neo2.core.agent.session import Session

    r = asyncio.run(
        Session(reflex=_Reflex("chat")).handle(
            "when I say good night, summarize what I copied", from_user=True
        )
    )
    assert r.route == "chat" and session_env.calls == []


def test_without_the_tool_the_hook_is_skipped(session_env):
    from neo2.core.agent.session import Session

    prev = catalog().remove("in_front")
    try:
        r = asyncio.run(Session(reflex=_Reflex("chat")).handle(ASKED, from_user=True))
        assert r.route == "chat"
    finally:
        catalog().register(prev)


# ---- Gemini Live ------------------------------------------------------------------------------------------


def test_live_declares_and_explains_in_front_only_while_it_is_there(world, monkeypatch):
    from neo2.core import config
    from neo2.core.voice import live as live_mod

    assert "in_front" in live_mod._direct_tools()
    assert "in_front" in [d.name for d in live_mod._declarations()]
    assert "in_front" in live_mod._live_extra()
    monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "0")
    monkeypatch.setattr(config, "_settings", None)
    assert "in_front" not in live_mod._live_extra()
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.delenv("NEO_CONTEXT_CAPTURE")
    prev = catalog().remove("in_front")
    try:
        assert "in_front" not in live_mod._live_extra()
        assert "in_front" not in [d.name for d in live_mod._declarations()]
    finally:
        catalog().register(prev)


def test_switched_off_neo_app_hides_it_from_every_model(world, monkeypatch):
    # NEO_CONTEXT_CAPTURE=0 is today's NEO: no in_front in Live's declarations or an agent's tool list
    # (not a tool that is offered and only refuses). NEO.app hides it at start, as it does shortcut_run.
    from neo2.core import config
    from neo2.core.agent.session import wants_context
    from neo2.core.agent.toolselect import ToolSelector
    from neo2.core.tools import context_now as cn
    from neo2.core.voice import live as live_mod

    t = catalog().get("in_front")
    before = t.hidden
    try:
        monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "0")
        monkeypatch.setattr(config, "_settings", None)
        cn.start()
        assert t.hidden
        assert "in_front" not in [d.name for d in live_mod._declarations()]
        assert "in_front" not in live_mod._live_extra()
        specs = ToolSelector().select(ASKED, required=("in_front",)).specs(registry())
        assert "in_front" not in [sp.name for sp in specs]
        assert not wants_context(ASKED)
        monkeypatch.delenv("NEO_CONTEXT_CAPTURE")
        monkeypatch.setattr(config, "_settings", None)
        cn.start()  # on (the default): there for every model, as before
        assert not t.hidden and "in_front" in [d.name for d in live_mod._declarations()]
    finally:
        t.hidden = before


def _live_voice(session=None):
    """A Gemini Live voice with no connection: tool responses land in `sent`."""
    import threading

    import neo2.core.voice.live as live_mod

    sent: list = []

    class FakeLive:
        async def send_tool_response(self, function_responses):
            sent.extend(function_responses)

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._agent_session, v._tool_tasks, v._goals, v._server_cancelled = session, {}, {}, set()
    v._live, v._ui_lock, v._call_labels, v._taint, v._dialog = FakeLive(), asyncio.Lock(), {}, [], []
    v._utt_started, v._last_answer, v._lookups, v._seen_calls, v._recent_idem = 0.0, (0.0, ""), {}, {}, {}
    v._speaking, v.spk = False, SimpleNamespace(playing=threading.Event())
    v._result_pending = v._result_speech = v._turn_open = False
    v._asked = v._voiced_action = None
    return v, sent


def test_live_direct_call_reads_for_the_users_sentence_fenced(world, monkeypatch):
    from neo2.core import events

    monkeypatch.setattr(events, "_bus", events.EventBus())
    v, sent = _live_voice()
    fc = SimpleNamespace(id="c1", name="in_front", args={"what": "auto"})
    asyncio.run(v._run_tool(v._live, fc, "summarize what I copied"))
    result = sent[0].response["result"]
    assert result.startswith('<untrusted source="in_front">') and "Meeting moved" in result
    assert v._taint == ["in_front"]


def test_live_direct_call_on_the_models_own_idea_is_refused(world, monkeypatch):
    from neo2.core import events

    monkeypatch.setattr(events, "_bus", events.EventBus())
    v, sent = _live_voice()
    fc = SimpleNamespace(id="c1", name="in_front", args={"what": "clipboard"})
    asyncio.run(v._run_tool(v._live, fc, "what's the weather tomorrow"))
    result = sent[0].response["result"]
    assert "Meeting moved" not in result and not result.startswith("<untrusted") and v._taint == []
    assert world.pb.read == []


@pytest.mark.parametrize(
    "said, handed",
    [
        ("summarize what I copied", ("summarize what I copied", "user")),  # the user's own words, as theirs
        ("ये कॉपी किया हुआ पढ़ो", ("ये कॉपी किया हुआ पढ़ो", "user")),  # in Hindi, as Gemini writes it
        ("what does this say", ("what does this say", "user")),
        ("explain this answer", ("Summarize the clipboard", "live")),  # the conversation's: the model's goal
        ("tell me something nice", ("Summarize the clipboard", "live")),  # the model's goal, as before
    ],
)
def test_live_agent_task_hands_over_the_users_own_words_for_this(world, monkeypatch, said, handed):
    from neo2.core import events
    from neo2.core.agent.session import Reply

    monkeypatch.setattr(events, "_bus", events.EventBus())
    seen: list = []

    class S:
        async def handle(self, text, **kw):
            seen.append((text, kw.get("origin")))
            return Reply("Done.", "agent")

    v, _ = _live_voice(S())
    fc = SimpleNamespace(id="c1", name="agent_task", args={"goal": "Summarize the clipboard"})
    asyncio.run(v._run_tool(v._live, fc, said))
    assert seen == [handed]


def test_live_agent_task_switched_off_keeps_the_models_goal(world, monkeypatch):
    from neo2.core import config, events
    from neo2.core.agent.session import Reply

    monkeypatch.setattr(events, "_bus", events.EventBus())
    monkeypatch.setenv("NEO_CONTEXT_CAPTURE", "0")
    monkeypatch.setattr(config, "_settings", None)
    seen: list = []

    class S:
        async def handle(self, text, **kw):
            seen.append((text, kw.get("origin")))
            return Reply("Done.", "agent")

    v, _ = _live_voice(S())
    fc = SimpleNamespace(id="c1", name="agent_task", args={"goal": "Summarize the clipboard"})
    asyncio.run(v._run_tool(v._live, fc, ASKED))
    assert seen == [("Summarize the clipboard", "live")]
