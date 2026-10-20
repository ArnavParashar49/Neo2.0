"""Zero-LLM dispatch: regex fast paths, and plans that cover a whole utterance with them.

Tools declare `fast_path=[(regex, args)]`, or `(regex, args, check)` when the words alone can't
tell ("search X on <W>" is a site search only if W is a site NEO knows). A check may also read
:func:`utterance`, the whole request a plan is covering ("as white" is no song in "open chess and
play as white"). `match()` finds the tool whose pattern accepts a text. `plan()` covers a whole
utterance with fast-path steps — "open notes and type salt and pepper" → open_app(notes),
type_text("salt and pepper") — or returns None so the utterance goes to a model. The rules that
make it safe:

* A *payload* tool (type_text, dictate, click_text, notes_create, memory, web_search) takes
  the rest of the sentence as its text: "salt and pepper" is never split at "and". Only a
  trailing clause that is itself a plain command ("… and press enter") is split off.
* A `chain=False` tool runs only as the first step: in "open youtube and search for cats" the
  search belongs to YouTube, so the plan fails and the model handles it.
* Anything the regexes can't cover — one unknown clause — makes the whole plan None.

:func:`by_utterance` is the lane beside the regexes: a tool registered with ``by_utterance=True``
gets the user's whole request and grounds it by meaning itself — only for the user's own words.
"""

from __future__ import annotations

import re
from contextvars import ContextVar

from neo2.core.agent.registry import RegisteredTool, registry, utterance_lane

Step = tuple[str, dict]

SPLIT = re.compile(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\bafter that\b)\s*", re.I)
_EDGE = " ,.;!?"
_QUOTED = re.compile(r"""^(["“'])([^"“”']*)(["”'])$""")
_utterance: ContextVar[str] = ContextVar("fastpath_utterance", default="")


def utterance() -> str:
    """The whole request :func:`plan` is covering while its checks run ("" for a lone
    :func:`match`)."""
    return _utterance.get()


def strip_quotes(v):
    """type 'Laptops' → Laptops: one balanced pair of quotes around a payload is punctuation."""
    if isinstance(v, str) and (m := _QUOTED.match(v.strip())):
        return m.group(2)
    return v


def match(text: str) -> Step | None:
    """The first tool whose fast-path regex accepts the text. System tools (volume,
    brightness, timer, clock) come first so a broad pattern can never shadow a specific one."""
    tools = sorted(registry().all(), key=lambda t: 0 if t.category == "system" else 1)
    for t in tools:
        for pattern, args, *check in t.fast_path:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            # "<name>" values are filled from the named regex group of the same name.
            filled: dict = {}
            for k, v in args.items():
                if isinstance(v, str) and v.startswith("<") and v.endswith(">"):
                    filled[k] = strip_quotes(m.groupdict().get(v[1:-1]))
                else:
                    filled[k] = v
            if check and not _claims(check[0], filled):
                continue
            if _names_text_in_front(filled):
                continue
            return t.name, filled
    return None


def _names_text_in_front(filled: dict) -> bool:
    """Is a capture only a name for what the user selected or copied ("search what I copied on
    YouTube" → the query "what I copied", "play the selected text")? Its words are in front of the
    user, not in the sentence: no fast path can run it, so it goes on to a model that reads them first
    (in_front: neo2.core.agent.context_words, the Session's hook, Gemini Live's model). Only while NEO
    can read them — in_front there and NEO_CONTEXT_CAPTURE on; otherwise the words run as they did."""
    texts = [v for v in filled.values() if isinstance(v, str) and v.strip()]
    if not texts:
        return False
    from neo2.core.agent import context_words

    if not any(context_words.names_text(v) for v in texts):
        return False
    t = registry().get("in_front")
    if t is None or t.hidden:
        return False
    try:
        from neo2.core.config import settings

        return bool(getattr(settings(), "context_capture", True))
    except Exception:  # noqa: BLE001 — no settings to read (an embedding host): the words as they were
        return False


def _claims(check, filled: dict) -> bool:
    """A pattern's check on its captures. One that fails can't tell: the utterance goes on to the
    next pattern, and in the end to a model, rather than to a guess."""
    try:
        return bool(check(dict(filled)))
    except Exception:  # noqa: BLE001
        return False


def _tool(name: str) -> RegisteredTool | None:
    return registry().get(name)


def is_payload(name: str) -> bool:
    t = _tool(name)
    return bool(t and t.payload)


def plan(text: str, *, first_index: int = 0, whole: str = "", origin: str = "user") -> list[Step] | None:
    """Fast-path steps that cover the *whole* text, in order, or None.

    `first_index` > 0 means steps already ran for this utterance (chain rules apply). `whole`:
    the utterance when `text` is only what is left of it (what the checks read as the request)."""
    text = text.strip(_EDGE)
    if not text:
        return []
    # A saved routine is an exact whole-utterance shortcut. It is available only for the user's
    # own words; a model's goal or a library host must never activate local personal routines.
    if origin == "user" and first_index == 0 and not _utterance.get():
        try:
            from neo2.core import runtime

            if not runtime.library_bound():
                from neo2.core.tools.routines import match as match_routine

                row = match_routine(text)
                if row is not None:
                    return [("routine_run", {"routine_id": str(row["id"])})]
        except (ImportError, KeyError, TypeError):
            pass
    if _utterance.get():  # a clause of the request being planned
        return _plan(text, first_index)
    token = _utterance.set(whole.strip(_EDGE) or text)
    try:
        return _plan(text, first_index)
    finally:
        _utterance.reset(token)


def _plan(text: str, first_index: int) -> list[Step] | None:
    whole = match(text)
    if whole is not None:
        tool, _ = whole
        t = _tool(tool)
        if first_index > 0 and t is not None and not t.chain:
            return None
        if t is not None and t.payload:
            return _split_trailing_commands(text, whole, first_index)
        return [whole]
    steps: list[Step] = []
    pos = 0
    for m in list(SPLIT.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        clause = text[pos:end].strip(_EDGE)
        nxt = m.end() if m else len(text)
        if not clause:
            pos = nxt
            continue
        hit = match(clause)
        if hit is None:
            return None
        t = _tool(hit[0])
        if (first_index + len(steps)) > 0 and t is not None and not t.chain:
            return None
        if t is not None and t.payload and pos > 0:
            # The payload runs to the end of the utterance ("… and type salt and pepper").
            rest = plan(text[pos:], first_index=first_index + len(steps))
            return None if rest is None else steps + rest
        # (A payload clause at the very start whose whole-text match failed — "click play and
        # then press enter" — is just this clause: its regex refused to swallow the rest.)
        steps.append(hit)
        pos = nxt
    return steps


# ---- the by-utterance lane -------------------------------------------------------------------
# A tool registered with ``by_utterance=True`` (click_text, menu_command…) can take the user's whole
# request and find its target by meaning (neo2.core.tools.computer.resolve): no model call to fill in a
# label. Only for the user's own words in NEO.app: a model's (the agent loop) and Gemini Live's
# relays never drive a by-meaning press, and neither does an embedding host (library mode: an
# AsyncAgent.ask defaults to origin "user", but the host's settings don't carry the resolver's keys);
# they keep the exact regex fast paths. ``settings.resolver = "off"`` closes the lane too.


def _by_utterance_tools(origin: str = "user") -> dict[str, RegisteredTool]:
    """The tools whose by-utterance lane is open for this origin (registry.utterance_lane: the user's
    own words to NEO.app only — never an embedded agent's, a model's, Live's or an approval's)."""
    return {t.name: t for t in registry().all() if not t.hidden and utterance_lane(t, origin)}


def _tool_floor() -> float:
    try:
        from neo2.core.config import settings

        return float(settings().reflex_tool_floor)
    except Exception:  # noqa: BLE001
        return 0.85


def by_utterance(text: str, *, origin: str, decision=None) -> Step | None:
    """One step for the whole request — ``(tool, {"utterance": text, "label": <the control's name>})``
    — when a by-utterance tool should take it without a model, else None.

    * ``origin`` must be "user" (the user's own words); anything else → None. Never in library mode,
      never with ``settings.resolver = "off"``.
    * Never a question (a trailing "?" read before any punctuation is trimmed, a wh-word or
      auxiliary first, kya / क्या …), a key ("press enter") or a vague name ("click it"), whoever
      picked the tool.
    * The reflex picked a by-utterance tool (``decision.intent == "quick_action"``, ``decision.tool``,
      ``tool_p`` at the reflex_tool_floor) → that tool.
    * Or the words themselves point at a control (phrases.pointing: "click X", "press X" that isn't a
      key, "X pe click karo", "X dabao", "X पर क्लिक करो"…), one clause, the reflex doesn't call it
      conversation, and isn't sure it's another tool's ("press pause" at play_music, "select dark
      mode" at applescript: that tool's one-tool lane takes it) → the click tool (the by-utterance
      tool that takes a press).
    """
    if origin != "user":
        return None
    raw = " ".join((text or "").split())
    tools = _by_utterance_tools(origin)
    if not raw or not tools or _closed():
        return None
    from neo2.core.tools.computer import phrases

    if phrases.is_question(raw):
        return None  # decided on the words as said: "click send?" asks, it doesn't press
    text = raw.strip(_EDGE)
    intent = str(getattr(decision, "intent", "") or "")
    picked = str(getattr(decision, "tool", "") or "")
    p = float(getattr(decision, "tool_p", 0.0) or 0.0)
    core = phrases.pointing(text)
    if intent == "quick_action" and picked in tools and p >= _tool_floor():
        if SPLIT.search(text) and core is None:
            return None  # two things ("reply and send"): the agent's job
        name = core or phrases.strip_frame(text)
        if not phrases.nameable(name):
            return None  # a key ("press enter"), or nothing named ("click it", "is button pe click karo")
        return picked, {"utterance": text, "label": name}
    if core is None or intent == "chat":
        return None
    if picked and picked not in tools and p >= _tool_floor() and (t := registry().get(picked)) is not None and not t.hidden:
        return None  # the reflex is sure it's another tool's: never a press by these words
    press = [n for n, t in tools.items() if _takes(t, "press")]
    if not press:
        return None
    return press[0], {"utterance": text, "label": core}


def _closed() -> bool:
    """The lane is shut: an embedding host's request (library mode), or ``settings.resolver = "off"``
    (nothing is pressed from words by itself). Can't tell → shut."""
    try:
        from neo2.core import runtime
        from neo2.core.tools.computer import resolve

        return runtime.library_bound() or resolve.mode() == "off"
    except Exception:  # noqa: BLE001
        return True


def _takes(t: RegisteredTool, want: str) -> bool:
    """What a by-utterance tool grounds: its ``by_utterance`` value when it names one ("press",
    "menu", "field"), else a press for a tool whose parameters take a label."""
    v = getattr(t, "by_utterance", False)
    if isinstance(v, str):
        return v == want
    return want == "press" and "label" in (t.parameters.get("properties") or {})


def _split_trailing_commands(text: str, whole: Step, first_index: int) -> list[Step]:
    """ "type hello and press enter" → type_text(hello), hotkey(return). Only a tail that starts
    with a plain (non-payload) command is split off; "salt and pepper" stays one payload."""
    tool = whole[0]
    for m in SPLIT.finditer(text):
        head = text[: m.start()].strip(_EDGE)
        tail = text[m.end() :].strip(_EDGE)
        if not head or not tail:
            continue
        h = match(head)
        if h is None or h[0] != tool:
            continue
        rest = plan(tail, first_index=first_index + 1)
        if rest and not is_payload(rest[0][0]):  # the tail starts with a real command
            return [h, *rest]
    return [whole]


# ---- the Search Screen lane -------------------------------------------------------------------
# "search my screen", "circle to search", "screen pe search karo": the user's own words open Search
# Screen (neo2.core.agent.lens_words decides by the words) — never a model's, never an approval
# replay, never an embedding host's request. Checked before the camera's and the web's lanes.

LENS_TOOL = "search_screen"


def lens(text: str, *, origin: str) -> Step | None:
    """Route a Search Screen ask, with the question "search my screen for X" carries."""
    if origin != "user":
        return None
    raw = " ".join((text or "").split())
    if not raw:
        return None
    from neo2.core import runtime
    from neo2.core.agent import lens_words

    if runtime.library_mode() or runtime.library_bound():
        return None  # an embedded agent has no overlay
    t = registry().get(LENS_TOOL)
    if t is None:
        return None
    asked, question = lens_words.wants_lens(raw)
    if not asked:
        return None
    return LENS_TOOL, ({"question": question} if question else {})


def lens_takes_voice() -> bool:
    """Is Search Screen open with an area picked, so what the user says now is about that area
    (neo2.core.lens.live.takes_voice)? Never loads it: a core that never opened it has none."""
    import sys

    mod = sys.modules.get("neo2.core.lens.live")
    try:
        return bool(mod is not None and mod.lens_live.takes_voice())
    except Exception:  # noqa: BLE001
        return False


# ---- the camera lane --------------------------------------------------------------------------
# "what am I holding", "ye kya hai", "मेरे हाथ में क्या है": the user's own words ask NEO to look
# through the camera (neo2.core.agent.camera_words decides by the words). The camera turns on only
# for the user: never a model's words (the agent loop, Live relaying its own goal), never an approval
# replay, never text NEO read somewhere, and never an embedding host's request unless the host gave
# its agent the camera tool itself.

CAMERA_TOOL = "camera_look"


def camera(text: str, *, origin: str) -> Step | None:
    """Route a camera ask or a contextual follow-up, carrying the original question forward."""
    if origin != "user":
        return None
    raw = " ".join((text or "").split())
    if not raw:
        return None
    from neo2.core import runtime
    from neo2.core.agent import camera_words

    if runtime.library_mode() and not runtime.library_bound():
        return None  # a library's process-wide catalog: no host chose the camera
    t = registry().get(CAMERA_TOOL)
    if t is None or t.hidden:
        return None
    explicit = camera_words.wants_camera(raw)
    follow_up = camera_words.follow_up(raw)
    if not (explicit or follow_up):
        return None
    question = raw if explicit else camera_words.follow_up_question(raw)
    return CAMERA_TOOL, {"question": question}


# ---- the song lane ----------------------------------------------------------------------------
# "what song is this", "ye kaunsa gaana hai", "यह कौन सा गाना है": the user's own words ask NEO to listen
# and name the song playing (neo2.core.agent.song_words decides by the words). NEO listens only for the
# user, exactly as the camera turns on only for them (above).

SONG_TOOL = "song_identify"


def song(text: str, *, origin: str) -> Step | None:
    """Route a song ask or a fresh no-match retry from the user's own words."""
    if origin != "user":
        return None
    raw = " ".join((text or "").split())
    if not raw:
        return None
    from neo2.core import runtime
    from neo2.core.agent import song_words

    if runtime.library_mode() and not runtime.library_bound():
        return None  # a library's process-wide catalog: no host chose to let it listen
    t = registry().get(SONG_TOOL)
    explicit = song_words.wants_song(raw)
    retry = song_words.retry_after_miss(raw)
    if t is None or t.hidden or not (explicit or retry):
        return None
    # The registry strips private arguments before a handler sees them. The quick route uses this
    # marker only to authorize the user's contextual retry after the same lane checks it again.
    return SONG_TOOL, {"_retry": True} if retry and not explicit else {}
