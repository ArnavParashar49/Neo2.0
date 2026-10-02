"""The memory tools, backed by the local SQLite store (neo2.core.memory).

- ``memory_search`` — what NEO remembers about something: facts, conversation summaries and the
  journal of what was said, searched together (keyword + meaning), optionally within a time
  ("yesterday", "last week"). Short, dated, id-tagged snippets: at most 8, at most 1,200
  characters.
- ``remember`` — keep a fact. A secret (a password, a PIN, a code, a key, a card number) is
  refused, never stored. A newer value for the same key supersedes the old one.
- ``forget`` — by id, or by what it is about: NEO finds the facts, the journal lines and the
  summaries that mention it, asks the user to confirm exactly that (the confirmation card), then
  deletes them; the conversations that lost a line are summarised again from what is left.
- ``memory`` — the older one tool with ``action`` remember / recall / forget, kept working for the
  models and hosts that use it; it runs the same code.

The user's own words reach them without a model (fast paths): "remember that …", "what do you
remember about …", "do you remember …", "what do you know about me", "what did we talk about
yesterday / last week / on Monday", "forget what I said about …" — and their Hinglish forms.
Asked by the user directly, a search answers in a sentence to be spoken; asked by a model, with
the tagged snippets.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from typing import Any

from neo2.core.agent.registry import ToolContext, ToolOutput, tool
from neo2.core.config import settings
from neo2.core.memory import timephrase
from neo2.core.memory.redact import has_secret
from neo2.core.memory.store import Hit, fact_label, snippet, store

_MAX_ITEMS = 8
_MAX_CHARS = 1200
_ABOUT_ME = re.compile(r"^\s*(?:about\s+)?(?:me|myself|mujhe|mere\s+baare\s+mein|मेरे\s+बारे\s+में)\s*$", re.I)
_VAGUE = re.compile(r"^\s*(?:about\s+)?(?:it|that|this|them|everything|all|all\s+of\s+it|sab|sab\s+kuch|ye|yeh|wo|woh)\s*$", re.I)
_NO_SECRETS = (
    "I don't store secrets like passwords, PINs, codes, keys or card numbers — that one is safer in a "
    "password manager. I haven't saved it."
)
_SCOPES = {
    "facts": ("profile", "lesson", "note"),
    "profile": ("profile",),
    "sessions": ("summary",),
    "turns": ("turn",),
    "all": None,
}


def _is_time(a: dict) -> bool:
    """A fast path's check: the words after "what did we talk about" are a time and nothing else."""
    w = timephrase.find(str(a.get("when") or ""), datetime.now())
    return w is not None and not w.rest.strip(" ?.!")


def _names_something(a: dict) -> bool:
    """A fast path's check: "forget what I said about X" names an X ("forget about it" doesn't)."""
    q = str(a.get("query") or "")
    return bool(q.strip()) and not _VAGUE.match(q)


# ---- search -------------------------------------------------------------------------------------
def _spoken(hits: list[Hit], *, label: str = "") -> str:
    """The hits as a sentence or two for the user's ears: the facts and the conversations about
    it — or, when there are neither, what the user said."""
    facts = [fact_label(h.key, h.text) for h in hits if h.table == "memory"][:4]
    talks = [h for h in hits if h.table == "sessions"][:2]
    lines = [h for h in hits if h.table == "turns" and h.role == "user"][:3]
    parts = []
    if facts:
        parts.append("; ".join(facts) + ".")
    for h in talks:
        day = label or time.strftime("%A %-d %B", time.localtime(h.ts))
        parts.append(f"{day[:1].upper()}{day[1:]}: {h.text}")
    if lines and not talks and not facts:  # (a fact says it better than the line it came from)
        day = label or time.strftime("on %A %-d %B", time.localtime(lines[0].ts))
        parts.append(f"{day[:1].upper()}{day[1:]} you said: " + "; ".join(f"“{h.text[:120]}”" for h in lines) + ".")
    return " ".join(parts)[:700]


def _tagged(hits: list[Hit]) -> str:
    """The hits for a model: one short, dated, id-tagged line each, within the limits."""
    out, used = [], 0
    for h in hits[:_MAX_ITEMS]:
        line = f"#{h.ref} {snippet(h, 220)}"
        if used + len(line) + 1 > _MAX_CHARS:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


async def _search(a: dict, c: ToolContext) -> str:
    level = settings().cloud_memory  # results go straight to a cloud model
    if level == "off":
        return "Memory isn't shared with the online model (NEO_CLOUD_MEMORY=off)."
    st = store()
    from neo2.core.memory import journal

    await asyncio.to_thread(journal.flush, 0.3)  # what was said a moment ago is searched too
    query = " ".join(str(a.get("query") or "").split())
    scope = str(a.get("scope") or "all").lower()
    when = " ".join(str(a.get("when") or "").split())
    spoken = c.origin == "user"  # the user's own words: the answer is said, not read by a model
    if not query and not when:
        query = c.user_text if scope not in ("profile", "facts") and not spoken else ""
    if scope == "profile" or _ABOUT_ME.match(query) or (not query and not when and scope == "facts"):
        prof = await asyncio.to_thread(st.by_kind, "profile", 24)
        if not prof:
            return "I don't know anything about you yet." if spoken else "Nothing about the user in memory yet."
        if spoken:
            more = f" — and {len(prof) - 8} more things" if len(prof) > 8 else ""
            return "Here's what I know about you: " + "; ".join(m.label for m in prof[:8]) + more + "."
        return _tagged([Hit("memory", m.id, m.kind, m.text, m.updated, key=m.key) for m in prof])
    since = until = None
    label = ""
    if when:
        w = timephrase.find(when, datetime.now())
        if w is not None:
            since, until, label = w.since, w.until, w.label
        else:
            query = f"{query} {when}".strip()  # not a time NEO can read: part of the question
    kinds = _SCOPES.get(scope)
    if level == "profile":
        kinds = ("profile",)
    def find(kinds_: Any) -> list[Hit]:
        # (more than asked for: lines that were themselves questions to memory are dropped below)
        got = st.search(query, kinds=kinds_, since=since, until=until, k=_MAX_ITEMS * 3, asking=c.user_text)
        return [h for h in got if not (h.table == "turns" and _asks_memory(h.text))][:_MAX_ITEMS]

    hits = await asyncio.to_thread(find, kinds)
    if not hits and scope == "sessions" and level == "full":
        # No summary of that time yet (the conversation is still open): what was said, then.
        hits = await asyncio.to_thread(find, ("turn",))
    if not label and since is None and (w := timephrase.find(query, datetime.now())) is not None:
        label = w.label
    if not hits:
        if spoken:
            return f"I don't have anything from {label}." if label else "I don't have anything about that."
        return "Nothing relevant in memory."
    return _spoken(hits, label=label) or "I don't have anything about that." if spoken else _tagged(hits)


# ---- remember -------------------------------------------------------------------------------------
async def _remember(tool_name: str, a: dict, c: ToolContext) -> str | ToolOutput:
    text = " ".join(str(a.get("text") or "").split())
    if not text:
        return "Error: text required"
    if c.tainted():
        # What NEO remembers shapes every later answer. After reading a page, an email or the
        # screen, a memory could be planted by that content rather than the user — so they OK it.
        from neo2.core.agent import confirm

        gate = confirm.gated(tool_name, a, f"remember “{text[:80]}” (after reading {c.tainted()})", detail=text)
        if isinstance(gate, str):
            return gate
    key = str(a.get("key") or "").strip()
    if has_secret(text) or (key and has_secret(f"{key} is {text}")):
        return ToolOutput(_NO_SECRETS, say=True)  # said, even where a memory write is silent
    kind = a.get("kind") if a.get("kind") in ("profile", "lesson", "note") else "note"
    source = f"tool:{c.origin}"

    def save() -> int:
        if kind == "note" and not key and settings().memory_extract != "off":
            # "remember that my sister's name is Priya": the fact it states (sister.name), where
            # the rules can read one, rather than the sentence as a note beside it. (Where facts
            # are taken from what is said at all: an embedded agent keeps the words it was given.)
            from neo2.core.memory import extract

            stated = [f for f in extract.rules(text) if f.kind != "note"]
            ids = [extract.keep(store(), f, source=source) for f in stated]
            if any(ids):
                return next(i for i in ids if i)
        return store().remember(text, kind, key, source=source)

    # Off the event loop: an embedding and a database write can take a moment.
    mid = await asyncio.to_thread(save)
    return f"Remembered (#{mid})."


# ---- forget ---------------------------------------------------------------------------------------
_REF = re.compile(r"^\s*#?([fst])?(\d+)\s*$", re.I)


def _plan_for(query: str) -> dict[str, Any] | None:
    """What forgetting `query` would delete, and how the card says it. None: nothing found."""
    from neo2.core.memory import journal

    journal.flush(1.0)  # the lines said a moment ago count too
    st = store()
    found = st.find_about(query)
    if not any(found.values()):
        return None
    facts = st.facts(found["memory"])
    n_f, n_t, n_s = len(found["memory"]), len(found["turns"]), len(found["sessions"])
    bits = []
    if n_f:
        bits.append(f"{n_f} fact{'s' if n_f != 1 else ''}")
    if n_t:
        bits.append(f"{n_t} line{'s' if n_t != 1 else ''} of our conversations")
    if n_s:
        bits.append(f"{n_s} conversation summar{'ies' if n_s != 1 else 'y'}")
    summary = f"forget what I have about “{query[:60]}”: " + ", ".join(bits)
    detail = "\n".join(f"• {m.label[:100]}" for m in facts[:8])
    if n_t or n_s:
        detail += ("\n" if detail else "") + "The conversations these lines came from are summarised again without them."
    return {**found, "summary": summary, "detail": detail}


def _forget_plan(query: str, plan: dict[str, Any]) -> dict[str, int]:
    from neo2.core.memory import journal

    journal.flush(1.0)
    st = store()
    turns = set(plan.get("turns") or [])
    # The request itself and NEO's question about it were said since: they name it too.
    cutoff = time.time() - 300
    recent = {t.id for t in st.recent_turns(since=cutoff, limit=40, spoken_only=False)}
    turns |= set(st.find_about(query)["turns"]) & recent
    return st.forget_items(memory=plan.get("memory") or [], turns=sorted(turns), sessions=plan.get("sessions") or [])


async def _forget(tool_name: str, a: dict, c: ToolContext) -> str | ToolOutput:
    from neo2.core.agent import confirm

    st = store()
    ref = a.get("id")
    query = " ".join(str(a.get("query") or a.get("text") or "").split())
    if ref in (None, "", 0) and query.startswith("#") and _REF.match(query):
        ref, query = query, ""  # "#f12": a tag from a search, not a subject
    if ref not in (None, "", 0):
        m = _REF.match(str(ref))
        if not m:
            return "Error: no such memory."
        table, rid = (m.group(1) or "f").lower(), int(m.group(2))
        if c.tainted():
            gate = confirm.gated(tool_name, a, f"forget memory #{table}{rid} (after reading {c.tainted()})")
            if isinstance(gate, str):
                return gate
        if table == "f":
            gone = await asyncio.to_thread(st.forget, rid)
        else:
            done = await asyncio.to_thread(lambda: st.forget_items(**{"turns" if table == "t" else "sessions": [rid]}))
            gone = any(done.values())
        return "Forgotten." if gone else "Error: no such memory."
    if not query or _VAGUE.match(query):
        return confirm.ask_user("what should I forget? (Everything at once is “Erase NEO's Memory” in the menu.)")
    plan = a.get("_plan") if isinstance(a.get("_plan"), dict) else await asyncio.to_thread(_plan_for, query)
    if plan is None:
        return ToolOutput("I don't have anything about that.", say=True)
    # Only the user can delete what NEO remembers: the card says exactly what goes.
    gate = confirm.gated(tool_name, {"action": "forget", "text": query, "_plan": plan}, plan["summary"], plan["detail"])
    if isinstance(gate, str):
        return gate
    done = await asyncio.to_thread(_forget_plan, query, gate.get("_plan") or plan)
    names = (("fact", "facts", done["memory"]), ("line", "lines", done["turns"]), ("summary", "summaries", done["sessions"]))
    bits = [f"{n} {one if n == 1 else many}" for one, many, n in names if n]
    # Said even though it is an action that worked: the user asked, and should hear it is gone.
    return ToolOutput("Forgotten: " + (", ".join(bits) if bits else "nothing was left") + ".", say=True)


# ---- the tools ------------------------------------------------------------------------------------
_REMEMBER_PATH = (
    # "remember that I park on level 3" — a real fact of 2+ words; "note that down",
    # "remember that" alone, and "remember to …" (a reminder) are not facts.
    r"^\s*(?:please\s+)?(?:remember|keep\s+in\s+mind|note)\s+(?!to\b)(?:that\s+|this:?\s+)?"
    r"(?!(?:down|it|this|that|them)\b)(?!.*\bdown\s*[.!]*\s*$)"
    r"(?P<text>\S+(?:\s+\S+)+?)\s*[.!]*\s*$"
)
_RECALL_PATH = (
    r"^\s*(?:what\s+did\s+i\s+(?:say|tell\s+you)\s+about|do\s+you\s+remember(?:\s+anything\s+about)?|"
    r"what\s+do\s+you\s+(?:know|remember)\s+about)\s+(?P<query>.{2,}?)\s*[?.!]*\s*$"
)
_RECALL_PATH_HI = (
    r"^\s*(?:tumhe|tumko|aapko|tujhe)\s+(?P<query>.{2,}?)\s+ke\s+baare\s+mein\s+kya\s+(?:yaad|pata)\s+hai\s*[?.!]*\s*$"
)
_WHEN_PATH = (
    r"^\s*what\s+(?:did|were|have)\s+(?:we|i)\s+(?:talk(?:ing|ed)?\s+about|discuss(?:ing|ed)?|speak(?:ing)?\s+about|"
    r"chat(?:ting|ted)?\s+about|do|ask(?:\s+you)?(?:\s+about)?|say)\s+(?P<when>.{3,40}?)\s*[?.!]*\s*$"
)
_WHEN_PATH_HI = (
    r"^\s*(?P<when>kal|parso[n]?|aaj|pichh?le\s+hafte|pichh?le\s+mahine|कल|परसों|आज)\s+(?:humne|hamne|maine|हमने|मैंने)\s+"
    r"(?:kya|क्या)\s+(?:baat\s+ki|baat\s+kari|bola|kaha|kiya|poocha|बात\s+की|कहा|किया)(?:\s+(?:th[ai]|थ[ाी]))?\s*[?.!]*\s*$"
)
_FORGET_PATH = (
    r"^\s*(?:please\s+)?forget\s+(?:(?:what|everything|anything)\s+i\s+(?:said|told\s+you)\s+about|about|that)\s+"
    r"(?P<query>.{2,}?)\s*[.!]*\s*$"
)
_FORGET_PATH_HI = r"^\s*(?P<query>.{2,}?)\s+(?:ke\s+baare\s+mein\s+)?(?:sab\s+)?bhool\s+ja(?:o|iye|na)\s*[.!]*\s*$"


_MEMORY_QUESTION = re.compile(
    "|".join(
        "(?:" + re.sub(r"\(\?P<\w+>", "(?:", p) + ")"  # (the same patterns, their groups unnamed)
        for p in (_RECALL_PATH, _RECALL_PATH_HI, _WHEN_PATH, _WHEN_PATH_HI, _FORGET_PATH, _FORGET_PATH_HI)
    ),
    re.I,
)


def _asks_memory(text: str) -> bool:
    """A line that was itself a question to memory ("what do you remember about parking"): it
    says nothing about its subject, so it is never an answer to the next such question."""
    return bool(_MEMORY_QUESTION.search(text))


@tool(
    "memory",
    "Long-term memory. action=remember stores a fact (kind: profile for stable facts about the user, "
    "lesson for feedback on how to behave, note for anything else; key optional e.g. 'city'). "
    "action=recall searches. action=forget deletes by id, or by query (asks the user). Save preferences "
    "and corrections silently.",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["remember", "recall", "forget"]},
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": ["profile", "lesson", "note"]},
            "key": {"type": "string"},
            "query": {"type": "string"},
            "when": {"type": "string"},
            "id": {"type": "integer"},
        },
        "required": ["action"],
    },
    parallel_safe=True,
    category="memory",
    chain=False,  # only as the whole utterance: a later clause means 'in that app'
    fast_path=[
        (_REMEMBER_PATH, {"action": "remember", "text": "<text>", "kind": "note"}),
        (_RECALL_PATH, {"action": "recall", "query": "<query>"}),
        (_RECALL_PATH_HI, {"action": "recall", "query": "<query>"}),
    ],
    payload=True,
)
async def memory(a: dict, c: ToolContext) -> str | ToolOutput:
    act = a.get("action")
    if act == "remember":
        return await _remember("memory", a, c)
    if act == "recall":
        return await _search({**a, "query": a.get("query") or c.user_text}, c)
    if act == "forget":
        return await _forget("memory", a, c)
    return "Error: unknown action"


@tool(
    "memory_search",
    "Search NEO's memory of the user: facts, summaries of past conversations and what was said. Use it "
    "when the user asks about something earlier or about themselves. query: what to look for (may be "
    "empty with `when`). when: 'yesterday', 'last week', 'on Monday'. scope: facts|sessions|turns|all.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "scope": {"type": "string", "enum": ["facts", "sessions", "turns", "all"]},
            "when": {"type": "string"},
        },
    },
    parallel_safe=True,
    category="memory",
    chain=False,
    read_only=True,
    fast_path=[
        (_WHEN_PATH, {"query": "", "when": "<when>", "scope": "sessions"}, _is_time),
        (_WHEN_PATH_HI, {"query": "", "when": "<when>", "scope": "sessions"}, _is_time),
    ],
    payload=True,
)
async def memory_search(a: dict, c: ToolContext) -> str:
    return await _search(a, c)


@tool(
    "remember",
    "Save a fact to NEO's long-term memory, silently, when the user states a preference, corrects you or "
    "shares something about themselves. text: the fact. kind: profile (a stable fact about the user), "
    "lesson (how to behave) or note. key: what it is, e.g. 'city', 'sister.name' — a new value for the "
    "same key replaces the old one. Never a password, a code or a key.",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": ["profile", "lesson", "note"]},
            "key": {"type": "string"},
        },
        "required": ["text"],
    },
    parallel_safe=True,
    category="memory",
    chain=False,
)
async def remember(a: dict, c: ToolContext) -> str | ToolOutput:
    return await _remember("remember", a, c)


@tool(
    "forget",
    "Delete something from NEO's memory when the user asks to forget it. query: what it is about (NEO "
    "finds the facts and the lines that mention it and asks the user to confirm), or id: a tag from "
    "memory_search (f12, t881, s5).",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}, "id": {"type": "string"}},
    },
    category="memory",
    chain=False,
    fast_path=[
        (_FORGET_PATH, {"query": "<query>"}, _names_something),
        (_FORGET_PATH_HI, {"query": "<query>"}, _names_something),
    ],
    payload=True,
)
async def forget(a: dict, c: ToolContext) -> str | ToolOutput:
    return await _forget("forget", a, c)
