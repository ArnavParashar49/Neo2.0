"""Facts, taken from what the user says — so NEO remembers without being told to.

Two passes, both ending in the same upsert (:func:`keep`):

1. **Rules** (:func:`rules`), on every line the user says or types, the moment the journal writes
   it — no model, well under a millisecond. Only sentences that state a fact in so many words:
   "my name is …", "call me …", "my sister's name is …", "my favourite colour is …", "I live in
   …", "I work at …", "I'm from …", "I like / love / prefer / hate …", "I'm allergic to …", "my
   birthday is …", "remember that …" — and the same in Hinglish and Hindi ("mera naam … hai",
   "mujhe … pasand hai", "main … mein rehta hoon", "yaad rakhna ki …"). A question, or a sentence
   about a passing state ("my screen is frozen"), is not a fact: "my X is Y" only counts for
   things that stay true (a name, a birthday, a relative, a favourite …).
2. **A model, while NEO is idle** (:func:`idle_pass`) — never on the voice path: the journal
   lines since the last pass go to the fast brain, which answers with strict JSON
   ``[{"key", "value", "kind", "importance"}]``. It catches what no rule can ("we moved to Pune
   last year"). It reads only what the user said (NEO's own lines are context, and never ones
   that quote a page, a mail or a file). Off with ``cloud_memory = off`` (nothing of the memory
   goes to a model then) and with ``memory_extract = rules``; a watermark in the database says
   how far it has read, so a pass missed at shutdown is caught up at the next start.

Keys are normalised (``sister.name``, ``favorite_color``, ``city``): a newer value for the same
key supersedes the old one in the store, so there are never two contradictory live facts. A
secret is never a fact (neo2.core.memory.redact).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from neo2.core import logger
from neo2.core.config import settings
from neo2.core.memory.redact import MASK, has_secret

log = logger(__name__)


@dataclass
class Fact:
    kind: str  # "profile" | "lesson" | "note"
    key: str  # "" for a note
    text: str  # the value ("Priya"), or the whole statement for a note / a preference
    importance: float = 0.6
    confidence: float = 1.0


# ---- the rule pass -------------------------------------------------------------------------------
_E = r"(?![\w\u0900-\u097F])"
# A question never states a fact.
_QUESTION = re.compile(
    r"\?\s*$|^\s*(?:what|who|whom|whose|where|when|why|how|which|do|does|did|is|are|was|were|am|can|could|would|"
    r"will|should|shall|have|has)\b"
    # Hindi puts its question word anywhere: "mera naam kya hai", "aap kahan rehte ho"
    r"|\b(?:kya|kaun|kaunsa|kahan|kab|kyun|kyon|kaise|kitn[aei])\b|क्या|कौन|कहाँ|कहां|कब|क्यों|कैसे",
    re.I,
)
_LEAD = re.compile(
    r"^\s*(?:(?:hey|hi|ok|okay|so|well|and|also|btw|by\s+the\s+way|actually|fyi|oh|arre|acha|achha|waise)\b[\s,]*)*"
    r"(?:neo\b[\s,:!.-]*)?(?:(?:just\s+so\s+you\s+know|for\s+the\s+record|you\s+should\s+know(?:\s+that)?)\b[\s,]*)?",
    re.I,
)
_REMEMBER = re.compile(
    r"^\s*(?:please\s+)?(?:remember|keep\s+in\s+mind|note|don'?t\s+forget|never\s+forget)\s+(?:that\s+|this:?\s+)?"
    r"(?!to\b)(?P<rest>\S+(?:\s+\S+)+?)\s*[.!]*\s*$"
    r"|^\s*(?:yaad\s+rakh(?:o|na|iye|ein)?|याद\s+रख(?:ो|ना|िए|िये)?)\s+(?:ki\s+|कि\s+)?(?P<rest_hi>\S+(?:\s+\S+)+?)\s*[.!।]*\s*$",
    re.I,
)
# Things that stay true: only these may be the X of "my X is Y".
_RELATIONS = {
    "sister": "sister", "brother": "brother", "mother": "mother", "mom": "mother", "mum": "mother", "mummy": "mother",
    "father": "father", "dad": "father", "papa": "father", "wife": "wife", "husband": "husband", "partner": "partner",
    "girlfriend": "girlfriend", "boyfriend": "boyfriend", "fiance": "fiance", "fiancee": "fiancee", "son": "son",
    "daughter": "daughter", "friend": "friend", "best friend": "best_friend", "boss": "boss", "manager": "manager",
    "dog": "dog", "cat": "cat", "pet": "pet", "doctor": "doctor", "dentist": "dentist", "roommate": "roommate",
    "flatmate": "roommate", "grandmother": "grandmother", "grandfather": "grandfather", "uncle": "uncle",
    "aunt": "aunt", "cousin": "cousin", "teacher": "teacher", "landlord": "landlord", "colleague": "colleague",
    "kid": "child", "child": "child", "baby": "child",
    # Hinglish / Hindi
    "behen": "sister", "bahan": "sister", "behan": "sister", "didi": "sister", "bhai": "brother", "bhaiya": "brother",
    "maa": "mother", "mummy ji": "mother", "pitaji": "father", "pita": "father", "biwi": "wife", "patni": "wife",
    "pati": "husband", "beta": "son", "beti": "daughter", "dost": "friend", "kutta": "dog", "kutte": "dog",
    "billi": "cat", "बहन": "sister", "भाई": "brother", "माँ": "mother", "मां": "mother", "पापा": "father",
    "पिता": "father", "पत्नी": "wife", "पति": "husband", "बेटा": "son", "बेटी": "daughter", "दोस्त": "friend",
    "कुत्ते": "dog", "कुत्ता": "dog", "बिल्ली": "cat",
}  # fmt: skip
_ATTRIBUTES = {
    "name": "name", "full name": "full_name", "first name": "first_name", "last name": "last_name", "surname": "last_name",
    "nickname": "nickname", "birthday": "birthday", "date of birth": "birthday", "dob": "birthday", "age": "age",
    "email": "email", "email address": "email", "phone number": "phone", "number": "phone", "mobile number": "phone",
    "address": "address", "home address": "address", "city": "city", "hometown": "hometown", "home town": "hometown",
    "country": "country", "job": "job", "profession": "job", "occupation": "job", "role": "job", "company": "company",
    "employer": "company", "school": "school", "college": "college", "university": "university", "major": "major",
    "language": "language", "native language": "language", "mother tongue": "language", "timezone": "timezone",
    "time zone": "timezone", "anniversary": "anniversary", "wedding anniversary": "anniversary", "hobby": "hobby",
    "blood group": "blood_group", "blood type": "blood_group", "shoe size": "shoe_size", "shirt size": "shirt_size",
    "height": "height", "diet": "diet", "pronouns": "pronouns", "zodiac sign": "zodiac",
    "star sign": "zodiac", "team": "team", "gym": "gym", "bank": "bank",
}  # fmt: skip
_IMPORTANCE = {"name": 0.95, "birthday": 0.85, "city": 0.85, "company": 0.8, "job": 0.8, "hometown": 0.75}
_REL = "|".join(sorted(map(re.escape, _RELATIONS), key=len, reverse=True))
_ATTR = "|".join(sorted(map(re.escape, _ATTRIBUTES), key=len, reverse=True))
_VAL = r"(?P<val>(?:[^,;.!?।]|\.(?=\S)){1,80}?)"  # a dot inside a word (an address, "Dr.Rao") is not the end
# where the value ends: the sentence does, or another clause begins ("… and I live in Dubai")
_END = r"\s*(?:[.!।]+(?:\s+|$)|,\s*|;\s*|\b(?:and|but|so|because|aur|lekin)\s+(?:i|i'm|my|we|he|she|they|it|main|mera|meri|mujhe)\b.*|$)"
_NOT_VALUE = re.compile(
    r"^(?:not|no|never|still|always|also|just|so|very|really|too|quite|a\s+bit|kind\s+of|sort\s+of|going|being|"
    r"getting|coming|here|there|fine|ok|okay|good|bad|great|broken|frozen|slow|dead|dying|missing|gone|wrong|"
    r"that|this|it|what|why|how|on|in|at|the\s+same|about|over|up|down|back)\b",
    re.I,
)

# (pattern, builder). Each builder gets the match and returns Facts (or none).
_RULES: list[tuple[re.Pattern[str], str]] = [
    # "my sister's name is Priya", "my dog is called Bruno"
    (re.compile(rf"\bmy\s+(?P<rel>{_REL})(?:'s|s)?\s+(?:name\s+is|is\s+(?:called|named))\s+{_VAL}{_END}", re.I), "rel_name"),
    # "my name is Arnav", "my name's Arnav"
    (re.compile(rf"\bmy\s+name(?:\s+is|'s)\s+{_VAL}{_END}", re.I), "name"),
    (re.compile(rf"\b(?:(?:you\s+can\s+|please\s+|just\s+)?call\s+me|i\s+go\s+by|people\s+call\s+me)\s+{_VAL}{_END}", re.I), "name"),
    # "my favourite colour is blue"
    (re.compile(rf"\bmy\s+(?:all[-\s]time\s+)?favou?rite\s+(?P<thing>[\w ]{{2,30}}?)\s+(?:is|are)\s+{_VAL}{_END}", re.I), "favorite"),
    # "my birthday is on March 4", "my email is …"
    (re.compile(rf"\bmy\s+(?P<attr>{_ATTR})\s+(?:is|are|was)\s+(?:on\s+|at\s+)?{_VAL}{_END}", re.I), "attr"),
    # "my wife is Sara" — a name, said with a capital or after "is"
    (re.compile(rf"\b[Mm]y\s+(?P<rel>(?i:{_REL}))\s+is\s+(?P<val>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?){_END}"), "rel_is"),
    (re.compile(rf"\bi\s+(?:live|stay|am\s+living|'m\s+living|am\s+based|'m\s+based)\s+in\s+{_VAL}{_END}", re.I), "city"),
    (re.compile(rf"\bi(?:\s+am|'m)\s+from\s+{_VAL}{_END}", re.I), "hometown"),
    (re.compile(rf"\bi\s+(?:work|am\s+working|'m\s+working)\s+(?:at|for)\s+{_VAL}{_END}", re.I), "company"),
    (re.compile(rf"\bi\s+work\s+as\s+(?:an?\s+)?{_VAL}{_END}", re.I), "job"),
    (re.compile(rf"\bi(?:\s+am|'m)\s+allergic\s+to\s+{_VAL}{_END}", re.I), "allergy"),
    (re.compile(rf"\bi\s+(?:really\s+|absolutely\s+|just\s+)?(?P<verb>like|love|adore|enjoy|prefer)\s+(?!to\s+(?:know|ask|see|hear|say|thank)\b){_VAL}{_END}", re.I), "like"),
    (re.compile(rf"\bi\s+(?:really\s+|absolutely\s+|just\s+)?(?P<verb>hate|dislike|can'?t\s+stand|don'?t\s+like|do\s+not\s+like)\s+{_VAL}{_END}", re.I), "dislike"),
    # Hinglish / Hindi
    (re.compile(rf"\bmer[aie]\s+(?P<rel>{_REL})\s+k[aie]\s+naam\s+(?P<val>.{{1,60}}?)\s+(?:hai|h){_E}", re.I), "rel_name"),
    (re.compile(rf"मेर[ाीे]\s+(?P<rel>{_REL})\s+(?:का|की|के)\s+नाम\s+(?P<val>.{{1,60}}?)\s+है"), "rel_name"),
    (re.compile(rf"\bmera\s+naam\s+(?P<val>.{{1,60}}?)\s+(?:hai|h){_E}", re.I), "name"),
    (re.compile(r"मेरा\s+नाम\s+(?P<val>.{1,60}?)\s+है"), "name"),
    (re.compile(rf"\b(?:mujhe|mujhko)\s+(?P<val>.{{1,60}}?)\s+(?:bula(?:o|na|iye)|keh\s*(?:o|na)|kaho){_E}", re.I), "name"),
    (re.compile(r"\b(?:mujhe|mujhko)\s+(?P<val>.{1,60}?)\s+(?:bilkul\s+)?(?:pasand|acch[ai]\s+lagt[aie])\s+(?P<neg>nahi|nahin)\b", re.I), "dislike"),
    (re.compile(rf"\b(?:mujhe|mujhko)\s+(?P<val>.{{1,60}}?)\s+(?:bahut\s+|bohot\s+)?(?:pasand|acch[ai]\s+lagt[aie])\s+(?:hai|h|hain){_E}", re.I), "like"),
    (re.compile(r"मुझे\s+(?P<val>.{1,60}?)\s+(?:बिलकुल\s+|बिल्कुल\s+)?पसंद\s+नहीं"), "dislike"),
    (re.compile(r"मुझे\s+(?P<val>.{1,60}?)\s+(?:बहुत\s+)?पसंद\s+(?:है|हैं)"), "like"),
    (re.compile(rf"\b(?:main|mai|mein)\s+(?P<val>[\w .'-]{{2,40}}?)\s+(?:mein|me|main)\s+reht[aie]\s+(?:hoon|hu|hun|hai){_E}", re.I), "city"),
    (re.compile(r"मैं\s+(?P<val>.{2,40}?)\s+में\s+रहत[ाी]\s+हूँ|मैं\s+(?P<val2>.{2,40}?)\s+में\s+रहत[ाी]\s+हूं"), "city"),
    (re.compile(rf"\b(?:main|mai|mein)\s+(?P<val>[\w .'&-]{{2,40}}?)\s+(?:mein|me|main)\s+kaam\s+kart[aie]\s+(?:hoon|hu|hun|hai){_E}", re.I), "company"),
    (re.compile(rf"\b(?:main|mai|mein)\s+(?P<val>[\w .'-]{{2,40}}?)\s+se\s+(?:hoon|hu|hun){_E}", re.I), "hometown"),
    (re.compile(rf"\bmera\s+(?:janamdin|janmdin|birthday|b'?day)\s+(?P<val>.{{1,40}}?)\s+(?:ko\s+)?(?:hai|h|hota\s+hai|aata\s+hai){_E}", re.I), "birthday"),
    (re.compile(rf"\bmujhe\s+(?P<val>.{{1,60}}?)\s+se\s+allergy\s+(?:hai|h){_E}", re.I), "allergy"),
]


def slug(s: str) -> str:
    """A key from a few words: "Favourite Colour" → "favorite_colour"."""
    s = s.strip().lower().replace("favourite", "favorite").replace("colour", "color")
    return re.sub(r"[^\w\u0900-\u097F]+", "_", s).strip("_")[:40]


def _value(m: re.Match[str]) -> str:
    g = m.groupdict()
    v = (g.get("val") or g.get("val2") or "").strip().strip("\"'“”‘’")
    v = re.sub(r"\s+(?:by\s+the\s+way|btw|you\s+know|actually|please|ok(?:ay)?|na|yaar)\s*$", "", v, flags=re.I).strip()
    return " ".join(v.split())


def _tidy(obj: str) -> str:
    """What a preference is about, without its article or a trailing "a lot"."""
    obj = re.sub(r"\s+(?:a\s+lot|so\s+much|very\s+much|the\s+most|more|too|as\s+well)\s*$", "", obj, flags=re.I)
    return re.sub(r"^(?:the|a|an|my|some)\s+", "", obj, flags=re.I).strip()


def _one(kind: str, m: re.Match[str]) -> Fact | None:
    v = _value(m)
    if not v or len(v) > 80 or _NOT_VALUE.match(v) or has_secret(m.group(0)):
        return None
    g = m.groupdict()
    if kind == "rel_name":
        rel = _RELATIONS.get(g["rel"].lower(), slug(g["rel"]))
        return Fact("profile", f"{rel}.name", v, 0.8)
    if kind == "rel_is":
        rel = _RELATIONS.get(g["rel"].lower(), slug(g["rel"]))
        return Fact("profile", f"{rel}.name", v, 0.75)
    if kind == "name":
        if len(v.split()) > 4:
            return None
        return Fact("profile", "name", v, _IMPORTANCE["name"])
    if kind == "favorite":
        return Fact("profile", "favorite_" + slug(g["thing"]), v, 0.6)
    if kind == "attr":
        key = _ATTRIBUTES[g["attr"].lower()]
        return Fact("profile", key, v, _IMPORTANCE.get(key, 0.7))
    if kind in ("city", "hometown", "company", "job", "birthday"):
        return Fact("profile", kind, v, _IMPORTANCE.get(kind, 0.7))
    if kind == "allergy":
        return Fact("profile", "allergy." + slug(_tidy(v)), f"allergic to {_tidy(v)}", 0.95)
    if kind in ("like", "dislike"):
        obj = _tidy(v)
        if (
            not obj
            or len(obj.split()) > 8
            or re.match(r"(?:it|this|that|these|those|them|you|to|how|when|what|where|your|his|her|their)\b", obj, re.I)
            or re.search(r"\b(?:one|ones|option|version|idea|way|part|bit)$", obj, re.I)
        ):
            return None  # "I like it", "I prefer the second one": nothing that lasts
        verb = (g.get("verb") or "").lower()
        word = "dislikes" if kind == "dislike" else ("prefers" if verb == "prefer" else "likes")
        return Fact("profile", "pref." + slug(obj), f"{word} {obj}", 0.6)
    return None


def rules(text: str) -> list[Fact]:
    """The facts this line of the user's states in so many words (see the module doc)."""
    text = " ".join((text or "").split())
    if not text or len(text) > 400 or MASK in text:  # (a line that held a secret states nothing to keep)
        return []
    body = _LEAD.sub("", text, count=1)
    note: Fact | None = None
    if m := _REMEMBER.match(body):
        body = (m.group("rest") or m.group("rest_hi") or "").strip()
        if body and not has_secret(body):
            note = Fact("note", "", body, 0.6)
    elif _QUESTION.search(body):
        return []
    out: list[Fact] = []
    seen: set[str] = set()
    for rx, kind in _RULES:
        for m in rx.finditer(body):
            f = _one(kind, m)
            if f is not None and f.key not in seen:
                seen.add(f.key)
                out.append(f)
    # "remember that I park on level 3": no rule knows it, so it is kept as said. One a rule
    # did read ("remember that my sister's name is Priya") is that fact, not a note as well.
    if note is not None and not out:
        out.append(note)
    return out


def keep(st: Any, fact: Fact, *, source: str = "", since: float | None = None) -> int:
    """Put a fact in the store (the upsert both passes share). 0 when it isn't one to keep.
    `since`: when the work that found it began (Store.remember: never written after a later erase)."""
    text = " ".join((fact.text or "").split())
    if not text or MASK in text or has_secret(text) or has_secret(f"{fact.key} is {text}"):
        return 0
    kind = fact.kind if fact.kind in ("profile", "lesson", "note") else "note"
    return st.remember(
        text, kind, fact.key if kind != "note" else "", source=source,  # type: ignore[arg-type]
        importance=fact.importance, confidence=fact.confidence, since=since,
    )  # fmt: skip


# ---- the idle pass (a model, never on the voice path) ---------------------------------------------
WATERMARK = "extract_upto"  # meta key: the last journal line the idle pass has read
_RETRY_S = 600.0  # after a pass no brain could answer, the next one waits this long
_retry_at = 0.0
_MAX_LINES = 60
_MAX_CHARS = 6000
_PROMPT = (
    "You read a short transcript of a user talking to their assistant NEO and list the lasting facts the USER "
    "stated about themselves, their people, their preferences or how they want NEO to behave. Only what the user "
    "said in their own lines (NEO's lines are context). No guesses, nothing about a passing task or state, no "
    "passwords, codes, keys or card numbers. Answer with a JSON array only (no prose, no code fence), at most 8 "
    'items: [{"key": "sister.name", "value": "Priya", "kind": "profile", "importance": 0.8}]. kind is "profile" '
    '(a stable fact about the user), "lesson" (how NEO should behave) or "note" (anything else worth keeping). '
    "key is a short lower_snake name of what the value is (use dots for a person's attribute: wife.birthday); "
    "value is the fact itself, short, in the user's language. importance is 0 to 1. If there is nothing, answer []."
)
_JSON = re.compile(r"\[.*\]", re.S)


def parse(reply: str) -> list[Fact]:
    """The model's facts, strictly: a JSON array of objects with a key and a value. Anything else
    (prose, a half-written array, the wrong types) is nothing."""
    m = _JSON.search(reply or "")
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return []
    out: list[Fact] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        key, value, kind = it.get("key"), it.get("value"), it.get("kind", "profile")
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            continue
        key = re.sub(r"[^\w.\u0900-\u097F]+", "_", str(key or "").strip().lower()).strip("_.")[:48]
        if kind not in ("profile", "lesson", "note"):
            kind = "note"
        if kind != "note" and not key:
            continue
        try:
            imp = min(1.0, max(0.0, float(it.get("importance", 0.6))))
        except (TypeError, ValueError):
            imp = 0.6
        out.append(Fact(kind, key, " ".join(value.split()), imp, 0.8))
        if len(out) >= 8:
            break
    return out


def llm_allowed() -> bool:
    """May the idle pass send journal lines to a model? Not with memory kept off the cloud, and
    not unless the user left it on."""
    s = settings()
    return bool(s.memory_enabled and s.memory_journal and s.memory_extract == "rules+llm" and s.cloud_memory != "off")


async def idle_pass(*, budget_s: float = 12.0) -> int:
    """Read the journal lines since the watermark with the fast brain and keep what it finds.
    Returns how many facts were kept. Moves the watermark only when the model answered, so a
    failed pass is tried again; an erase in the meantime drops the result."""
    global _retry_at
    if not llm_allowed() or time.time() < _retry_at:
        return 0
    from neo2.core import privacy
    from neo2.core.memory.store import store
    from neo2.core.providers import brain
    from neo2.core.providers.base import Message

    st = store()
    gen, began = privacy.generation(), time.monotonic()
    upto = int(st.get_meta(WATERMARK, "0") or 0)
    turns = await asyncio.to_thread(st.turns_after, upto, _MAX_LINES)
    said = [t for t in turns if t.role == "user" and t.source != "tool"]
    if not turns:
        return 0
    last = turns[-1].id
    if not said:
        st.set_meta(WATERMARK, str(last))
        return 0
    lines = []
    for t in turns:
        if t.source == "tool" or (t.role != "user" and t.tainted):
            continue  # never what NEO read somewhere: only words that were the user's or NEO's own
        lines.append(f"User (t{t.id}): {t.text}" if t.role == "user" else f"NEO: {t.text[:200]}")
    text = "\n".join(lines)[-_MAX_CHARS:]
    known = sorted({m.key for m in await asyncio.to_thread(st.by_kind, "profile", 60) if m.key})
    if known:  # the names only, never the values: the same fact keeps the key it has
        text = "Keys already in use (reuse one when the fact is the same thing): " + ", ".join(known) + "\n\n" + text
    try:
        turn = await asyncio.wait_for(
            brain("fast").complete([Message.user(text)], system=_PROMPT, effort="low", max_tokens=500), budget_s
        )
    except Exception as e:  # noqa: BLE001 — no brain right now: the same lines are read later
        log.warning(f"[memory] idle fact pass skipped ({str(e)[:60]})")
        _retry_at = time.time() + _RETRY_S
        return 0
    if privacy.generation() != gen:
        return 0
    facts = parse(turn.text or "")
    src = ",".join(f"t{t.id}" for t in said[-6:])
    kept = 0
    for f in facts:
        if privacy.generation() != gen:
            return kept
        kept += bool(await asyncio.to_thread(keep, st, f, source=src, since=began))
    if privacy.generation() != gen:
        return kept
    st.set_meta(WATERMARK, str(last))
    if kept:
        log.info(f"[memory] idle pass kept {kept} fact(s) from {len(said)} line(s)")  # never what they say
    return kept

