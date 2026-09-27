"""Who "Priya", "mom" or "+91 98765 43123" is: the user's words resolved against their Contacts.

Never a guess: a name matches a contact only when every word of it IS a word of the contact's
name, nickname or organisation (so "Priya" never finds Priyanka); two matches are a question for
the user, never a pick. Devanagari names are compared with Latin ones through a fixed
romanisation, not a similarity score, and the card always shows the contact's real name.

Nothing here asks for access: without it (or before the user decided) a name stays the spoken
name. Nothing is cached beyond one request, and the only thing ever learned is which contact a
phrase meant (by the contact's id and the handle's label, never the number or address itself).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from neo import effects
from neo.decide._objc import pooled

if TYPE_CHECKING:
    from neo.decide.inventory import App

__all__ = [
    "Handle",
    "Person",
    "Resolution",
    "status",
    "usage_declared",
    "normalize",
    "phrase_key",
    "resolve",
    "bind",
    "pick_handle",
    "app_evidence",
]

Need = Literal["message", "email"]
_STATUS = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized", 4: "limited"}
_ENTITY_CONTACTS = 0
_READABLE = ("authorized", "limited")
MAX_OPTIONS = 3
MAX_SCAN = 20000  # contacts compared for a Devanagari name
_KEYS = (
    "CNContactIdentifierKey",
    "CNContactGivenNameKey",
    "CNContactFamilyNameKey",
    "CNContactNicknameKey",
    "CNContactOrganizationNameKey",
    "CNContactPhoneNumbersKey",
    "CNContactEmailAddressesKey",
    "CNContactInstantMessageAddressesKey",
    "CNContactSocialProfilesKey",
    "CNContactRelationsKey",
)
# Words for a relative, in the user's languages -> the me card's relation label (a language table).
KIN: dict[str, str] = {
    **dict.fromkeys(("mom", "mother", "mum", "mummy", "mommy", "maa", "ma", "amma", "मम्मी", "माँ", "मां"), "Mother"),
    **dict.fromkeys(("dad", "father", "papa", "pitaji", "पापा", "पिताजी"), "Father"),
    **dict.fromkeys(("bhai", "bhaiya", "brother", "भाई", "भैया"), "Brother"),
    **dict.fromkeys(("didi", "behen", "sister", "दीदी", "बहन"), "Sister"),
    **dict.fromkeys(("wife", "biwi", "patni", "पत्नी", "husband", "pati", "पति"), "Spouse"),
    **dict.fromkeys(("son", "beta", "बेटा"), "Son"),
    **dict.fromkeys(("daughter", "beti", "बेटी"), "Daughter"),
    "partner": "Partner",
}
# Respectful words after a name ("Sharma ji"): not part of it.
_HONORIFICS = frozenset({"ji", "sahab", "saheb", "sir", "madam", "ma'am", "maam", "जी", "साहब"})
_MY = frozenset({"my", "mera", "meri", "mere", "मेरा", "मेरी", "मेरे"})  # "my mom", "meri didi"
_DEVANAGARI = re.compile("[ऀ-ॿ]")
_PHONE_CHARS = re.compile(r"^[+\d\s().\-]+$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class Handle:
    kind: Literal["phone", "email", "im"]
    value: str
    label: str  # the Contacts label ('mobile', 'home', ...) as shown to the user
    service: str = ""  # an instant-message / social service's name


@dataclass(frozen=True)
class Person:
    display: str
    contact_id: str = ""
    handles: tuple[Handle, ...] = ()
    source: Literal["contacts", "literal", "spoken"] = "spoken"
    org: str = ""


@dataclass(frozen=True)
class Resolution:
    status: Literal["ok", "ambiguous", "not_found", "denied", "unavailable"]
    person: Person | None = None
    options: tuple[str, ...] = ()  # display strings for a question, at most MAX_OPTIONS
    question: str = ""
    label: str = ""  # the handle label a learned binding chose


def _cn() -> Any:
    import Contacts

    return Contacts


@pooled
def status() -> str:
    """What the user decided about NEO and their Contacts (never asks)."""
    try:
        return _STATUS.get(int(_cn().CNContactStore.authorizationStatusForEntityType_(_ENTITY_CONTACTS)), "denied")
    except Exception:  # noqa: BLE001 - no Contacts framework
        return "unavailable"


def usage_declared() -> bool:
    from neo.decide import evidence

    return "NSContactsUsageDescription" in evidence.responsible_usage()


# ---- words ----------------------------------------------------------------------------------------


def _strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def normalize(s: str) -> str:
    """Casefolded words; Latin letters lose their accents (Devanagari keeps its marks, which are
    part of the letters); honorifics after a name are dropped."""
    s = unicodedata.normalize("NFKC", s or "").casefold()
    if not _DEVANAGARI.search(s):
        s = _strip_marks(s)
    words = re.findall(r"[\w'ऀ-ॿ]+", s)
    while len(words) > 1 and words[-1] in _HONORIFICS:
        words.pop()
    return " ".join(words)


def phrase_key(phrase: str) -> str:
    """What a learned binding is filed under: a hash of the phrase, never the phrase."""
    return hashlib.sha256(normalize(phrase).encode()).hexdigest()[:16]


@pooled
def _latin(s: str) -> str:
    """Devanagari romanised by the OS's own transform ('प्रिया' -> 'priya')."""
    try:
        from Foundation import NSString

        return str(NSString.stringWithString_(s).stringByApplyingTransform_reverse_("Any-Latin; Latin-ASCII", False))
    except Exception:  # noqa: BLE001
        return ""


def _skeleton(word: str) -> str:
    """A fixed spelling-insensitive form of a romanised word, for cross-script comparison only:
    Hindi's inherent final vowel and its sh/s, w/v, long vowels and doubled letters fold together."""
    w = re.sub(r"[^a-z]", "", word.lower())
    for a, b in (("sh", "s"), ("ph", "f"), ("w", "v"), ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u")):
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w)
    return w[:-1] if len(w) > 2 and w.endswith("a") else w


def _tokens(s: str) -> list[str]:
    return normalize(s).split()


# ---- contacts --------------------------------------------------------------------------------------


def _keys() -> list[Any]:
    cn = _cn()
    return [getattr(cn, k) for k in _KEYS]


def _label(raw: Any) -> str:
    if not raw:
        return ""
    try:
        return str(_cn().CNLabeledValue.localizedStringForLabel_(raw))
    except Exception:  # noqa: BLE001
        return str(raw)


def _person(c: Any, source: Literal["contacts", "literal"] = "contacts") -> Person:
    handles: list[Handle] = []
    for lv in c.phoneNumbers() or []:
        handles.append(Handle("phone", str(lv.value().stringValue()), _label(lv.label())))
    for lv in c.emailAddresses() or []:
        handles.append(Handle("email", str(lv.value()), _label(lv.label())))
    for lv in c.instantMessageAddresses() or []:
        v = lv.value()
        handles.append(Handle("im", str(v.username()), _label(lv.label()), str(v.service() or "")))
    for lv in c.socialProfiles() or []:
        v = lv.value()
        handles.append(Handle("im", str(v.username()), _label(lv.label()), str(v.service() or "")))
    full = " ".join(x for x in (str(c.givenName() or ""), str(c.familyName() or "")) if x)
    org = str(c.organizationName() or "")
    return Person(
        display=full or str(c.nickname() or "") or org,
        contact_id=str(c.identifier()),
        handles=tuple(handles),
        source=source,
        org=org,
    )


def _name_words(c: Any) -> list[str]:
    return [
        w for part in (c.givenName(), c.familyName(), c.nickname(), c.organizationName()) for w in _tokens(str(part or ""))
    ]


def _matches(phrase: str, c: Any) -> bool:
    """Every word of ``phrase`` is a whole word of the contact's names (Latin: exactly; Devanagari:
    by romanised skeleton)."""
    words = _tokens(phrase)
    if not words:
        return False
    if _DEVANAGARI.search(phrase):
        mine = {_skeleton(w) for w in _tokens(_latin(" ".join(words)))}
        theirs = {_skeleton(w) for w in _name_words(c)}
        return bool(mine) and mine <= theirs
    return set(words) <= set(_name_words(c))


def _store() -> Any:
    cls = _cn().CNContactStore
    effects.guard("contacts", "open the user's Contacts", via=cls)
    return cls.alloc().init()


def _by_name(store: Any, phrase: str) -> list[Any]:
    cn = _cn()
    if _DEVANAGARI.search(phrase):  # the store matches Latin names only: compare every contact here
        req = cn.CNContactFetchRequest.alloc().initWithKeysToFetch_(_keys())
        found: list[Any] = []

        def each(c: Any, stop: Any) -> None:
            if len(found) < MAX_SCAN and _matches(phrase, c):
                found.append(c)

        store.enumerateContactsWithFetchRequest_error_usingBlock_(req, None, each)
        return found
    pred = cn.CNContact.predicateForContactsMatchingName_(normalize(phrase))
    got, _err = store.unifiedContactsMatchingPredicate_keysToFetch_error_(pred, _keys(), None)
    return [c for c in got or [] if _matches(phrase, c)]


def _by_id(store: Any, cid: str) -> Any | None:
    got, _err = store.unifiedContactWithIdentifier_keysToFetch_error_(cid, _keys(), None)
    return got


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _literal(phrase: str) -> Handle | None:
    p = phrase.strip()
    if _EMAIL.match(p):
        return Handle("email", p, "")
    if _PHONE_CHARS.match(p) and len(_digits(p)) >= 7:
        return Handle("phone", p, "")
    return None


def _owner(store: Any, h: Handle) -> Any | None:
    """The contact a literal number (by its last 10 digits) or address belongs to, if exactly one."""
    cn = _cn()
    if h.kind == "email":
        pred = cn.CNContact.predicateForContactsMatchingEmailAddress_(h.value)
    else:
        pred = cn.CNContact.predicateForContactsMatchingPhoneNumber_(cn.CNPhoneNumber.phoneNumberWithStringValue_(h.value))
    got, _err = store.unifiedContactsMatchingPredicate_keysToFetch_error_(pred, _keys(), None)
    tail = _digits(h.value)[-10:]
    hits = [
        c
        for c in got or []
        if any(
            (h.kind == "email" and x.kind == "email" and x.value.casefold() == h.value.casefold())
            or (h.kind == "phone" and x.kind == "phone" and _digits(x.value)[-10:] == tail)
            for x in _person(c).handles
        )
    ]
    return hits[0] if len(hits) == 1 else None


def _kin_word(phrase: str) -> str:
    """The word for a relative the phrase is ("my mom" -> "mom"); '' when it isn't one."""
    words = normalize(phrase).split()
    w = " ".join(words[1:] if len(words) > 1 and words[0] in _MY else words)
    return w if w in KIN else ""


def _kin(store: Any, word: str) -> list[Any]:
    """The contacts the me card names as this relative."""
    cn = _cn()
    want = getattr(cn, f"CNLabelContactRelation{KIN[word]}", None)
    me, _err = store.unifiedMeContactWithKeysToFetch_error_(_keys(), None)
    if me is None or want is None:
        return []
    names = [str(lv.value().name()) for lv in me.contactRelations() or [] if lv.label() == want]
    return [c for n in names for c in _by_name(store, n)]


def _options(people: list[Person]) -> tuple[str, ...]:
    """Display strings that tell the matches apart: a repeated name gets its organisation, else
    its first handle's label (a label, never the number or address)."""
    seen: dict[str, int] = {}
    for p in people:
        seen[p.display] = seen.get(p.display, 0) + 1
    out: list[str] = []
    for n, p in enumerate(people, 1):
        tag = (p.org or next((h.label for h in p.handles if h.label), "") or str(n)) if seen[p.display] > 1 else ""
        out.append(f"{p.display} ({tag})" if tag else p.display)
    return tuple(dict.fromkeys(out))[:MAX_OPTIONS]


def _or(opts: tuple[str, ...]) -> str:
    return opts[0] if len(opts) == 1 else ", ".join(opts[:-1]) + " or " + opts[-1]


@pooled
def _resolve(phrase: str, need: Need) -> Resolution:
    literal = _literal(phrase)
    spoken = Person(display=phrase.strip(), source="spoken")
    st = status()
    if st not in _READABLE:
        if literal is not None:
            return Resolution("ok", Person(literal.value, handles=(literal,), source="literal"))
        return Resolution("unavailable" if st == "unavailable" else "denied", spoken)
    store = _store()
    if literal is not None:
        owner = _owner(store, literal)
        if owner is None:
            return Resolution("ok", Person(literal.value, handles=(literal,), source="literal"))
        p = _person(owner)
        return Resolution("ok", Person(p.display, p.contact_id, (literal,), "literal", p.org))
    from neo.decide import learned

    key = phrase_key(phrase)
    if bound := learned.get("bindings", key):
        c = _by_id(store, str(bound.get("contact_id", "")))
        p = _person(c) if c is not None else None
        if p is not None and any(h.label == bound.get("label") for h in p.handles):
            return Resolution("ok", p, label=str(bound["label"]))
        learned.drop("bindings", key)  # that contact, or that number of theirs, is gone
    word = _kin_word(phrase)
    # a relative by the me card's relation; else, like any name, a contact saved under the word ("Mom")
    found = (_kin(store, word) if word else []) or _by_name(store, word or phrase)
    people = list({p.contact_id: p for p in map(_person, found)}.values())
    if len(people) == 1:
        return Resolution("ok", people[0])
    if len(people) > 1:
        opts = _options(people)
        return Resolution("ambiguous", options=opts, question=f"Which {phrase.strip()}: {_or(opts)}?")
    ask = "What's their email address?" if need == "email" else "What's the number, or which app's chat?"
    return Resolution("not_found", spoken, question=f"I couldn't find {phrase.strip()} in your contacts. {ask}")


async def resolve(phrase: str, need: Need) -> Resolution:
    """Who ``phrase`` is, in this order: a literal number or address (named after its contact when
    it has one), the contact the user confirmed for this phrase before, a relative on the me card,
    then a contact by name (a relative's word too: many save a parent as "Mom"). ``need`` is what
    will be sent (message or email)."""
    return await asyncio.to_thread(_resolve, phrase, need)


def bind(phrase: str, p: Person, h: Handle) -> None:
    """The user approved a card for this person: remember which contact (and which of their
    handles, by label) the phrase meant - never the number or address itself."""
    if p.contact_id:
        from neo.decide import learned

        learned.put("bindings", phrase_key(phrase), {"contact_id": p.contact_id, "label": h.label})


def _mobile(label: str) -> bool:
    return label.strip().casefold() in ("mobile", "iphone", "cell")


def _serves(h: Handle, app: App | None) -> bool:
    """An instant-message handle for this app: its service is one of the app's names."""
    if app is None or not h.service:
        return False
    return h.service.strip().casefold() in {a.casefold() for a in (app.name, *app.aliases)}


def pick_handle(p: Person, need: Need, app: App | None = None, prefer_label: str = "") -> Handle | Resolution:
    """The one handle to use, or the question that picks it ("Priya has two numbers...")."""
    if need == "email":
        pool = [h for h in p.handles if h.kind == "email"]
    else:
        pool = [h for h in p.handles if h.kind == "phone" or (h.kind == "im" and _serves(h, app))]
    if prefer_label and len(chosen := [h for h in pool if h.label == prefer_label]) == 1:
        return chosen[0]
    if len(pool) == 1:
        return pool[0]
    if len(mobiles := [h for h in pool if h.kind == "phone" and _mobile(h.label)]) == 1:
        return mobiles[0]
    what = "email addresses" if need == "email" else "numbers"
    if not pool:
        noun = "email address" if need == "email" else "number"
        return Resolution("not_found", p, question=f"{p.display} has no {noun} in your contacts. What is it?")
    labels = tuple(dict.fromkeys(h.label or h.kind for h in pool))[:MAX_OPTIONS]
    return Resolution(
        "ambiguous",
        p,
        options=labels,
        question=f"{p.display} has {len(pool)} {what}, {_or(labels)}: which one?",
    )


def app_evidence(p: Person, apps: list[App]) -> dict[str, str]:
    """bundle id -> why this person can be reached with that app ("has a <service> handle for this
    app"), for the words Laya reads about each candidate."""
    out: dict[str, str] = {}
    for h in p.handles:
        for a in apps:
            if h.kind == "im" and _serves(h, a):
                out.setdefault(a.bundle_id, f"has a {h.service} handle for this app")
    return out
