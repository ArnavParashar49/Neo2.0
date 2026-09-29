"""One camera question, one frame: :func:`look`.

Apple Vision (labels, words, hands) and the object detector (neo2.core.vision.detect: YOLOE-26s, or
D-FINE) run side by side in threads, for at most :data:`LOCAL_WAIT_S` — a detector still loading is
left to finish in the background and the answer goes on without its boxes. What they found is handed
to ``on_local`` (the live card draws the boxes while the answer is on its way), then, as short hints
with the one still, to the cloud brains that see (neo2.core.vision.answer: Gemini, then Cerebras; the
on-device labels alone when those are off or down). "What am I holding?": the object whose box meets
a hand's.

The boxes are drawn the way NEO 1 drew its YOLO's, which the user liked (:func:`objects`): every
thing the detector is sure of, people included (as "person", whatever role word the open vocabulary
gave them), one box per thing, at most :data:`MAX_SHOWN` — never a body part, how someone looks, a
group, or the room itself. Apple Vision naming the same thing adds a box the detector was less sure
of; it never hides one. The brains hear of things, never people: what a hand holds first, then what
nobody in view is wearing (:func:`hint_objects`). A finger count isn't told what the hand holds.

:func:`preview` is the live card's per-frame boxes (the detector only; nothing leaves the Mac).
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import Callable, Iterable
from typing import Any

from PIL import Image

from neo2.core import effects, logger
from neo2.core.vision import answer as _answer
from neo2.core.vision import apple, detect, yolo

log = logger(__name__)

LOCAL_WAIT_S = 1.2  # the on-device layers' share of the ≤3 s answer
_REACH = 0.5  # a hand's box grown by this much of its size on each side: what it's holding is in there

MAX_SHOWN = 10  # boxes on the card at most
MAX_HINTS = 4  # objects named to the brains at most
SAME_THING = 0.6  # two boxes overlapping this much (IoU) are one thing, whatever their words: the surer stays
SAME_WORD = 0.4  # ...this much, when they have the same word
SCENE = 0.8  # a box (not a person's) covering this share of the frame is the room or the table, not a thing
SPECK = 0.002  # smaller than this share of the frame: a speck, not something to show
HELD = 0.6  # a thing in the hand covers less of the frame than this

# The open vocabulary's words for a person are one word on the card: "person", as NEO 1 said. (Its
# roles, jobs and kin: the card never calls the user a "preacher" or a "jazz artist", and the brains
# never hear NEO's guess at who someone is.)
PERSON = frozenset(
    {
        "person", "man", "woman", "boy", "girl", "child", "baby", "adult", "toddler", "teenager", "player", "actor",
        "actress", "bride", "businessman", "business woman", "chef", "dancer", "ballet dancer", "doctor", "nurse",
        "officer", "soldier", "student", "college student", "worker", "construction worker", "singer",
        "lead singer", "musician", "athlete", "photographer", "designer", "fashion girl", "fashion model",
        "flower girl", "child actor", "tv actor", "theatre actor", "baseball player", "basketball player",
        "football player", "hockey player", "ice hockey player", "rugby player", "tennis player",
        "volleyball player", "senior", "youth", "twin", "dude", "brunette", "bodybuilder", "celebrity", "hero",
        "mascot", "pirate", "witch", "samurai", "warrior", "king", "queen", "prince", "princess", "royal",
        "duchess", "socialite", "host", "commentator", "journalist", "politician", "president", "protester",
        "pope", "monk", "nun", "preacher", "prophet", "pilgrim", "judge", "lawyer", "police", "surgeon",
        "paramedic", "dentist", "scientist", "researcher", "engineer", "technician", "mechanic", "electrician",
        "plumber", "welder", "carpenter", "craftsman", "blacksmith", "tailor", "barber", "baker", "butcher",
        "waiter", "barista", "bartender", "secretary", "manager", "customer", "client", "shopper", "vendor",
        "passenger", "commuter", "pedestrian", "tourist", "hiker", "camper", "climber", "mountaineer", "explorer",
        "astronaut", "pilot", "sailor", "captain", "paratrooper", "marine", "veteran", "lifeguard", "rescuer",
        "farmer", "gardener", "fisherman", "shepherd", "lumberjack", "miner", "cowboy", "jockey", "rider",
        "motorcyclist", "driver", "teacher", "educator", "mother", "father", "daughter", "cousin", "grandmother",
        "grandfather", "grandparent", "neighbor", "nanny", "patient", "volunteer", "participant", "competitor",
        "champion", "spectator", "supporter", "referee", "goalkeeper", "runner", "swimmer", "diver", "surfer",
        "skier", "skater", "skateboarder", "snowboarder", "gymnast", "wrestler", "golfer", "cricketer", "striker",
        "defender", "artist", "painter", "sculptor", "animator", "architect", "author", "writer", "composer",
        "conductor", "guitarist", "bassist", "drummer", "pianist", "violinist", "saxophonist", "rapper", "rocker",
        "dj", "entertainer", "magician", "director", "philosopher", "historian", "physicist", "chemist",
        "astronomer", "detective", "hacker", "gladiator", "groom", "bridesmaid", "tv personality",
        "business executive", "auto mechanic", "street vendor", "tour guide", "rock climber", "motorcycle racer",
        "bus driver", "truck driver", "soccer goalkeeper", "head coach", "football coach", "basketball coach",
        "figure skater", "martial artist", "street artist", "folk artist", "pop artist", "rock artist",
        "jazz artist", "jazz fusion artist", "blue artist", "country artist", "country pop artist",
        "folk rock artist", "hard rock artist", "hip hop artist", "metal artist", "makeup artist", "tattoo artist",
        "fashion designer", "film director", "film producer", "wedding photographer",
    }
)  # fmt: skip
# Never a box of their own: a part of a body (the person's box has them), how someone looks (their
# hair, a scar, a tattoo: never NEO's to name), several people at once.
BODY = frozenset(
    {
        "hand", "arm", "shoulder", "face", "eye", "nose", "mouth", "ear", "hair", "long hair", "short hair", "skin",
        "beard", "head", "neck", "leg", "foot", "muscle", "body", "forehead", "eyebrow", "toe", "belly", "back",
        "chest", "finger", "thumb", "palm", "fist", "wrist", "lip", "nail", "tooth", "tongue", "muzzle", "beak",
        "fang", "whisker", "paw", "tail", "claw", "fur", "feather", "wing", "waist", "hip", "jaw", "ankle",
        "afro", "bangs", "braid", "ponytail", "haircut", "hairstyle", "hair color", "freckle", "scar", "wrinkle",
        "tattoo", "tan",
    }
)  # fmt: skip
GROUPS = frozenset(
    {"crowd", "couple", "wedding couple", "family", "group", "audience", "people", "army", "gang", "crew", "squad",
     "choir", "tribe", "troop", "navy", "community"}
)  # fmt: skip
# What someone wears (their clothes aren't drawn at all: yolo.NOT_SHOWN): inside a person's box it's
# theirs, not a thing they hold up — the brains hear of it last, and a hand beside it isn't holding it.
WORN = frozenset(
    {
        "tie", "glasses", "sunglasses", "hat", "cap", "baseball hat", "keycard", "necklace", "earring", "bracelet",
        "ring", "watch", "earphone", "glove", "sock", "shoe", "backpack", "mask", "scarf", "headscarf", "veil",
        "wristband", "armband",
    }
)  # fmt: skip
INSIDE = 0.8  # this share of a box within a person's box: it's on them
HAND_ITSELF = 0.6  # a box overlapping Apple Vision's hand this much (IoU) is the hand, whatever its word
_FINGER_COUNT = re.compile(r"\b(how many|count|number of)\b[^?.!]*\bfingers?\b", re.I)


def _words(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z]+", s.lower()) if len(w) > 2}


def name(label: str) -> str | None:
    """The word a box shows: "person" for any person word, None for a body part, a look or a group."""
    n = " ".join(str(label).split())
    low = n.lower()
    if low in PERSON:
        return "person"
    if low in BODY or low in yolo.NOT_SHOWN or low in GROUPS or low.endswith(" team"):
        return None  # (clothes too: the person's box has them)
    return n


def _area(b: dict[str, float]) -> float:
    return max(0.0, b["w"]) * max(0.0, b["h"])


def _inter(a: dict[str, float], b: dict[str, float]) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return ix * iy


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    inter = _inter(a, b)
    return inter / max(1e-9, _area(a) + _area(b) - inter)


def _worn(n: str, b: dict[str, float], people: list[dict[str, float]]) -> bool:
    """Is the thing ``n`` at ``b`` worn by someone in view (a word for what's worn, inside their box)?"""
    return n.lower() in WORN and any(_inter(b, p) >= INSIDE * _area(b) for p in people)


def _hand_itself(b: dict[str, float], hands: Iterable[apple.Hand]) -> bool:
    """Is the box ``b`` one of the hands (the detector's word for it aside: YOLOE calls a rendered emoji
    hand a "banana")?"""
    return any(_iou(b, dict(zip(("x", "y", "w", "h"), h.box, strict=True))) >= HAND_ITSELF for h in hands)


def objects(
    dets: Iterable[detect.Det], labels: list[tuple[str, float]] | None = None, *,
    scores: tuple[float, float] | None = None,
) -> list[detect.Det]:  # fmt: skip
    """The boxes worth drawing and telling the brains of, best first: the detector's from its
    ``sure`` score up (``scores`` = (floor, sure), the detector's own: detect.scores), and from its
    floor up the ones Apple Vision's ``labels`` name too; person words as "person"; no body part,
    group, speck or whole-scene box; one box per thing; at most :data:`MAX_SHOWN`."""
    floor, sure = scores if scores is not None else detect.scores()
    seen: set[str] = set().union(*(_words(n) for n, _ in labels)) if labels else set()
    out: list[detect.Det] = []
    for d in sorted(dets, key=lambda d: -float(d["score"])):
        n, b, s = name(d["label"]), d["box"], float(d["score"])
        if n is None or s < floor or _area(b) < SPECK or (n != "person" and _area(b) >= SCENE):
            continue
        if s < sure and not _words(n) & seen:
            continue
        if any(_iou(b, k["box"]) >= (SAME_WORD if k["label"] == n else SAME_THING) for k in out):
            continue  # the same thing again, less sure
        out.append(detect.Det(label=n, score=round(s, 3), box=dict(b)))
        if len(out) >= MAX_SHOWN:
            break
    return out


def hint_objects(
    dets: Iterable[detect.Det], held: str | None = None, hands: Iterable[apple.Hand] = ()
) -> list[str]:
    """The things (never people, nor a box that is one of the ``hands``) to name to the brains, a few,
    each once: the ``held`` one first, then what nobody in view is wearing, then what they wear (their
    glasses, their tie) — surest first within each, as ``dets`` come."""
    dets, hands = list(dets), list(hands)
    people = [d["box"] for d in dets if name(d["label"]) == "person"]
    ranked = [
        (n != held, _worn(n, d["box"], people), i, n)
        for i, d in enumerate(dets)
        if (n := name(d["label"])) not in (None, "person") and not _hand_itself(d["box"], hands)
    ]
    return list(dict.fromkeys(n for *_, n in sorted(ranked)))[:MAX_HINTS]


def ui_boxes(dets: list[detect.Det]) -> list[dict[str, Any]]:
    """Boxes as the camera card draws them: {x, y, w, h, label, score}, 0..1 of the frame."""
    return [{**d["box"], "label": d["label"], "score": d["score"]} for d in dets]


def holding(hands: list[apple.Hand], dets: list[detect.Det]) -> str | None:
    """The thing by a hand (never a person or a part of one, nor the hand itself under another word,
    nor the table it's over, nor what someone in view wears — the glasses beside a raised hand are on
    the face): the box that overlaps the hand's reach most, None when nothing is within reach."""
    people = [d["box"] for d in dets if name(d["label"]) == "person"]
    best, most = None, 0.0
    for h in hands:
        x, y, w, hh = h.box
        reach = {"x": x - w * _REACH, "y": y - hh * _REACH, "w": w * (1 + 2 * _REACH), "h": hh * (1 + 2 * _REACH)}
        for d in dets:
            n, b = name(d["label"]), d["box"]
            if n in (None, "person") or _area(b) >= HELD or _worn(n, b, people) or _hand_itself(b, hands):
                continue
            share = _inter(b, reach) / max(1e-6, _area(b)) * d["score"]  # how much of it is in reach, by confidence
            if share > most:
                best, most = n, share
    return best if most >= 0.1 else None


def _apple(img: Image.Image) -> tuple[apple.Seen | None, str, float]:
    t0 = time.perf_counter()
    try:
        return apple.analyse(img), "", _ms(t0)
    except (Exception, effects.EffectRefused) as e:  # noqa: BLE001 — Unavailable, a Vision error, effects off
        return None, _why("Apple Vision", e), _ms(t0)


def _detect(img: Image.Image) -> tuple[list[detect.Det] | None, str, float]:
    t0 = time.perf_counter()
    try:
        return detect.detect(img), "", _ms(t0)
    except (Exception, effects.EffectRefused) as e:  # noqa: BLE001
        return None, _why("the object detector", e), _ms(t0)


def _why(who: str, e: BaseException) -> str:
    from neo2.core.vision import Unavailable

    return str(e) if isinstance(e, Unavailable) else f"{who} failed ({type(e).__name__})"


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


async def _tell(cb: Callable[[dict[str, Any]], Any] | None, found: dict[str, Any]) -> None:
    if cb is None:
        return
    try:
        r = cb(found)
        if inspect.isawaitable(r):
            await r
    except Exception as e:  # noqa: BLE001 — the card's trouble never stops the answer
        log.info(f"[vision] on_local failed: {type(e).__name__}")


async def look(
    question: str,
    img: Image.Image,
    *,
    on_local: Callable[[dict[str, Any]], Any] | None = None,
    local_wait_s: float = LOCAL_WAIT_S,
) -> dict[str, Any]:
    """Answer ``question`` about the frame ``img``: {answer, by, cloud, labels, boxes, text, hands,
    holding, notes, ms: {apple, detect, answer, total}}. ``boxes`` are the card's (:func:`ui_boxes`);
    ``notes``: why a layer was left out (not installed, not downloaded), for the user."""
    t0 = time.perf_counter()
    _answer.prewarm()  # (Cerebras's connection, in case Gemini can't answer: nothing when warm)
    ta = asyncio.ensure_future(asyncio.to_thread(_apple, img))
    td = asyncio.ensure_future(asyncio.to_thread(_detect, img))
    await asyncio.wait({ta, td}, timeout=max(0.0, local_wait_s))
    notes: list[str] = []
    seen, why_a, ms_a = ta.result() if ta.done() else (None, "", None)
    dets, why_d, ms_d = td.result() if td.done() else (None, "", None)
    if not ta.done():
        notes.append("Apple Vision was too slow for this answer")
    if not td.done():
        notes.append("the object detector is still loading")
    notes += [w for w in (why_a, why_d) if w]
    dets = objects(dets or [], seen.labels if seen else None)
    hands_found = seen.hands if seen else []
    found: dict[str, Any] = {
        "labels": [[n, c] for n, c in (seen.labels if seen else [])],
        "boxes": ui_boxes(dets),
        "text": list(seen.text) if seen else [],
        "hands": apple.hands_summary(hands_found),
        # (not for a finger count: the fingers are what's "held up", not a thing beside them)
        "holding": None if _FINGER_COUNT.search(question or "") else holding(hands_found, dets),
    }
    await _tell(on_local, found)
    t1 = time.perf_counter()
    hints = {**found, "objects": hint_objects(dets, found["holding"], hands_found)}
    reply = await _answer.ask(img, question, hints)
    ms = {"apple": ms_a, "detect": ms_d, "answer": _ms(t1), "total": _ms(t0)}
    return {**found, "answer": reply.text, "by": reply.by, "cloud": reply.cloud, "notes": notes, "ms": ms}


def preview(img: Image.Image) -> list[dict[str, Any]]:
    """The live card's boxes for one preview frame (the detector, on this Mac; [] when it can't run).
    Blocking: call it in a thread."""
    try:
        return ui_boxes(objects(detect.detect(img)))
    except (Exception, effects.EffectRefused) as e:  # noqa: BLE001 — no boxes is a fine preview
        log.debug(f"[vision] preview boxes: {type(e).__name__}")
        return []


__all__ = [
    "LOCAL_WAIT_S", "MAX_SHOWN", "PERSON", "WORN", "hint_objects", "holding", "look", "name", "objects", "preview",
    "ui_boxes",
]  # fmt: skip
