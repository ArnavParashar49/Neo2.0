"""Public datasets for the reflex: where each comes from, its licence, and how it maps onto NEO.

Every source is pinned to a revision (like neo/models.py pins models), and only what a loader
reads is fetched: parquet columns by projection, the head of a 185 MB CSV and the task texts
inside a 259 MB zip by HTTP range requests. Downloads go to the Hugging Face cache or to
``<data_dir>/hf_cache``. The mapping is the point: we don't learn *their* labels, we learn how
people phrase *ours* (route, tool, task kind, destructive, needs the screen, wants a reply).

| source             | pinned at                                                    | licence                        |
|--------------------|--------------------------------------------------------------|--------------------------------|
| clinc              | clinc/clinc_oos@155b9c7 (plus)                               | CC-BY-3.0                      |
| massive            | mteb/amazon_massive_intent@940fd47 (en, hi)                  | CC-BY-4.0 (AmazonScience/massive) |
| hinglish_top       | WillHeld/hinglish_top@53404b8                                | Apache-2.0 (google-research-datasets/Hinglish-TOP-Dataset), translations of TOPv2: CC-BY-SA-4.0-derived |
| hinglish_top_synth | rvv-karma/English-Hinglish-TOP@5ab1056 (synthetic rows)      | Apache-2.0, translations of TOPv2: CC-BY-SA-4.0-derived |
| hinglish_chat      | Abhishekcr448/Hinglish-Everyday-Conversations-1M@cf040b1     | MIT                            |
| topv2              | WillHeld/top_v2@5e382a8                                      | CC-BY-SA-4.0 (TOPv2)           |
| mtop               | WillHeld/mtop@13054fc (en, hi)                               | CC-BY-SA-4.0 (MTOP)            |
| sgd                | github google-research-datasets/dstc8-schema-guided-dialogue@e852981 | CC-BY-SA-4.0          |
| glaive             | glaiveai/glaive-function-calling-v2@8862406 (parquet branch) | Apache-2.0                     |
| agentnet           | xlangai/AgentNet@d76ee50 (meta_data_merged.jsonl)            | MIT                            |
| omniact            | Writer/omniact@5e4bcb2 (task texts in data.zip)              | MIT                            |
| hwu64              | DeepPavlov/hwu64@0dd289c                                     | CC-BY-4.0 (xliuhw/NLU-Evaluation-Data) |
| sentence_types (opt-in) | TigreGotico/sentence-types@4274d78                      | none stated (sibling set is Apache-2.0) |
| leyzer (opt-in)    | github cartesinus/leyzer@3214355 (branch 0.2.0)              | CC-BY-NC-ND-4.0 in the data repo (HF card says CC-BY-4.0) |

The HF mirrors of TOP, MTOP, Hinglish-TOP and HWU64 carry no licence tag; the upstream one is
cited. Hinglish-TOP and rvv-karma's set are Apache-2.0, but their English column is TOPv2 text and
the Hinglish a translation of it, so they are treated as ShareAlike like TOPv2 itself. Rows from
CC-BY-SA sources (and anything derived from them) are generated on the user's machine only and
never committed; test fixtures are invented sentences in the same format.

Off by default, ``--with <name>`` opts in:
- leyzer: the repository that hosts its data is licensed CC-BY-NC-ND-4.0 (non-commercial, no
  derivatives) — the LICENSE file at the pinned commit,
  https://github.com/cartesinus/leyzer/blob/32143559641d28fbc9659b8ff699c5b68eade93d/LICENSE —
  although its HF card (cartesinus/leyzer-fedcsis) says CC-BY-4.0.
- sentence_types: no licence is stated, which grants no rights; and its polite requests are the
  noisiest mapping here (about 3 in 10 sampled rows wrong).

Skipped on purpose: non-commercial (SWDA, DailyDialog, APIGen-MT-5k, SLURP audio), use-restricted
(AgentHarm), gated (Salesforce xLAM), LDC-licensed (ATIS), RICO-ScreenQA (its questions never
mention a screen, so they would teach "any factual question needs the screen"), and sets that
barely map onto NEO (Banking77, MultiWOZ, xSID, ToolBench, ToolACE, hermes, Mind2Web,
AndroidControl). Eval-only sets (BFCL, ScreenSpot-v2, Agent-SafetyBench) aren't loaded.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from neo.reflex.schema import asks_something

INTENTS = ("chat", "quick_action", "agent_task", "stop")
# neo/decide TASKS (engine.py), mirrored so building data never imports the decide engine.
KINDS = ("play_media", "send_message", "send_email", "write_note", "add_reminder", "add_event", "open_app",
         "web_search", "other")
LANGS = ("en", "hi", "hinglish")  # hi = Devanagari, hinglish = romanised Hindi

PINS: dict[str, tuple[str, str]] = {
    "clinc": ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec"),
    "massive": ("mteb/amazon_massive_intent", "940fd47a81eaa7f2cc7b129674d945d618ac38c2"),
    "hinglish_top": ("WillHeld/hinglish_top", "53404b8688a8bb2504a3717a345f8fc85c29ee61"),
    "hinglish_top_synth": ("rvv-karma/English-Hinglish-TOP", "5ab10562888b4dd9735e434496f7e6112859329d"),
    "hinglish_chat": ("Abhishekcr448/Hinglish-Everyday-Conversations-1M", "cf040b1b80857c6038eb73ddad441f0b435e02a6"),
    "topv2": ("WillHeld/top_v2", "5e382a8497d4dd28842cc0bfa85387f965ac9d8d"),
    "mtop": ("WillHeld/mtop", "13054fc9d7475eebe9919802a5ae36f36abdc567"),
    "hwu64": ("DeepPavlov/hwu64", "0dd289ccdeb185ec065d1ebcf5de1c443cd1620f"),
    "sentence_types": ("TigreGotico/sentence-types", "4274d78b96eaadab5e566b44b88bc8e01f0df2d3"),
    # the dataset ships one JSON file; its auto-converted parquet (refs/convert/parquet) allows projection
    "glaive": ("glaiveai/glaive-function-calling-v2", "8862406ff857565709af3dca7fed589b84edf235"),
    "agentnet": ("xlangai/AgentNet", "d76ee50a63fad81cfdbe576416757d7c2091ed50"),
    "omniact": ("Writer/omniact", "5e4bcb26745afc8d58ba9519ea9c9fb832386d26"),
    "sgd": ("google-research-datasets/dstc8-schema-guided-dialogue", "e852981ae34990f4358979625854259302feaa78"),
    "leyzer": ("cartesinus/leyzer", "32143559641d28fbc9659b8ff699c5b68eade93d"),
}


def _hf_url(name: str, path: str) -> str:
    repo, rev = PINS[name]
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def _gh_url(name: str, path: str) -> str:
    repo, rev = PINS[name]
    return f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"


# ---- rows ------------------------------------------------------------------------------------
# Not a bare "google ": "google the capital of peru" is a search, not an address.
_WAKE_WORDS = ("olly ", "alexa ", "hey computer, ", "computer, ", "ok google ", "hey google ", "hey siri ")
_DEVANAGARI = re.compile("[\u0900-\u097f]")


def clean(text: str) -> str:
    t = " ".join(str(text).strip().split())
    low = t.lower()
    for w in _WAKE_WORDS:  # MASSIVE addresses "olly"; NEO's wake word is stripped before the reflex
        if low.startswith(w):
            t = t[len(w) :]
            break
    return t.strip(" ,.")


def lang_of(text: str, latin: str = "en") -> str:
    """Devanagari anywhere → "hi"; otherwise what a Latin-script row of this source is."""
    return "hi" if _DEVANAGARI.search(text) else latin


def text_key(text: str) -> str:
    """The identity of a sentence for dedupe and the benchmark holdout: NFKC, case-folded, with
    every punctuation, space, symbol and control character removed, so re-tokenised copies match
    ("mom ' s" / "mom's", "5 : 30 am" / "5:30am", "to do" / "todo"). Combining marks (Unicode M)
    stay: Devanagari vowel signs carry meaning (करो ≠ करें)."""
    t = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(ch for ch in t if unicodedata.category(ch)[0] not in "PZSC")


def group_id(ns: str, text: str) -> str:
    """A group for a sentence: its re-tokenised copies ("9 : 30 am ." / "9:30am") share it, and a
    translation passes its source sentence's text here to join it."""
    norm = re.sub(r"[\W_]+", "", str(text).lower())
    return f"{ns}:{hashlib.md5(norm.encode()).hexdigest()[:12]}"


def row(
    text: str,
    intent: str,
    tool: str,
    source: str,
    label: str,
    *,
    lang: str,
    split: str = "train",
    group: str | None = None,
    kind: str | None = None,
    destructive: bool | None = None,
    screen: bool | None = None,
    reply: bool | None = None,
) -> dict | None:
    """One training row; optional labels are left out when the source can't say (unknown ≠ no)."""
    t = clean(text)
    if not t:
        return None
    assert intent in INTENTS and lang in LANGS and split in ("train", "eval"), (intent, lang, split)
    assert kind is None or kind in KINDS, kind
    r = {"text": t, "intent": intent, "tool": tool or "", "source": f"{source}:{label}", "hf_intent": label,
         "lang": lang, "split": split, "group": group or group_id(source, t)}
    for k, v in (("kind", kind), ("destructive", destructive), ("screen", screen), ("reply", reply)):
        if v is not None:
            r[k] = bool(v) if k != "kind" else v
    return r


# What the one tool implies about the task kind; everything else an action does is "other".
_KIND_BY_TOOL = {"play_music": "play_media", "mail_send": "send_email", "calendar_add": "add_event",
                 "reminder_add": "add_reminder", "notes_create": "write_note", "open_app": "open_app",
                 "web_search": "web_search", "search_site": "web_search"}


def kind_for(intent: str, tool: str, override: str | None = None) -> str | None:
    if intent not in ("quick_action", "agent_task"):
        return None  # only actions reach neo/decide
    return override or _KIND_BY_TOOL.get(tool, "other")


# ---- what the words say, whatever the source's label ---------------------------------------------
# A source labels an intent; some of its sentences say something else. These read the sentence —
# for a translation, the English it was translated from, so every language of a sentence gets one
# label. Translations without an English source fall back to their own text.

# "how do i report a lost card" asks for advice (chat), it doesn't report one.
_HOW_TO = re.compile(
    r"^(?:(?:please|so|ok|okay|hey|hi)[\s,]+)?(?:how (?:do|can|should|would|could|does) (?:i|we|one|you)\b|"
    r"how to\b|what (?:do|should|must|can) i do\b|what (?:are|is) the (?:steps|process|procedure)\b|"
    r"where (?:can|do|should) i (?:find|get|go|receive|recieve|apply|report|sign up)\b|"
    r"who (?:do|can|should) i\b|(?:(?:can|could|would) you )?(?:tell|show|explain to) me how (?:to|i|do|can|should)\b|"
    r"what'?s the (?:best )?way to\b)",
    re.I,
)


def how_to(text: str) -> bool:
    return bool(_HOW_TO.search(text))


# NEO's calendar_today lists today's events only; any other day needs the Calendar app.
_W = r"\w\u0900-\u097f"  # a word character, Devanagari vowel signs included (they aren't \w)
_OTHER_DAY = re.compile(
    rf"(?<![{_W}])(?:tomorrow'?s?|tmrw|yesterday'?s?|last|previous|past|upcoming|coming up|later this|next day|"
    r"day after|"
    r"weekends?|weeks?|weekly|months?|monthly|years?|(?:mon|tues|wednes|thurs|fri|satur|sun)days?|"
    r"january|february|march|april|june|july|august|september|october|november|december|"
    r"(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?may|may\s+\d)|\d{1,2}(?:st|nd|rd|th)|\d{1,2}/\d{1,2}|"
    r"(?:the|on|for)\s+(?:twenty[- ]|thirty[- ])?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"ninth|tenth|eleventh|twelfth|\w+teenth|twentieth|thirtieth)\b|"
    r"कल|परसों|बीते|पिछले|पिछली|आने वाल[ीेा]|सप्ताह|हफ्ते|हफ़्ते|महीने|साल|"
    r"सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|"
    rf"सितंबर|सितम्बर|अक्टूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर)(?![{_W}])",
    re.I,
)
_REMINDERS = re.compile(r"remind|\bdue\b|\bpending\b|\bnotif|\balert me\b|don'?t let me forget|याद|रिमाइंडर|अनुस्मारक|लंबित", re.I)


def calendar_route(text: str) -> tuple[str, str]:
    """(route, tool) of a calendar lookup: today's events are calendar_today; another day, or
    reminders ("do i have any reminders set", "when is my mortgage due"), needs an agent."""
    if _REMINDERS.search(text) or _OTHER_DAY.search(text):
        return "agent_task", ""
    return "quick_action", "calendar_today"


# Deletes, clears and "take it off my list": hard to undo, whatever intent they were filed under.
_REMOVES = re.compile(
    r"\b(?:remove[sd]?|removing|delete[sd]?|deleting|clear(?:ed|ing)?|erase[sd]?|cancel(?:led|ed|ing)?|"
    r"empty|get rid|wipe|cross\b.{0,40}\boff|take\b.{0,40}\boff|(?:don'?t|do not|no longer) need)\b",
    re.I,
)


def removes(text: str) -> bool:
    return bool(_REMOVES.search(text))


# A stop said to a ringing alarm or a call, naming nothing ("be quiet", "stop now", "चुप रहो",
# "बंद हो जाओ") is NEO's stop. A bare "turn it off" / "band karo" is not: it usually means the
# thing that's on (neo/reflex/__init__.py makes the same call).
_STOP_WORD = (
    r"stop(?:\s+(?:it|that|now|talking|ringing|making\s+(?:noise|sounds?)))?|cancel(?:\s+(?:it|that))?|"
    r"never\s*mind|quiet|be\s+(?:quiet|silent)|silence|shut\s+(?:up|off|down)|hush|sh+|enough|"
    r"that'?s\s+enough|quit\s+it|go\s+away|"
    r"chup(?:\s+(?:raho|rahe|rahiye|ho\s+ja(?:o)?|kar(?:o)?))?|sh?aant?h?(?:\s+(?:raho|rahe|ho\s+jao))?|"
    r"ruko|ruk\s+ja(?:o)?|bas(?:\s+kar(?:o|\s+do)?)?|khamosh|"
    r"चुप(?:\s+(?:रहो|रहें|रहिए|हो\s+जाओ|करो))?|शांत(?:\s+(?:रहो|रहें|हो\s+जाओ))?|रुको|रुक\s+जाओ|रुकें|"
    r"बस(?:\s+करो)?|बंद\s*हो\s*जाओ|ख़?ामोश(?:\s+(?:रहो|हो\s+जाओ))?"
)
_FILLER = r"please|pls|now|ok|okay|oh|just|already|neo|na|yaar|abhi|ab|अभी|प्लीज़|प्लीज|कृपया|जी"
_BARE_STOP = re.compile(
    rf"^\W*(?:(?:{_FILLER})[\s,]+)*(?:{_STOP_WORD})(?:[\s,!.।]+(?:{_FILLER}|{_STOP_WORD}))*[\s,!.।?]*$", re.I
)


def bare_stop(text: str) -> bool:
    return bool(_BARE_STOP.match(text.strip()))


# ---- fetching ---------------------------------------------------------------------------------
class Fetcher:
    """Reads pinned files, downloading each once: whole small files into the Hugging Face cache,
    projected parquet columns, CSV heads and zip members into ``cache`` (``<data_dir>/hf_cache``)."""

    def __init__(self, cache: Path):
        self.cache = Path(cache)

    def _path(self, *parts: str) -> Path:
        p = self.cache.joinpath(*(re.sub(r"[^\w.@-]+", "__", s) for s in parts))
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def records(self, name: str, path: str, columns: list[str] | None = None) -> list[dict]:
        """Rows of a parquet file in a pinned HF dataset, reading only `columns` over HTTP."""
        import pyarrow.parquet as pq

        repo, rev = PINS[name]
        out = self._path(f"{repo}@{rev[:12]}", f"{path}.{'-'.join(columns or ['all'])}.parquet")
        if not out.exists():
            from huggingface_hub import HfFileSystem

            with HfFileSystem().open(f"datasets/{repo}@{rev}/{path}", "rb", block_size=4 << 20) as fh:
                table = _decode_class_labels(pq.ParquetFile(fh).read(columns=columns))
            tmp = out.with_suffix(".tmp")
            pq.write_table(table, tmp)
            tmp.replace(out)
        return pq.read_table(out).to_pylist()

    def text(self, name: str, path: str) -> str:
        """A whole (small) file from a pinned HF dataset, via the Hugging Face cache."""
        from huggingface_hub import hf_hub_download

        repo, rev = PINS[name]
        p = Path(hf_hub_download(repo, path, repo_type="dataset", revision=rev))
        data = p.read_bytes()
        return (gzip.decompress(data) if path.endswith(".gz") else data).decode("utf-8")

    def url_file(self, url: str) -> Path:
        """A pinned URL (a commit-addressed GitHub file), downloaded once."""
        out = self._path("urls", _url_name(url))
        if not out.exists():
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(_http(url))
            tmp.replace(out)
        return out

    def url_head(self, url: str, nbytes: int) -> bytes:
        """The first `nbytes` of a large file (one range request)."""
        out = self._path("urls", f"{_url_name(url)}.head{nbytes}")
        if not out.exists():
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(_http(url, (0, nbytes - 1)))
            tmp.replace(out)
        return out.read_bytes()

    def zip_texts(self, url: str, want: Callable[[str], bool], tag: str) -> dict[str, str]:
        """The text members of a remote zip that `want` picks, fetching only the central directory
        and the byte ranges those members sit in. `tag` names the cached selection."""
        out = self._path("urls", f"{_url_name(url)}.{tag}.json")
        if not out.exists():
            f = _RangeFile(url)
            with zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16)) as z:
                texts = {n: z.read(n).decode("utf-8", "replace") for n in z.namelist() if want(n)}
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(texts))
            tmp.replace(out)
        return json.loads(out.read_text())


def _url_name(url: str) -> str:
    tail = "__".join(url.split("?")[0].rstrip("/").split("/")[-2:])
    return f"{hashlib.md5(url.encode()).hexdigest()[:10]}-{tail}"


def _http(url: str, rng: tuple[int, int] | None = None) -> bytes:
    import urllib.request

    headers = {"User-Agent": "neo-reflex-data"}
    if rng:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        return r.read()


def _probe(url: str) -> tuple[int, str]:
    """(size, final URL) of a remote file — the final URL is the CDN one, so later range requests
    skip the redirect."""
    import urllib.request

    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "neo-reflex-data"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Range"].split("/")[-1]), r.url


def _decode_class_labels(table):
    """Integer ClassLabel columns → their names (the parquet's Hugging Face metadata lists them)."""
    import pyarrow as pa

    meta = json.loads((table.schema.metadata or {}).get(b"huggingface", b"{}") or b"{}")
    feats = (meta.get("info") or {}).get("features") or {}
    for col, spec in feats.items():
        if col in table.column_names and isinstance(spec, dict) and spec.get("names"):
            names = spec["names"]
            vals = [names[v] if v is not None else None for v in table.column(col).to_pylist()]
            table = table.set_column(table.column_names.index(col), col, pa.array(vals, pa.string()))
    return table


class _RangeFile(io.RawIOBase):
    """A remote file read by HTTP range requests in 1 MB blocks (zipfile seeks around in it)."""

    BLOCK = 1 << 20

    def __init__(self, url: str):
        self.size, self.url = _probe(url)
        self.pos = 0
        self.blocks: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        first, last = self.pos // self.BLOCK, (self.pos + n - 1) // self.BLOCK
        missing = [i for i in range(first, last + 1) if i not in self.blocks]
        if missing:  # one request for the whole missing run
            lo, hi = missing[0] * self.BLOCK, min(self.size, (missing[-1] + 1) * self.BLOCK) - 1
            data = _http(self.url, (lo, hi))
            for i in missing:
                self.blocks[i] = data[(i - missing[0]) * self.BLOCK : (i - missing[0] + 1) * self.BLOCK]
        buf = b"".join(self.blocks[i] for i in range(first, last + 1))
        off = self.pos - first * self.BLOCK
        b[:n] = buf[off : off + n]
        self.pos += n
        return n


# ---- CLINC150 + MASSIVE ------------------------------------------------------------------------
# (intent, tool). A tool means "this one tool, called once, does it".
CLINC: dict[str, tuple[str, str]] = {
    # ---- conversation / knowledge → chat
    **dict.fromkeys(
        ["nutrition_info", "oil_change_how", "gas_type", "meaning_of_life", "improve_credit_score", "measurement_conversion", "flip_coin", "do_you_have_pets", "tell_joke", "exchange_rate", "meal_suggestion", "tire_change", "user_name", "what_are_your_hobbies", "jump_start", "vaccines", "food_last", "who_do_you_work_for", "international_visa", "translate", "carry_on", "what_is_your_name", "where_are_you_from", "cook_time", "roll_dice", "who_made_you", "how_old_are_you", "recipe", "yes", "no", "maybe", "ingredients_list", "what_can_i_ask_you", "are_you_a_bot", "plug_type", "oil_change_when", "thank_you", "fun_fact", "ingredient_substitution", "calories", "calculator", "definition", "next_holiday", "mpg", "spelling", "greeting", "goodbye", "repeat", "change_ai_name", "change_user_name", "change_accent", "change_speed", "whisper_mode", "change_language", "timezone", "redeem_rewards", "oos", "gas", "tire_pressure", "last_maintenance", "international_fees"],
        ("chat", ""),
    ),
    # ---- needs current information from the web → quick_action + web_search
    **dict.fromkeys(
        ["directions", "distance", "traffic", "direct_deposit", "flight_status", "travel_alert", "travel_suggestion", "restaurant_suggestion", "how_busy", "exchange_rate",
         "restaurant_reviews", "accept_reservations"],
        ("quick_action", "web_search"),
    ),
    # ---- one-shot commands → quick_action (+ the NEO tool when there is one)
    "weather": ("quick_action", "weather"),
    "time": ("quick_action", "clock"),
    "date": ("quick_action", "clock"),
    "timer": ("quick_action", "timer"),
    "alarm": ("quick_action", "reminder_add"),
    "reminder_update": ("quick_action", "reminder_add"),
    "calendar": ("quick_action", "calendar_today"),
    "meeting_schedule": ("quick_action", "calendar_today"),
    "calendar_update": ("quick_action", "calendar_add"),
    "change_volume": ("quick_action", "volume"),
    "play_music": ("quick_action", "play_music"),
    "next_song": ("quick_action", ""),
    "what_song": ("quick_action", ""),
    "smart_home": ("quick_action", ""),
    # ---- needs an app, a website, an account or several steps → agent_task
    **dict.fromkeys(
        ["apr", "interest_rate", "min_payment", "insurance", "reset_settings", "account_blocked", "report_lost_card", "order", "schedule_meeting", "freeze_account", "restaurant_reservation", "make_call", "text", "bill_balance", "balance", "uber", "car_rental", "credit_limit", "shopping_list", "expiration_date", "routing", "todo_list", "card_declined", "rewards_balance", "share_location", "book_flight", "insurance_change", "todo_list_update", "cancel_reservation", "transactions", "credit_score", "report_fraud", "spending_history", "reminder", "payday", "find_phone", "order_status", "confirm_reservation", "damaged_card", "pin_change", "replacement_card_duration", "new_card", "income", "taxes", "pto_request", "rollover_401k", "pto_request_status", "application_status", "update_playlist", "credit_limit_change", "pay_bill", "lost_luggage", "book_hotel", "w2", "shopping_list_update", "pto_balance", "order_checks", "schedule_maintenance", "transfer", "current_location", "travel_notification", "pto_used", "bill_due"],
        ("agent_task", ""),
    ),
    "cancel": ("stop", ""),
}
# "sync to my phone": nothing a Mac assistant does.
CLINC_SKIP = frozenset(["sync_device"])
CLINC_KIND = {"text": "send_message", "share_location": "send_message", "schedule_meeting": "add_event",
              "todo_list_update": "add_reminder", "shopping_list_update": "add_reminder"}
# Money moved, something sent, booked or changed on an account: hard to undo.
CLINC_DESTRUCTIVE = frozenset(
    ["pay_bill", "transfer", "order", "book_flight", "book_hotel", "car_rental", "restaurant_reservation",
     "cancel_reservation", "uber", "make_call", "text", "share_location", "freeze_account", "pin_change",
     "report_lost_card", "report_fraud", "order_checks", "rollover_401k", "insurance_change", "pto_request",
     "credit_limit_change", "schedule_maintenance", "reset_settings"]
)

MASSIVE: dict[str, tuple[str, str]] = {
    "alarm_set": ("quick_action", "reminder_add"),
    "alarm_query": ("agent_task", ""),
    "alarm_remove": ("agent_task", ""),
    "audio_volume_down": ("quick_action", "volume"),
    "audio_volume_up": ("quick_action", "volume"),
    "audio_volume_mute": ("quick_action", "volume"),  # rows that name no audio are "stop" (see _is_stop_mute)
    "audio_volume_other": ("quick_action", "volume"),
    "calendar_query": ("quick_action", "calendar_today"),
    "calendar_set": ("quick_action", "calendar_add"),
    "calendar_remove": ("agent_task", ""),
    "cooking_query": ("chat", ""),
    "cooking_recipe": ("chat", ""),
    "datetime_convert": ("chat", ""),
    "datetime_query": ("quick_action", "clock"),
    "email_addcontact": ("agent_task", ""),
    "email_query": ("quick_action", "mail_unread"),
    "email_querycontact": ("agent_task", ""),
    "email_sendemail": ("agent_task", "mail_send"),
    "general_greet": ("chat", ""),
    "general_joke": ("chat", ""),
    "general_quirky": ("chat", ""),
    **dict.fromkeys(
        ["iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on"],
        ("quick_action", ""),
    ),
    "lists_createoradd": ("agent_task", ""),
    "lists_query": ("agent_task", ""),
    "lists_remove": ("agent_task", ""),
    "music_dislikeness": ("chat", ""),
    "music_likeness": ("chat", ""),
    "music_query": ("chat", ""),
    "music_settings": ("quick_action", ""),
    "news_query": ("quick_action", "web_search"),
    "play_audiobook": ("quick_action", "play_music"),
    "play_game": ("quick_action", ""),
    "play_music": ("quick_action", "play_music"),
    "play_podcasts": ("quick_action", "play_music"),
    "play_radio": ("quick_action", "play_music"),
    "qa_currency": ("quick_action", "web_search"),  # rates change: needs the web
    "qa_definition": ("chat", ""),
    "qa_factoid": ("chat", ""),
    "qa_maths": ("chat", ""),
    "qa_stock": ("quick_action", "web_search"),
    "recommendation_events": ("quick_action", "web_search"),
    "recommendation_locations": ("quick_action", "web_search"),
    "recommendation_movies": ("quick_action", "web_search"),
    "social_post": ("agent_task", ""),
    "social_query": ("agent_task", ""),
    "takeaway_order": ("agent_task", ""),
    "takeaway_query": ("agent_task", ""),
    "transport_query": ("quick_action", "web_search"),
    "transport_taxi": ("agent_task", ""),
    "transport_ticket": ("agent_task", ""),
    "transport_traffic": ("quick_action", "web_search"),
    "weather_query": ("quick_action", "weather"),
}
MASSIVE_KIND = {"lists_createoradd": "add_reminder"}
MASSIVE_DESTRUCTIVE = frozenset(["email_sendemail", "social_post", "takeaway_order", "transport_ticket",
                                 "transport_taxi", "alarm_remove", "calendar_remove", "lists_remove"])

# 42 of MASSIVE's 110 English mute rows name no audio at all ("be quiet", "shut up", "good night"):
# that is NEO's "stop", and CLINC's cancel already says so for the same words. A mode ("silent mode
# on for three hours") is a device setting, and a row that says neither is noise ("clear data",
# "not use"): dropped.
_AUDIO_EN = re.compile(
    r"\b(?:volume|sounds?|audio|music|speakers?|mute|muted|unmute|songs?|tv|television|radio|podcasts?|"
    r"ring(?:er|tone)?|notifications?|alarms?|video|media|play(?:ing|back)?|mic|microphone|headphones?|"
    r"player|loud|louder|quieter|noise|beeps?|modes?)\b",
    re.I,
)
_AUDIO_HI = re.compile("आवाज|ध्वनि|वॉल्यूम|म्यूट|संगीत|म्यूजिक|स्पीकर|ऑडियो|साउंड|रेडियो|टेलीविजन|टीवी|मीडिया|"
                       "नोटिफिकेशन|सूचना|अलार्म|बजने|शोर|कम्पन|मोड")  # no \b: matras aren't \w
_STOP_TALK = re.compile(
    r"\b(?:quiet|silent|silence|shut|hush|talk(?:ing)?|speak(?:ing)?|stop|enough|good ?night)\b|"
    r"चुप|ख़ामोश|खामोश|शांत|मौन|मूक|साइलेंट|मत बोल|बोलना बंद|बात मत|बात करना बंद|रुक|शुभ रात्रि",
    re.I,
)


# "बंद करो" / "turn it off" on its own: turn the sound off, not "stop" (see bare_stop).
_BARE_OFF = re.compile(r"^\W*(?:(?:please|कृपया)\s+)?(?:band\s+kar(?:o|\s+do|ein)?|बंद\s+कर(?:ो|ें|\s+दो|\s+दें)|"
                       r"(?:turn|switch)\s+(?:it\s+)?off)\W*$", re.I)


def mute_route(text: str, own: str = "") -> str | None:
    """A MASSIVE mute row: "volume" when it names audio or a mode (or is a bare "turn it off"),
    "stop" when it tells NEO to be quiet, None (drop it) when it says neither. `text` is the English
    source, `own` the row's own words when it is a translation."""
    if _AUDIO_EN.search(text) or _AUDIO_HI.search(text) or _BARE_OFF.match(own or text):
        return "volume"
    return "stop" if _STOP_TALK.search(text) else None


# "bye", "talk to you later", "that's all", "not right now": the user is done — NEO's stop, as SGD's
# goodbyes and HWU64's commandstop say. A thanks or "nice chatting" on its own is conversation.
_CLOSER = re.compile(
    r"\b(?:bye|goodbye|good\s*bye|good ?night|see (?:you|ya)|talk (?:to you )?later|catch you later|later|"
    r"until next time|(?:have|got|gotta|need) to (?:go|run|leave)|i'?m (?:out|off|leaving)|signing off|"
    r"that'?s (?:all|it|everything)|thats (?:all|it)|that'?all|that is (?:all|it|everything)|"
    r"that(?:'ll| will| would)? be (?:all|it|everything)|all for now|nothing (?:else|more|for now|at all)|"
    r"it'?s all|all set|all good|(?:i'?m|i am|we'?re|we are) (?:good|done|fine)|that will do|"
    r"everything i need(?:ed)?|all (?:the \w+ )?i need(?:ed)?|(?:don'?t|do not|won'?t) (?:need|want|wish)|"
    r"not (?:right )?now|not at (?:the|this) (?:moment|time)|not (?:just )?yet|not today|(?:some ?)?other time|"
    r"another time|no need|not necessary|leave (?:it|them))\b",
    re.I,
)
_DECLINE = re.compile(r"^\W*(?:(?:no|nope|nah)\b[\s,.!;]*(?:but\s+)?(?:thanks?|thank (?:you|u)|i appreciate)|nothing\b)",
                      re.I)
_THANKS = re.compile(r"\b(?:thanks?|thank (?:you|u)|appreciated?|obliged|grateful|helpful|great help)\b", re.I)


def closes(text: str) -> bool:
    """Does the user end the conversation or decline ("no thanks", "not right now", "that's all")?"""
    return bool(_CLOSER.search(text) or _DECLINE.search(text))


# Does an intent, by its name, want an answer? CLINC and MASSIVE name intents after what they ask.
# (finetune's fallback for rows without a `reply` label; applied here so every row carries one.)
_ASKS_LABEL = re.compile(
    r"query|qa_|status|balance|what_|how_|when|where|who_|which|info|reviews|rate|_time|(?<!up)date|weather|"
    r"traffic|directions|distance|calendar$|calendar_today|reminder$|todo_list$|shopping_list$|"
    r"spending|transactions|expiration|routing|limit$|payday|income|taxes|due|used|pto_balance|"
    r"meeting_schedule|flight_status|order_status|current_location|find_phone|what_song|"
    r"suggestion|recommendation|definition|spelling|translate|calculator|conversion|fun_fact|"
    r"joke|greeting|thank_you|repeat|oos|history|alerts?$|timezone|holiday|vaccines|visa|carry_on|"
    r"plug_type|mpg|gas$|tire_pressure|jump_start|recipe|cook|ingredients|calories|nutrition|"
    r"food_last|exchange_rate|credit_score|apr$|interest_rate|min_payment|international_fees|"
    r"insurance$|w2|next_holiday|how_busy|accept_reservations|confirm_reservation|duration|^time$|"
    r"are_you|do_you|what_are|what_is|where_are|who_made|how_old|meaning_of_life|what_can"
)
# How-to wording in these intents still wants the live thing (directions, a contact, where I am).
_NOT_HOW_TO = frozenset(["directions", "distance", "traffic", "how_busy", "current_location", "transport_query",
                         "transport_traffic", "recommendation_locations", "recommendation_events",
                         "recommendation_movies", "email_querycontact"])
_CALENDAR_LOOKUP = frozenset(["calendar", "meeting_schedule", "calendar_query"])
_LIST_EDITS = frozenset(["calendar_update", "shopping_list_update", "todo_list_update"])
# "save this song to my favorites" / "don't play this again": a command (like TOP LIKE/DISLIKE_MUSIC),
# not the chat about taste the rest of music_likeness is.
_MUSIC_CMD = re.compile(r"\b(?:save|add|favou?rites?|remember|rate|rating|stars?|put it on|(?:don'?t|do not|never) play|"
                        r"skip)\b", re.I)


def _intent_row(src: str, table: dict, kinds: dict, destructive: frozenset, label: str, text: str, *,
                en: str | None, lang: str, split: str, group: str | None, skip: Counter) -> dict | None:
    """A CLINC / MASSIVE row: the intent's labels, refined by what the sentence (for a translation,
    its English source `en`) says."""
    intent, tool = table[label]
    kind, d = kinds.get(label), label in destructive
    reply = intent == "chat" or bool(_ASKS_LABEL.search(label))
    probe = en or clean(text)
    if label == "audio_volume_mute":
        route = mute_route(probe, clean(text))
        if route is None:
            skip[f"{src}:audio_volume_mute_noise"] += 1
            return None
        if route == "stop":
            intent, tool = "stop", ""
    elif label == "goodbye":
        intent = "stop" if closes(probe) else "chat"
    elif label in _CALENDAR_LOOKUP:
        intent, tool = calendar_route(probe)
    elif label == "calendar_set" and _REMINDERS.search(probe):
        tool, kind = "reminder_add", "add_reminder"
    elif label in _LIST_EDITS and removes(probe):
        intent, tool, kind, d = "agent_task", "", "other", True
    elif label in ("music_likeness", "music_dislikeness") and _MUSIC_CMD.search(probe):
        intent, tool, reply = "quick_action", "", False
    if intent in ("quick_action", "agent_task") and label not in _NOT_HOW_TO and how_to(probe):
        intent, tool, kind, d, reply = "chat", "", None, False, True  # asks how, doesn't do it
    if intent == "stop":
        reply = False
    return row(text, intent, tool, src, label, lang=lang, split=split, group=group,
               kind=kind_for(intent, tool, kind), destructive=d, screen=False, reply=reply)


_CLINC_FILES = {"train": "plus/train-00000-of-00001.parquet", "eval": "plus/test-00000-of-00001.parquet"}


def load_clinc(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    for split, path in _CLINC_FILES.items():
        for r in fx.records("clinc", path, ["text", "intent"]):
            if r["intent"] in CLINC_SKIP:
                skip[f"clinc:{r['intent']}"] += 1
                continue
            if r["intent"] not in CLINC:
                skip[f"clinc:unmapped:{r['intent']}"] += 1
                continue
            out = _intent_row("clinc", CLINC, CLINC_KIND, CLINC_DESTRUCTIVE, r["intent"], r["text"], en=None,
                              lang="en", split=split, group=None, skip=skip)
            if out:
                yield out


def load_massive(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    """en + hi (Devanagari). Translations share MASSIVE's id, so they share a group, and a Hindi row
    is labelled from its English source sentence."""
    files = {(cfg, split): [json.loads(line) for line in fx.text("massive", f"{name}/{cfg}.json.gz").split("\n")
                            if line.strip()]
             for cfg in ("en", "hi") for split, name in (("train", "train"), ("eval", "test"))}
    english = {r["id"]: clean(r["text"]) for (cfg, _), rs in files.items() if cfg == "en" for r in rs}
    for (cfg, split), rs in files.items():
        for r in rs:
            label = r["label"]
            if label not in MASSIVE:
                skip[f"massive:unmapped:{label}"] += 1
                continue
            t = clean(r["text"])
            en = english.get(r["id"]) if cfg != "en" else None
            if len(t.split()) < 2 and not (label == "audio_volume_mute" and mute_route(en or t, t) == "stop"):
                continue  # "emails" alone is too ambiguous to learn from ("quiet" is a clear stop)
            out = _intent_row("massive", MASSIVE, MASSIVE_KIND, MASSIVE_DESTRUCTIVE, label, t, en=en,
                              lang=lang_of(t, "en" if cfg == "en" else "hinglish"), split=split,
                              group=f"massive:{r['id']}", skip=skip)
            if out:
                yield out


# ---- TOP-style parses: TOPv2, Hinglish-TOP, MTOP -------------------------------------------------
_WEB = ("GET_EVENT", "GET_DATE_TIME_EVENT", "GET_CATEGORY_EVENT", "GET_DIRECTIONS", "UPDATE_DIRECTIONS",
        "GET_LOCATION", "GET_DISTANCE", "GET_ESTIMATED_ARRIVAL", "GET_ESTIMATED_DEPARTURE", "GET_ESTIMATED_DURATION",
        "GET_INFO_ROAD_CONDITION", "GET_INFO_ROUTE", "GET_INFO_TRAFFIC", "GET_STORIES_NEWS", "GET_DETAILS_NEWS",
        "QUESTION_NEWS")
_PLAYER = ("PAUSE_MUSIC", "STOP_MUSIC", "RESUME_MUSIC", "SKIP_TRACK_MUSIC", "PREVIOUS_TRACK_MUSIC", "REPLAY_MUSIC",
           "LOOP_MUSIC", "UNLOOP_MUSIC", "START_SHUFFLE_MUSIC", "STOP_SHUFFLE_MUSIC", "REPEAT_ALL_MUSIC",
           "REPEAT_ALL_OFF_MUSIC", "FAST_FORWARD_MUSIC", "REWIND_MUSIC", "LIKE_MUSIC", "DISLIKE_MUSIC",
           "GET_TRACK_INFO_MUSIC", "QUESTION_MUSIC", "GET_LYRICS_MUSIC")
# NEO's timer tool only starts timers: managing one is a quick action with no single tool.
_TIMER_CTL = ("GET_TIMER", "PAUSE_TIMER", "RESUME_TIMER", "RESTART_TIMER", "ADD_TIME_TIMER", "SUBTRACT_TIME_TIMER",
              "UPDATE_TIMER", "DELETE_TIMER", "SILENCE_ALARM", "SNOOZE_ALARM")
# Calls, messages, playlists and settings, and looking things up in apps NEO has no tool for
# (alarms, reminders, call and message history, a contact's number): the agent does them.
_APP = ("ADD_TO_PLAYLIST_MUSIC", "CREATE_PLAYLIST_MUSIC", "FOLLOW_MUSIC", "ANSWER_CALL", "END_CALL", "HOLD_CALL",
        "IGNORE_CALL", "MERGE_CALL", "RESUME_CALL", "SWITCH_CALL", "UPDATE_CALL", "UPDATE_METHOD_CALL",
        "IGNORE_MESSAGE", "SET_DEFAULT_PROVIDER_CALLING", "SET_DEFAULT_PROVIDER_MUSIC", "GET_ALARM", "GET_REMINDER",
        "GET_REMINDER_AMOUNT", "GET_REMINDER_DATE_TIME", "GET_REMINDER_LOCATION", "GET_CALL", "GET_CALL_CONTACT",
        "GET_CALL_TIME", "GET_MESSAGE", "GET_MESSAGE_CONTACT", "GET_CONTACT_METHOD")
_TOP_STOP = ("stop", "", None, False)
# intent → (route, tool, kind, destructive). An intent that isn't here (or in _TOP_SKIP) is skipped and
# counted as unmapped, so a new one shows up in the build report instead of becoming an agent task.
TOP: dict[str, tuple[str, str, str | None, bool]] = {
    "CREATE_ALARM": ("quick_action", "reminder_add", "add_reminder", False),
    "CREATE_REMINDER": ("quick_action", "reminder_add", "add_reminder", False),
    "CREATE_TIMER": ("quick_action", "timer", "other", False),
    **dict.fromkeys(("GET_WEATHER", "GET_SUNRISE", "GET_SUNSET", "GET_AIRQUALITY"), ("quick_action", "weather", "other", False)),
    **dict.fromkeys(("PLAY_MUSIC", "PLAY_MEDIA"), ("quick_action", "play_music", "play_media", False)),
    **dict.fromkeys(("SEND_MESSAGE", "SEND_TEXT_MESSAGE", "REACT_MESSAGE"), ("agent_task", "", "send_message", True)),
    **dict.fromkeys(_WEB, ("quick_action", "web_search", "web_search", False)),
    **dict.fromkeys(_PLAYER + _TIMER_CTL, ("quick_action", "", "other", False)),
    **dict.fromkeys(("DELETE_ALARM", "DELETE_REMINDER", "DELETE_PLAYLIST_MUSIC", "REMOVE_FROM_PLAYLIST_MUSIC",
                     "CREATE_CALL", "SHARE_EVENT", "SET_RSVP_YES", "SET_RSVP_NO", "SET_RSVP_INTERESTED"),
                    ("agent_task", "", "other", True)),
    **dict.fromkeys(("UPDATE_ALARM", "UPDATE_REMINDER", "UPDATE_REMINDER_DATE_TIME", "UPDATE_REMINDER_TODO",
                     "UPDATE_REMINDER_LOCATION"), ("agent_task", "", "add_reminder", False)),
    **dict.fromkeys(("GET_RECIPES", "GET_INFO_RECIPES", "IS_TRUE_RECIPES", "HELP_REMINDER"), ("chat", "", None, False)),
    **dict.fromkeys(_APP, ("agent_task", "", "other", False)),
    # "stop sending that", "cancel this call": said while NEO is mid-send, exactly when stop matters
    **dict.fromkeys(("CANCEL_MESSAGE", "CANCEL_CALL"), _TOP_STOP),
}
# Not requests NEO acts on: preferences, unsupported, and MTOP's people / social-graph domain ("Is Nick
# Foles married?", "who is Ross?" — general knowledge or Facebook data), presence ("show me as busy":
# a status NEO can't set) and who is going to an event (Facebook's).
_TOP_SKIP = re.compile(
    r"^UNSUPPORTED|^(?:PREFER|DISPREFER|GET_GROUP|GET_EVENT_ATTENDEE\w*|GET_EVENT_ORGANIZER|GET_ATTENDEE_EVENT|"
    r"GET_(?:INFO_)?CONTACT|GET_EMPLOYER|GET_AGE|GET_GENDER|GET_LANGUAGE|GET_EDUCATION_\w+|GET_UNDERGRAD|"
    r"GET_MAJOR|GET_JOB|GET_EMPLOYMENT_TIME|GET_LIFE_EVENT\w*|GET_MUTUAL_FRIENDS|GET_AVAILABILITY|"
    r"SET_(?:UN)?AVAILABLE)$"
)
# MTOP's GET_LOCATION is where a person lives or works (the people domain again); Hinglish-TOP's is
# navigation ("Macys kaha hai"), a web search.
_MTOP_SKIP = frozenset(["GET_LOCATION"])
_TOP_INTENT = re.compile(r"^\s*\[IN:([A-Z_]+)")
# Said to a ringing alarm, a call or an incoming message, words that name nothing are NEO's stop.
_STOPPABLE = frozenset(["SILENCE_ALARM", "END_CALL", "IGNORE_MESSAGE"])
# "turn off tomorrow's alarm", "no alarms on weekends": not silencing one that rings — deleting.
_FUTURE_ALARM = re.compile(
    r"\b(?:tomorrow'?s?|tonight'?s?|every|weekdays?|weekends?|(?:mon|tues|wednes|thurs|fri|satur|sun)day'?s?|"
    r"next|(?:don'?t|do not|no longer) need|no alarms?|any alarms)\b",
    re.I,
)
# "Remind Erik that …", "send the team a reminder": a message to someone, not a reminder for me.
_FOR_OTHERS = re.compile(
    r"^\W*(?:please\s+)?(?:remind\s+(?!(?:me|us|myself|to|about|of|that|at|in|on|for|every|tomorrow|today|"
    r"tonight|this|next)\b)\w|send\s+(?!(?:a|an|the|me|us|myself|my)\b)\w+(?:\s+\w+)?\s+(?:an?\s+)?reminders?\b|"
    r"send\b.{0,40}?\breminders?\b.{0,30}?\bto\s+(?:the|my|all|everyone|everybody|each|every)\b)",
    re.I,
)


def top_labels(parse_or_intent: str, text: str = "", en: str = "") -> tuple[str, dict] | None:
    """(label, row kwargs) for a TOP/MTOP intent ("IN:GET_WEATHER" or a whole parse), None to skip.
    `text` is the sentence and `en` the English it translates (the sentence itself when empty):
    a few intents are refined by what they say."""
    name = _top_name(parse_or_intent)
    if not name or _TOP_SKIP.search(name) or name not in TOP:
        return None
    route, tool, kind, destructive = TOP[name]
    probe = en or text
    if name in _STOPPABLE and text and bare_stop(text):
        route, tool, kind, destructive = _TOP_STOP
    elif name == "SILENCE_ALARM" and _FUTURE_ALARM.search(probe):
        route, tool, kind, destructive = TOP["DELETE_ALARM"]
    elif name == "REMOVE_FROM_PLAYLIST_MUSIC" and probe and not removes(probe):
        route, tool, kind, destructive = TOP["DISLIKE_MUSIC"]  # "don't play this again", "skip all Bieber"
    elif name == "CREATE_REMINDER" and _FOR_OTHERS.search(probe):
        route, tool, kind, destructive = "agent_task", "", "send_message", True
    asks = name.startswith(("GET_", "QUESTION_", "IS_TRUE_"))
    return name.lower(), {"intent": route, "tool": tool, "kind": kind, "destructive": destructive,
                          "screen": False, "reply": route == "chat" or (asks and route != "stop")}


def _top_name(parse_or_intent: str) -> str | None:
    m = _TOP_INTENT.match(parse_or_intent) or re.match(r"^\s*(?:IN:)?([A-Z_]+)\s*$", parse_or_intent)
    return m.group(1) if m else None


def _top_row(src: str, text: str, parse: str, *, en: str = "", lang: str, split: str, group: str,
             skip: Counter) -> dict | None:
    lab = top_labels(parse, clean(text), clean(en) if en else "")
    if lab is None:
        name = _top_name(parse) or "?"
        known = name in TOP or _TOP_SKIP.search(name)
        skip[f"{src}:{'' if known else 'unmapped:'}{name.lower()}"] += 1
        return None
    label, kw = lab
    return row(text, kw.pop("intent"), kw.pop("tool"), src, label, lang=lang, split=split, group=group, **kw)


_HT_FILES = {"data/train-00000-of-00001-edd2764b201a5c1f.parquet": "train",
             "data/validation-00000-of-00001-5abcf62ed020ba5a.parquet": "train",
             "data/test-00000-of-00001-cf691a6b0d345ef6.parquet": "eval"}  # NEO's Hinglish benchmark


def load_hinglish_top(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    """Human Hinglish (+ the English it was translated from). Both are TOPv2 sentences, so the group
    is keyed on the English: a TOPv2 or synthetic copy of a test sentence is held out with it."""
    for path, split in _HT_FILES.items():
        for r in fx.records("hinglish_top", path, ["en_query", "cs_query", "cs_parse"]):
            g = group_id("top", r["en_query"])
            for text, lang, en in ((r["en_query"], "en", ""),
                                   (r["cs_query"], lang_of(r["cs_query"], "hinglish"), r["en_query"])):
                out = _top_row("hinglish_top", text, r["cs_parse"], en=en, lang=lang, split=split, group=g, skip=skip)
                if out:
                    yield out


def load_hinglish_synth(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    """rvv-karma's machine-made Hinglish only: its "human" rows are Hinglish-TOP's test split."""
    cols = ["en", "hi_en", "hi_en_parse", "generated_by"]
    for r in fx.records("hinglish_top_synth", "data/train-00000-of-00001.parquet", cols):
        if r["generated_by"] != "synthetic":
            continue
        out = _top_row("hinglish_top_synth", r["hi_en"], r["hi_en_parse"], en=r["en"],
                       lang=lang_of(r["hi_en"], "hinglish"), split="train", group=group_id("top", r["en"]), skip=skip)
        if out:
            yield out


def load_topv2(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    """Messages, reminders, alarms, timers, music, weather and events. Not navigation: CLINC, MASSIVE,
    SGD and Hinglish-TOP already make web_search the biggest tool class."""
    cols = ["domain", "utterance", "semantic_parse"]
    for r in fx.records("topv2", "data/train-00000-of-00001-4f5cf905029cbf9d.parquet", cols):
        if r["domain"] == "navigation":
            skip["topv2:navigation"] += 1
            continue
        out = _top_row("topv2", r["utterance"], r["semantic_parse"], lang="en", split="train",
                       group=group_id("top", r["utterance"]), skip=skip)
        if out:
            yield out


_MTOP_FILES = {"data/train_en-00000-of-00001-fc176d2de42e5a89.parquet": ("train", "en"),
               "data/train_hi-00000-of-00001-c43ac9eb28e5c7c5.parquet": ("train", "hi"),
               "data/test_en-00000-of-00001-32bc89f7a6f05de1.parquet": ("eval", "en"),
               "data/test_hi-00000-of-00001-b9aa5bde874a8000.parquet": ("eval", "hi")}


def load_mtop(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    """English + Devanagari; the two languages share MTOP's id, so they share a group (and a Hindi
    row is labelled from its English source sentence)."""
    files = {path: fx.records("mtop", path, ["id", "intent", "utterance"]) for path in _MTOP_FILES}
    english = {r["id"]: r["utterance"] for path, rs in files.items() if _MTOP_FILES[path][1] == "en" for r in rs}
    for path, (split, cfg) in _MTOP_FILES.items():
        for r in files[path]:
            if _top_name(r["intent"]) in _MTOP_SKIP:
                skip[f"mtop:{_top_name(r['intent']).lower()}"] += 1
                continue
            out = _top_row("mtop", r["utterance"], r["intent"], en=english.get(r["id"], "") if cfg != "en" else "",
                           lang=lang_of(r["utterance"], "en" if cfg == "en" else "hinglish"),
                           split=split, group=f"mtop:{r['id']}", skip=skip)
            if out:
                yield out


# ---- Hinglish chat -----------------------------------------------------------------------------
# Friends chatting ("kal party hai na?"); drop the lines that are really a command to someone, and
# weather and time questions ("kal ka weather kaisa rahega?"): NEO answers those with a tool.
_HI_COMMAND = re.compile(
    r"\b(?:chala(?:o|do|na)?|bajao|baja do|laga do|lagao|set kar\w*|band kar\w*|khol\w*|open|play|call kar\w*|"
    r"message kar\w*|msg kar\w*|bhej\w*|yaad dila\w*|remind\w*|alarm|timer|search|google kar\w*|dhoondh?\w*|"
    r"order kar\w*|book kar\w*|delete|download|install|weather|mausam|baa?rish|temperature|garmi kitni|"
    r"thand kitni|kitne baje|(?:kya|kitna) time|time kya)\b",
    re.I,
)
_HINGLISH_CSV = "hinglish_conversations.csv"
HINGLISH_CHAT_BYTES = 1_500_000


def load_hinglish_chat(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    head = fx.url_head(_hf_url("hinglish_chat", _HINGLISH_CSV), HINGLISH_CHAT_BYTES).decode("utf-8", "ignore")
    head = head[: head.rfind("\n") + 1]  # the last line is cut mid-way
    for r in csv.DictReader(io.StringIO(head)):
        t = clean(r.get("input") or "")
        if not 3 <= len(t.split()) <= 20 or _HI_COMMAND.search(t):
            skip["hinglish_chat:filtered"] += 1
            continue
        out = row(t, "chat", "", "hinglish_chat", "chat", lang=lang_of(t, "hinglish"), destructive=False,
                  screen=False, reply=True)
        if out:
            yield out


# ---- Leyzer (opt-in) ------------------------------------------------------------------------------
_LEYZER = "corpora/0.2.0/leyzer-0.2.0-fedcsis.tar.gz"
_OPEN = re.compile(r"\b(?:open|launch|start|bring up|go to|show me)\b", re.I)
# (domain, intent regex) → (route, tool, kind, destructive, screen, reply); first match wins.
LEYZER_RULES: list[tuple[str, str, tuple[str, str, str, bool, bool, bool]]] = [
    ("Console", r"ConsoleRM|ConsoleMV|ConsoleCP|ConsoleEdit", ("agent_task", "shell", "other", True, False, False)),
    ("Console", r"ConsoleLS|ConsoleCD", ("agent_task", "shell", "other", False, False, True)),
    ("Email", r"^Send|^Reply", ("agent_task", "mail_send", "send_email", True, False, False)),
    ("Email", r"^Show", ("quick_action", "mail_unread", "other", False, False, True)),
    ("Calendar", r"^AddEvent", ("quick_action", "calendar_add", "add_event", False, False, False)),
    ("Calendar", r"^Check", ("quick_action", "calendar_today", "other", False, False, True)),
    ("Calendar", r"Notify", ("agent_task", "", "add_reminder", False, False, False)),
    ("Spotify", r"^Play", ("quick_action", "play_music", "play_media", False, False, False)),
    ("Spotify", r"^(?:Next|Previous|Pause|Repeat|Shuffle|Seek)", ("quick_action", "", "other", False, False, False)),
    ("Spotify", r"^(?:Add|Create)", ("agent_task", "", "other", False, False, False)),
    ("Speaker", r"Volume|Mute", ("quick_action", "volume", "other", False, False, False)),
    ("Weather", r"^(?!Open)", ("quick_action", "weather", "other", False, False, True)),
    ("Websearch", r"^(?!Open)", ("quick_action", "web_search", "web_search", False, False, True)),
    ("Youtube", r"^Find", ("quick_action", "search_site", "web_search", False, False, True)),
    ("Wikipedia", r"Scroll|GoToElement", ("quick_action", "", "other", False, True, False)),
    ("Wikipedia", r"SearchQuery|RandomPage", ("quick_action", "search_site", "web_search", False, False, True)),
    ("Wikipedia", r"DownloadAsPdf", ("agent_task", "", "other", False, True, False)),
    ("Phone", r"^SMS", ("agent_task", "", "send_message", True, False, False)),
    ("Phone", r"^Call", ("agent_task", "", "other", True, False, False)),
    ("Slack", r"^Send", ("agent_task", "", "send_message", True, False, False)),
    ("Slack", r"^Set", ("agent_task", "", "other", True, False, False)),
    ("Slack", r"^Check", ("agent_task", "", "other", False, False, True)),
    ("Twitter", r"Send", ("agent_task", "", "send_message", True, False, False)),
    ("Twitter", r"Follow|Unfollow", ("agent_task", "", "other", True, False, False)),
    ("Facebook", r"^Post", ("agent_task", "", "other", True, False, False)),
    ("Instagram", r"^Post", ("agent_task", "", "other", True, False, False)),
    ("Instagram", r"^TakePicture", ("agent_task", "", "other", False, False, False)),
    ("Gdrive", r"^Share", ("agent_task", "", "other", True, False, False)),
    ("Gdrive", r"^Create", ("agent_task", "write_file", "other", False, False, False)),
    ("Contacts", r"^Edit|^Create", ("agent_task", "", "other", True, False, False)),
    ("Translate", r"^Translate|Detect", ("chat", "", "", False, False, True)),
    ("Yelp", r".", ("quick_action", "web_search", "web_search", False, False, True)),
    ("News", r"^Show", ("quick_action", "web_search", "web_search", False, False, True)),
]


def leyzer_labels(domain: str, intent: str, text: str) -> tuple[str, str, str, bool, bool, bool] | None:
    if domain == "Email" and intent == "OpenEmail" and not _OPEN.search(text):
        return ("quick_action", "mail_unread", "other", False, False, True)  # "do i have new emails"
    for d, rx, lab in LEYZER_RULES:
        if d == domain and re.search(rx, intent):
            return lab
    if intent.startswith("Open"):
        return ("quick_action", "open_app", "open_app", False, False, False)
    return None


def load_leyzer(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    """English only, all train: part of Leyzer is ASR-noised ("the firmest dout"), which is good
    for robustness and bad for a benchmark."""
    with tarfile.open(fx.url_file(_gh_url("leyzer", _LEYZER))) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("en-US.jsonl"))
        lines = tar.extractfile(member).read().decode("utf-8").split("\n")  # type: ignore[union-attr]
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        lab = leyzer_labels(r["domain"], r["intent"], r["utterance"])
        label = f"{r['domain']}.{r['intent']}"
        if lab is None:
            skip[f"leyzer:{label}"] += 1
            continue
        intent, tool, kind, destructive, screen, reply = lab
        if tool == "calendar_today":
            intent, tool = calendar_route(r["utterance"])
        out = row(r["utterance"], intent, tool, "leyzer", label, lang="en", kind=kind_for(intent, tool, kind or None),
                  destructive=destructive, screen=screen, reply=intent == "chat" or reply)
        if out:
            yield out


# ---- Schema-Guided Dialogue --------------------------------------------------------------------
# A spread of files so every service (incl. Payment/Messaging/Alarm, only in dev/test) shows up;
# each is ~3.4 MB, the whole train folder 439 MB.
SGD_FILES = [f"train/dialogues_{n:03d}.json" for n in (1, 17, 33, 49, 65, 81, 97, 113)] + [
    f"dev/dialogues_{n:03d}.json" for n in (1, 10, 20)] + [f"test/dialogues_{n:03d}.json" for n in (1, 12, 24, 34)]
_SGD_SCHEMAS = ("train/schema.json", "dev/schema.json", "test/schema.json")
# SGD's is_transactional, refined to NEO's "hard to undo": playing, alarms and calendar entries are
# reversible; money, bookings and anything sent to someone else are not.
SGD_INTENTS: dict[str, tuple[str, str, str, bool]] = {
    "GetWeather": ("quick_action", "weather", "other", False),
    "GetEvents": ("quick_action", "calendar_today", "other", False),
    "GetAvailableTime": ("quick_action", "calendar_today", "other", False),
    "AddEvent": ("quick_action", "calendar_add", "add_event", False),
    "AddAlarm": ("quick_action", "reminder_add", "add_reminder", False),
    "GetAlarms": ("agent_task", "", "other", False),
    "CheckBalance": ("agent_task", "", "other", False),
    "PlaySong": ("quick_action", "play_music", "play_media", False),
    "PlayMedia": ("quick_action", "play_music", "play_media", False),
    "PlayMovie": ("agent_task", "", "play_media", False),
    "RentMovie": ("agent_task", "", "play_media", True),
    "ShareLocation": ("agent_task", "", "send_message", True),
}
_SGD_UNCLEAR = frozenset(["LookupSong", "LookupMusic"])  # "find me songs" — then play? search? skipped


def sgd_labels(intent: str, transactional: bool) -> tuple[str, str, str, bool]:
    if intent in SGD_INTENTS:
        return SGD_INTENTS[intent]
    if transactional:  # Buy*, Reserve*, Book*, GetRide, TransferMoney, MakePayment, ScheduleVisit, …
        return ("agent_task", "", "other", True)
    return ("quick_action", "web_search", "web_search", False)  # Find*/Search*: look it up


def load_sgd(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    """The user turn that starts an intent (route, tool, kind, destructive, reply), turns that decline
    or end it ("no, not now", "that's all") as stop, and pure follow-up questions as chat. A goodbye
    or negation that only thanks ("Great, thank you") is chat; one that only answers ("No, I didn't")
    makes no sense without the question and is dropped."""
    transactional: dict[str, bool] = {}
    for p in _SGD_SCHEMAS:
        for svc in json.loads(fx.url_file(_gh_url("sgd", p)).read_text()):
            for it in svc["intents"]:
                transactional[it["name"]] = transactional.get(it["name"], False) or it["is_transactional"]
    for p in SGD_FILES:
        for d in json.loads(fx.url_file(_gh_url("sgd", p)).read_text()):
            for t in d["turns"]:
                if t["speaker"] != "USER":
                    continue
                acts = [a for fr in t["frames"] for a in fr["actions"]]
                names = {a["act"] for a in acts}
                text = t["utterance"]
                if "INFORM_INTENT" in names:
                    intent = next(a["canonical_values"][0] for a in acts if a["act"] == "INFORM_INTENT")
                    if intent in _SGD_UNCLEAR:
                        skip[f"sgd:{intent}"] += 1
                        continue
                    route, tool, kind, destructive = sgd_labels(intent, transactional.get(intent, False))
                    if tool == "calendar_today":  # "am i free on the 5th?" isn't today's calendar
                        route, tool = calendar_route(clean(text))
                    out = row(text, route, tool, "sgd", intent, lang="en", kind=kind_for(route, tool, kind),
                              destructive=destructive, screen=False,
                              reply=not transactional.get(intent, False) or "REQUEST" in names)
                elif names & {"NEGATE_INTENT", "GOODBYE"} or names == {"NEGATE", "THANK_YOU"}:
                    label = "negate_intent" if "NEGATE_INTENT" in names else "goodbye"
                    if closes(text):
                        out = row(text, "stop", "", "sgd", label, lang="en", destructive=False, screen=False, reply=False)
                    elif _THANKS.search(text):
                        out = row(text, "chat", "", "sgd", label, lang="en", destructive=False, screen=False, reply=True)
                    else:
                        skip[f"sgd:{label}_answer"] += 1
                        continue
                elif names == {"REQUEST"}:  # "what's their address?" — answered from the conversation
                    out = row(text, "chat", "", "sgd", "request", lang="en", destructive=False, screen=False, reply=True)
                else:
                    continue
                if out:
                    yield out


# ---- glaive function calling -------------------------------------------------------------------
# function-name regex → NEO tool (registry names); first match wins.
FN2NEO: list[tuple[re.Pattern, str]] = [(re.compile(p, re.I), t) for p, t in [
    (r"^(?:send|compose|draft|reply|forward)_?(?:an_)?e?mail", "mail_send"),
    # an inbox read, not check_email (glaive's address validator), check_email_availability, validate_email
    (r"^(?:get|read|fetch|list)_(?:my_)?(?:unread_|new_|latest_|recent_)?e?mails?$|^check_(?:my_)?(?:unread|new|inbox)|inbox",
     "mail_unread"),
    (r"(?:create|add|schedule|book|set)_(?:calendar_)?(?:event|meeting|appointment)|schedule_meeting|calendar_add", "calendar_add"),
    (r"(?:get|list|check|fetch)_(?:calendar|events|schedule|agenda|meetings)", "calendar_today"),
    (r"(?:create|add|set)_(?:todo|to_do|reminder|task)|remind", "reminder_add"),
    (r"(?:create|add|save|write|take)_(?:a_)?notes?$|create_note", "notes_create"),
    (r"play_(?:music|song|playlist|track|podcast|video|audio|media|radio)|music_play|start_playback", "play_music"),
    (r"(?:set|adjust|change|increase|decrease)_(?:system_)?volume|mute", "volume"),
    (r"brightness", "brightness"),
    (r"(?:set|start|create)_(?:a_)?timer|countdown", "timer"),
    (r"^(?:get_)?(?:current_)?(?:weather|forecast)|^weather_", "weather"),
    (r"^get_(?:current_)?(?:time|date)$|current_time|get_time_in|get_date", "clock"),
    (r"(?:search|query)_(?:the_)?(?:web|internet|google|online)|web_search|google_search|bing", "web_search"),
    (r"(?:get|fetch|read)_(?:web)?page|scrape|fetch_url|extract_(?:text|content)_from_url|get_url", "web_fetch"),
    (r"(?:get|search|fetch)_(?:latest_|top_)?news|news_headlines", "web_search"),
    (r"(?:open|launch|start)_(?:app|application|program)", "open_app"),
    (r"(?:quit|close|kill)_(?:app|application|program)", "quit_app"),
    (r"(?:delete|remove|trash)_(?:file|folder|directory)", "trash"),
    (r"(?:read|open|get)_(?:file|document)_?(?:content)?$|read_file|list_(?:files|directory)", "read_file"),
    (r"(?:write|create|save)_(?:file|document)|write_to_file", "write_file"),
    (r"(?:run|execute)_(?:shell|command|script|bash|terminal)", "shell"),
    (r"(?:take|capture)_screenshot|screenshot", "screenshot"),
    (r"(?:search|find)_(?:on_)?(?:youtube|amazon|wikipedia)", "search_site"),
]]
# Things NEO answers itself: a calculator call is a chat turn. Currency, stock and distance are not:
# they need today's numbers (qa_currency, qa_stock and distance map to web_search).
CHAT_FN = re.compile(
    r"^(?:calculate|compute|convert|get_(?:random_)?(?:joke|fact|quote|definition)|translate|generate_"
    r"(?:password|random|username|quote|poem|story|color)|analyze_sentiment|check_(?:palindrome|anagram|prime|"
    r"spelling)|is_|solve|reverse|count|sort|find_(?:max|min|median|gcd|lcm)|fibonacci|factorial|"
    r"(?:check|validate|verify)_email(?:_address)?$)",  # "is john.doe@example.com a valid address?"
    re.I,
)
_NOT_CHAT_FN = re.compile(r"currency|exchange|stock|distance|route|travel_time|shipping|crypto", re.I)
# tool → (route, kind, destructive, reply)
GLAIVE_TOOLS: dict[str, tuple[str, str, bool | None, bool]] = {
    "mail_send": ("agent_task", "send_email", True, False),
    "mail_unread": ("quick_action", "other", False, True),
    "calendar_add": ("quick_action", "add_event", False, False),
    "calendar_today": ("quick_action", "other", False, True),
    "reminder_add": ("quick_action", "add_reminder", False, False),
    "notes_create": ("quick_action", "write_note", False, False),
    "play_music": ("quick_action", "play_media", False, False),
    "volume": ("quick_action", "other", False, False),
    "brightness": ("quick_action", "other", False, False),
    "timer": ("quick_action", "other", False, False),
    "weather": ("quick_action", "other", False, True),
    "clock": ("quick_action", "other", False, True),
    "web_search": ("quick_action", "web_search", False, True),
    "search_site": ("quick_action", "web_search", False, True),
    "web_fetch": ("agent_task", "other", False, True),
    "open_app": ("quick_action", "open_app", False, False),
    "quit_app": ("quick_action", "other", False, False),
    "screenshot": ("quick_action", "other", False, False),
    "trash": ("agent_task", "other", True, False),
    "read_file": ("agent_task", "other", False, True),
    "write_file": ("agent_task", "other", True, False),
    "shell": ("agent_task", "other", None, False),  # "run ls" is harmless, "rm -rf" isn't: unknown
}
_CALL = re.compile(r'<functioncall>\s*\{\s*"name"\s*:\s*"([^"]+)"')


def neo_tool(fn: str) -> str | None:
    for rx, t in FN2NEO:
        if rx.search(fn):
            return t
    return None


def glaive_call(chat: str) -> tuple[str, str | None]:
    """(first user message, the function called in answer to it) — the call has to come before the
    user's third message, or the first message wasn't what it was for."""
    first, users = "", 0
    for turn in chat.split("\n\n\n"):
        turn = turn.strip()
        if turn.startswith("USER:"):
            users += 1
            if users == 1:
                first = turn[5:].strip()
            if users > 2:
                break
        elif turn.startswith("ASSISTANT:") and "<functioncall>" in turn:
            m = _CALL.search(turn)
            return first, m.group(1) if m else None
    return first, None


def _glaive_pairs(fx: Fetcher, skip: Counter, src: str) -> Iterator[tuple[str, str]]:
    for r in fx.records("glaive", "default/train/0000.parquet", ["chat"]):
        text, fn = glaive_call(r["chat"] or "")
        if not fn:
            continue
        if "\n" in text or "```" in text or not 3 <= len(text.split()) <= 30:
            skip[f"{src}:long_or_multiline"] += 1
            continue
        yield text, fn


# "what events are happening in New York on the 25th": public events, not the user's calendar.
_PUBLIC_EVENTS = re.compile(r"\bhappening\b|\b(?:in|near|around)\s+[A-Z][a-z]")


def load_glaive(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    for text, fn in _glaive_pairs(fx, skip, "glaive"):
        tool = neo_tool(fn)
        if not tool:
            continue
        route, kind, destructive, reply = GLAIVE_TOOLS[tool]
        if tool == "calendar_today":
            if _PUBLIC_EVENTS.search(text):
                skip["glaive:public_events"] += 1
                continue
            route, tool = calendar_route(text)
        out = row(text, route, tool, "glaive", fn, lang="en", kind=kind, destructive=destructive, screen=False,
                  reply=reply)
        if out:
            yield out


def load_glaive_chat(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    for text, fn in _glaive_pairs(fx, skip, "glaive_chat"):
        if neo_tool(fn) or not CHAT_FN.search(fn) or _NOT_CHAT_FN.search(fn):
            continue
        out = row(text, "chat", "", "glaive_chat", fn, lang="en", destructive=False, screen=False, reply=True)
        if out:
            yield out


# ---- computer-use instructions: AgentNet, OmniACT ------------------------------------------------
# needs_screen only when the wording points at something visible; "rename the recording in Zoom"
# might or might not need the screen, so it stays unknown rather than becoming a negative. A click
# ("click the Send button", "click Python on python.org") is not a pointer: NEO's own rows label
# those False (click_text finds a label without looking), so they stay unknown too.
DEICTIC = re.compile(
    r"\b(?:this|these)\s+(?:\w+\s+){0,2}?(?:window|page|tab|file|document|doc|image|picture|photo|pdf|e-?mail|"
    r"message|slide|cell|row|column|table|text|paragraph|sentence|word|link|button|icon|folder|item|song|track|"
    r"video|chart|sheet|spreadsheet|form|field|dialog|menu|app|screen|one|article|site|website|playlist|note|"
    r"event|contact|list|selection|error)s?\b"
    r"|\bcurrent(?:ly)?\s+(?:open(?:ed)?|selected|active|displayed|shown|visible|page|window|tab|file|document|"
    r"slide|sheet|image|folder|directory|view|screen|cell|row|web\s?page|article|e-?mail|message)\b"
    r"|\b(?:selected|highlighted)\b|\bon[- ](?:the |my )?screen\b|\bvisible\b"
    r"|\bthe\s+(?:open(?:ed)?|active|frontmost)\s+(?:window|tab|page|document|file|app)\b"
    r"|\b(?:that|those)\s+(?:[\"'\w-]+\s+){0,3}(?:button|icon|checkbox|check\s+box|dropdown|drop-down|toggle)s?\b",
    re.I,
)
# A single click / key / scroll / typed text is a NEO quick action (click_text, hotkey, scroll,
# type_text), not an agent task.
_BARE_CONTROL = re.compile(
    r"^(?:click|tap|press|hit|scroll|zoom|double[- ]click|right[- ]click|hover|select|type|enter|input)\b", re.I)
_MULTI_STEP = re.compile(r",|;|\b(?:and|then|after|before|to (?:open|see|view|go|find|create|add|save))\b", re.I)


def _single_control(text: str) -> bool:
    return bool(_BARE_CONTROL.match(text)) and not _MULTI_STEP.search(text)


def _computer_use_row(src: str, label: str, text: str) -> dict | None:
    return row(text, "agent_task", "", src, label, lang="en", screen=True if DEICTIC.search(text) else None,
               reply=asks_something(clean(text)))


_NOT_AN_INSTRUCTION = re.compile(r"^Recording-|\(Untitled\)|pyautogui", re.I)  # a file name, a script


def load_agentnet(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    """Short single-sentence tasks only: most AgentNet instructions are numbered step lists, and
    nobody says those to a voice assistant."""
    for line in fx.text("agentnet", "meta_data_merged.jsonl").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        raw = (r.get("instruction") or "").strip()
        text = " ".join(raw.split())
        domain = str(r.get("domains") or "other")
        if ("\n" in raw or not 3 <= len(text.split()) <= 25 or domain in ("infeasible", "error_correction")
                or r.get("task_description_alignment") in ("Low Alignment", "Not Aligned") or _single_control(text)
                or _NOT_AN_INSTRUCTION.search(text)):
            skip["agentnet:filtered"] += 1
            continue
        out = _computer_use_row("agentnet", re.sub(r"\W+", "_", domain.lower()).strip("_"), text)
        if out:
            yield out


def load_omniact(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    texts = fx.zip_texts(_hf_url("omniact", "data.zip"), lambda n: "/tasks/" in n and n.endswith(".txt"), "tasks")
    for name, body in sorted(texts.items()):
        task = re.split(r"Output [Ss]cript|pyautogui\.", body)[0].replace("Task:", "").strip()
        text = " ".join(task.split())
        parts = name.split("/")
        if not 3 <= len(text.split()) <= 25 or _single_control(text):
            skip["omniact:filtered"] += 1
            continue
        out = _computer_use_row("omniact", f"{parts[-3]}/{parts[-2]}", text)
        if out:
            yield out


# ---- small cleanups: HWU64 stop, TigreGotico polite requests -------------------------------------
def load_hwu64(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    """general_commandstop ("olly that's enough, cancel", "please shut up"); the rest of HWU64 is
    MASSIVE again (both come from SLURP)."""
    names = {r["id"]: r["name"] for r in fx.records("hwu64", "intents/intents-00000-of-00001.parquet", ["id", "name"])}
    for path in ("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"):
        for r in fx.records("hwu64", path, ["utterance", "label"]):
            if names.get(r["label"], r["label"]) == "general_commandstop":
                out = row(r["utterance"], "stop", "", "hwu64", "general_commandstop", lang="en", destructive=False,
                          screen=False, reply=False)
                if out:
                    yield out


# A polite request ("Could you turn off the lights, please") wants doing, not an answer: the reply
# head's hard negatives. Requests for information, advice or checking are dropped (they do want an
# answer: "Can I have the latest weather update", "Could you help me with a budget plan"), and so are
# requests only a person could do ("lend me a hand with this box", "help me carry these boxes").
_ASKS_INFO = re.compile(
    r"\b(?:tell|show|explain|describe|recommend|suggest|list|name|find|search|look up|check|read|give me|"
    r"let me know|what|which|who|where|when|how|information|details|overview|summary of|biography|"
    r"profiles?|adult|directions?|address|location|nearest|weather|(?:latest|an|any|status)\s+updates?|verify|"
    r"feedback|advi[cs]e|plan|"
    r"routine|outline|translate|help me with|assist me)\b",
    re.I,
)
_PHYSICAL = re.compile(r"\b(?:hand me|pass me|lend me|borrow|feed|bring me|a hand|glass of|cup of tea|carry|lift|a ride|drive me|"
                       r"pick me up|your (?:coat|umbrella|book|notes|pen))\b", re.I)
_DEVICE = re.compile(
    r"\b(?:turn|switch)\s+(?:[\w']+\s+){0,3}?(?:on|off|up|down)\b|\b(?:volume|brightness)\b|"
    r"\b(?:mute|unmute|pause|resume|skip|dim|brighten|louder|quieter)\b|"
    r"\bplay(?:ing)?\b|\b(?:set|start)\s+(?:up\s+)?(?:an?\s+|the\s+)?(?:new\s+)?(?:alarm|timer|reminder|volume|"
    r"thermostat|temperature)\b|\b(?:set|get|create|add)\s+(?:an?\s+|the\s+)?(?:\w+\s+){0,2}reminder\b|"
    r"\bremind me\b|\b(?:increase|decrease|raise|lower|adjust)\s+(?:the\s+)?(?:room\s+)?(?:volume|brightness|"
    r"temperature|thermostat|bass|treble)\b|\b(?:fast forward|rewind|next track|previous track|channel)\b|"
    r"\b(?:room temperature|thermostat|lights?|coffee|fan|tv|dehumidifier|humidifier|heater)\b",
    re.I,
)
_TIGRE_TOOL = [(re.compile(p, re.I), t) for p, t in [
    (r"\balarm\b|\bremind", "reminder_add"), (r"\btimer\b", "timer"), (r"\bplay(?:ing)?\b", "play_music"),
    (r"\bvolume\b|\bmute\b|\bunmute\b|\blouder\b|\bquieter\b", "volume"), (r"\bbrightness\b|\bdim\b", "brightness"),
]]


def load_sentence_types(fx: Fetcher, skip: Counter) -> Iterator[dict]:
    for r in csv.DictReader(io.StringIO(fx.text("sentence_types", "utterance_tags_v0.3.csv"))):
        if (r.get("label"), r.get("sublabel")) != ("QUESTION", "REQUEST"):
            continue
        t = clean(r.get("sentence") or "")
        if _ASKS_INFO.search(t) or _PHYSICAL.search(t):
            skip["sentence_types:asks_or_physical"] += 1
            continue
        if _DEVICE.search(t):
            tool = next((tl for rx, tl in _TIGRE_TOOL if rx.search(t)), "")
            intent = "quick_action"
        else:
            tool, intent = "", "agent_task"
        out = row(t, intent, tool, "sentence_types", "question_request", lang="en", reply=False)
        if out:
            yield out


# ---- the registry ------------------------------------------------------------------------------
Loader = Callable[[Fetcher, Counter], Iterable[dict]]


@dataclass(frozen=True)
class Source:
    """A public source and how much of it to keep: at most `cap` train rows per key (by default the
    source intent × language), `total` overall, drawn round-robin across keys so rare intents keep
    what they have."""

    name: str
    load: Loader
    licence: str
    cap: int
    total: int | None = None
    key: Callable[[dict], str] | None = None
    default: bool = True
    note: str = ""

    def key_of(self, r: dict) -> str:
        return self.key(r) if self.key else f"{r['hf_intent']}|{r['lang']}"


# In priority order: a sentence two sources share is kept from the first (human before synthetic).
SOURCES: list[Source] = [
    Source("clinc", load_clinc, "CC-BY-3.0", cap=60),
    Source("massive", load_massive, "CC-BY-4.0", cap=60),
    Source("hinglish_top", load_hinglish_top, "Apache-2.0; translations of TOPv2 (CC-BY-SA-4.0): ShareAlike", cap=300),
    Source("hinglish_top_synth", load_hinglish_synth, "Apache-2.0; translations of TOPv2 (CC-BY-SA-4.0): ShareAlike",
           cap=80),
    Source("hinglish_chat", load_hinglish_chat, "MIT", cap=1500),
    Source("topv2", load_topv2, "CC-BY-SA-4.0", cap=150),
    Source("leyzer", load_leyzer, "CC-BY-NC-ND-4.0 (data repo LICENSE) / CC-BY-4.0 (HF card)", cap=50, default=False,
           note="its data repository is licensed non-commercial, no-derivatives; opt in with --with leyzer"),
    Source("mtop", load_mtop, "CC-BY-SA-4.0", cap=80),
    Source("sgd", load_sgd, "CC-BY-SA-4.0", cap=60),
    Source("glaive", load_glaive, "Apache-2.0", cap=250, key=lambda r: r["tool"]),
    Source("glaive_chat", load_glaive_chat, "Apache-2.0", cap=4, total=600),
    # keyed by needs_screen as well, so the scarce screen positives aren't sampled away
    Source("agentnet", load_agentnet, "MIT", cap=100, total=1500, key=lambda r: f"{r['hf_intent']}|{r.get('screen')}"),
    Source("omniact", load_omniact, "MIT", cap=40, total=1200, key=lambda r: f"{r['hf_intent']}|{r.get('screen')}"),
    Source("hwu64", load_hwu64, "CC-BY-4.0", cap=200),
    Source("sentence_types", load_sentence_types, "none stated", cap=500, default=False,
           note="no licence stated (no rights granted), and the noisiest mapping; opt in with --with sentence_types"),
]
# The benchmark's sources: their eval splits are always loaded, so the holdout doesn't depend on
# which sources a build uses.
BENCHMARK = ("clinc", "massive", "hinglish_top", "mtop")
BY_NAME = {s.name: s for s in SOURCES}
