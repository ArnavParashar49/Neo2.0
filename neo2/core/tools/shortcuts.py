"""shortcut_run: the user's own macOS Shortcuts, run by name as NEO verbs.

"Run my Send ETA shortcut", "good morning shortcut chalao": NEO runs the shortcut the user made in the
Shortcuts app, through the system's own command-line tool (``/usr/bin/shortcuts``), and says what it
returned. Nothing here knows any shortcut, app or site: the names are whatever the user's own list holds.

**Their list.** ``shortcuts list --show-identifiers`` (capped at 5 s), parsed into names and identifiers
and kept in RAM for an hour: read when NEO starts (:func:`start`), again when the user starts talking to
NEO after it went stale (a wake warmer), and by the tool itself when it is stale or misses a name. It is
never written anywhere and never logged: the core log has its length. The fast paths below read only
what is cached (a ``check`` must never wait), so a shortcut made a minute ago is found by the tool,
through a model, until the next refresh.

**Who decides it runs.** A shortcut can do anything its maker built into it (send a message, turn off
the lights), so the user's own words are the only thing that runs one straight away:

* the user's sentence names it — a fast path ("run my <name> shortcut", the name exactly as their list
  has it) on the words the Session marked as theirs, or the sentence Gemini Live hands its own direct
  call (marked LIVE_CALL), which must name the same shortcut;
* anything else — a model inside the agent loop, Live's model on its own idea, a goal Live hands the
  Session in its model's words (agent_task: origin "live", but no direct call), an approval replay of
  a card someone else staged, or a request that has read something untrusted (a page, a mail) — stages
  a card first, "Run your shortcut “<name>”?", whatever NEO's confirmation setting says.

A name the user didn't say exactly is matched by the memory's embedder (cosine at least 0.85) only when
that model is already loaded: it is never loaded for this. No match lists the three closest names.

**Running it.** ``shortcuts run <identifier> [--input-path <file>] --output-path <file>`` in a fresh
private (0700) temporary folder that is removed afterwards, in a session of its own (``start_new_session``):
after 30 s, or when the call is cancelled, the whole process group is killed. (macOS runs the shortcut's
actions in its own background runner: one that is already under way may still finish there; NEO only
stops waiting.) Text output is capped at 2,000 characters and reaches the model fenced as untrusted
content (the request is tainted, so a send that follows asks first); anything else it returns is
described by its size and dropped with the folder.

**What is kept.** Nothing new: the core log has the call (the name's length, never the name) and how it
ended; a staged card goes to the audit log like every confirmation (the shortcut's name; the input's
length, never its words). ``NEO_SHORTCUTS=0`` turns it all off (the tool is hidden from every model and
refuses), and the names reach a cloud model's tool list only with ``NEO_SHORTCUTS_NAMES_IN_PROMPT=1``.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from neo2.core import logger, runtime
from neo2.core.agent.registry import LIVE_CALL, USERS_WORDS, ToolContext, ToolOutput, catalog, tool
from neo2.core.config import settings

log = logger(__name__)

TOOL = "shortcut_run"
SHORTCUTS = "/usr/bin/shortcuts"
LIST_TIMEOUT_S = 5.0
RUN_TIMEOUT_S = 30.0
TTL_S = 3600.0  # how long a read of the list is fresh
_RETRY_S = 60.0  # a name the list doesn't have reads it again, at most this often
MAX_OUTPUT = 2000  # characters of what a shortcut returned that reach the model
_READ_BYTES = 64 * 1024  # what is read of its output file at most
MAX_INPUT = 4000  # characters of text handed to a shortcut as its input
FUZZY = 0.85  # the embedder's cosine for a name the user didn't say exactly
NAMES_IN_PROMPT = 30
_NAME_CHARS = 40  # a name in the tool's description, at most
_MAC = frozenset({"macos"})

BASE_DESCRIPTION = (
    "Run one of the user's own Shortcuts (the macOS Shortcuts app's, by name). shortcut: its name as the "
    "user said it; text: optional text to hand it as input. Only when the user asks to run one of their "
    "shortcuts. A run the user didn't ask for in their own words asks them first."
)


@dataclass(frozen=True)
class Shortcut:
    name: str
    id: str  # its identifier, as `shortcuts list --show-identifiers` prints it ("" when none was printed)


# ---- the user's list ----------------------------------------------------------------------------------

# "Name (UUID)": the identifier is the last parenthesis, so a name may hold parentheses of its own.
_LINE = re.compile(
    r"^(?P<name>.*\S)\s+\((?P<id>[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\)\s*$"
)


def parse(text: str) -> list[Shortcut]:
    """`shortcuts list --show-identifiers` as Shortcuts, in its order. A line without an identifier is
    a name alone (run by its name)."""
    out: list[Shortcut] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE.match(line)
        out.append(Shortcut(m["name"].strip(), m["id"].upper()) if m else Shortcut(line, ""))
    return out


_lock = threading.Lock()  # one read of the list at a time: a second caller waits and takes its result
_list: list[Shortcut] | None = None  # None: never read
_read_at = 0.0  # time.monotonic() of the last read that worked


def _list_text() -> str:
    """The CLI's list, capped at LIST_TIMEOUT_S (the seam the tests fake)."""
    out = subprocess.run(
        [SHORTCUTS, "list", "--show-identifiers"],
        capture_output=True, text=True, timeout=LIST_TIMEOUT_S, stdin=subprocess.DEVNULL, check=False,
    )  # fmt: skip
    if out.returncode != 0:
        raise RuntimeError(f"shortcuts list exited {out.returncode}")
    return out.stdout


def cached() -> list[Shortcut] | None:
    """The list as last read (None: never read). Never reads it."""
    return _list


def _age() -> float:
    return time.monotonic() - _read_at if _list is not None else float("inf")


def refresh(max_age: float = 0.0) -> list[Shortcut] | None:
    """Read the list now (blocking, at most LIST_TIMEOUT_S), unless a read that finished while this
    waited for the one before it is younger than `max_age`. A read that fails keeps what was there."""
    global _list, _read_at
    with _lock:
        if _age() <= max_age:
            return _list
        try:
            got = parse(_list_text())
        except (OSError, subprocess.SubprocessError, RuntimeError, UnicodeError) as e:
            log.info(f"[shortcuts] couldn't read the list ({type(e).__name__})")
            return _list
        _list, _read_at = got, time.monotonic()
    log.info(f"[shortcuts] {len(got)} in the list")
    _describe(got)
    return got


async def listing(max_age: float = TTL_S) -> list[Shortcut] | None:
    """The list, read again first when it is older than `max_age` (off the event loop)."""
    if _age() <= max_age:
        return _list
    return await asyncio.to_thread(refresh, max_age)


def forget() -> None:
    """Drop the cached list (tests; a fresh start)."""
    global _list, _read_at
    with _lock:
        _list, _read_at = None, 0.0


# ---- on and off --------------------------------------------------------------------------------------


def enabled() -> bool:
    """NEO_SHORTCUTS: on unless the user switched it off. Never an embedded agent's."""
    if runtime.library_mode() or runtime.library_bound():
        return False
    return bool(getattr(settings(), "shortcuts", True))


def _names_in_prompt() -> bool:
    return bool(getattr(settings(), "shortcuts_names_in_prompt", False))


def _describe(have: list[Shortcut]) -> None:
    """The tool's description: with NEO_SHORTCUTS_NAMES_IN_PROMPT on, the first NAMES_IN_PROMPT names
    too (they then go to whichever model gets the tool, a cloud one included); off, none."""
    t = catalog().get(TOOL)
    if t is None:
        return
    try:
        names = [_spoken(s.name)[:_NAME_CHARS] for s in have[:NAMES_IN_PROMPT]] if _names_in_prompt() else []
    except Exception:  # noqa: BLE001 — no settings (an embedding host, unbound): no names
        names = []
    names = [n for n in names if n]
    t.description = BASE_DESCRIPTION + (f" The user's shortcuts: {'; '.join(names)}." if names else "")


def start() -> None:
    """NEO.app is starting: hide the tool when the user switched it off; otherwise read the list in the
    background, and again when the user starts talking to NEO once it is stale."""
    t = catalog().get(TOOL)
    on = enabled()
    if t is not None:
        t.hidden = not on
    if not on:
        return
    threading.Thread(target=refresh, args=(TTL_S,), name="neo-shortcuts", daemon=True).start()
    from neo2.core.agent import prefetch

    prefetch.register_warmer(_warm_if_stale)


def _warm_if_stale() -> None:
    if enabled() and _age() > TTL_S:
        refresh(TTL_S)


# ---- which shortcut the words mean ---------------------------------------------------------------------

_LEAD = re.compile(r"^(?:my|the|a|mera|meri|mere)\s+")


def _spoken(name: str) -> str:
    """A name as it reads aloud: no quotes, no line breaks, one space."""
    return " ".join(str(name or "").replace('"', "").replace("“", "").replace("”", "").split())


def norm(s: str) -> str:
    """Lower case, letters and digits only ("Mom's ETA ⏱" → "moms eta", "Work & Play" → "work and play")."""
    s = unicodedata.normalize("NFKC", str(s or "")).casefold()
    s = re.sub(r"[’'`]", "", s).replace("&", " and ")
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def _candidates(asked: str) -> list[str]:
    """What the user's words may name: as said, without a leading "my"/"the", and with "shortcut"
    after it (a shortcut called "Morning Shortcut")."""
    key = norm(asked)
    bare = _LEAD.sub("", key)
    out = [key, bare, f"{key} shortcut", f"{bare} shortcut"]
    return [k for k in dict.fromkeys(out) if k and k != "shortcut"]


def exact(asked: str, have: list[Shortcut]) -> Shortcut | None:
    """The shortcut these words name exactly (case, punctuation and a leading "my" aside)."""
    by = {}
    for s in have:
        by.setdefault(norm(s.name), s)
    for k in _candidates(asked):
        if k in by:
            return by[k]
    return None


def _similar(asked: str, have: list[Shortcut]) -> Shortcut | None:
    """The closest name by meaning, at FUZZY or more — only when the memory's embedder is loaded
    already (it is never loaded for this)."""
    try:
        from neo2.core.memory import embed

        if not have or not embed.ready():
            return None
        v = embed.embed_texts([norm(asked)] + [norm(s.name) for s in have])
    except Exception:  # noqa: BLE001 — a guess is never worth a failure
        return None
    if v is None or len(v) != len(have) + 1:
        return None
    sims = v[1:] @ v[0]
    i = int(sims.argmax())
    return have[i] if float(sims[i]) >= FUZZY else None


def resolve(asked: str, have: list[Shortcut]) -> Shortcut | None:
    return exact(asked, have) or _similar(asked, have)


def closest(asked: str, have: list[Shortcut], n: int = 3) -> list[str]:
    key = norm(asked)
    ranked = sorted(
        have, key=lambda s: difflib.SequenceMatcher(None, key, norm(s.name)).ratio(), reverse=True
    )
    return [_spoken(s.name) for s in ranked[:n]]


# ---- the words ----------------------------------------------------------------------------------------
# Whole-utterance and anchored, each with the word "shortcut": "good morning" alone is a greeting, never a
# run, whatever the user named a shortcut. The check reads the cached list only: a name it doesn't have
# goes on to the other routes.
_P = (
    r"^\s*(?:(?:hey\s+|ok(?:ay)?\s+)?neo[\s,:!.-]+)?(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+)*"
)
_END = r"(?:\s+(?:for\s+me|please|now|na|yaar))*[\s?.!।]*$"
_RUN = r"(?:run|start|trigger|launch|fire(?:\s+off)?|do)"
_HI_RUN = r"(?:chalao|chalaao|chala\s*do|chala\s+dijiye|chalaiye|run\s+karo|run\s+kar\s*do|chalu\s+karo|start\s+karo)"
_WALA = r"(?:\s+(?:wala|waala|vala|wale|waale))?"


def _named(args: dict) -> bool:
    """The fast paths' check: the words name a shortcut the cached list has exactly (and running them
    is on). Never reads the list: a check must not wait."""
    have = _list
    return bool(have) and enabled() and exact(str(args.get("shortcut") or ""), have) is not None


_PATTERNS = [
    # "run my Send ETA shortcut", "could you start the good morning shortcut"
    _P + _RUN + r"\s+(?P<shortcut>.+?)\s+shortcut" + _END,
    # "run the shortcut called Send ETA", "run shortcut Send ETA"
    _P + _RUN + r"\s+(?:my\s+|the\s+)?shortcut\s+(?:called\s+|named\s+)?(?P<shortcut>.+?)" + _END,
    # "good morning shortcut chalao", "mera send ETA wala shortcut chala do"
    r"^\s*(?P<shortcut>.+?)" + _WALA + r"\s+shortcut\s+" + _HI_RUN + _END,
    # "गुड मॉर्निंग शॉर्टकट चलाओ"
    r"^\s*(?P<shortcut>.+?)(?:\s+वाला)?\s+शॉर्टकट\s+(?:चलाओ|चला\s+दो|चलाइए|चलाइये|रन\s+करो)" + _END,
]
FAST_PATHS = [(p, {"shortcut": "<shortcut>"}, _named) for p in _PATTERNS]


def named_in(words: str, have: list[Shortcut]) -> Shortcut | None:
    """The shortcut a sentence of the user's names through the fast paths' own words, said whole or as
    one of its clauses ("open Notes and run my Send ETA shortcut"): None when it names none exactly."""
    from neo2.core.agent.fastpath import SPLIT

    words = words or ""
    for part in (words, *SPLIT.split(words)):
        for pattern in _PATTERNS:
            m = re.search(pattern, part.strip(" ,.;!?"), re.I)
            if m and (sc := exact(m["shortcut"] or "", have)) is not None:
                return sc
    return None


def _users_pick(ctx: ToolContext, sc: Shortcut, have: list[Shortcut]) -> bool:
    """Did the user's own words choose this shortcut — a fast path's sentence the Session marked as
    theirs (USERS_WORDS), or the sentence Gemini Live hands its own direct call (LIVE_CALL) — in a
    request that has read nothing untrusted? A request Live hands the Session in its model's words
    (agent_task's goal: origin "live", but no direct call) is a model's choice, however it is worded."""
    if ctx.tainted():
        return False
    if ctx.origin == "user":
        words = str(ctx.get(USERS_WORDS, "") or "")
    elif ctx.origin == "live" and ctx.get(LIVE_CALL) is True:
        words = ctx.user_text
    else:
        return False
    return bool(words) and named_in(words, have) == sc


# ---- running it ----------------------------------------------------------------------------------------

_spawn = asyncio.create_subprocess_exec  # (the tests' seam)
_killpg = os.killpg


async def _stop(proc) -> None:
    """Kill the run's whole process group (it started a session of its own), then reap it."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        _killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), 2.0)


def _why(err: bytes) -> str:
    """The CLI's failure in NEO's own words (its text is never passed on: a shortcut can put words there)."""
    e = (err or b"").decode("utf-8", "replace").casefold()
    if ("couldn" in e and "find" in e) or "not found" in e or "no shortcut" in e:
        return "Shortcuts couldn't find it"
    if "permission" in e or "not allowed" in e or "denied" in e:
        return "Shortcuts wasn't allowed to do something it needs"
    return "it stopped with an error"


def _output(path: Path) -> tuple[str, int]:
    """(its text, or "" when it isn't text; its size in bytes). (0 bytes: it returned nothing.)"""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return "", 0
    with path.open("rb") as f:
        data = f.read(_READ_BYTES)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "", size
    return ("" if "\x00" in text else text.strip()), size


async def run(sc: Shortcut, text: str = "") -> ToolOutput:
    name = _spoken(sc.name)
    tmp = Path(tempfile.mkdtemp(prefix="neo-shortcut-"))  # 0700, removed below whatever happens
    try:
        argv = [SHORTCUTS, "run", sc.id or sc.name]
        if text:
            src = tmp / "input.txt"
            with os.fdopen(
                os.open(src, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8"
            ) as f:
                f.write(text)
            argv += ["--input-path", str(src)]
        dst = tmp / "output"
        argv += ["--output-path", str(dst)]
        t0 = time.monotonic()
        proc = await _spawn(
            *argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), RUN_TIMEOUT_S)
        except TimeoutError:
            await _stop(proc)
            log.info(f"[shortcuts] run timed out after {RUN_TIMEOUT_S:.0f}s")
            return ToolOutput(
                f"Error: your shortcut “{name}” didn't finish within {RUN_TIMEOUT_S:.0f} seconds, so NEO stopped "
                "waiting. It may be waiting for a permission prompt in Shortcuts.",
                ok=False,
            )
        except asyncio.CancelledError:
            await _stop(proc)
            raise
        log.info(f"[shortcuts] run exit={proc.returncode} s={time.monotonic() - t0:.1f}")
        if proc.returncode != 0:
            return ToolOutput(f"Error: your shortcut “{name}” didn't work: {_why(err)}.", ok=False)
        body, size = _output(dst)
        if body:
            if len(body) > MAX_OUTPUT:
                body = body[:MAX_OUTPUT].rstrip() + " …(cut)"
            return ToolOutput(f"Ran your shortcut “{name}”. It returned:\n{body}", say=True)
        if size:
            kb = max(1, round(size / 1024))
            return ToolOutput(
                f"Ran your shortcut “{name}”. It returned something that isn't text ({kb} KB), which NEO doesn't keep.",
                say=True,
            )
        return ToolOutput(f"Ran your shortcut “{name}”.", say=False)  # done: nothing to say
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- the tool -----------------------------------------------------------------------------------------


def _detail(sc: Shortcut, text: str) -> str:
    line = f"Shortcut: {_spoken(sc.name)}"
    if text:
        shown = " ".join(text.split())
        line += f"\nInput: {shown[:200]}{'…' if len(shown) > 200 else ''}"
    return line


@tool(
    TOOL,
    BASE_DESCRIPTION,
    {
        "type": "object",
        "properties": {
            "shortcut": {"type": "string", "description": "The shortcut's name, as the user said it."},
            "text": {"type": "string", "description": "Optional text to hand the shortcut as its input."},
        },
        "required": ["shortcut"],
    },
    risk="reversible",
    category="shortcuts",
    slow=True,
    fast_path=FAST_PATHS,
    timeout=RUN_TIMEOUT_S + LIST_TIMEOUT_S + 10,
    untrusted=True,
    requires=_MAC,
)
async def shortcut_run(a: dict, c: ToolContext) -> str | ToolOutput:
    if runtime.library_mode() or runtime.library_bound():
        return "Error: shortcut_run runs only NEO.app's own user's Shortcuts."
    if not enabled():
        return "Error: running your Shortcuts is off in NEO's settings."
    asked = " ".join(str(a.get("shortcut") or a.get("name") or "").split())
    text = str(a.get("text") or a.get("input") or "")[:MAX_INPUT]
    # Only in the replay of a card NEO staged: the registry drops `_` args anywhere else.
    sid = str(a.get("_id") or "")
    if not asked and not sid:
        return "Error: say which shortcut to run."
    have = await listing()
    if have is None:
        return "Error: NEO couldn't read your list of Shortcuts."
    sc = next((s for s in have if (s.id or s.name) == sid), None) if sid else resolve(asked, have)
    if sc is None and not sid and _age() > _RETRY_S:  # made since the list was read?
        have = await listing(max_age=0) or have
        sc = resolve(asked, have)
    if sc is None:
        if sid:
            return "Error: that shortcut isn't in your Shortcuts any more."
        if not have:
            return "You don't have any shortcuts in the Shortcuts app."
        said = _spoken(asked)[:60]
        return f"You don't have a shortcut called “{said}”. The closest: {', '.join(closest(asked, have))}."
    if not _users_pick(c, sc, have):
        from neo2.core.agent import confirm

        params = {"shortcut": sc.name, "_id": sc.id or sc.name, **({"text": text} if text else {})}
        gate = confirm.gated(
            TOOL, params, f"run your shortcut “{_spoken(sc.name)}”", _detail(sc, text), always=True
        )
        if isinstance(gate, str):
            return ToolOutput(gate, halt=True, staged=getattr(gate, "action_id", ""))
        text = str(gate.get("text") or "")
    return await run(sc, text)
