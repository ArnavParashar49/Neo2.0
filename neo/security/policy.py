"""What NEO may do without asking, in one place.

The rule is an allowlist: a shell command or AppleScript runs straight away only when it can be
shown to be read-only, and a file write only when it creates a new, ordinary file in the user's
own folders. Everything else goes through the confirm gate with the exact command on the card.
Secrets (keys, tokens, browser profiles, NEO's own data) are never read at all.
"""

from __future__ import annotations

import glob
import ipaddress
import os
import re
import shlex
import socket
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.parse import urlsplit

HOME = Path.home()

# ---- secrets ---------------------------------------------------------------------------------
_SECRET_DIRS = [
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".neo", ".password-store",
    ".config/gcloud", ".config/gh", ".config/op", ".config/hub",
    "Library/Keychains", "Library/Cookies", "Library/Messages", "Library/Mail",
    "Library/Safari", "Library/Application Support/NEO",
    "Library/Application Support/Google/Chrome", "Library/Application Support/BraveSoftware",
    "Library/Application Support/Firefox", "Library/Application Support/Microsoft Edge",
    "Library/Application Support/Arc", "Library/Application Support/com.apple.TCC",
]
_SECRET_NAME = re.compile(
    r"^(?:\.env(?:\..*)?|.*\.(?:pem|key|p12|pfx|keychain|keychain-db|kdbx|ovpn)|id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?"
    r"|\.netrc|\.git-credentials|\.npmrc|\.pypirc|\.pgpass|\.(?:zsh|bash|python|node_repl|psql|mysql)_history"
    r"|credentials(?:\.json)?|ws-token|api_keys\.json|service[-_]account.*\.json)$",
    re.I,
)


_DATA_VOLUME = ("/", "System", "Volumes", "Data")  # the same files, reached through the data volume


def absolute(path: str | os.PathLike, base: str | os.PathLike | None = None) -> Path:
    """`path` as NEO's tools mean it: ~ expanded, and relative to `base` (the folder a command
    runs in; the home folder by default) — never to wherever NEO's own process happens to be."""
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = Path(os.path.expanduser(str(base))) / p if base else HOME / p
    return Path(os.path.normpath(p))


def _variants(path: str | os.PathLike, base: str | os.PathLike | None = None) -> list[Path]:
    p = absolute(path, base)
    out = [p]
    try:
        r = p.resolve()
        if r != p:
            out.append(r)
    except (OSError, RuntimeError):
        pass
    for v in list(out):
        out.extend(_off_data_volume(v))
    return out


def _fold(part: str) -> str:
    # APFS ignores case and Unicode normalisation: ~/.SSH is ~/.ssh.
    return unicodedata.normalize("NFC", part).casefold()


def _under(p: Path, root: Path) -> bool:
    pp, rp = [_fold(x) for x in p.parts], [_fold(x) for x in root.parts]
    return pp[: len(rp)] == rp


def _extra_roots() -> list[Path]:
    """NEO's data wherever else it lives: every embedded agent's data dir and extra secret paths,
    and NEO.app's own data dir when NEO_DATA_DIR moved it (neo.runtime.secret_roots — which
    never builds settings to find out)."""
    from neo.runtime import secret_roots

    out: list[Path] = []
    for r in secret_roots():
        for v in (r, _resolved(r)):
            # named through the data volume: its files are reached the usual way too (_variants)
            for w in (v, *_off_data_volume(v)):
                if w not in out:
                    out.append(w)
    return out


def _off_data_volume(p: Path) -> list[Path]:
    """/System/Volumes/Data/Users/x → /Users/x (the firmlinked, usual name), else nothing."""
    if tuple(_fold(x) for x in p.parts[:4]) == tuple(_fold(x) for x in _DATA_VOLUME):
        return [Path("/", *p.parts[4:])]
    return []


def _resolved(p: Path) -> Path:
    try:
        return p.resolve()
    except (OSError, RuntimeError):
        return p


def _in_agent_folder(p: Path) -> bool:
    """Embedded (neo.runtime): `p` lies in a folder NEO made an agent's own (its marker) — a live
    agent's, a closed one's, or one from another process. NEO.app's own tools: unchanged."""
    from neo import runtime

    if not runtime.library_mode():
        return False
    from neo.config import AGENT_MARKER

    return any(os.path.isfile(d / AGENT_MARKER) for d in (p, *p.parents))


def is_secret(path: str | os.PathLike, base: str | os.PathLike | None = None) -> bool:
    """Keys, tokens, password stores, browser profiles, chat and mail stores, NEO's own data."""
    roots: list[Path] | None = None
    for p in _variants(path, base):
        if _SECRET_NAME.match(p.name):
            return True
        if any(_under(p, HOME / d) for d in _SECRET_DIRS):
            return True
        if _under(p, Path("/Library/Keychains")) or _under(p, Path("/private/var/db")):
            return True
        roots = _extra_roots() if roots is None else roots
        if any(_under(p, r) for r in roots):
            return True
        if _in_agent_folder(p):
            return True
    return False


def covers_secret(path: str | os.PathLike, base: str | os.PathLike | None = None) -> bool:
    """Would a recursive read of this directory reach a secret? (the home folder, ~/Library…)"""
    roots = _extra_roots()
    return any(
        is_secret(p) or any(_under(HOME / d, p) for d in _SECRET_DIRS) or any(_under(r, p) for r in roots)
        for p in _variants(path, base)
    )


# ---- file writes -----------------------------------------------------------------------------
_OWN_FOLDERS = ["Desktop", "Documents", "Downloads", "Movies", "Music", "Pictures"]
# Double-clicking these runs something, or macOS loads them by itself.
_RUNNABLE = re.compile(
    r"\.(?:command|tool|terminal|app|pkg|mpkg|plist|workflow|scpt|scptd|applescript|mobileconfig|"
    r"webloc|inetloc|fileloc|kext|dylib|so|bundle|plugin|saver|prefpane|action|definition)$",
    re.I,
)


def write_risk(path: str | os.PathLike, base: str | os.PathLike | None = None) -> str:
    """Why writing this file needs the user's OK ("" = it doesn't)."""
    vs = _variants(path, base)
    p = vs[-1]
    if is_secret(path, base):
        return "it holds secrets"
    if any(v.exists() for v in vs):
        return "it already exists and would be overwritten"
    tmp = [Path("/tmp"), Path("/private/tmp"), Path(os.environ.get("TMPDIR", "/tmp")).resolve()]
    for v in vs:
        # Compared case-insensitively (APFS), so take the parts below home by position.
        below = v.parts[len(HOME.parts) :] if _under(v, HOME) else v.parts[1:]
        if any(part.startswith(".") for part in below):
            return "it's a hidden settings file or folder"
    if _RUNNABLE.search(p.name) or any(_RUNNABLE.search(part) for part in p.parts):
        return "macOS can run or load that kind of file"
    if any(_under(p, HOME / f) for f in _OWN_FOLDERS) or any(_under(p, t) for t in tmp):
        return ""
    return "it's outside your Desktop, Documents and Downloads folders"


# ---- shell -----------------------------------------------------------------------------------
_READ_ONLY_CMDS = {
    "ls", "cat", "head", "tail", "wc", "pwd", "which", "whoami", "id", "date", "cal", "uptime",
    "echo", "grep", "egrep", "fgrep", "rg", "find", "du", "df", "file", "stat", "sw_vers", "uname",
    "hostname", "mdfind", "mdls", "sort", "uniq", "cut", "tr", "basename", "dirname", "realpath",
    "readlink", "diff", "cmp", "tree", "ps", "brew", "pip", "pip3", "defaults", "pmset",
    "system_profiler", "sysctl", "lsof", "netstat", "vm_stat", "nl", "column", "true", "mkdir",
    "touch", "say", "sleep",
}
# For every shell command: git may not start programs from a repository's own config (a
# downloaded folder can carry a .git/config that sets them).
GIT_SAFE_ENV = {
    "GIT_CONFIG_COUNT": "2",
    "GIT_CONFIG_KEY_0": "core.fsmonitor",
    "GIT_CONFIG_VALUE_0": "false",
    "GIT_CONFIG_KEY_1": "core.hooksPath",
    "GIT_CONFIG_VALUE_1": "/dev/null",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
}
_SEPARATORS = {"|", "||", "&&", ";"}
_FIND_ACTIONS = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
_BREW_READ = {"list", "ls", "info", "search", "outdated", "deps", "leaves", "config", "--version", "-v"}
_PIP_READ = {"list", "show", "freeze", "--version", "-V"}


def _words(cmd: str) -> list[str] | None:
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        return list(lex)
    except ValueError:  # unbalanced quotes
        return None


def _paths_ok(args: list[str], base: str | os.PathLike, *, recursive: bool, write: bool) -> bool:
    for a in args:
        if a.startswith("-"):
            continue
        whole = str(absolute(a, base))
        cands = glob.glob(whole) if any(c in a for c in "*?[") else [whole]
        for c in cands or [whole]:
            if is_secret(c) or (recursive and covers_secret(c)):
                return False
            if write and write_risk(c) and write_risk(c) != "it already exists and would be overwritten":
                return False
    return True


# Long options that make an otherwise read-only command run a program or write a file.
_RUNS_OR_WRITES = re.compile(r"^--(?:comp|out|exec|files0|pager|editor|use-comp|command|filter|tmp|temp)", re.I)
_SORT_FLAGS = set("bdfghiMnRrsuVcCz")  # sort's short options that only change the ordering / checks
_SORT_LONG = {
    "--numeric-sort", "--reverse", "--unique", "--ignore-case", "--human-numeric-sort", "--version-sort",
    "--stable", "--general-numeric-sort", "--month-sort", "--ignore-leading-blanks", "--dictionary-order",
    "--random-sort", "--check", "--zero-terminated",
}


def _sort_ok(args: list[str]) -> bool:
    it = iter(range(len(args)))
    for i in it:
        a = args[i]
        if a == "--" or not a.startswith("-") or a == "-":
            continue
        if a.startswith("--"):
            name = a.split("=", 1)[0]
            if name in ("--key", "--field-separator", "--buffer-size"):
                continue
            if a not in _SORT_LONG:
                return False  # --compress-program, --output, --files0-from, --temporary-directory…
            continue
        opts = a[1:]
        if opts[0] in "ktS":  # -k 2 / -t , / -S 1K take a value
            if len(opts) == 1:
                next(it, None)
            continue
        if not set(opts) <= _SORT_FLAGS:
            return False
    return True


def _simple_ok(argv: list[str], base: str | os.PathLike) -> bool:
    name, args = argv[0], argv[1:]
    if name not in _READ_ONLY_CMDS:
        return False
    if any(_RUNS_OR_WRITES.match(a) for a in args):
        return False
    flags = [a for a in args if a.startswith("-")]
    plain = [a for a in args if not a.startswith("-")]
    if name == "find":
        if _FIND_ACTIONS & set(args):
            return False
        return _paths_ok(plain[:1], base, recursive=False, write=False)  # names only, never contents
    if name == "sort" and not _sort_ok(args):
        return False
    if name == "brew":
        return bool(plain) and plain[0] in _BREW_READ or args[:1] in (["--version"], ["-v"])
    if name in ("pip", "pip3"):
        return bool(args) and args[0] in _PIP_READ
    if name == "defaults":
        # Apple's own preferences only: other apps keep tokens in theirs.
        return (
            len(args) >= 2 and args[0] in ("read", "read-type")
            and (args[1].startswith("com.apple.") or args[1] in ("-g", "NSGlobalDomain"))
        ) or args[:1] == ["domains"]
    if name == "pmset":
        return args[:1] == ["-g"]
    if name == "sysctl":
        return "-w" not in args and not any("=" in a for a in args)
    if name in ("sort", "tree") and any(f in ("-o", "--output") or f.startswith("--output") for f in flags):
        return False
    if name == "uniq" and len(plain) > 1:  # uniq IN OUT writes OUT
        return False
    if name == "say" and any(f in ("-o", "--output-file", "-n", "--network-send") for f in flags):
        return False
    if name in ("mkdir", "touch"):
        return _paths_ok(plain, base, recursive=False, write=True)
    if name in ("grep", "egrep", "fgrep") and any(
        f in ("--recursive", "--dereference-recursive") or (not f.startswith("--") and ("r" in f or "R" in f))
        for f in flags
    ):
        return False  # reads every file below, hidden ones (.env) included
    if name == "rg":
        # rg walks directories (the home folder when given none) and reads every file it finds,
        # keys and credentials included: only explicit files run without asking.
        targets = plain[1:]
        if not targets or any(absolute(t, base).is_dir() for t in targets):
            return False
        return _paths_ok(targets, base, recursive=False, write=False)
    if name == "diff" and any(f in ("-r", "--recursive") or (not f.startswith("--") and "r" in f) for f in flags):
        return False  # compares (and prints) every file below two folders
    if name in ("grep", "egrep", "fgrep"):
        return _paths_ok(plain[1:], base, recursive=False, write=False)
    return _paths_ok(plain, base, recursive=False, write=False)


def shell_is_read_only(cmd: str, cwd: str | os.PathLike | None = None) -> bool:
    """True only for commands that can't change anything: every simple command is on the
    read-only list (with its writing flags excluded), nothing is redirected or substituted, and
    no argument reaches a secret."""
    if not cmd.strip() or any(ch in cmd for ch in "`$\n\r\\") or "<(" in cmd or ">(" in cmd:
        return False
    words = _words(cmd)
    if not words:
        return False
    segments: list[list[str]] = [[]]
    i = 0
    while i < len(words):
        w = words[i]
        if w in _SEPARATORS:
            segments.append([])
        elif w in (">", ">>", "&>") and i + 1 < len(words) and words[i + 1] == "/dev/null":
            if segments[-1] and segments[-1][-1] in ("1", "2"):
                segments[-1].pop()  # the fd in 2>/dev/null
            i += 1  # output thrown away: nothing written
        elif w == ">&" and i + 1 < len(words) and words[i + 1] in ("1", "2"):
            if segments[-1] and segments[-1][-1] in ("1", "2"):
                segments[-1].pop()  # 2>&1
            i += 1
        elif set(w) <= set("<>&|();"):
            return False  # other redirects, background jobs, subshells
        else:
            if not segments[-1] and "=" in w:
                return False  # VAR=value cmd
            segments[-1].append(w)
        i += 1
    if segments and not segments[-1] and len(segments) > 1 and words[-1] == ";":
        segments.pop()  # a trailing ";"
    base = cwd or HOME
    return all(seg and _simple_ok(seg, base) for seg in segments)


# ---- AppleScript -----------------------------------------------------------------------------
# A script runs without asking only when it can be *shown* to read: it is read statement by
# statement (a small lexer and parser, not a pattern over the text), and every word outside a
# string must be a word of reading — a property, an element class, a reference form, an operator —
# or one of the script's own variables. Anything else (a command, a word NEO doesn't know, a raw
# «event» code, a handler, the ObjC bridge) makes it a script that acts, and the user is asked.
#
# Read-only statements: get / return / log <expr>, count / exists / a plain reference as a
# statement, `set <variable> to <expr>` (never `set <property> [of …] to`), if / else, repeat,
# try / on error, `with timeout`, `local`, `delay <n>`, `error <expr>`, and `tell <target>` blocks
# and one-liners around them.

def _terms(text: str) -> frozenset[str]:
    return frozenset(text.split())


def _phrases(text: str) -> frozenset[str]:
    """ "a b|c d|…" over several lines: each phrase with its words one space apart."""
    return frozenset(" ".join(p.split()) for p in text.replace("\n", " ").split("|") if p.strip())


_AS_READ = _terms(
    """
    of in every whose where first second third fourth fifth sixth seventh eighth ninth tenth last
    middle front back some index named id thru through before after beginning end it its me my
    the a an is isn't not and or equal equals than greater less contains contain does doesn't
    aren't wasn't begins starts ends comes as mod div true false null yes no result return
    linefeed tab space quote pi get count exists exist with number length reverse rest contents
    text string list lists record records integer real boolean date dates class reference alias
    file files folder folders item items character characters word words paragraph paragraphs
    applescript value values
    application applications app process processes window windows sheet sheets drawer drawers
    button buttons checkbox checkboxes group groups image images menu menus row rows column
    columns cell cells table tables outline outlines slider sliders splitter splitters toolbar
    toolbars incrementor incrementors browser browsers attribute attributes action actions
    name class role subrole title description help position size enabled focused selected
    frontmost visible orientation properties version running bounds zoomed miniaturized
    closeable resizable titled modal floating collapsed expanded minimized
    document documents tab tabs url source loading track tracks playlist playlists artist album
    genre duration time rating composer lyrics artwork bpm kind year comment played shuffling
    repeating playing paused stopped loved disliked favorited
    disk disks container desktop home trash owner target selection locked extension
    message messages mailbox mailboxes inbox account accounts subject sender recipient recipients
    address content headers
    calendar calendars event events summary location uid status reminder reminders priority
    completed flagged body note notes plaintext shared
    weekday month day year hours minutes seconds days weeks
    january february march april may june july august september october november december
    monday tuesday wednesday thursday friday saturday sunday
    """
)
# Terms of more than one word (matched first, longest first). None starts with "to": a `to`
# always stays a word of its own for `set … to` and `tell … to`.
_AS_PHRASES = frozenset(
    """
    ui element|ui elements|ui elements enabled|pop up button|pop up buttons|menu button|
    menu buttons|menu bar|menu bars|menu bar item|menu bar items|menu item|menu items|text field|
    text fields|text area|text areas|static text|static texts|scroll area|scroll areas|scroll bar|
    scroll bars|radio button|radio buttons|radio group|radio groups|check box|check boxes|
    tab group|tab groups|splitter group|splitter groups|combo box|combo boxes|busy indicator|
    progress indicator|progress indicators|value indicator|level indicator|relevance indicator|
    color well|grow area|list item|list items|application process|application processes|
    role description|accessibility description|bundle identifier|displayed name|unix id|
    background only|short name|file type|creator type|entire contents|minimum value|
    maximum value|current tab|active tab|active tab index|front window|front document|
    current track|current playlist|current stream title|player state|player position|
    sound volume|shuffle enabled|shuffle mode|song repeat|album artist|played count|
    track number|disc number|date added|start date|end date|allday event|due date|
    remind me date|completion date|creation date|modification date|date received|date sent|
    read status|flagged status|junk mail status|deleted status|unread count|message id|
    all headers|name extension|label index|physical size|application file|application files|
    document file|document files|alias file|alias files|finder window|finder windows|
    insertion location|desktop position|current view|missing value|text item|text items|
    text item delimiters|quoted form|posix path|posix file|unicode text|current date|
    time string|date string|short date string|path to|equal to|home folder|desktop folder|
    documents folder|downloads folder|applications folder|library folder|pictures folder|
    music folder|movies folder|startup disk|frontmost application|begins with|starts with|
    ends with|comes before|comes after|is not|is in|is not in|is equal to|is not equal to|
    is greater than|is less than|greater than|less than|greater than or equal to|
    less than or equal to|is greater than or equal to|is less than or equal to|does not contain|
    doesn't contain|does not equal|doesn't equal|number of|count of|folder action|folder actions
    """.replace("\n", "").split("|")
)
_AS_PHRASES = frozenset(" ".join(p.split()) for p in _AS_PHRASES if p.strip())
# Commands that read like terms, and must never pass for one: kept whole so they are refused.
_AS_DENIED_PHRASES = frozenset(
    {"list folder", "list disks", "info for", "open location", "set volume", "the clipboard", "back track"}
)
_AS_PHRASE_MAX = max(len(p.split()) for p in _AS_PHRASES | _AS_DENIED_PHRASES)
_AS_PHRASE_WORDS = frozenset(w for p in _AS_PHRASES for w in p.split())
# Properties a model may mean to change with `set <word> to …`: never taken for a script's own
# variable. Belt and braces only: inside an app's tell, what keeps the app's property from passing
# for a variable is the reader's `known` and _own_name, not this list.
_AS_SETTABLE = _terms(
    """
    mute muted shuffle volume label tag mode view state level brightness speed rate loop looping
    theme wallpaper background color font style format language timezone alarm alert sound
    current default active hidden checked starred picture preset audio video
    """
)
# Commands (AppleScript's own, Standard Additions', and the usual app verbs) and the language's
# reserved words: never a script's own variable, whatever the spelling.
_AS_NEVER_LOCAL = _terms(
    """
    set copy put run load store do call using considering ignoring script property global on to
    tell if then else repeat while until times exit try error timeout transaction local use
    delay log say beep display choose open close quit launch activate reveal select click perform
    keystroke key make delete remove erase empty move duplicate save send reply forward redirect
    print read write mount eject restart shutdown sleep play pause stop resume next previous
    skip rewind playpause fastforward backtrack search go add attach sign accept decline join leave
    post submit buy create import export convert update mark enable disable toggle install reload
    refresh share clean sync backup download upload rename restore hide show bring zoom minimize
    minimise miniaturize collapse expand check uncheck unmount offset round summarize random
    info clipboard system scripting framework current into onto without given returning since
    against apart aside around beside between beneath behind below above about over out ref prop
    but for from by at contain equal increment decrement pick cancel confirm sort arrange bounce
    synchronize subscribe switch present trim step fastforward
    """
)
_AS_SYMS = frozenset(
    ["(", ")", "{", "}", ",", ":", "&", "=", "≠", "<", ">", "<=", ">=", "≤", "≥", "+", "-", "*", "/", "÷", "^", "'s"]
)
# What may follow one of the script's own variables: an operator, never an argument (`x "a"`,
# `x(1)` would be a command or handler call).
_AS_AFTER_LOCAL = _terms(
    """
    of in as is isn't and or contains contain does doesn't aren't wasn't mod div equals thru
    through whose where then
    """
) | {p for p in _AS_PHRASES if p.split()[0] in ("is", "begins", "starts", "ends", "comes", "greater", "less", "does", "doesn't", "equal")}
_AS_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_AS_NUM = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")
_AS_QUOTES = '"“”'
# Apps and processes named by what they are at run time, not by name: whichever is in front, all
# of them. A script with one of these never counts as reading, and no grant ever covers it.
_AS_PICKED = frozenset({"processes", "application processes", "applications", "frontmost application"})
_AS_APP_WORDS = ("application", "app")
_AS_PROC_WORDS = ("process", "application process")
_SE = "system events"

# The language's own words: they say nothing about which app's objects a statement touches.
_AS_GENERIC = _terms(
    """
    of in every whose where first second third fourth fifth sixth seventh eighth ninth tenth last
    middle front back some index named id thru through before after beginning end it its me my
    the a an is isn't not and or equal equals than greater less contains contain does doesn't
    aren't wasn't begins starts ends comes as mod div true false null yes no result return
    linefeed tab space quote pi get count exists exist with number length reverse rest contents
    text string list lists record records integer real boolean date dates class reference item
    items character characters word words paragraph paragraphs applescript value values
    weekday month day year hours minutes seconds days weeks
    january february march april may june july august september october november december
    monday tuesday wednesday thursday friday saturday sunday
    """
) | {
    p for p in _AS_PHRASES
    if p.split()[0] in ("is", "begins", "starts", "ends", "comes", "greater", "less", "does", "doesn't", "equal")
} | {
    "number of", "count of", "missing value", "text item", "text items", "text item delimiters", "unicode text",
    "current date", "time string", "date string", "short date string", "quoted form",
}
# System Events' words for an app's window and what is in it (reading them takes Accessibility).
_AS_UI_ELEMENTS = _terms(
    """
    window windows sheet sheets drawer drawers button buttons checkbox checkboxes group groups image
    images menu menus row rows column columns cell cells table tables outline outlines slider sliders
    splitter splitters toolbar toolbars incrementor incrementors browser browsers attribute
    attributes action actions
    """
) | _phrases(
    """
    ui element|ui elements|ui elements enabled|pop up button|pop up buttons|menu button|menu buttons|
    menu bar|menu bars|menu bar item|menu bar items|menu item|menu items|text field|text fields|
    text area|text areas|static text|static texts|scroll area|scroll areas|scroll bar|scroll bars|
    radio button|radio buttons|radio group|radio groups|check box|check boxes|tab group|tab groups|
    splitter group|splitter groups|combo box|combo boxes|busy indicator|progress indicator|
    progress indicators|value indicator|level indicator|relevance indicator|color well|grow area|
    list item|list items
    """
)
# …and what System Events knows about a process and its elements.
_AS_UI_PROPS = _terms(
    """
    name class role subrole title description help position size enabled focused selected value
    frontmost visible orientation properties process
    """
) | {
    "application process", "role description", "accessibility description", "minimum value", "maximum value",
    "entire contents", "bundle identifier", "displayed name", "unix id", "background only", "short name",
}
# Acts a yes "for this task" may cover inside System Events: clicking, pressing, choosing and filling
# in one named app's window. Everything else a script does is asked about every time.
_AS_UI_VERBS = frozenset({"click", "perform", "select"})
_AS_UI_SET = frozenset({"value", "selected", "focused", "frontmost"})
# `perform action "…"` a grant may cover: pressing (an element named by its label), showing its
# menu, raising a window. Not AXConfirm (Return), AXCancel, AXIncrement / AXDecrement (a value).
_AS_UI_ACTIONS = frozenset({"axpress", "axshowmenu", "axraise"})
# Where `set value of … to` is typing, whichever the field: what it changes is the text it shows.
_AS_UI_TEXT_ENTRY = frozenset({"text field", "text area", "combo box"})
# Apps no task-wide yes ever covers, whatever the script does in them: they run commands or
# scripts (a click or a typed line there is a shell), or hold NEO's own cards and the Mac's
# security settings. Names and bundle ids, compared case-insensitively.
_NEVER_GRANT = _phrases(
    """
    terminal|iterm|iterm2|warp|ghostty|kitty|alacritty|wezterm|hyper|tabby|script editor|
    applescript editor|script debugger|automator|shortcuts|shortcuts events|system events|osascript|
    neo|neo-ui|system settings|system preferences|keychain access|passwords|installer|
    com.apple.terminal|com.googlecode.iterm2|dev.warp.warp-stable|com.mitchellh.ghostty|
    net.kovidgoyal.kitty|org.alacritty|io.alacritty|com.github.wez.wezterm|co.zeit.hyper|
    com.apple.scripteditor2|com.latenightsw.scriptdebugger8|com.apple.automator|com.apple.shortcuts|
    com.apple.shortcuts.events|com.apple.systemevents|com.arnav.neo|com.apple.systempreferences|
    com.apple.keychainaccess|com.apple.passwords|com.apple.installer
    """
)


def _app_key(name: str) -> str:
    n = " ".join(name.split()).casefold()
    return n[:-4] if n.endswith(".app") else n


def _is_se(name: str, by_id: bool = False) -> bool:
    return _app_key(name) == ("com.apple.systemevents" if by_id else _SE)


def never_grant(app: str) -> bool:
    """An app no yes "for this task" may ever cover (see _NEVER_GRANT)."""
    return _app_key(app) in _NEVER_GRANT


def _by_path(name: str) -> bool:
    """`application "/Users/me/Downloads/Thing.app"` (or an HFS path, or "Thing.app"): AppleScript
    launches whatever bundle that is to send it the event — open_app's check never sees it."""
    n = name.strip()
    return "/" in n or ":" in n or n.casefold().endswith(".app") or n.startswith("~")


class _Tok:
    __slots__ = ("kind", "text", "raw")

    def __init__(self, kind: str, text: str, raw: str = "") -> None:
        self.kind = kind  # "w" word or phrase, "s" string, "n" number, "p" punctuation, "nl" line end
        self.text = text  # words lower-case, phrases joined by one space; a string's contents
        self.raw = raw or text  # a word as written (a variable's camelCase)

    def is_(self, *words: str) -> bool:
        return self.kind == "w" and self.text in words

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.kind}:{self.text}"


def _as_lex(src: str) -> list[_Tok] | None:
    """AppleScript's tokens, or None for anything this reader won't vouch for: an unterminated
    string or comment, a quote inside a block comment, «raw codes», |pipe identifiers|, a
    character outside the language's own (lookalike letters, backslashes, semicolons…)."""
    out: list[_Tok] = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch in " \t":
            i += 1
        elif ch in "\r\n":
            out.append(_Tok("nl", "\n"))
            i += 1
        elif ch == "¬":  # the line goes on
            j = i + 1
            while j < n and src[j] in " \t":
                j += 1
            if j >= n or src[j] not in "\r\n":
                return None
            i = j + (2 if src.startswith("\r\n", j) else 1)
        elif src.startswith("--", i) or ch == "#":
            while i < n and src[i] not in "\r\n":
                i += 1
        elif src.startswith("(*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("(*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*)", j):
                    depth, j = depth - 1, j + 2
                elif src[j] in _AS_QUOTES:
                    return None  # how quotes nest in comments is not worth betting on
                else:
                    j += 1
            if depth:
                return None
            i = j
        elif ch in _AS_QUOTES:
            j, buf = i + 1, []
            while j < n and src[j] not in _AS_QUOTES:
                if src[j] == "\\":
                    if j + 1 >= n:
                        return None
                    buf.append(src[j + 1])
                    j += 2
                    continue
                buf.append(src[j])
                j += 1
            if j >= n:
                return None
            out.append(_Tok("s", "".join(buf)))
            i = j + 1
        elif ch.isascii() and (ch.isalpha() or ch == "_"):
            m = _AS_WORD.match(src, i)
            assert m is not None
            word, j = m.group(0), m.end()
            if word.lower().endswith("n") and src.startswith(("'t", "’t"), j):
                word, j = word + "'t", j + 2  # isn't, doesn't
            out.append(_Tok("w", word.lower(), word))
            i = j
        elif ch.isascii() and (ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit())):
            m = _AS_NUM.match(src, i)
            assert m is not None
            out.append(_Tok("n", m.group(0)))
            i = m.end()
        elif src.startswith(("'s", "’s"), i) and out and out[-1].kind in "wsnp":
            if i + 2 < n and (src[i + 2].isalnum() or src[i + 2] == "_"):
                return None
            out.append(_Tok("p", "'s"))
            i += 2
        elif src.startswith(("<=", ">="), i):
            out.append(_Tok("p", src[i : i + 2]))
            i += 2
        elif ch in "(){},:&=≠<>≤≥+-*/÷^":
            out.append(_Tok("p", ch))
            i += 1
        else:
            return None
    return _as_phrases(out)


def _as_phrases(toks: list[_Tok]) -> list[_Tok]:
    """Multi-word terms ("menu bar item", "is not equal to") become one token, longest first."""
    out: list[_Tok] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "w":
            for size in range(min(_AS_PHRASE_MAX, len(toks) - i), 1, -1):
                run = toks[i : i + size]
                p = " ".join(x.text for x in run)
                if all(x.kind == "w" for x in run) and (p in _AS_PHRASES or p in _AS_DENIED_PHRASES):
                    out.append(_Tok("w", p, " ".join(x.raw for x in run)))
                    i += size
                    break
            else:
                out.append(t)
                i += 1
        else:
            out.append(t)
            i += 1
    return out


def _find(toks: list[_Tok], word: str, start: int = 0) -> int:
    """Index of the first `word` outside brackets, or -1."""
    depth = 0
    for i in range(start, len(toks)):
        t = toks[i]
        if t.kind == "p" and t.text in "({":
            depth += 1
        elif t.kind == "p" and t.text in ")}":
            depth -= 1
        elif depth == 0 and t.is_(word):
            return i
    return -1


def _valid_local(t: _Tok) -> bool:
    return (
        t.kind == "w"
        and " " not in t.text
        and t.text not in _AS_READ
        and t.text not in _AS_SETTABLE
        and t.text not in _AS_NEVER_LOCAL
        and t.text not in _AS_PHRASE_WORDS
    )


def _own_name(t: _Tok) -> bool:
    """A name no app's dictionary uses: its terms are whole lower-case words, a space between two
    (Apple's scripting guidelines) — so `theRule`, `track_2`, `n` are the script's own, where
    `recurrence` may be the app's property (`set recurrence to …` in Calendar's event changes it)."""
    raw = t.raw
    return (
        len(raw) == 1
        or "_" in raw
        or any(c.isdigit() for c in raw)
        or (any(c.islower() for c in raw) and any(c.isupper() for c in raw[1:]))
    )


# Where a statement is: (terms, app, proc, known). `terms`: whose terminology its words are — the
# innermost `tell application "X"`'s name folded ("id:…" for an app named by its id), "" at the
# top. `app`: that app as written. `proc`: inside System Events, the process "X" a `tell` block is
# about ("" = none: System Events' own objects). `known`: every one-word property the statement's
# object can have is a word of reading, so `set <word> to` can only make a variable — true at the
# top, at System Events' app level and in a process's UI (a fixed dictionary NEO has read); false
# inside any other app's tell and on System Events' own objects (files, disks, preferences), where
# only a name no app term could be (_own_name) is the script's own.
_Ctx = tuple[str, str, str, bool]
_TOP: _Ctx = ("", "", "", True)


def _is_term(t: _Tok) -> bool:
    return t.kind == "w" and (t.text in _AS_READ or t.text in _AS_PHRASES)


def _literal_after(toks: list[_Tok], i: int) -> str | None:
    """The name in `application "X"`, `application id "X"` or `process "X"` at toks[i]; None when
    the app or process is picked by an expression (`process 1`, `every process whose …`, a variable)."""
    nxt = toks[i + 1] if i + 1 < len(toks) else None
    if nxt is not None and nxt.kind == "s" and nxt.text.strip():
        return nxt.text.strip()
    if toks[i].text in _AS_APP_WORDS and nxt is not None and nxt.is_("id") and i + 2 < len(toks):
        after = toks[i + 2]
        if after.kind == "s" and after.text.strip():
            return after.text.strip()
    return None


class _ScriptReader:
    """Reads a script statement by statement; `ok` says whether every one only reads — or, with
    `grant`, only reads or clicks, presses, chooses and fills in one named app's window (what a
    yes "for this task" may cover, when every press can be checked by its label). As it reads, it
    notes what each statement touches (`targets`), whether it reaches into an app's UI through
    System Events (`ui`), and the labels of what it presses (`labels`) or why they can't be read
    (`unchecked`).

    Terminology follows the innermost `tell application "…"`: inside it, a word the app
    defines means the app's command or property. So a variable counts as the script's own only
    where it was assigned (a compiled `set x to` there shows x isn't the app's command) or at
    the top level."""

    def __init__(self, lines: list[list[_Tok]], *, grant: bool = False) -> None:
        self.lines = lines
        self.grant = grant
        self.assigned: dict[str, set[str]] = {}
        self.collect = True
        self.stack: list[tuple[str, _Ctx]] = []
        self.targets: dict[str, str] = {}  # every app and process a statement touches (folded → as written)
        self.ui = False
        self.labels: list[str] = []  # the labels of what it presses or chooses
        self.unchecked: list[str] = []  # why what it presses can't be checked by its label

    def ok(self) -> bool:
        for collect in (True, False):  # first learn the variables, then check every use
            self.collect, self.stack = collect, []
            self.targets, self.ui, self.labels, self.unchecked = {}, False, [], []
            for line in self.lines:
                ctx = self.stack[-1][1] if self.stack else _TOP
                if not self._stmt(line, ctx, line=True):
                    return False
            if self.stack:
                return False  # a block left open
        return True

    # ---- variables ----------------------------------------------------------------------
    def _assign(self, t: _Tok, ctx: _Ctx) -> bool:
        if not _valid_local(t) or not (ctx[3] or _own_name(t)):
            return False
        self.assigned.setdefault(t.text, set()).add(ctx[0])
        return True

    def _local_ok(self, t: _Tok, ctx: _Ctx) -> bool:
        if not _valid_local(t):
            return False
        if self.collect:
            return True
        where = self.assigned.get(t.text)
        # Inside an app's tell, only a name assigned in that app's terms: `set volumeUp to 0` at
        # the top says nothing about VLC, whose volumeUp is a command.
        return bool(where) and (ctx[0] == "" or ctx[0] in where)

    # ---- what a statement touches -------------------------------------------------------
    def _add(self, name: str) -> None:
        self.targets.setdefault(_app_key(name), name)

    def _hit(self, toks: list[_Tok], ctx: _Ctx, *, act: bool = False) -> None:
        """Note the apps and processes a statement reads or acts on (the checking pass). Inside
        System Events a statement belongs to a process only when all it touches is that named
        process's window: anything else (login items, preferences, disks, the Mac itself) is
        System Events' own, and it stays a target — a process name never relabels it."""
        if self.collect:
            return
        terms, app, proc, _ = ctx
        procs: list[str] = []
        for i, x in enumerate(toks):
            if x.kind == "w" and x.text in _AS_APP_WORDS + _AS_PROC_WORDS and (name := _literal_after(toks, i)):
                if x.text in _AS_APP_WORDS:
                    self._add(name)
                else:
                    procs.append(name)
        content = [x.text for x in toks if _is_term(x) and x.text not in _AS_GENERIC]
        if terms == _SE:
            about = procs + ([proc] if proc else [])
            if act or content:
                if not about or any(w not in _AS_UI_ELEMENTS and w not in _AS_UI_PROPS for w in content):
                    self._add("System Events")
                for p in about:
                    self._add(p)
            if act or any(w in _AS_UI_ELEMENTS for w in content):
                self.ui = True
            return
        for p in procs:
            self._add(p)
        if app and (act or content):
            self._add(app)

    # ---- expressions --------------------------------------------------------------------
    def _expr(self, toks: list[_Tok], ctx: _Ctx) -> bool:
        if not toks:
            return False
        depth: list[str] = []
        for i, t in enumerate(toks):
            if t.kind in ("s", "n"):
                continue
            if t.kind == "p":
                if t.text not in _AS_SYMS:
                    return False
                if t.text in "({":
                    depth.append(t.text)
                elif t.text in ")}" and (not depth or depth.pop() != ("(" if t.text == ")" else "{")):
                    return False
                continue
            if t.kind != "w":
                return False
            if t.text in _AS_PICKED:
                return False  # whichever app is in front, every process: not one NEO can name
            if t.text in _AS_APP_WORDS + _AS_PROC_WORDS:
                name = _literal_after(toks, i)
                if name is None or (t.text in _AS_APP_WORDS and _by_path(name)):
                    return False
                continue
            if t.text in _AS_READ or t.text in _AS_PHRASES:
                continue
            if not self._local_ok(t, ctx):
                return False
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if nxt is not None and not (
                (nxt.kind == "p" and nxt.text not in "({") or (nxt.kind == "w" and nxt.text in _AS_AFTER_LOCAL)
            ):
                return False  # `x "a"`, `x(1)`, `x y`: a call, not a variable
        return not depth

    # ---- statements ---------------------------------------------------------------------
    def _stmt(self, t: list[_Tok], ctx: _Ctx, *, line: bool) -> bool:
        if not t:
            return False
        h = t[0].text if t[0].kind == "w" else ""
        if h == "tell":
            k = _find(t, "to")
            new = self._tell_target(t[1:k] if k != -1 else t[1:], ctx)
            if new is None:
                return False
            if k != -1:
                return self._stmt(t[k + 1 :], new, line=False)
            return line and self._open("tell", new)
        if h == "end":
            if not line or not self.stack:
                return False
            kind = self.stack.pop()[0]
            return len(t) == 1 or (len(t) == 2 and t[1].is_(kind))
        self._hit(t, ctx)
        if h == "if":
            k = _find(t, "then")
            if k == -1 or not self._expr(t[1:k], ctx):
                return False
            if k == len(t) - 1:
                return line and self._open("if", ctx)
            return self._stmt(t[k + 1 :], ctx, line=False)
        if h == "else":
            if not line or not self.stack or self.stack[-1][0] != "if":
                return False
            if len(t) == 1:
                return True
            k = _find(t, "then")
            return t[1].is_("if") and k == len(t) - 1 and self._expr(t[2:k], ctx)
        if h == "repeat":
            return line and self._repeat(t[1:], ctx) and self._open("repeat", ctx)
        if h == "exit":
            return len(t) == 1 or (len(t) == 2 and t[1].is_("repeat"))
        if h == "try":
            return line and len(t) == 1 and self._open("try", ctx)
        if h == "on":  # on error [message] [number n], inside a try block
            if not line or not self.stack or self.stack[-1][0] != "try" or len(t) < 2 or not t[1].is_("error"):
                return False
            rest = t[2:]
            if rest and not rest[0].is_("number"):
                if not self._assign(rest[0], ctx):
                    return False
                rest = rest[1:]
            if rest:
                return len(rest) == 2 and rest[0].is_("number") and self._assign(rest[1], ctx)
            return True
        if h == "with":  # with timeout [of] N second(s)
            words = [x.text for x in t[1:]]
            if words[:1] == ["timeout"] and words[1:2] == ["of"]:
                words = ["timeout"] + words[2:]
            return (
                line
                and len(words) == 3
                and words[0] == "timeout"
                and t[-2].kind == "n"
                and words[2] in ("second", "seconds")
                and self._open("timeout", ctx)
            )
        if h == "set":
            return self._set(t[1:], ctx)
        if h in ("get", "log"):
            return self._expr(t[1:], ctx)
        if h in ("return", "error"):
            return len(t) == 1 or self._expr(t[1:], ctx)
        if h == "local":
            names = [x for x in t[1:] if not (x.kind == "p" and x.text == ",")]
            return bool(names) and all(self._assign(x, ctx) for x in names)
        if h == "delay":
            return len(t) == 2 and t[1].kind == "n"
        if h == "use":
            words = [x.kind + x.text for x in t[1:]]
            return words == ["wscripting", "wadditions"] or (
                len(t) == 4 and t[1].is_("applescript") and t[2].is_("version") and t[3].kind == "s"
            )
        if self.grant and (acted := self._act(t, ctx)) is not None:
            return acted
        # A plain expression as a statement: `count windows`, `name of window 1`. It must start
        # with a word of reading — a statement that starts with anything else is a command.
        if t[0].kind == "w" and t[0].text not in _AS_READ and t[0].text not in _AS_PHRASES:
            return False
        return self._expr(t, ctx)

    def _act(self, t: list[_Tok], ctx: _Ctx) -> bool | None:
        """The acts a yes for this task may cover (None: the statement isn't one of them).
        `activate` inside an app's tell; inside System Events, `click` / `select` / `perform
        action "AX…" of` an element of a named process's window. What a grant presses is checked
        by its label (_COMMIT, a bare "OK"), so pressing an element picked by its place (`button
        3`, `first button whose …`) or an action other than a press, a menu or a raise is noted in
        `unchecked`: the script keeps its app on its card, and no grant covers it."""
        terms, app, proc, _ = ctx
        h = t[0].text if t[0].kind == "w" else ""
        if h == "activate" and len(t) == 1 and app and terms != _SE and not proc:
            self._hit([], ctx, act=True)
            return True
        if terms == _SE and h in _AS_UI_VERBS and len(t) > 1:
            obj = t[1:]
            if h == "perform" and (len(obj) < 2 or not obj[0].is_("action")):
                return False
            if not self._expr(obj, ctx):
                return False  # `click at {x, y}`: a point on the screen, in whichever app is there
            if h != "perform":
                self._press(obj)
            elif obj[1].kind != "s" or (what := obj[1].text.strip().casefold()) not in _AS_UI_ACTIONS:
                self._uncheck("it performs an action other than a press, a menu or raising a window")
            elif what == "axpress":
                self._press(obj[3:] if len(obj) > 3 and obj[2].is_("of") else [])
            self._hit(obj, ctx, act=True)
            return True
        return None

    def _press(self, elem: list[_Tok]) -> None:
        """Pressing or choosing `elem`: its label goes to `labels` when it's named by one (`button
        "Start" [of …]`); picked by its place, nothing can say what it does."""
        if self.collect:
            return
        if (
            len(elem) >= 2
            and elem[0].kind == "w"
            and elem[0].text in _AS_UI_ELEMENTS
            and elem[0].text not in ("action", "actions", "attribute", "attributes")
            and elem[1].kind == "s"
            and elem[1].text.strip()
            and (len(elem) == 2 or elem[2].is_("of"))
        ):
            self.labels.append(elem[1].text)
        else:
            self._uncheck("it presses something picked by its place, not by its label")

    def _uncheck(self, why: str) -> None:
        if not self.collect:
            self.unchecked.append(why)

    def _open(self, kind: str, ctx: _Ctx) -> bool:
        self.stack.append((kind, ctx))
        return True

    def _tell_target(self, t: list[_Tok], ctx: _Ctx) -> _Ctx | None:
        """Where a tell block's statements are: the app's terms for `tell application "X"`, else
        the enclosing block's (a process, a window, a sheet). None: not a target NEO can read."""
        if not t or not self._expr(t, ctx):
            return None
        if t[0].kind == "w" and not _is_term(t[0]):
            return None  # tell <variable>: whatever it holds
        terms, app, proc, known = ctx
        apps = [i for i, x in enumerate(t) if x.kind == "w" and x.text in _AS_APP_WORDS]
        procs = [i for i, x in enumerate(t) if x.kind == "w" and x.text in _AS_PROC_WORDS]
        if len(apps) > 1 or len(procs) > 1:
            return None
        if apps:
            i = apps[0]
            name = _literal_after(t, i) or ""
            by_id = t[i + 1].is_("id")
            if i + (3 if by_id else 2) != len(t):
                return None  # the app ends the target: tell window 1 of application "X"
            terms, app, proc = ("id:" if by_id else "") + name.casefold(), name, ""
            if _is_se(name, by_id):
                terms = _SE  # what it touches is worked out statement by statement
            elif not self.collect:
                self._add(name)  # compiling `tell application "X"` may launch X: it counts
        # tell disk 1 / tell folder "x": System Events' own objects, not a process's UI
        se_own = terms == _SE and any(
            _is_term(x)
            and x.text not in _AS_GENERIC
            and x.text not in _AS_UI_ELEMENTS
            and x.text not in _AS_UI_PROPS
            and x.text not in _AS_APP_WORDS
            for x in t
        )
        if procs:
            proc = _literal_after(t, procs[0]) or ""
            known = terms == _SE and not se_own  # a process and its window
        elif se_own:
            proc, known = "", False
        elif apps:
            known = terms == _SE and apps[0] == 0  # System Events itself, not an object in another app
        else:
            known = known and terms == _SE and bool(proc)  # a window or a sheet of a process
        return (terms, app, proc, known)

    def _repeat(self, t: list[_Tok], ctx: _Ctx) -> bool:
        if not t:
            return True
        if t[0].is_("while", "until"):
            return self._expr(t[1:], ctx)
        if t[-1].is_("times"):
            return self._expr(t[:-1], ctx)
        if t[0].is_("with") and len(t) >= 4:
            var, how = t[1], t[2]
            if how.is_("in"):
                return self._assign(var, ctx) and self._expr(t[3:], ctx)
            if how.is_("from"):
                k = _find(t, "to", 3)
                if k == -1:
                    return False
                b = _find(t, "by", k)
                end = t[k + 1 : b] if b != -1 else t[k + 1 :]
                step_ok = b == -1 or self._expr(t[b + 1 :], ctx)
                return self._assign(var, ctx) and self._expr(t[3:k], ctx) and self._expr(end, ctx) and step_ok
        return False

    def _set(self, t: list[_Tok], ctx: _Ctx) -> bool:
        k = _find(t, "to")
        if k <= 0 or not self._expr(t[k + 1 :], ctx):
            return False
        target = t[:k]
        words = [x.kind + x.text for x in target]
        if self.grant and ctx[0] == _SE and target[0].kind == "w" and target[0].text in _AS_UI_SET:
            # set frontmost of process "X" to true; inside tell process "X": set frontmost to true;
            # set value of text field 1 of window 1 to …: typing. Choosing (a checkbox's value, a
            # row's selected) is pressing: checked by the element's label, as a click is.
            prop, elem = target[0].text, target[2:]
            if len(target) == 1:
                if not ctx[2]:
                    return False
                if prop not in ("frontmost", "focused"):
                    self._press([])  # the tell's own element: no label to read
            elif len(target) >= 3 and target[1].is_("of") and self._expr(elem, ctx):
                typing = prop == "value" and elem[0].kind == "w" and elem[0].text in _AS_UI_TEXT_ENTRY
                if prop not in ("frontmost", "focused") and not typing:
                    self._press(elem)
            else:
                return False
            self._hit(target, ctx, act=True)
            return True
        if len(target) == 1:
            return self._assign(target[0], ctx)
        if words[0] == "p{" and words[-1] == "p}":  # set {a, b} to {1, 2}
            names = target[1:-1]
            return len(names) % 2 == 1 and all(
                self._assign(x, ctx) if i % 2 == 0 else (x.kind == "p" and x.text == ",") for i, x in enumerate(names)
            )
        if len(target) == 3 and words[0] in ("wend", "wbeginning") and words[1] == "wof":
            return self._local_ok(target[2], ctx)  # set end of theList to …: the script's own list
        return words in (
            ["wapplescript", "p's", "wtext item delimiters"],
            ["wtext item delimiters", "wof", "wapplescript"],
            ["wmy", "wtext item delimiters"],
        )


def _as_lines(toks: list[_Tok]) -> list[list[_Tok]]:
    lines: list[list[_Tok]] = [[]]
    for t in toks:
        if t.kind == "nl":
            lines.append([])
        else:
            lines[-1].append(t)
    return [ln for ln in lines if ln]


# Why a task-wide "yes" can never cover a script (see neo/agent/grants.py), before the allowlist
# (_ScriptReader with `grant`) is even asked. Over the whole text, strings included: a button
# called "Send" is clicked with a string.
_SCRIPT_RUNS_CODE = re.compile(
    r"«|»|\||\bdo\s+(?:shell\s+)?(?:script|javascript)\b|\b(?:run|load|store)\s+script\b|\bcall\s+method\b|"
    r"\bcurrent\s+application\b|\buse\s+framework\b|\bexecute\b|\bdo\b|\bwith\s+administrator\b",
    re.I,
)
_SCRIPT_KEYS = re.compile(r"\bkeystroke\b|\bkey\s+(?:code|down|up)\b", re.I)
# Outside strings only: words that reach past the app (files, links, the Mac itself).
_SCRIPT_REACHES_OUT = re.compile(
    r"\b(?:open|launch|run|save|move|duplicate|export|write|eject|erase|empty|restart|shut\s*down|"
    r"log\s*out|sleep|mount|set\s+volume|clipboard)\b",
    re.I,
)
# Strings included: the Mac's own menus, whichever app's menu bar they are reached through — the
# Apple menu (sleep, restart, shut down, lock, recent items) and Services (other apps' workflows).
_SCRIPT_MAC_MENUS = re.compile(
    r"\bmenu\s+bar\s+item\s+(?:1\b|[\"“”]\s*apple\b)|\b(?:services|restart|shut\s*down|sleep|lock\s+screen|"
    r"force\s+quit|recent\s+items)\b",
    re.I,
)


def _ui_anyway(toks: list[_Tok] | None) -> bool:
    """For a script the reader can't follow: does it drive an app's UI through System Events?
    Read from the code alone (never its strings or comments), whole words only."""
    if not toks:
        return False
    if not any(
        t.kind == "w"
        and t.text in _AS_APP_WORDS
        and _is_se(_literal_after(toks, i) or "", i + 1 < len(toks) and toks[i + 1].is_("id"))
        for i, t in enumerate(toks)
    ):
        return False
    return any(
        t.kind == "w"
        and (
            t.text in _AS_UI_ELEMENTS
            or t.text in ("click", "perform", "keystroke")
            or (t.text == "key" and i + 1 < len(toks) and toks[i + 1].is_("code", "down", "up"))
        )
        for i, t in enumerate(toks)
    )


class ScriptFacts:
    """What NEO knows about a script before running it.

    `read_only`: every statement only reads. `apps`: what it touches — the apps it names, and a
    process "X" in System Events' place only where all a statement touches is that process's window.
    `app`: the one app it acts on ("" = none, several, or picked at run time). `ui`: it reads or
    drives an app's UI through System Events, which takes Accessibility. `danger`: why a task-wide
    yes never covers it ("" = one can: it only reads, fills in, or clicks by label in one ordinary
    app's window)."""

    __slots__ = ("read_only", "apps", "app", "ui", "danger")

    def __init__(self, read_only: bool, apps: tuple[str, ...], app: str, ui: bool, danger: str) -> None:
        self.read_only, self.apps, self.app, self.ui, self.danger = read_only, apps, app, ui, danger

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ScriptFacts(read_only={self.read_only}, app={self.app!r}, ui={self.ui}, danger={self.danger!r})"


def script_facts(script: str) -> ScriptFacts:
    src = script or ""
    toks = _as_lex(src)
    lines = _as_lines(toks or [])
    reader = _ScriptReader(lines, grant=True)
    shaped = bool(toks) and reader.ok()  # only reads, or reads and clicks in a named app's window
    read_only = shaped and _ScriptReader(lines).ok()
    code = src if toks is None else " ".join(t.text for t in toks if t.kind != "s")  # strings left out
    # Every app and process it names; one picked while it runs (or a script NEO can't read); an app
    # named by where it's stored.
    named: dict[str, str] = {}
    picked, by_path = toks is None, False
    for i, t in enumerate(toks or []):
        if t.kind != "w":
            continue
        if t.text in _AS_PICKED:
            picked = True
        elif t.text in _AS_APP_WORDS + _AS_PROC_WORDS:
            name = _literal_after(toks or [], i)
            if name is None:
                picked = True  # `every process`, `process 1`, `application (x)`, `process appName`
                continue
            by_path = by_path or (t.text in _AS_APP_WORDS and _by_path(name))
            named.setdefault(_app_key(name), name)
    apps = list(reader.targets.values()) if shaped else list(named.values())
    ui = reader.ui if shaped else _ui_anyway(toks)
    danger = ""
    if toks is None or _SCRIPT_RUNS_CODE.search(src):
        danger = "it runs code or raw Apple events"
    elif _SCRIPT_KEYS.search(code):
        danger = "keystrokes go to whichever app is in front"
    elif _COMMIT.search(src):
        danger = "it may send, buy, delete or grant something"
    elif shaped and reader.unchecked:
        danger = reader.unchecked[0]
    elif shaped and any(_GENERIC_OK.match(label) for label in reader.labels):
        danger = "its OK or Continue says nothing about what it confirms"
    elif _SCRIPT_REACHES_OUT.search(code):
        danger = "it reaches outside the app (files, links or the Mac itself)"
    elif _SCRIPT_MAC_MENUS.search(src):
        danger = "it reaches the Mac's own menus (the Apple menu, Services)"
    elif picked:
        danger = "it picks its app while it runs"
    elif by_path:
        danger = "it names an app by where it's stored"
    elif len(apps) > 1:
        danger = "it talks to more than one app"
    elif [_app_key(a) for a in apps] == [_SE]:
        danger = "System Events acts on the whole Mac"
    elif any(never_grant(a) for a in apps):
        danger = "that app runs commands, or holds NEO's or the Mac's own settings"
    elif not shaped:
        danger = "it does more than read, click or fill in one app's window"
    app = apps[0] if len(apps) == 1 and not picked and not by_path else ""
    return ScriptFacts(read_only, tuple(apps), app, ui, danger)


def script_is_read_only(script: str) -> bool:
    """An AppleScript that only asks questions (get/count/exists/name of…) — see the reader
    above; anything it can't show only reads counts as acting."""
    return script_facts(script).read_only


# ---- clicks and keys -------------------------------------------------------------------------
_COMMIT = re.compile(
    r"\b(?:send|submit|buy|pay|purchase|order|checkout|check\s+out|delete|remove|erase|trash|"
    r"transfer|confirm|post|publish|tweet|share|invite|allow|authori[sz]e|grant|accept|agree|"
    r"install|download|subscribe|unsubscribe|sign\s+out|log\s+out|deactivate|empty|reset|"
    r"uninstall|approve|withdraw|deposit|donate|book|reserve|sell|upgrade|renew|restore|wipe|"
    r"format|overwrite|replace|discard|leave|block|report|unfriend|unfollow|close\s+account)\b",
    re.I,
)


_GENERIC_OK = re.compile(r"^\s*(?:ok(?:ay)?|yes|continue|proceed|apply|done|confirm|i\s+agree|got\s+it|allow)\s*[.!]?\s*$", re.I)
_CONSEQUENCE = re.compile(
    r"\b(?:delete|erase|remove|permanently|cannot\s+be\s+undone|can'?t\s+be\s+undone|overwrite|replace|"
    r"discard|send|pay|purchase|buy|order|transfer|uninstall|reset|wipe|format|sign\s+out|log\s+out|"
    r"unsubscribe|close\s+account|deactivate|publish|post|share|grant|allow|access)\b",
    re.I,
)


# What a press in an app's own UI can't take back, that scripts and drops were never gated on: a
# forward (it sends), ending the app, the session or the Mac (Quit, Restart…, Shut Down…, Log Off,
# Force Quit), throwing work away (Revert, Don't Save, Clear History), giving a game up (Resign).
# Labels only: commits() reads it; the AppleScript and drop checks don't. (Moving forward - a browser's
# "Go Forward", a player's "Skip Forward" - is navigation, not a forward.)
_UI_COMMIT = re.compile(
    r"\b(?:(?<!go )(?<!skip )(?<!fast )(?<!fast-)(?<!move )forward|quit|restart|reboot|shut\s*down|power\s+off|log\s+off|sign\s+off|force\s+quit|kill|"
    r"terminate|revert|don['’]?t\s+save|do\s+not\s+save|resign|forfeit|surrender|concede|abandon|"
    r"end\s+(?:game|match|call|meeting|session)|clear\s+(?:all|history|browsing|everything|cache|cookies|data|recents?)|"
    r"update\s+now)\b",
    re.I,
)
# Words in a sheet or dialog saying that answering it ends, loses or throws away something (the
# current game, unsaved changes): with them NEO never answers the sheet by itself, whatever button.
_LOSS = re.compile(
    r"\b(?:lose|loses|lost|losing|unsaved|not\s+(?:been\s+)?saved|won['’]?t\s+be\s+saved|will\s+not\s+be\s+saved|"
    r"can(?:not|['’]?t)\s+be\s+(?:undone|recovered|restored|reversed)|irreversibl[ey]|permanent(?:ly)?|"
    r"will\s+be\s+(?:closed|deleted|removed|erased|lost|replaced|overwritten|ended|reset|discarded|cancell?ed|stopped)|"
    r"(?:current|unfinished|ongoing)\s+(?:game|match|document|changes|session|progress|call|download)|"
    r"in\s+progress|resign|forfeit|quit|restart|shut\s*down|log\s+(?:out|off)|sign\s+out|end\s+(?:the|this|your))\b",
    re.I,
)

# English source strings of a localized label or dialog text (a Hindi "भेजें" is the OS's "Send"):
# registered by the Mac side (neo.decide.uistrings, from the OS's own string tables). Data handed
# in, never imported: this module stays pure stdlib.
_back_translate: Callable[[str], Iterable[str]] | None = None


def use_back_translations(fn: Callable[[str], Iterable[str]] | None) -> None:
    """Register (None: drop) where a localized label's English source strings come from."""
    global _back_translate
    _back_translate = fn


def back_translations(text: str) -> tuple[str, ...]:
    """The English source strings of a localized label or dialog text, from the registered source;
    () for plain ASCII (it is English already), without a source, or when it fails."""
    fn = _back_translate
    if fn is None or not text or text.isascii():
        return ()
    try:
        return tuple(s for s in fn(text) if isinstance(s, str) and s)[:32]
    except Exception:  # noqa: BLE001 - a broken lookup adds nothing; today's checks still run
        return ()


def generic_ok(label: str, sources: Iterable[str] | None = None) -> bool:
    """A plain "OK" / "Continue" / "Yes" (in any language the back-translations know): it says
    nothing about what it confirms, so the dialog's own words decide."""
    srcs = back_translations(label or "") if sources is None else tuple(sources)
    return any(_GENERIC_OK.match(t or "") for t in (label or "", *srcs))


def commits(
    label: str,
    context: str = "",
    *,
    sources: Iterable[str] | None = None,
    context_sources: Iterable[str] | None = None,
) -> bool:
    """A button that sends, buys, deletes, grants or publishes something, or ends or throws away
    something (Quit, Restart…, Don't Save, Resign) — or a plain "OK" / "Continue" in a dialog whose
    text says it does. A localized label (or dialog text) also counts as its English source strings
    (`sources` / `context_sources`; by default the registered back-translations): they are only
    ever OR'd in, so the gate never gets looser."""
    label, context = label or "", context or ""
    srcs = back_translations(label) if sources is None else tuple(sources)
    if any(_COMMIT.search(t or "") or _UI_COMMIT.search(t or "") for t in (label, *srcs)):
        return True
    if not generic_ok(label, srcs):
        return False
    ctx = back_translations(context) if context_sources is None else tuple(context_sources)
    return any(_CONSEQUENCE.search(t or "") for t in (context, *ctx))


def sheet_consequence(text: str, sources: Iterable[str] | None = None) -> bool:
    """Does a sheet's or dialog's own text say that answering it commits, ends or loses something
    ("the current game will be lost", "can't be undone", "send to 40 people")? Checked for every
    button before NEO answers a sheet by itself, not only for a plain OK."""
    srcs = back_translations(text or "") if sources is None else tuple(sources)
    return any(_CONSEQUENCE.search(t or "") or _LOSS.search(t or "") for t in (text or "", *srcs))


def drop_commits(label: str, window_words: str = "") -> bool:
    """A drop that deletes, sends or shares where it lands: onto something whose label says so (the
    Trash), or into a window whose own words say it does (an AirDrop window: "share instantly with
    people nearby"; its recipients are just names). What is dragged never counts — a file called
    "report.pdf" is a file, not a button."""
    return bool(_COMMIT.search(label or "") or _CONSEQUENCE.search(window_words or ""))


# ---- URLs ------------------------------------------------------------------------------------
_DOC_EXT = re.compile(
    r"\.(?:pdf|txt|md|rtf|rtfd|docx?|xlsx?|pptx?|key|pages|numbers|csv|tsv|json|png|jpe?g|gif|heic|"
    r"webp|tiff?|bmp|svg|mp3|m4a|aac|wav|aiff?|flac|mp4|m4v|mov|avi|mkv|epub|html?|log)$",
    re.I,
)
_APP_DIRS = [Path("/Applications"), Path("/System/Applications"), HOME / "Applications"]


def open_risk(target: str) -> str:
    """Why opening this with `open` needs the user's OK ("" = it doesn't)."""
    t = target.strip()
    if re.match(r"^[\w.-]+:\d+(?:/|$)", t):
        return ""  # localhost:3000
    if "://" in t or (re.match(r"^[a-z][\w+.-]*:", t, re.I) and not t.startswith(("/", "~"))):
        scheme = t.split(":", 1)[0].lower()
        if scheme in ("http", "https", "mailto"):
            return ""
        return f"{scheme}: links can make other apps do things"
    if not t.startswith(("/", "~")):
        return ""  # an app or site name
    p = _variants(t)[-1]
    if p.suffix.lower() == ".app":
        return "" if any(_under(p, d) for d in _APP_DIRS) else "it's an app from outside Applications"
    if p.is_dir():
        return "" if not is_secret(p) else "it holds secrets"
    if is_secret(p):
        return "it holds secrets"
    if _DOC_EXT.search(p.name):
        return ""
    return "opening it could run it"


def url_carries_data(url: str) -> bool:
    """The shape of a URL that smuggles data out: a long query, fragment or path segment, or a
    long or random-looking subdomain (data can ride in the hostname as well)."""
    try:
        u = urlsplit(url if "://" in url else "https://" + url)
    except ValueError:
        return True
    if len(u.query) + len(u.fragment) > 48 or any(len(seg) > 48 for seg in u.path.split("/")):
        return True
    labels = (u.hostname or "").split(".")[:-2]  # everything left of the registrable domain
    return any(
        len(lb) > 24 or (len(lb) >= 12 and re.search(r"\d", lb) and re.search(r"[a-z]", lb)) for lb in labels
    )


def private_host(url: str) -> str:
    """The reason a background fetch must not go there ("" = public internet)."""
    try:
        u = urlsplit(url if "://" in url else "https://" + url)
    except ValueError:
        return "that isn't a valid URL"
    if u.scheme not in ("http", "https"):
        return f"{u.scheme or 'that'} URLs aren't fetched"
    host = (u.hostname or "").strip("[]").lower()
    if not host:
        return "that URL has no host"
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        return "it's on this Mac or the local network"
    try:
        infos = socket.getaddrinfo(host, u.port or (443 if u.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError):
        return ""  # doesn't resolve: the fetch itself will fail
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return "it's on this Mac or the local network"
        # a VPN's or carrier's shared range (100.64/10: Tailscale peers), IPv6 site-local (fec0::/10)
        if not ip.is_global or getattr(ip, "is_site_local", False):
            return "it's on this Mac or the local network"
    return ""
