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
_SCRIPT_ACTS = re.compile(
    r"do\s+(?:shell\s+)?script|do\s+javascript|run\s+script|load\s+script|store\s+script|"
    r"current\s+application|use\s+framework|«|\bcall\s+method\b|\busing\s+terms\b|System\s+Events|"
    r"\b(?:set|make|send|delete|save|open|close|quit|move|duplicate|remove|erase|empty|restart|"
    r"shut\s*down|log\s*out|sleep|play|pause|activate|launch|reveal|select|print|display|create|"
    r"import|export|convert|copy|put|write|read|mount|eject|update|forward|reply|redirect|mark|"
    r"keystroke|key\s+code|click|perform|tell\s+process|choose|beep|say|lock|install|reload|"
    r"refresh|go|search|add|attach|sign|accept|decline|join|leave|post|submit|buy|move|enable|"
    r"disable|toggle|stop|resume|advance|increase|decrease|next|previous|skip|mute|unmute|"
    r"rename|clean|restore|sync|backup|download|upload|share|log|run|launch|quit|hide|show|"
    r"bring|zoom|minimi[sz]e|miniaturize|collapse|expand|check|uncheck|mount|unmount)\b",
    re.I,
)


def script_is_read_only(script: str) -> bool:
    """An AppleScript that only asks questions (get/count/exists/name of…) runs without asking."""
    return not _SCRIPT_ACTS.search(script or "")


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


def commits(label: str, context: str = "") -> bool:
    """A button that sends, buys, deletes, grants or publishes something — or a plain "OK" /
    "Continue" in a dialog whose text says it does."""
    if _COMMIT.search(label or ""):
        return True
    return bool(_GENERIC_OK.match(label or "") and _CONSEQUENCE.search(context or ""))


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
    return ""
