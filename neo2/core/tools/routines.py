"""User-defined spoken routines, kept locally in NEO's data directory.

Routines are deliberately small and declarative: a user phrase maps to an ordered list of
``open_app`` and ``open_url`` calls. They never contain shell commands, scripts, keystrokes or
arbitrary model-generated actions. Every launch still goes through NEO's normal tool handlers and
their safety checks.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from neo2.core.agent.registry import ToolContext, ToolOutput, registry, tool
from neo2.core.config import settings
from neo2.core.memory.redact import has_secret

_FILE = "routines.json"
_LOCK = threading.RLock()
_MAX_ROUTINES = 50
_MAX_ACTIONS = 8
_WAKE = re.compile(r"^\s*(?:(?:hey|hi|okay|ok)\s+)?neo[\s,:!.-]*", re.I)
_SPACE = re.compile(r"[^\w\s]", re.UNICODE)
_SETUP = re.compile(
    r"\b(?:when|whenever)\s+i\s+say\b|\b(?:create|make|set\s+up|save)\s+(?:a\s+|my\s+)?"
    r"(?:spoken\s+|voice\s+)?(?:routine|shortcut|custom\s+phrase)\b",
    re.I,
)
_LOCAL_SETUP = re.compile(
    r"^\s*(?:when|whenever)\s+i\s+say\s+(?P<trigger>.+?)"
    r"(?:\s*[,;:]\s*|\s+then\s+|\s+(?=(?:please\s+)?(?:open|launch|start|bring\s+up)\b))"
    r"(?:open|launch|start|bring\s+up)\s+(?P<targets>.+?)\s*[.!?]*\s*$",
    re.I,
)
_URL_TARGET = re.compile(
    r"^(?:https?://)?(?:www\.)?[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?(?::\d+)?"
    r"(?:[/?#][^\s]*)?$",
    re.I,
)


def _path() -> Path:
    return settings().data_dir / _FILE


def normalize(text: str) -> str:
    text = _WAKE.sub("", " ".join(str(text or "").split()), count=1)
    return " ".join(_SPACE.sub(" ", text.casefold()).split())


def is_setup_request(text: str) -> bool:
    """Recognize an explicit user instruction to teach a routine; never infer one from a goal."""
    return bool(_SETUP.search(_WAKE.sub("", str(text or ""), count=1)))


def parse_local_setup(text: str) -> dict | None:
    """Parse the unambiguous routine form locally, without sending its words to a model.

    Examples: ``When I say “I want to study”, open Notion, Safari, and youtube.com`` and the
    spoken equivalent ``When I say I want to study, then open Notion and Safari``. More open-ended
    definitions can still go through the constrained routine manager.
    """
    raw = _WAKE.sub("", " ".join(str(text or "").split()), count=1)
    m = _LOCAL_SETUP.match(raw)
    if m is None:
        return None
    trigger = m.group("trigger").strip().strip(" \t,;:'\"“”‘’")
    if len(trigger.split()) < 2:
        return None
    targets = m.group("targets").strip().rstrip(".!?").strip()
    parts = [
        re.sub(r"\s+please$", "", part.strip(), flags=re.I).strip()
        for part in re.split(r"\s*(?:,|;|\band\b|\bthen\b)\s*", targets, flags=re.I)
    ]
    actions: list[dict] = []
    for target in parts:
        target = re.sub(r"^(?:(?:please\s+)?(?:open|launch|start|visit|go\s+to|navigate\s+to)\s+)+", "", target, flags=re.I)
        target = re.sub(r"^(?:the\s+)?(?:website|site|app)\s+", "", target, flags=re.I).strip()
        target = target.strip(" \t,;:'\"“”‘’")
        if not target:
            return None
        if re.match(r"^(?:https?://|www\.)", target, re.I):
            kind = "open_url"
        elif _URL_TARGET.fullmatch(target) and "." in target.split(":", 1)[0]:
            kind = "open_url"
        else:
            # A dotted first token usually means the user intended a site but omitted a valid URL.
            if re.match(r"^[\w-]+\.[\w.-]+", target) and any(ch.isspace() for ch in target):
                return None
            kind = "open_app"
        try:
            actions.append(_safe_action({"kind": kind, "target": target}))
        except (TypeError, ValueError):
            return None
    if not 1 <= len(actions) <= _MAX_ACTIONS:
        return None
    return {"action": "create", "trigger": trigger, "actions": actions}


def _read() -> list[dict]:
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        rows = value.get("routines", []) if isinstance(value, dict) else []
        return [r for r in rows if isinstance(r, dict)]
    except (OSError, ValueError, TypeError):
        return []


def _write(rows: list[dict]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "routines": rows[-_MAX_ROUTINES:]}, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _safe_action(raw: dict) -> dict:
    kind = str(raw.get("kind") or raw.get("type") or "").strip().lower()
    target = " ".join(str(raw.get("target") or "").split())
    if not target or len(target) > 180 or any(ord(ch) < 32 for ch in target):
        raise ValueError("Each routine action needs a short app name or website address.")
    if has_secret(target):
        raise ValueError("Don't save a password, access token, or other secret in a routine destination.")
    if kind in ("app", "open_app"):
        if "/" in target or "://" in target:
            raise ValueError(f"{target!r} does not look like an app name.")
        return {"kind": "open_app", "target": target}
    if kind in ("website", "url", "open_url"):
        raw_url = target if "://" in target else "https://" + target
        parts = urlsplit(raw_url)
        host = (parts.hostname or "").rstrip(".").lower()
        if parts.scheme.lower() not in ("http", "https") or not host or parts.username or parts.password:
            raise ValueError("Routine websites must be ordinary http or https addresses.")
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise ValueError("Local network addresses can't be saved in a spoken routine.")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("Private or local network addresses can't be saved in a spoken routine.")
        # Fragments are browser state, not part of a useful launch destination. Keep a path and
        # query the user explicitly gave (the regular open_url policy still checks it at runtime).
        return {"kind": "open_url", "target": urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, parts.query, ""))}
    raise ValueError("Routine actions can only open an app or a website.")


def match(text: str) -> dict | None:
    """Return an exact normalized match without mutating usage stats.

    Planning is also used for previews and outcome classification, so resolving a phrase must be
    side-effect free. ``routine_run`` records a use when execution actually begins.
    """
    phrase = normalize(text)
    if not phrase or len(phrase.split()) < 2:
        return None
    with _LOCK:
        rows = _read()
        found = next((r for r in reversed(rows) if r.get("match") == phrase and r.get("enabled", True)), None)
        if found is None:
            return None
        return dict(found)


def _record_use(routine_id: str) -> None:
    with _LOCK:
        rows = _read()
        found = next((r for r in rows if str(r.get("id")) == routine_id), None)
        if found is None:
            return
        try:
            uses = int(found.get("uses", 0))
        except (TypeError, ValueError):
            uses = 0
        found["uses"] = uses + 1
        found["last_used"] = time.time()
        _write(rows)


def has_actions(actions: list[dict]) -> bool:
    """Whether the user already saved this ordered launch sequence."""
    try:
        safe = [_safe_action(a) for a in actions]
    except (TypeError, ValueError):
        return False
    with _LOCK:
        return any(r.get("actions") == safe for r in _read())


def forget() -> int:
    """Delete the user's routines as part of NEO's erase-everything action."""
    path = _path()
    with _LOCK:
        rows = _read()
        path.unlink(missing_ok=True)
        return len(rows)


def _routine_label(row: dict) -> str:
    actions = [f"open {a['target']}" for a in row.get("actions", [])]
    return f"{row.get('name') or row.get('trigger')} (say “{row.get('trigger')}”): " + " → ".join(actions)


def _join_targets(actions: list[dict]) -> str:
    targets = [str(a["target"]) for a in actions]
    if len(targets) < 2:
        return targets[0] if targets else "nothing"
    if len(targets) == 2:
        return " and ".join(targets)
    return ", ".join(targets[:-1]) + ", and " + targets[-1]


def _create(args: dict, ctx: ToolContext) -> str | ToolOutput:
    if ctx.origin != "approval" and not is_setup_request(ctx.user_text):
        return "Error: only save a routine when the user's own request explicitly says to define a spoken phrase or shortcut."
    trigger = " ".join(str(args.get("trigger") or "").split()).strip(" .,!?")
    name = " ".join(str(args.get("name") or "").split())[:60] or trigger[:60]
    key = normalize(trigger)
    raw_actions = args.get("actions")
    if not trigger or len(trigger) > 100 or len(key.split()) < 2:
        return "Error: choose a distinct spoken phrase of at least two words."
    if not isinstance(raw_actions, list) or not 1 <= len(raw_actions) <= _MAX_ACTIONS:
        return f"Error: a routine needs between 1 and {_MAX_ACTIONS} app or website actions."
    try:
        actions = [_safe_action(a) for a in raw_actions if isinstance(a, dict)]
    except ValueError as e:
        return f"Error: {e}"
    if len(actions) != len(raw_actions):
        return "Error: every routine action must be an app or website."
    if ctx.tainted():
        from neo2.core.agent import confirm

        detail = f"Phrase: {trigger}\n" + "\n".join(f"{a['kind']}: {a['target']}" for a in actions)
        gate = confirm.gated("routine_manage", {"action": "create", "trigger": trigger, "name": name, "actions": actions},
                             f"save a spoken routine based on {ctx.tainted()}", detail=detail)
        if isinstance(gate, str):
            return ToolOutput(gate, halt=True, staged=getattr(gate, "action_id", ""))
    with _LOCK:
        rows = _read()
        found = next((r for r in rows if r.get("match") == key), None)
        if found:
            found.update({"name": name, "trigger": trigger, "actions": actions, "enabled": True})
            verb = "Updated"
        else:
            if len(rows) >= _MAX_ROUTINES:
                rows = rows[1:]
            rows.append({
                "id": uuid.uuid4().hex[:12], "name": name, "trigger": trigger, "match": key,
                "actions": actions, "enabled": True, "created": time.time(), "uses": 0, "last_used": 0.0,
            })
            verb = "Created"
        _write(rows)
    status = "Saved locally" if verb == "Created" else "Updated locally"
    return f'{status}: when you say “{trigger}”, I’ll open {_join_targets(actions)}.'


def _list() -> str:
    with _LOCK:
        rows = _read()
    if not rows:
        return "You don't have any spoken routines yet. Tell me: “When I say [phrase], open [apps or websites].”"
    return "Your spoken routines:\n" + "\n".join(f"• {_routine_label(r)}" for r in rows)


def _delete(args: dict, ctx: ToolContext) -> str | ToolOutput:
    target = " ".join(str(args.get("name") or args.get("trigger") or "").split())
    target = target.strip(" \'\"“”‘’.,!? ").casefold()
    if not target:
        return "Error: tell me which routine to delete."
    with _LOCK:
        rows = _read()
        row = next((r for r in rows if target in (str(r.get("id", "")).casefold(),
                                                   str(r.get("name", "")).casefold(),
                                                   str(r.get("trigger", "")).casefold())), None)
        if row is None:
            return "Error: I couldn't find that routine. Ask me to list your routines."
        from neo2.core.agent import confirm

        detail = _routine_label(row)
        gate = confirm.gated("routine_manage", {"action": "delete", "name": row["name"]},
                             f"delete the spoken routine “{row['name']}”", detail=detail)
        if isinstance(gate, str):
            return ToolOutput(gate, halt=True, staged=getattr(gate, "action_id", ""))
        rows.remove(row)
        _write(rows)
    return f'Deleted the “{row["name"]}” routine.'


@tool(
    "routine_manage",
    "Create, list or delete the user's local spoken routines. Use this when they say 'when I say X, do Y', "
    "want a custom voice phrase/shortcut/workflow, or ask what routines they have. A routine trigger is an exact "
    "spoken phrase and its actions may only open named Mac apps or http/https websites; do not invent actions.",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "delete"]},
            "name": {"type": "string", "description": "A short routine name, or the name to delete."},
            "trigger": {"type": "string", "description": "The exact phrase the user will say, without the wake word."},
            "actions": {
                "type": "array", "maxItems": _MAX_ACTIONS,
                "items": {"type": "object", "properties": {
                    "kind": {"type": "string", "enum": ["open_app", "open_url"]},
                    "target": {"type": "string"},
                }, "required": ["kind", "target"]},
            },
        },
        "required": ["action"],
    },
    category="memory",
    fast_path=[
        (r"^\s*(?:list|show|what are)\s+(?:my\s+)?(?:voice\s+)?routines\s*[?.!]*$", {"action": "list"}),
        (r"^\s*(?:delete|remove|forget)\s+(?:my\s+)?(?:the\s+)?routine\s+(?P<name>.+?)\s*[?.!]*$",
         {"action": "delete", "name": "<name>"}),
        (r"^\s*(?:delete|remove|forget)\s+(?:my\s+)?(?:the\s+)?(?P<name>.+?)\s+routine\s*[?.!]*$",
         {"action": "delete", "name": "<name>"}),
    ],
)
async def routine_manage(args: dict, ctx: ToolContext) -> str | ToolOutput:
    action = str(args.get("action") or "").strip().lower()
    if action == "list":
        return _list()
    if action == "create":
        return _create(args, ctx)
    if action == "delete":
        return _delete(args, ctx)
    return "Error: choose create, list or delete."


@tool(
    "routine_run",
    "Run one saved spoken routine.",
    {"type": "object", "properties": {"routine_id": {"type": "string"}}, "required": ["routine_id"]},
    category="apps", quiet=False, hidden=True, timeout=90.0,
)
async def routine_run(args: dict, ctx: ToolContext) -> ToolOutput:
    rid = str(args.get("routine_id") or "")
    with _LOCK:
        row = next((dict(r) for r in _read() if str(r.get("id")) == rid), None)
    if row is None:
        return ToolOutput("That routine is no longer available. Ask me to list your routines.", ok=False, say=True)
    _record_use(rid)
    job = str(ctx.get("neo_job") or "")
    for index, action in enumerate(row.get("actions", []), 1):
        try:
            action = _safe_action(action)
        except (TypeError, ValueError):
            return ToolOutput(f'“{row.get("name", row.get("trigger"))}” contains an invalid saved action.',
                              ok=False, say=True)
        tool_name = action["kind"]
        call_args = {"target": action["target"]} if tool_name == "open_app" else {"url": action["target"]}
        from neo2.core.events import bus

        await bus().tool_start(tool_name, call_args, job=job)
        out = await registry().invoke(tool_name, call_args, ctx)
        await bus().tool_end(tool_name, out.ok, out.text, job=job)
        if out.halt:
            return ToolOutput(out.text, ok=False, halt=True, staged=out.staged, say=True)
        if not out.ok:
            return ToolOutput(f'“{row.get("name", row.get("trigger"))}” stopped at action {index}: {out.text}',
                              ok=False, say=True)
    return ToolOutput(f'Started your “{row.get("name", row.get("trigger"))}” setup.', say=True, fence=False)
