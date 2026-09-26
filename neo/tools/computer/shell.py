"""Shell + file primitives behind the confirm gate (see policy.py for what runs without asking)."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from neo.agent import confirm
from neo.tools.computer import policy

_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|SESSION|COOKIE|AUTH", re.I)


def looks_destructive(cmd: str) -> bool:
    """Kept for callers: anything that isn't provably read-only is treated as consequential."""
    return not policy.shell_is_read_only(cmd)


def _env() -> dict[str, str]:
    """The user's environment minus anything that looks like a credential (NEO's API keys too)."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "/usr/bin:/bin")
    return env


async def run(cmd: str, *, cwd: str | None = None, timeout: float = 60, args: dict | None = None) -> str:
    if not policy.shell_is_read_only(cmd):
        where = f" (in {cwd})" if cwd else ""
        gate = confirm.gated(
            "shell", args if args is not None else {"command": cmd, "cwd": cwd}, f"run a command{where}",
            detail=cmd,
        )
        if isinstance(gate, str):
            return gate
    wd = os.path.expanduser(cwd) if cwd else str(Path.home())
    p = await asyncio.create_subprocess_shell(
        cmd,
        cwd=wd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        env=_env(),
    )
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
    except TimeoutError:
        p.kill()
        return f"Error: command timed out after {timeout}s"
    text = out.decode(errors="replace")
    code = p.returncode
    return (text.strip() or "(no output)") + ("" if code == 0 else f"\n[exit {code}]")


def _secret_refusal(path: str) -> str:
    return (
        f"Error: {path} holds keys, passwords or private app data, so NEO doesn't read it. "
        "If the user needs it, they can open it themselves."
    )


def read_file(path: str, *, start: int = 1, lines: int = 400) -> str:
    if policy.is_secret(path):
        return _secret_refusal(path)
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"Error: {path} does not exist"
    if p.is_dir():
        items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        return "\n".join(("📁 " if i.is_dir() else "   ") + i.name for i in items[:300]) or "(empty dir)"
    if p.suffix.lower() in (".pdf",):
        try:
            from pypdf import PdfReader

            r = PdfReader(str(p))
            return "\n".join((pg.extract_text() or "") for pg in r.pages[:30])[:20000]
        except Exception as e:  # noqa: BLE001
            return f"Error reading PDF: {e}"
    try:
        content = p.read_text(errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
    all_lines = content.splitlines()
    start = max(1, int(start))
    lines = max(1, int(lines))
    chunk = all_lines[start - 1 : start - 1 + lines]
    body = "\n".join(f"{i + start:5d}| {line}" for i, line in enumerate(chunk))
    if len(all_lines) > start - 1 + lines:
        body += f"\n… {len(all_lines) - (start - 1 + lines)} more lines"
    return body


def _preview(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… ({len(text) - limit} more characters)"


def write_file(path: str, content: str, args: dict) -> str:
    p = Path(os.path.expanduser(path))
    if why := policy.write_risk(path):
        if policy.is_secret(path):
            return f"Error: {p} holds keys or private app data — NEO doesn't write there."
        gate = confirm.gated(
            "write_file", args, f"write {p.name} ({why})", detail=f"{p}\n\n{_preview(content)}"
        )
        if isinstance(gate, str):
            return gate
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {p}"


def edit_file(path: str, old: str, new: str, args: dict) -> str:
    if policy.is_secret(path):
        return f"Error: {path} holds keys or private app data — NEO doesn't edit it."
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"Error: {path} does not exist"
    text = p.read_text()
    n = text.count(old)
    if n == 0:
        return "Error: old_string not found (must match exactly, including whitespace)"
    if n > 1:
        return f"Error: old_string matches {n} places; include more context to make it unique"
    gate = confirm.gated(
        "edit_file", args, f"edit {p.name}", detail=f"{p}\n\n− {_preview(old, 700)}\n+ {_preview(new, 700)}"
    )
    if isinstance(gate, str):
        return gate
    p.write_text(text.replace(old, new, 1))
    return f"Edited {p.name}"


def trash(path: str, args: dict) -> str:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"Error: {path} does not exist"
    gate = confirm.gated("trash", args, f"move {p.name} to the Trash", detail=str(p.resolve()))
    if isinstance(gate, str):
        return gate
    import subprocess

    r = subprocess.run(
        ["osascript", "-e", 'on run argv\ntell application "Finder" to delete (POSIX file (item 1 of argv))\nend run', str(p)],
        capture_output=True,
        text=True,
    )
    return f"Moved {p.name} to Trash" if r.returncode == 0 else f"Error: {r.stderr.strip()}"
