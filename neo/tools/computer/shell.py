"""Shell + file primitives with a destructive-command gate."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from neo.agent import confirm

_DESTRUCTIVE = re.compile(
    r"("
    r"\brm\b(?:\s+\S+)*?\s+-[a-z]*[rf]|\brm\s+(?!-)\S+\s*$|"  # rm with -r/-f anywhere, or bare rm of a file
    r"\bfind\b.*\s-delete\b|\bsudo\b|\bmkfs|\bdd\s+if=|>\s*/dev/(?!null)|\bkillall\b|\bpkill\b|\bkill\s+-9|"
    r"\bshutdown\b|\breboot\b|\bdiskutil\s+(?:erase|partition|unmount)|"
    r"\bgit\s+(?:push\s+(?:-f|--force)|reset\s+--hard|clean\s+-\S*f|branch\s+-D|checkout\s+--|stash\s+drop)|"
    r"\b(?:brew|pip3?|npm|pnpm|yarn|cargo|gem)\s+(?:uninstall|remove|rm|un)\b|"
    r"\|\s*(?:ba|z|k|da)?sh\b|\bchmod\s+-R|\bchown\s+-R|\bdefaults\s+(?:delete|write)|"
    r"\blaunchctl\s+(?:unload|remove|bootout)|\bsecurity\s+delete|\bosascript\b|\btruncate\b|\b:>\s*\S|"
    r"\bmv\b.*\s/(?:Users|Applications|Library|System)\b|\bcrontab\s+-r|\bpmset\b|\bnvram\b|\bcsrutil\b"
    r")",
    re.I,
)


def looks_destructive(cmd: str) -> bool:
    return bool(_DESTRUCTIVE.search(cmd))


async def run(
    cmd: str, *, cwd: str | None = None, timeout: float = 60, confirm_arg: dict | None = None
) -> str:
    if looks_destructive(cmd):
        gate = confirm.gated(
            "shell:" + cmd[:60], "shell", {**(confirm_arg or {}), "command": cmd}, f"run `{cmd[:120]}`"
        )
        if isinstance(gate, str):
            return gate
    home = str(Path.home())
    wd = os.path.expanduser(cwd) if cwd else home
    p = await asyncio.create_subprocess_shell(
        cmd,
        cwd=wd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "PATH": "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")},
    )
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
    except TimeoutError:
        p.kill()
        return f"Error: command timed out after {timeout}s"
    text = out.decode(errors="replace")
    code = p.returncode
    return (text.strip() or "(no output)") + ("" if code == 0 else f"\n[exit {code}]")


def read_file(path: str, *, start: int = 1, lines: int = 400) -> str:
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


def write_file(path: str, content: str, args: dict) -> str:
    p = Path(os.path.expanduser(path))
    if p.exists():
        gate = confirm.gated(f"write:{p}", "write_file", {**args, "path": str(p)}, f"overwrite {p.name}")
        if isinstance(gate, str):
            return gate
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {p}"


def edit_file(path: str, old: str, new: str) -> str:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"Error: {path} does not exist"
    text = p.read_text()
    n = text.count(old)
    if n == 0:
        return "Error: old_string not found (must match exactly, including whitespace)"
    if n > 1:
        return f"Error: old_string matches {n} places; include more context to make it unique"
    p.write_text(text.replace(old, new, 1))
    return f"Edited {p.name}"


def trash(path: str, args: dict) -> str:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"Error: {path} does not exist"
    gate = confirm.gated(f"trash:{p}", "trash", {**args, "path": str(p)}, f"move {p.name} to the Trash")
    if isinstance(gate, str):
        return gate
    import subprocess

    r = subprocess.run(
        ["osascript", "-e", f'tell application "Finder" to delete POSIX file "{p}"'],
        capture_output=True,
        text=True,
    )
    return f"Moved {p.name} to Trash" if r.returncode == 0 else f"Error: {r.stderr.strip()}"
