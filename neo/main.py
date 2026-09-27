"""NEO entry point.

python -m neo            # full assistant: voice + overlay server
python -m neo --text     # text REPL in the terminal (no voice, no UI)
python -m neo --doctor   # check keys, permissions, models
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from neo.config import settings
from neo.events import bus


def _print_event(ev) -> None:
    d = ev.data
    if ev.type == "state":
        print(f"  · {d['state']}", file=sys.stderr)
    elif ev.type == "tool_start":
        print(f"  ▶ {d['name']} {d['args']}", file=sys.stderr)
    elif ev.type == "tool_end":
        print(f"  ◀ {d['name']} {'ok' if d['ok'] else 'FAIL'}: {d['summary'][:160]}", file=sys.stderr)
    elif ev.type == "reflex":
        print(
            f"  ~ {d['intent']} ({d['intent_confidence']:.2f}) destructive={d['destructive']} screen={d['needs_screen']} via {d['source']} {d['latency_ms']:.0f}ms",
            file=sys.stderr,
        )


async def text_repl(verbose: bool) -> None:
    from neo.agent.session import Session
    from neo.memory.store import store
    from neo.reflex import warmup
    from neo.tools import load_all

    load_all()
    sess = Session()
    sess.memory_context = store().prompt_context()
    warmup(sess.reflex)
    if verbose:
        bus().subscribe(_print_event)
    print("NEO ready. Type a request (Ctrl-D to quit).")
    import time

    started = time.time()
    loop = asyncio.get_event_loop()
    while True:
        try:
            text = await loop.run_in_executor(None, lambda: input("\nyou> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            continue
        sess.memory_context = store().prompt_context(text)
        reply = await sess.handle(text, from_user=True)
        print(f"neo> {reply.text}")
    from neo.memory.summarize import summarize_session
    from neo.tools import render
    from neo.tools.browser import shutdown as close_browser

    await close_browser()
    await render.shutdown()
    if summary := await summarize_session(sess.history, started):
        print(f"[memory] session saved: {summary[:120]}", file=sys.stderr)


def doctor() -> int:
    s = settings()
    ok = True

    def row(label: str, good: bool, note: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {'✓' if good else '✗'} {label}{(' — ' + note) if note else ''}")

    print("Keys")
    row(
        "GEMINI_API_KEY",
        bool(s.gemini_api_key),
        "free at aistudio.google.com" if not s.gemini_api_key else "",
    )
    row(
        "GROQ_API_KEY (optional, fast chat)",
        bool(s.groq_api_key),
        "free at console.groq.com" if not s.groq_api_key else "",
    )
    print("macOS permissions (grant to the app that runs NEO: Terminal / the NEO app)")
    if sys.platform == "darwin":
        from neo.tools.computer import ax, screen

        row("Accessibility", ax.is_trusted(), "System Settings → Privacy & Security → Accessibility")
        from neo.tools.computer.input import can_post

        row(
            "Sending clicks and keys",
            can_post(),
            "System Settings → Privacy & Security → Accessibility: turn NEO off and on again",
        )
        row(
            "Screen Recording",
            screen.can_capture(),
            "System Settings → Privacy & Security → Screen Recording",
        )
    print("Local models")
    try:
        import laya  # noqa: F401

        row("Laya reflex", True)
    except Exception as e:  # noqa: BLE001
        row("Laya reflex", False, str(e))
    try:
        import mlx_audio  # noqa: F401

        row("mlx-audio (local voice)", True)
    except Exception as e:  # noqa: BLE001
        row("mlx-audio (local voice)", False, str(e))
    print(f"Data dir: {s.data_dir}")
    return 0 if ok else 1


async def _erase_everything() -> dict:
    """A running NEO keeps the conversation in memory and would write it back: ask it to erase
    (over its token-protected socket). Only when none is running, wipe the files here."""
    import json

    import websockets

    from neo.config import settings

    s = settings()
    token_file = s.data_dir / "ws-token"
    token = token_file.read_text().strip() if token_file.exists() else ""
    try:
        async with websockets.connect(f"ws://{s.ws_host}:{s.ws_port}/?token={token}", open_timeout=3) as ws:
            await ws.send(json.dumps({"cmd": "forget_all"}))
            async with asyncio.timeout(60):
                async for raw in ws:
                    ev = json.loads(raw)
                    if ev.get("type") == "erased":
                        print("(NEO is running: it erased everything, including this session)")
                        return ev.get("data", {}).get("counts", {})
    except ConnectionRefusedError:
        pass  # NEO isn't running: do it here
    except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as e:
        # Something is listening but didn't erase: wiping the files underneath a running NEO
        # would only have it write the conversation back.
        raise SystemExit(
            f"NEO is running but didn't confirm the erase ({str(e)[:80] or 'no answer'}). "
            "Use the menu-bar icon → Erase NEO's Memory…, or quit NEO and run this again."
        ) from e
    from neo.privacy import forget_everything

    return await forget_everything()


def main() -> None:
    os.umask(0o077)  # everything NEO writes (memory, logs, learned examples) is private to this user
    ap = argparse.ArgumentParser(prog="neo")
    ap.add_argument("--text", action="store_true", help="text REPL, no voice/UI")
    ap.add_argument("--doctor", action="store_true", help="check setup")
    ap.add_argument(
        "--local", action="store_true", help="download + serve the offline model (Gemma 4 12B, ~6.7 GB)"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--forget-all", action="store_true", help="erase everything NEO has kept about you")
    args = ap.parse_args()
    if args.doctor:
        sys.exit(doctor())
    if args.forget_all:
        if input("Erase NEO's memories, playbooks, history, learned requests, logs and browser logins? [y/N] ").strip().lower() != "y":
            print("Nothing erased.")
            return
        counts = asyncio.run(_erase_everything())
        print("Erased: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
        return
    if args.local:
        from neo.providers.local_server import ensure_running

        ok = asyncio.run(ensure_running(wait_s=3600))
        print(
            "local model server ready on http://127.0.0.1:8080 — leave this running"
            if ok
            else "failed to start (see ~/.neo/local_server.log)"
        )
        if ok:
            import signal as _sig

            _sig.pause()
        return
    if args.text:
        asyncio.run(text_repl(args.verbose))
        return
    from neo.app import run

    asyncio.run(run())


if __name__ == "__main__":
    main()
