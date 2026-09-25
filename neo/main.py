"""NEO entry point.

python -m neo            # full assistant: voice + overlay server
python -m neo --text     # text REPL in the terminal (no voice, no UI)
python -m neo --doctor   # check keys, permissions, models
"""

from __future__ import annotations

import argparse
import asyncio
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
        reply = await sess.handle(text)
        print(f"neo> {reply.text}")


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


def main() -> None:
    ap = argparse.ArgumentParser(prog="neo")
    ap.add_argument("--text", action="store_true", help="text REPL, no voice/UI")
    ap.add_argument("--doctor", action="store_true", help="check setup")
    ap.add_argument(
        "--local", action="store_true", help="download + serve the offline model (Gemma 4 12B, ~6.7 GB)"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.doctor:
        sys.exit(doctor())
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
