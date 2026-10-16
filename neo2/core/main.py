"""NEO entry point.

python -m neo2                   # full assistant: voice + overlay server
python -m neo2 --text            # text REPL in the terminal (no voice, no UI)
python -m neo2 --doctor [--test] # check keys and brains (--test: ask each provider), permissions, models
python -m neo2 keys …            # the API keys NEO uses (neo2.core.cli_keys): list | set | remove | test
python -m neo2 brain …           # which brain each role uses: show | set

The core's stdout and stderr (NEO.app's core.log) go through neo2.core.logredact from the start: a
key NEO has loaded never reaches the log whole.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from neo2.core.config import settings
from neo2.core.events import bus


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
    from neo2.core.agent.session import Session
    from neo2.core.reflex import warmup
    from neo2.core.tools import load_all

    load_all()
    sess = Session()  # (each request works out its own memory context: neo2.core.agent.session)
    warmup(sess.reflex)
    if verbose:
        bus().subscribe(_print_event)
    print("NEO ready. Type a request (Ctrl-D to quit).")
    import time

    from neo2.core.memory import journal, summarize

    started = time.time()
    if journal.enabled():  # conversations an earlier run left without a summary get one now
        asyncio.create_task(summarize.catch_up())
    loop = asyncio.get_event_loop()
    while True:
        try:
            text = await loop.run_in_executor(None, lambda: input("\nyou> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            continue
        reply = await sess.handle(text, from_user=True)
        print(f"neo> {reply.text}")
    from neo2.core.tools import render
    from neo2.core.tools.browser import shutdown as close_browser

    await close_browser()
    await render.shutdown()
    if journal.enabled():  # the conversation is in the journal: its summary is written from there
        await asyncio.to_thread(journal.flush, 2.0)
        if n := await summarize.summarize_due(idle_s=0.0):
            print(f"[memory] {n} conversation(s) summarised", file=sys.stderr)
        await asyncio.to_thread(journal.close, 1.0)
    elif summary := await summarize.summarize_session(sess.history, started):
        print(f"[memory] session saved: {summary[:120]}", file=sys.stderr)


def doctor(test: bool = False) -> int:
    """Keys (each provider: where its key comes from, its last 4), the roles and whether each can
    answer, what runs on this Mac, the permissions and the local models. `test`: ask each provider
    whose key is set whether it works (no tokens spent). Exit 1 when the Brain can't answer, NEO has
    no model at all, or something it needs is missing — never for an optional provider left unset."""
    s = settings()
    ok = True

    def row(label: str, good: bool, note: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {'✓' if good else '✗'} {label}{(' — ' + note) if note else ''}")

    def info(label: str, good: bool, note: str = "") -> None:  # (shown, never a reason to fail)
        print(f"  {'✓' if good else '·'} {label}{(' — ' + note) if note else ''}")

    from neo2.core import brainref, brains, health, keystore
    from neo2.core.cli_keys import _where
    from neo2.core.providers import catalog

    customs = list(getattr(s, "custom_providers", None) or ())
    try:
        asyncio.run(brains.refresh_probes(s, force=True))  # (what answers on this Mac: loopback only)
    except Exception as e:  # noqa: BLE001
        print(f"  · couldn't look for models on this Mac ({type(e).__name__})")
    print("Keys (Brains & Keys: the window, or python -m neo2 keys set <provider>)")
    srcs = keystore.sources(custom=customs)
    for pid, e in srcs.items():
        p = catalog.provider(pid)
        cp = catalog.custom_endpoint(pid, s) if pid.startswith(brainref.CUSTOM) else None
        label = p.label if p is not None else (cp.label if cp is not None else pid)
        if pid == "cerebras" and e.get("set"):
            from neo2.core.providers import cerebras

            st = cerebras.status()
            row(f"{label} (paid)", not st.get("refused"), st.get("refused") or (
                f"{_where(e, pid)}; ${st.get('today_usd', 0):.4f} of ${st.get('budget_usd', 0):.2f} spent today"))  # fmt: skip
        elif e.get("set"):
            info(label, True, _where(e, pid))
        elif cp is not None and cp.on_device:
            info(label, True, "on this Mac (no key needed)")
        else:
            info(label, False, "not set" + (f" ({p.key_url})" if p is not None and p.key_url else ""))
    for problem in health._store_problems():
        row(problem, False)
    print("Brains")
    now = brains.saved()
    for title, f, role in (("Brain", "brain", "brain"), ("Quick replies", "fast_brain", "quick"), ("Eyes", "eyes_brain", "eyes")):
        ref = str(getattr(s, f, "") or "")
        src = {"env": "environment", "saved": "saved in NEO", "dotenv": "~/.neo/.env", "checkout_env": "checkout .env",
               "default": "default"}.get(brains.setting_source(f, saved_now=now), "")  # fmt: skip
        if not ref:
            info(f"{title}: the Brain, then the seeing fallbacks", True, src)
            continue
        r = brainref.parse(ref)
        if not r.valid:
            typed = brainref.shown(ref)
            row(f"{title}: {typed + ' is' if typed else 'it is'} not a brain NEO knows", False, src)
            continue
        got = brains._role(ref, role, s)
        note = f"{r.text}, {src}" + ("" if got["ok"] else " — not set up")
        (row if role == "brain" else info)(f"{title}: {got['label']}", got["ok"], note)
    ready = brains.ready(s)
    row("NEO has a model to ask", ready, "" if ready else "connect one in Brains & Keys")
    if ready:
        for line in health.model_problems(s):  # (its brain can't see, and nothing else that sees is set up)
            info(line, False)
    here = sorted({u for n in brains._named(s) if (u := brains._probe_url(n, s))})
    if here:
        print("On this Mac")
        for u in here:
            note = "answering" if brains.reachable(u) else (
                "not running (python -m neo2 --local)" if u == brains.LOCAL_MODELS_URL else "not answering")  # fmt: skip
            info(u.rsplit("/models", 1)[0], brains.reachable(u), note)
    if test:
        from neo2.core.providers import keycheck

        print("Key checks (no tokens spent)")
        for pid, e in srcs.items():
            cp = catalog.custom_endpoint(pid, s) if pid.startswith(brainref.CUSTOM) else None
            if not e.get("set") and not (cp is not None and cp.on_device):
                continue
            r = asyncio.run(keycheck.check(pid))
            row(catalog.provider_label(pid, s), r.ok, r.message)
    print("macOS permissions (grant to the app that runs NEO: Terminal / the NEO app)")
    if sys.platform == "darwin":
        from neo2.core.tools.computer import ax, screen

        row("Accessibility", ax.is_trusted(), "System Settings → Privacy & Security → Accessibility")
        from neo2.core.tools.computer.input import can_post

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

    from neo2.core.config import settings

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
    from neo2.core.privacy import forget_everything

    return await forget_everything()


def main() -> None:
    os.umask(0o077)  # everything NEO writes (memory, logs, learned examples) is private to this user
    from neo2.core import log_to_stdout, logredact

    logredact.install()  # first: no key NEO loads ever reaches core.log whole
    log_to_stdout()  # NEO's status lines, as NEO.app's core.log and the terminal have always shown them
    if len(sys.argv) > 1 and sys.argv[1] in ("keys", "brain"):
        from neo2.core.cli_keys import main as keys_main

        sys.exit(keys_main(sys.argv[1:]))
    ap = argparse.ArgumentParser(
        prog="neo2",
        epilog="Brains & Keys in a terminal: neo2 keys list | set <provider> | remove <provider> | test <provider>, "
        "and neo2 brain show | set --brain REF --quick REF --eyes REF.",
    )
    ap.add_argument("--text", action="store_true", help="text REPL, no voice/UI")
    ap.add_argument("--doctor", action="store_true", help="check setup")
    ap.add_argument("--test", action="store_true", help="with --doctor: ask each provider whether its key works")
    ap.add_argument(
        "--local", action="store_true", help="download + serve the offline model (Gemma 4 12B, ~6.7 GB)"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--forget-all", action="store_true", help="erase everything NEO has kept about you")
    args = ap.parse_args()
    if args.doctor:
        sys.exit(doctor(test=args.test))
    if args.forget_all:
        if input("Erase NEO's memories, playbooks, history, learned requests, logs and browser logins? [y/N] ").strip().lower() != "y":
            print("Nothing erased.")
            return
        counts = asyncio.run(_erase_everything())
        print("Erased: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
        return
    if args.local:
        from neo2.core.providers.local_server import ensure_running

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
    from neo2.core.app import run

    asyncio.run(run())


if __name__ == "__main__":
    main()
