"""Brains & Keys in a terminal: ``python -m neo2 keys …`` and ``python -m neo2 brain …``.

    python -m neo2 keys list                       # each provider: where its key comes from, ••••last4
    python -m neo2 keys set <provider> [--extra] [--no-test]
    python -m neo2 keys remove <provider>
    python -m neo2 keys test <provider>
    python -m neo2 brain show
    python -m neo2 brain set [--brain REF] [--quick REF] [--eyes REF] [--light REF] [--backups REFS]
                             [--cloud-eyes auto|on|off] [--voice auto|live|local|off] [--budget PROVIDER=USD]

A key is read with getpass — no echo, never on the command line (where `ps` and the shell's history
would keep it) — and written to NEO's key store directly (~/.neo/api_keys.json: neo2.core.keystore,
under its lock, as the core writes it). Nothing printed here ever holds more of a key than its last 4
characters. After a change a running NEO is told to read its files again (`brains_reload`, as a
local tool over its websocket); with none running, the change waits for it to start.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from collections.abc import Sequence
from typing import Any

from neo2.core import brainref, brains, keystore
from neo2.core.providers import catalog

_WHERE = {
    "env": "from the environment",
    "saved": "saved in NEO",
    "checkout_env": "from this checkout's .env",
    "dotenv": "from ~/.neo/.env",
    "default": "the default",
    "none": "not set",
}
_REFUSING = ("bad_key", "no_access", "no_credit")


def _settings() -> Any:
    from neo2.core.config import settings

    return settings()


def _label(pid: str) -> str:
    p = catalog.provider(pid)
    if p is not None:
        return p.label
    cp = catalog.custom_endpoint(pid, _settings())
    return cp.label if cp is not None else pid


def _env_of(pid: str) -> str | None:
    p = catalog.provider(pid)
    if p is not None:
        return p.env
    cp = catalog.custom_endpoint(pid, _settings())
    return cp.env if cp is not None else None


def _store() -> keystore.FileKeyStore | None:
    from neo2.core import config

    p = config.KEY_STORE_PATH
    return None if p is None else keystore.FileKeyStore(p)


def _saved_config() -> keystore.SavedConfig | None:
    from neo2.core import config

    p = config.SAVED_PATH
    return None if p is None else keystore.SavedConfig(p)


def _provider(value: str) -> str | None:
    """A provider id as typed ("claude" is anthropic, "google" is gemini) that takes a key."""
    pid = brains.provider_id(value)
    if not pid:
        return None
    if pid.startswith(brainref.CUSTOM):
        return pid if catalog.custom_endpoint(pid, _settings()) is not None else None
    p = catalog.provider(pid)
    return pid if p is not None and p.key_field else None


def _known() -> str:
    s = _settings()
    ids = [pid for pid, p in catalog.PROVIDERS.items() if p.key_field]
    ids += [cp.id for cp in getattr(s, "custom_providers", None) or ()]
    return ", ".join(ids)


def _where(entry: dict[str, Any], pid: str) -> str:
    src = entry.get("source", "none")
    if src == "none":
        return "not set"
    words = _WHERE.get(src, src)
    if src == "env" and (env := _env_of(pid)):
        words = f"from the environment ({env})"
    end = entry.get("last4")
    shown = f" ••••{end}" if end else " ••••"
    extra = f" (+{entry['extra']} more)" if entry.get("extra") else ""
    also = [a for a in entry.get("also") or [] if a in _WHERE]
    behind = f"; also {', '.join(_WHERE[a] for a in also)}" if also else ""
    return f"{words}{shown}{extra}{behind}"


async def _tell_core() -> bool:
    """A running NEO reads its files again (brains_reload, as a local tool). False: none answered."""
    import websockets

    s = _settings()
    token_file = s.data_dir / "ws-token"
    try:
        token = token_file.read_text().strip()
    except OSError:
        return False
    try:
        async with websockets.connect(f"ws://{s.ws_host}:{s.ws_port}/?token={token}", open_timeout=3) as ws:
            await ws.send(json.dumps({"cmd": "brains_reload"}))
            async with asyncio.timeout(10):
                async for raw in ws:
                    ev = json.loads(raw)
                    if ev.get("type") == "brains_reload_result":
                        return bool((ev.get("data") or {}).get("ok"))
    except (OSError, TimeoutError, ValueError, websockets.exceptions.WebSocketException):
        return False
    return False


def _reload(out: Any) -> None:
    if asyncio.run(_tell_core()):
        print("NEO is running and uses it now.", file=out)
    else:
        print("NEO isn't running here: it uses it when it starts (or restart NEO).", file=out)


# ---- keys ------------------------------------------------------------------------------------------------


def keys_list(out: Any = None) -> int:
    out = out or sys.stdout
    s = _settings()
    srcs = keystore.sources(custom=list(getattr(s, "custom_providers", None) or ()))
    for pid, entry in srcs.items():
        mark = "✓" if entry.get("set") else "·"
        print(f"  {mark} {_label(pid)} ({pid}) — {_where(entry, pid)}", file=out)
    return 0


def _ask_key(pid: str) -> str | None:
    """The key, typed or pasted without an echo (never an argument). None: none was given."""
    try:
        raw = getpass.getpass(f"{_label(pid)} API key (it isn't shown as you type): ")
    except (EOFError, KeyboardInterrupt):
        return None
    return raw if raw.strip() else None


def keys_set(pid: str, *, extra: bool = False, test: bool = True, out: Any = None) -> int:
    from neo2.core import keyshape, logredact
    from neo2.core.providers import keycheck, spend

    out = out or sys.stdout
    raw = _ask_key(pid)
    if raw is None:
        print("No key given: nothing saved.", file=out)
        return 1
    try:
        key = keystore.clean_key(raw)
    except ValueError as e:
        print(f"That key can't be saved: {e}.", file=out)
        return 1
    logredact.remember(key)
    cp = catalog.custom_endpoint(pid, _settings()) if pid.startswith("custom:") else None
    if why := brains.wrong_provider(pid, key, cp):
        print(f"{why} Nothing saved.", file=out)
        return 1
    if hint := keyshape.hint(pid, key):
        print(hint, file=out)
    more: list[str] = []
    if extra and pid == "gemini":
        while True:
            try:
                x = getpass.getpass("Another Gemini key (Enter when done): ")
            except (EOFError, KeyboardInterrupt):
                break
            if not x.strip():
                break
            try:
                more.append(keystore.clean_key(x))
            except ValueError as e:
                print(f"That one can't be saved: {e}.", file=out)
        logredact.remember(*more)
    elif extra:
        print("--extra is for Gemini's extra keys only.", file=out)
        return 2
    if test:
        r = asyncio.run(keycheck.check(pid, key))
        print(r.message, file=out)
        if r.status in _REFUSING:
            print("Not saved. (--no-test saves it anyway.)", file=out)
            return 1
    store = _store()
    if store is None:
        print("NEO's key store isn't available here.", file=out)
        return 1
    try:
        store.put(pid, key, more)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Not saved: {type(e).__name__}.", file=out)
        return 1
    print(f"Saved the {_label(pid)} key {keystore.mask(key)} in NEO.", file=out)
    s = _settings()
    cap = brains.auto_cap(pid, s)  # (decision 4: a paid key saved here, and the one NEO uses)
    sc = _saved_config()
    if cap is not None and sc is not None:
        try:
            now = sc.update({}, add_to={"daily_budgets": {pid: cap}})
            if (now.get("daily_budgets") or {}).get(pid) == cap:
                print(
                    f"NEO stops using {catalog.provider_label(pid, s)} for the day at ${cap:.2f} "
                    f"(python -m neo2 brain set --budget {pid}=5 changes it).",
                    file=out,
                )
        except (OSError, RuntimeError, ValueError) as e:
            print(f"(The daily cap wasn't saved: {type(e).__name__}.)", file=out)
    elif spend.prefill(pid, "saved", on_device=bool(cp and cp.on_device)) is not None and pid not in (getattr(s, "daily_budgets", None) or {}):
        print(f"(No daily cap was set for it: python -m neo2 brain set --budget {pid}=2 sets one.)", file=out)
    env = _env_of(pid)
    if env and brains._in_environ(env):
        print(f"But {env} is set in your environment, and it wins over what's saved here.", file=out)
    _reload(out)
    return 0


def keys_remove(pid: str, out: Any = None) -> int:
    out = out or sys.stdout
    store = _store()
    if store is None:
        print("NEO's key store isn't available here.", file=out)
        return 1
    try:
        gone = store.remove(pid)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Not removed: {type(e).__name__}.", file=out)
        return 1
    print(f"Removed the {_label(pid)} key saved in NEO." if gone else f"No {_label(pid)} key was saved in NEO.", file=out)
    entry = keystore.sources(custom=list(getattr(_settings(), "custom_providers", None) or ())).get(pid, {})
    if entry.get("set") and entry.get("source") != "saved":
        print(f"A key is still set {_where(entry, pid)}.", file=out)
    if gone:
        _reload(out)
    return 0


def keys_test(pid: str, out: Any = None) -> int:
    from neo2.core.providers import keycheck

    out = out or sys.stdout
    r = asyncio.run(keycheck.check(pid))
    print(r.message, file=out)
    if r.credit:
        print(f"Credit: {json.dumps(r.credit)}", file=out)
    return 0 if r.ok else 1


# ---- brain ----------------------------------------------------------------------------------------------


def brain_show(out: Any = None) -> int:
    out = out or sys.stdout
    s = _settings()
    now = brains.saved()
    for f, title in (
        ("brain", "Brain"), ("fast_brain", "Quick replies"), ("eyes_brain", "Eyes"), ("light_brain", "Light actions"),
        ("offline_brain", "Offline"),
    ):  # fmt: skip
        ref = str(getattr(s, f, "") or "")
        src = _WHERE.get(brains.setting_source(f, saved_now=now), "")
        if not ref:
            what = {"eyes_brain": "the Brain, then the seeing fallbacks", "light_brain": "Quick replies"}.get(f, "none")
            print(f"  · {title}: {what} — {src}", file=out)
            continue
        r = brainref.parse(ref)
        if not r.valid:
            typed = brainref.shown(ref)
            print(f"  ✗ {title}: {typed + ' is' if typed else 'it is'} not a brain NEO knows — {src}", file=out)
            continue
        ok = brains.usable(ref, s)
        print(f"  {'✓' if ok else '✗'} {title}: {catalog.label_for(ref, s=s)} ({ref}) — {src}"
              f"{'' if ok else ' — not set up'}", file=out)  # fmt: skip
    b = getattr(s, "backup_brains", None)
    print(f"  · Backups: {'Automatic' if b is None else (', '.join(b) or 'none')}", file=out)
    print(f"  · Screenshots: {'only the brains picked' if s.cloud_eyes == 'off' else 'others may help'} ({s.cloud_eyes})", file=out)
    print(f"  · Voice: {brains.voice(s)} (NEO_VOICE={s.voice})", file=out)
    if s.daily_budgets:
        print("  · Daily caps: " + ", ".join(f"{k} ${v:.2f}" for k, v in sorted(s.daily_budgets.items())), file=out)
    print(f"  {'✓' if brains.ready(s) else '✗'} NEO {'has a model to ask' if brains.ready(s) else 'has no model to ask yet'}", file=out)
    return 0


def brain_set(args: argparse.Namespace, out: Any = None) -> int:
    out = out or sys.stdout
    changes: dict[str, Any] = {}
    for opt, f in (("brain", "brain"), ("quick", "fast_brain"), ("eyes", "eyes_brain"), ("light", "light_brain"),
                   ("offline", "offline_brain"), ("cloud_eyes", "cloud_eyes"), ("voice", "voice")):  # fmt: skip
        v = getattr(args, opt, None)
        if v is not None:
            changes[f] = v
    if args.backups is not None:
        changes["backup_brains"] = brainref.backups(args.backups)
    if args.budget:
        try:
            got = {}
            for item in args.budget:
                k, _, v = item.partition("=")
                got[brains.provider_id(k) or k.strip()] = float(v.strip().lstrip("$"))
        except ValueError:
            print("--budget takes PROVIDER=DOLLARS (anthropic=2).", file=out)
            return 2
        saved = dict(brains.saved().get("daily_budgets") or {})
        changes["daily_budgets"] = {**saved, **got}
    if not changes:
        print("Nothing to set: see python -m neo2 brain set --help.", file=out)
        return 2
    try:
        kept, warnings = asyncio.run(brains.check_roles(changes))
    except brains.Refused as r:
        words = {
            "invalid": "isn't a brain or a value NEO knows",
            "not_configured": "isn't set up (its key isn't set, or nothing answers on this Mac)",
            "no_tools": "can't call tools, so it can't be the Brain",
            "cant_see": "can't see pictures, so it can't be the Eyes",
        }.get(r.error, r.error)
        print(f"Not saved: {r.field or 'that'} {words}.", file=out)
        return 1
    sc = _saved_config()
    if sc is None:
        print("NEO's saved brains aren't available here.", file=out)
        return 1
    try:
        sc.update(kept)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Not saved: {type(e).__name__}.", file=out)
        return 1
    for w in warnings:
        print(f"Note: {w}.", file=out)
    for f in kept:
        if brains._in_environ(brains._env_name(f)):
            print(f"But {brains._env_name(f)} is set in your environment, and it wins over what's saved here.", file=out)
    print("Saved.", file=out)
    _reload(out)
    return 0


# ---- the command line --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neo2", description="NEO's Brains & Keys, in a terminal.")
    sub = ap.add_subparsers(dest="what", required=True)
    keys = sub.add_parser("keys", help="the API keys NEO uses").add_subparsers(dest="do", required=True)
    keys.add_parser("list", help="each provider: where its key comes from (••••last 4)")
    st = keys.add_parser("set", help="save a key in NEO (typed without an echo)")
    st.add_argument("provider")
    st.add_argument("--extra", action="store_true", help="Gemini: also ask for extra keys")
    st.add_argument("--no-test", action="store_true", help="save without asking the provider first")
    keys.add_parser("remove", help="forget the key saved in NEO").add_argument("provider")
    keys.add_parser("test", help="ask the provider whether the key works (no tokens spent)").add_argument("provider")
    brain = sub.add_parser("brain", help="which brain each role uses").add_subparsers(dest="do", required=True)
    brain.add_parser("show", help="the roles and where each comes from")
    bs = brain.add_parser("set", help="save roles in NEO (<brain>[@<model>]: claude@claude-opus-5-5)")
    bs.add_argument("--brain")
    bs.add_argument("--quick")
    bs.add_argument("--eyes", help='"" for none')
    bs.add_argument("--light", help='"" for Quick replies')
    bs.add_argument("--offline")
    bs.add_argument("--backups", help='"auto", "none" or a comma-separated list')
    bs.add_argument("--cloud-eyes", dest="cloud_eyes", choices=("auto", "on", "off"))
    bs.add_argument("--voice", choices=("auto", "live", "local", "off"))
    bs.add_argument("--budget", action="append", help="a daily cap, PROVIDER=DOLLARS (repeatable)")
    return ap


def main(argv: Sequence[str]) -> int:
    """`python -m neo2 keys …` / `python -m neo2 brain …` (argv without the program's name)."""
    args = _parser().parse_args(list(argv))
    if args.what == "brain":
        return brain_show() if args.do == "show" else brain_set(args)
    if args.do == "list":
        return keys_list()
    pid = _provider(args.provider)
    if pid is None:
        typed = brainref.shown(args.provider)  # (a key typed in its place is never printed back)
        print(f"{repr(typed) if typed else 'That'} isn't a provider NEO keeps a key for (one of: {_known()}).")
        return 2
    if args.do == "set":
        return keys_set(pid, extra=args.extra, test=not args.no_test)
    if args.do == "remove":
        return keys_remove(pid)
    return keys_test(pid)
