#!/usr/bin/env python3
"""The release engine's id: what an engine is made of, as 16 hex characters.

    scripts/engine_id.py              the id
    scripts/engine_id.py --explain    the id, and each input with its size and SHA-256
    scripts/engine_id.py --json       {"engine_id": …, "inputs": {…}}

The engine is Python plus every package NEO.app's core runs (scripts/build_engine.sh). Its id is the
first 16 hex characters of the SHA-256 of its inputs, each framed by its name and size:

    "neo-engine-id 1\\n"  then, for each input in this order:  "<path>\\n<size in bytes>\\n<contents>"

with the inputs scripts/engine.env (the interpreter, its hash, the macOS target, the recipe),
requirements.release.txt (every package, by hash) and scripts/engine.prune (what's removed).
build_engine.sh computes the same in bash (its --print-id; the tests hold the two together).
NEO.app keeps engines in engines/<id> and checks an id against ^[0-9a-f]{16}$, so a release whose
inputs didn't change reuses the engine, and its users download nothing but the app.
Standard library only: CI runs it with any python3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("scripts/engine.env", "requirements.release.txt", "scripts/engine.prune")
SCHEME = b"neo-engine-id 1\n"


def engine_id(root: Path = ROOT) -> str:
    h = hashlib.sha256(SCHEME)
    for name in INPUTS:
        data = (root / name).read_bytes()
        h.update(f"{name}\n{len(data)}\n".encode())
        h.update(data)
    return h.hexdigest()[:16]


def inputs(root: Path = ROOT) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name in INPUTS:
        data = (root / name).read_bytes()
        out[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)  # the tests' copy
    ap.add_argument("--explain", action="store_true", help="also list each input's size and SHA-256")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    missing = [n for n in INPUTS if not (root / n).is_file()]
    if missing:
        print("missing: " + ", ".join(missing), file=sys.stderr)
        return 1
    eid = engine_id(root)
    if args.json:
        print(json.dumps({"engine_id": eid, "inputs": inputs(root)}, indent=2))
    elif args.explain:
        print(eid)
        for name, i in inputs(root).items():
            print(f"  {i['sha256']}  {i['bytes']:>8} bytes  {name}")
    else:
        print(eid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
