#!/usr/bin/env python3
"""The licences of what is compiled into NEO.app itself: its Rust crates and its page's npm packages.

    python3 scripts/app_licenses.py --out ui/src-tauri/resources/Licenses [--strict]

Writes two files into --out, each listing every package (name, version, licence, where it comes
from) and then each distinct licence text once, as the packages ship them:

    rust-crates.txt    the crates cargo builds into NEO.app for macOS on Apple silicon (`cargo metadata
                       --offline --filter-platform aarch64-apple-darwin`; build and runtime
                       dependencies, not dev-dependencies), texts from cargo's registry
    npm-packages.txt   the page's runtime packages (ui/package-lock.json without its dev packages),
                       texts from ui/node_modules

scripts/stage_resources.sh runs it for every bundle (NEO.app's Contents/Resources/Licenses). Offline:
cargo reads its own registry cache and nothing is downloaded. --strict: fail if a list can't be made
or a package has neither a licence file nor a licence the texts in scripts/licenses cover (a release);
otherwise say so and carry on (a developer's build).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT|UNLICENSE|AUTHORS|PATENTS)", re.I)
SPDX = re.compile(r"[A-Za-z0-9.+-]+")
TARGET = "aarch64-apple-darwin"


def files_in(folder: Path, extra: list[str] = ()) -> list[Path]:  # type: ignore[assignment]
    found = sorted(p for p in folder.iterdir() if p.is_file() and NAMES.match(p.name)) if folder.is_dir() else []
    for e in extra:
        p = folder / e
        if p.is_file() and p not in found:
            found.append(p)
    return found


def standard_texts(expression: str, texts: dict[str, Path]) -> list[Path]:
    """The full texts of the licences an SPDX expression names, from scripts/licenses."""
    return [texts[i] for i in SPDX.findall(expression or "") if i in texts]


def cargo() -> str | None:
    for c in [shutil.which("cargo"), *glob.glob(os.path.expanduser("~/.cargo/bin/cargo")),
              *sorted(glob.glob(os.path.expanduser("~/.rustup/toolchains/stable-*/bin/cargo")))]:  # fmt: skip
        if c and os.access(c, os.X_OK):
            return c
    return None


def rust_crates(manifest: Path) -> list[dict[str, object]]:
    c = cargo()
    if c is None:
        raise RuntimeError("cargo isn't installed")
    env = dict(os.environ, PATH=f"{Path(c).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    out = subprocess.run(
        [c, "metadata", "--offline", "--locked", "--format-version", "1", "--filter-platform", TARGET,
         "--manifest-path", str(manifest)],  # fmt: skip
        capture_output=True, text=True, env=env, timeout=300,
    )  # fmt: skip
    if out.returncode != 0:
        raise RuntimeError("cargo metadata: " + out.stderr.strip()[-400:])
    meta = json.loads(out.stdout)
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen: set[str] = set()
    todo = [root]
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for d in nodes[pid]["deps"]:
            if any(k["kind"] in (None, "build") for k in d["dep_kinds"]):
                todo.append(d["pkg"])
    seen.discard(root)
    crates = []
    for pid in seen:
        p = packages[pid]
        folder = Path(p["manifest_path"]).parent
        extra = [p["license_file"]] if p.get("license_file") else []
        crates.append({
            "name": p["name"], "version": p["version"], "license": p.get("license") or "",
            "source": p.get("repository") or f"https://crates.io/crates/{p['name']}/{p['version']}",
            "files": files_in(folder, extra),
        })  # fmt: skip
    return sorted(crates, key=lambda c: (str(c["name"]).lower(), str(c["version"])))


def npm_packages(ui: Path) -> list[dict[str, object]]:
    lock = json.loads((ui / "package-lock.json").read_text())
    out = []
    for key, info in lock.get("packages", {}).items():
        if not key or info.get("dev") or info.get("devOptional"):
            continue
        folder = ui / key
        pj = folder / "package.json"
        if not pj.is_file():  # an optional package for another platform: not installed, not bundled
            continue
        meta = json.loads(pj.read_text())
        repo = meta.get("repository")
        repo = repo.get("url") if isinstance(repo, dict) else repo
        lic = meta.get("license") or info.get("license") or ""
        if isinstance(lic, dict):
            lic = lic.get("type", "")
        out.append({
            "name": meta.get("name") or key.rsplit("node_modules/", 1)[-1], "version": meta.get("version") or info.get("version", ""),
            "license": lic, "source": repo or f"https://www.npmjs.com/package/{meta.get('name')}", "files": files_in(folder),
        })  # fmt: skip
    return sorted(out, key=lambda p: (str(p["name"]).lower(), str(p["version"])))


def write(path: Path, title: str, intro: str, items: list[dict[str, object]], texts: dict[str, Path]) -> list[str]:
    """One file: the list, then each distinct text once. Returns the packages with no text at all."""
    blobs: dict[str, tuple[str, str]] = {}
    lines = [title, "=" * len(title), "", intro, ""]
    missing = []
    for it in items:
        files: list[Path] = list(it["files"])  # type: ignore[arg-type]
        note = ""
        if not files:
            files = standard_texts(str(it["license"]), texts)
            note = " (no licence file in the package: the standard text)" if files else ""
        refs = []
        for f in files:
            data = f.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            owner = f"{it['name']} {it['version']}: {f.name}" if f.parent.name != "licenses" else f"{f.stem} (standard text)"
            blobs.setdefault(sha, (owner, data.decode("utf-8", "replace")))
            refs.append(sha)
        if not refs:
            missing.append(f"{it['name']} {it['version']} ({it['license'] or 'no licence stated'})")
        ids = {s: i + 1 for i, s in enumerate(blobs)}
        lines.append(f"{it['name']} {it['version']}  —  {it['license'] or 'licence not stated'}{note}")
        lines.append(f"    {it['source']}")
        lines.append("    texts: " + (", ".join(f"[{ids[s]}]" for s in refs) or "none shipped"))
    ids = {s: i + 1 for i, s in enumerate(blobs)}
    for sha, (owner, text) in blobs.items():
        lines += ["", "-" * 72, f"[{ids[sha]}] {owner}", "-" * 72, text.rstrip()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--strict", action="store_true", help="fail on anything missing (a release)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    texts = {p.stem: p for p in sorted((root / "scripts" / "licenses").glob("*.txt"))}
    problems: list[str] = []
    try:
        crates = rust_crates(root / "ui" / "src-tauri" / "Cargo.toml")
        miss = write(args.out / "rust-crates.txt", "The Rust crates compiled into NEO.app",
                     f"NEO.app's binary is built from NEO's own code (MIT) and these {len(crates)} crates, for macOS on Apple\n"
                     "silicon. Each is listed with its licence (an SPDX expression: where it offers a choice, NEO uses\n"
                     "it under any one of them) and its source; the texts follow. The MPL-2.0 crates are unmodified;\n"
                     "their source is at crates.io under the same name and version.", crates, texts)  # fmt: skip
        problems += [f"crate {m}" for m in miss]
        print(f"app_licenses: {len(crates)} Rust crates", file=sys.stderr)
    except (RuntimeError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        problems.append(f"the Rust list: {e}")
    try:
        pkgs = npm_packages(root / "ui")
        miss = write(args.out / "npm-packages.txt", "The npm packages in NEO.app's page",
                     f"NEO's page (its overlay, chat, Brains & Keys and Setup windows) is bundled with these {len(pkgs)}\n"
                     "packages. Each is listed with its licence and its source; the texts follow.", pkgs, texts)  # fmt: skip
        problems += [f"npm {m}" for m in miss]
        print(f"app_licenses: {len(pkgs)} npm packages", file=sys.stderr)
    except (OSError, json.JSONDecodeError) as e:
        problems.append(f"the npm list: {e}")
    for p in problems:
        print(f"app_licenses: {'' if args.strict else 'warning: '}{p}", file=sys.stderr)
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
