#!/usr/bin/env python3
"""The release engine's licences: what every package in it says about itself, gathered in one place.

    <engine>/bin/python3 -I scripts/engine_licenses.py --engine <engine> --texts scripts/licenses \\
        --pbs 20260924 [--combined <file>]

scripts/build_engine.sh runs it with the engine's own Python, before packing. It writes
<engine>/licenses/:

    README.md            every package: version, licence (as its metadata says it), its files here
    packages/<name>-<version>/   each package's own licence files (its dist-info's, or its package's)
    python/LICENSE.txt   CPython's licence (PSF-2.0 and what it incorporates)
    python/COMPONENTS.md the libraries python-build-standalone builds into Python, at their versions
    SOURCES.md           the copyleft packages (GPL, LGPL, MPL…) and where their exact sources are
    <SPDX>.txt           the full texts NEO's notices name (GPL-3.0, LGPL-2.1, Apache-2.0, 0BSD, MPL-2.0)
    licenses.json        the same, for scripts (stage_resources.sh, the tests)

--combined also writes one text file of all of it (each distinct licence text once): NEO.app's
Licenses/engine-packages.txt. Standard library only, and it never reads outside the engine and --texts.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

LICENSE_NAMES = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|AUTHORS|COPYRIGHT|THIRD.PARTY|PATENTS)", re.I)
COPYLEFT = re.compile(r"\b(A?GPL|LGPL|MPL|EPL|CDDL|EUPL|General Public|Mozilla Public|Lesser General)", re.I)

# Where the sources of what the engine redistributes under the GPL are, exactly (the user's decision:
# point to the upstream sources, don't bundle them). Package name -> lines for SOURCES.md.
GPL_SOURCES = {
    "espeakng-loader": [
        "Bundles libespeak-ng {espeak} and espeak-ng-data (eSpeak NG, GPL-3.0-or-later), built from",
        "eSpeak NG's own source by the package's build. The wheel's metadata names no licence.",
        "- eSpeak NG {espeak}: https://github.com/espeak-ng/espeak-ng/releases/tag/{espeak}",
        "  (source: https://github.com/espeak-ng/espeak-ng/archive/refs/tags/{espeak}.tar.gz)",
        "- espeakng-loader {version} (the build that made the library and the loader):",
        "  https://github.com/thewh1teagle/espeakng-loader and https://pypi.org/project/espeakng-loader/{version}/",
    ],
    "phonemizer-fork": [
        "GPL-3.0-or-later. Its source distribution, the exact source of the installed wheel:",
        "- https://pypi.org/project/phonemizer-fork/{version}/#files (phonemizer_fork-{version}.tar.gz)",
        "- upstream project: https://github.com/bootphon/phonemizer",
    ],
}


# Packages whose metadata names no licence, with the one their project states.
KNOWN = {"vosk": "Apache-2.0"}
# How licences are written in metadata, as SPDX ids (for the standard texts in scripts/licenses).
WORDS = (
    (re.compile(r"\bGPL-3\.0|GPLv3|General Public License v3|GENERAL PUBLIC LICENSE", re.I), "GPL-3.0"),
    (re.compile(r"\bLGPL|Lesser General", re.I), "LGPL-2.1"),
    (re.compile(r"\bMIT\b", re.I), "MIT"),
    (re.compile(r"\bApache", re.I), "Apache-2.0"),
    (re.compile(r"\bBSD-3-Clause|\bBSD 3-Clause|\bBSD License\b|^BSD$", re.I), "BSD-3-Clause"),
    (re.compile(r"\bMPL-2\.0|Mozilla Public License 2", re.I), "MPL-2.0"),
    (re.compile(r"\b0BSD\b", re.I), "0BSD"),
)


def standard(lic: str) -> list[str]:
    """The SPDX ids a package's licence words name, most specific first, each once."""
    out: list[str] = []
    for rx, spdx in WORDS:
        if rx.search(lic) and spdx not in out and not (spdx == "GPL-3.0" and "LGPL-2.1" in out):
            out.append(spdx)
    return out


def metadata(dist: Path) -> dict[str, object]:
    msg = email.parser.Parser().parsestr((dist / "METADATA").read_text(encoding="utf-8", errors="replace"))
    lic = (msg.get("License") or "").strip()
    first = lic.splitlines()[0].strip() if lic else ""
    classifiers = [c.split("::")[-1].strip() for c in msg.get_all("Classifier") or [] if c.startswith("License ::")]
    urls = [u.split(",", 1)[-1].strip() for u in msg.get_all("Project-URL") or []]
    if msg.get("Home-page"):
        urls.insert(0, msg["Home-page"].strip())
    return {
        "name": (msg.get("Name") or dist.name.split("-")[0]).strip(),
        "version": (msg.get("Version") or "").strip(),
        "expression": (msg.get("License-Expression") or "").strip(),
        "license": first[:200],
        "classifiers": classifiers,
        "license_files": [f.strip() for f in msg.get_all("License-File") or []],
        "urls": list(dict.fromkeys(u for u in urls if u.startswith("http"))),
    }


def licence_files(dist: Path, site: Path, meta: dict[str, object]) -> list[Path]:
    """The package's own licence files: its dist-info's (PEP 639's licenses/ too), else its package's."""
    found: list[Path] = []
    lic_dir = dist / "licenses"
    if lic_dir.is_dir():
        found += sorted(p for p in lic_dir.rglob("*") if p.is_file())
    found += sorted(p for p in dist.iterdir() if p.is_file() and LICENSE_NAMES.match(p.name))
    for name in meta["license_files"]:  # type: ignore[union-attr]
        for p in (dist / name, lic_dir / name):
            if p.is_file() and p not in found:
                found.append(p)
    if not found:
        top = dist / "top_level.txt"
        names = top.read_text().split() if top.is_file() else [dist.name.split("-")[0]]
        for n in names:
            pkg = site / n
            if pkg.is_dir():
                found += sorted(p for p in pkg.iterdir() if p.is_file() and LICENSE_NAMES.match(p.name))
    return found


def said(meta: dict[str, object]) -> str:
    """The licence in the package's own words: its SPDX expression, else its License field or classifiers."""
    if meta["expression"]:
        return str(meta["expression"])
    lic = str(meta["license"])
    if lic and len(lic) <= 80 and not lic.lower().startswith(("copyright", "the ", "permission")):
        return lic
    if meta["classifiers"]:
        return " / ".join(meta["classifiers"])  # type: ignore[arg-type]
    return lic[:80] if lic else "not stated in its metadata"


def components() -> list[tuple[str, str, str]]:
    """The libraries inside python-build-standalone's Python, at the versions this Python reports."""
    out: list[tuple[str, str, str]] = []

    def version(get):  # noqa: ANN001
        try:
            return str(get())
        except Exception:  # (a module this build doesn't have)
            return "as built by python-build-standalone"

    import ssl  # noqa: PLC0415

    out.append(("OpenSSL", ssl.OPENSSL_VERSION, "Apache-2.0 (Apache-2.0.txt)"))
    out.append(("SQLite", version(lambda: __import__("sqlite3").sqlite_version), "public domain (blessing)"))
    out.append(("zlib", version(lambda: __import__("zlib").ZLIB_RUNTIME_VERSION), "Zlib"))
    out.append(("XZ Utils (liblzma)", version(lambda: __import__("lzma") and "as built"), "0BSD (0BSD.txt)"))
    out.append(("bzip2", version(lambda: __import__("bz2") and "as built"), "bzip2-1.0.6"))
    out.append(("libffi", version(lambda: __import__("ctypes") and "as built"), "MIT"))
    out.append(("Expat", version(lambda: __import__("pyexpat").EXPAT_VERSION), "MIT"))
    out.append(("mpdecimal", version(lambda: __import__("decimal").__libmpdec_version__), "BSD-2-Clause"))
    out.append(("ncurses", version(lambda: ".".join(map(str, __import__("curses").ncurses_version))), "X11 (MIT-style)"))
    out.append(("libedit (readline)", version(lambda: __import__("readline")._READLINE_LIBRARY_VERSION), "BSD-3-Clause"))
    out.append(("Tcl/Tk", version(lambda: f"{__import__('tkinter').TclVersion}"), "TCL"))
    out.append(("libuuid (util-linux)", "as built", "BSD-3-Clause"))
    out.append(("HACL* (hashes, in CPython)", "as in CPython", "MIT (in python/LICENSE.txt)"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--texts", type=Path, required=True, help="the full licence texts (scripts/licenses)")
    ap.add_argument("--pbs", default="", help="python-build-standalone's release (engine.env PBS_RELEASE)")
    ap.add_argument("--combined", type=Path, help="also write everything as one text file here")
    args = ap.parse_args(argv)
    engine = args.engine.resolve()
    libs = sorted(engine.glob("lib/python3.*"))
    if not libs or not (libs[0] / "site-packages").is_dir():
        print(f"engine_licenses: no lib/python3.*/site-packages in {engine}", file=sys.stderr)
        return 1
    lib = libs[0]
    site = lib / "site-packages"
    out = engine / "licenses"
    shutil.rmtree(out, ignore_errors=True)
    (out / "packages").mkdir(parents=True)
    (out / "python").mkdir()

    texts = {p.stem: p for p in sorted(args.texts.glob("*.txt"))}
    for p in texts.values():
        shutil.copy2(p, out / p.name)

    py_license = lib / "LICENSE.txt"
    if py_license.is_file():
        shutil.copy2(py_license, out / "python" / "LICENSE.txt")
    pyver = ".".join(map(str, sys.version_info[:3]))
    comps = components()
    with open(out / "python" / "COMPONENTS.md", "w", encoding="utf-8") as f:
        f.write(f"# Python {pyver} (python-build-standalone {args.pbs or '?'})\n\n")
        f.write("CPython is under the PSF License Agreement (PSF-2.0): `LICENSE.txt` beside this file, which also\n")
        f.write("covers the code CPython incorporates. python-build-standalone (https://github.com/astral-sh/\n")
        f.write("python-build-standalone, MPL-2.0 for its build scripts) builds these libraries into it:\n\n")
        f.write("| Library | Version | Licence |\n|---|---|---|\n")
        for name, ver, lic in comps:
            f.write(f"| {name} | {ver} | {lic} |\n")
        f.write("\nThe libraries' own licence texts are in python-build-standalone's \"full\" archive of the same\n")
        f.write(f"release (python/licenses/), https://github.com/astral-sh/python-build-standalone/releases/tag/{args.pbs}.\n")
        f.write("No GPL library is built in: python-build-standalone uses libedit, not GNU readline, and leaves out gdbm.\n")

    entries: list[dict[str, object]] = []
    blobs: dict[str, tuple[str, str]] = {}  # sha -> (first owner, text)
    for dist in sorted(site.glob("*.dist-info"), key=lambda d: d.name.lower()):
        if not (dist / "METADATA").is_file():
            continue
        meta = metadata(dist)
        name, version = str(meta["name"]), str(meta["version"])
        files = licence_files(dist, site, meta)
        dest = out / "packages" / f"{name}-{version}"
        copied: list[str] = []
        shas: list[str] = []
        for p in files:
            rel = p.relative_to(dist) if p.is_relative_to(dist) else Path(p.name)
            rel = Path(*rel.parts[1:]) if rel.parts[:1] == ("licenses",) and len(rel.parts) > 1 else rel
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            copied.append(str(target.relative_to(out)))
            data = p.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            shas.append(sha)
            blobs.setdefault(sha, (f"{name} {version}: {rel}", data.decode("utf-8", "replace")))
        lic = said(meta)
        key = name.lower().replace("_", "-")
        copyleft = bool(COPYLEFT.search(lic)) or key in GPL_SOURCES
        if key in KNOWN and lic in ("UNKNOWN", "not stated in its metadata", ""):
            lic = KNOWN[key]
        if key == "espeakng-loader":
            lic = "GPL-3.0-or-later (eSpeak NG)"
        if not copied:
            # a package that names its licence but ships no text: the standard text (scripts/licenses)
            for spdx in standard(lic):
                if spdx in texts:
                    copied.append(f"{spdx}.txt")
                    data = texts[spdx].read_bytes()
                    shas.append(hashlib.sha256(data).hexdigest())
                    blobs.setdefault(shas[-1], (f"{spdx} (the standard text)", data.decode("utf-8", "replace")))
        entries.append({"name": name, "version": version, "license": lic, "files": copied, "texts": shas,
                        "urls": meta["urls"], "copyleft": copyleft})  # fmt: skip

    espeak = "1.52.0"
    for p in site.glob("espeakng_loader/libespeak-ng.*.*.*.dylib"):
        espeak = p.name.removeprefix("libespeak-ng.").removesuffix(".dylib")
    with open(out / "SOURCES.md", "w", encoding="utf-8") as f:
        f.write("# Where the copyleft parts' sources are\n\n")
        f.write("NEO's engine redistributes these packages unmodified, as their authors published them on PyPI.\n")
        f.write("Their exact sources are at the addresses below; NEO's own source is https://github.com/ArnavParashar49/Neo2.0.\n\n")
        for e in entries:
            if not e["copyleft"]:
                continue
            key = str(e["name"]).lower().replace("_", "-")
            f.write(f"## {e['name']} {e['version']}: {e['license']}\n\n")
            for line in GPL_SOURCES.get(key, []):
                f.write(line.format(version=e["version"], espeak=espeak) + "\n")
            if key not in GPL_SOURCES:
                f.write(f"- https://pypi.org/project/{e['name']}/{e['version']}/#files\n")
                for u in e["urls"][:2]:  # type: ignore[index]
                    f.write(f"- {u}\n")
            f.write("\n")

    with open(out / "README.md", "w", encoding="utf-8") as f:
        f.write(f"# What NEO's engine contains, and under which licences\n\nPython {pyver}: python/ (PSF-2.0).\n")
        f.write("Copyleft packages and their sources: SOURCES.md. Each package's own licence files: packages/.\n\n")
        f.write("| Package | Version | Licence | Files |\n|---|---|---|---|\n")
        for e in entries:
            files = ", ".join(e["files"]) or "none shipped (see the package's project page)"  # type: ignore[arg-type]
            f.write(f"| {e['name']} | {e['version']} | {e['license']} | {files} |\n")
    (out / "licenses.json").write_text(json.dumps({"python": pyver, "pbs": args.pbs, "packages": entries}, indent=1) + "\n")

    if args.combined:
        with open(args.combined, "w", encoding="utf-8") as f:
            f.write("NEO's engine: the software in it and its licences\n")
            f.write("=" * 52 + "\n\n")
            f.write(f"Python {pyver}, python-build-standalone {args.pbs}: PSF-2.0 (text [PY] below).\n")
            for name, ver, lic in comps:
                f.write(f"  {name} {ver}: {lic}\n")
            f.write("\nPackages (each one's licence text is listed by its number below):\n\n")
            ids = {sha: i + 1 for i, sha in enumerate(blobs)}
            for e in entries:
                refs = ", ".join(f"[{ids[s]}]" for s in e["texts"]) or "no licence file shipped"  # type: ignore[union-attr]
                f.write(f"{e['name']} {e['version']}: {e['license']} — {refs}\n")
            f.write("\nThe copyleft packages' sources: see the engine's licenses/SOURCES.md, also in THIRD_PARTY_NOTICES.md.\n")
            if py_license.is_file():
                f.write("\n" + "-" * 72 + f"\n[PY] Python {pyver}: LICENSE.txt\n" + "-" * 72 + "\n")
                f.write(py_license.read_text(encoding="utf-8", errors="replace"))
            for sha, (owner, text) in blobs.items():
                f.write("\n" + "-" * 72 + f"\n[{ids[sha]}] {owner}\n" + "-" * 72 + "\n" + text.rstrip() + "\n")
    print(f"engine_licenses: {len(entries)} packages, {len(blobs)} distinct licence texts, "
          f"{sum(1 for e in entries if e['copyleft'])} copyleft, "
          f"{sum(1 for e in entries if not e['files'])} without a licence file")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
