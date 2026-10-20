#!/usr/bin/env bash
# NEO's release engine: Python and every package NEO.app's core runs, as one archive NEO.app downloads
# and unpacks into ~/Library/Application Support/NEO/engines/<id> (ui/src-tauri/src/setup.rs).
#
#   scripts/build_engine.sh [--out DIR] [--wheel neo2-X.Y.Z-py3-none-any.whl]
#   scripts/build_engine.sh --print-id        the engine id (the same as scripts/engine_id.py)
#
# It writes, in --out (default dist/engines):
#   neo-engine-<id>-macos14-arm64.tar.xz       the archive: entries at its root (bin/python3, lib/…)
#   neo-engine-<id>.json                       its facts: id, asset, sha256, bytes, unpacked_bytes,
#                                              files, python, min_macos (stage_resources.sh reads it)
#   neo-engine-<id>-licenses.txt               every licence in it, as one file (NEO.app's Licenses/)
# and does nothing when they're already there for this id (--force builds again).
#
# The steps: python-build-standalone's install_only_stripped archive (scripts/engine.env: its URL
# and SHA-256; downloaded once into --cache), unpacked; every package of requirements.release.txt
# installed into it with uv (--require-hashes, macOS 14's wheels: MACOSX_DEPLOYMENT_TARGET=14.0);
# console scripts made to find their Python wherever the engine is; scripts/engine.prune applied;
# every Mach-O file checked with codesign --verify (never stripped, never relinked); licences
# gathered (scripts/engine_licenses.py); packed with /usr/bin/tar (xz). Then it tests that the engine
# moves: unpacked into a folder with spaces and non-ASCII letters in its path, compiled, given NEO's
# wheel (--wheel) and imported as Setup's smoke test does, with no file naming the build folder.
#
# Options: --cache DIR (the downloaded Python; default dist/engine-cache), --work DIR (scratch, about
# 4 GB; default dist/engine-work, removed afterwards unless --keep-work), --no-selftest, --force.
# Needs uv, the network (python-build-standalone and any wheel not in uv's cache) and about 6 GB.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/dist/engines" CACHE="$ROOT/dist/engine-cache" WORK="$ROOT/dist/engine-work"
WHEEL="" KEEP=0 FORCE=0 PRINT_ID=0 SELFTEST=1
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --wheel) WHEEL="$2"; shift 2 ;;
    --keep-work) KEEP=1; shift ;;
    --no-selftest) SELFTEST=0; shift ;;
    --force) FORCE=1; shift ;;
    --print-id) PRINT_ID=1; shift ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "build_engine.sh: unknown option $1" >&2; exit 2 ;;
  esac
done

# The id: the first 16 hex characters of a SHA-256 over the inputs, each framed by its path and
# size (scripts/engine_id.py says the same in Python; tests hold the two together).
engine_id() {
  {
    printf 'neo-engine-id 1\n'
    for f in scripts/engine.env requirements.release.txt scripts/engine.prune; do
      printf '%s\n%s\n' "$f" "$(wc -c < "$ROOT/$f" | tr -d ' ')"
      cat "$ROOT/$f"
    done
  } | shasum -a 256 | cut -c1-16
}
ID="$(engine_id)"
[ "$PRINT_ID" = 1 ] && { echo "$ID"; exit 0; }

say() { printf '→ %s\n' "$*"; }
die() { printf 'build_engine.sh: %s\n' "$*" >&2; exit 1; }
now() { perl -MTime::HiRes=time -e 'printf "%.1f", time'; }
since() { perl -e 'printf "%.1f", $ARGV[1] - $ARGV[0]' "$1" "$(now)"; }

# engine.env: KEY=value lines (ours; checked before use)
while IFS='=' read -r k v; do
  case "$k" in ''|\#*) continue ;; esac
  [[ "$k" =~ ^[A-Z_][A-Z0-9_]*$ ]] && [[ "$v" =~ ^[A-Za-z0-9._:/%+=-]*$ ]] || die "engine.env: a line I won't read: $k=$v"
  printf -v "$k" '%s' "$v"
done < "$ROOT/scripts/engine.env"
for k in ENGINE_RECIPE PYTHON_VERSION PBS_RELEASE PBS_ASSET PBS_URL PBS_SHA256 MACOSX_DEPLOYMENT_TARGET PYTHON_PLATFORM ENGINE_PLATFORM; do
  [ -n "${!k:-}" ] || die "engine.env has no $k"
done
case "$PBS_URL" in https://github.com/astral-sh/python-build-standalone/releases/download/*) ;; *) die "engine.env: PBS_URL isn't python-build-standalone's release" ;; esac
[[ "$PBS_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "engine.env: PBS_SHA256 isn't a SHA-256"
PYMM="${PYTHON_VERSION%.*}"

ASSET="neo-engine-$ID-$ENGINE_PLATFORM.tar.xz"
FACTS="neo-engine-$ID.json"
LICS="neo-engine-$ID-licenses.txt"
mkdir -p "$OUT" "$CACHE"
OUT="$(cd "$OUT" && pwd)" CACHE="$(cd "$CACHE" && pwd)"
if [ "$FORCE" = 0 ] && [ -f "$OUT/$ASSET" ] && [ -f "$OUT/$FACTS" ] && [ -f "$OUT/$LICS" ]; then
  want="$(plutil -extract sha256 raw -o - "$OUT/$FACTS" 2>/dev/null || true)"
  if [ -n "$want" ] && [ "$(shasum -a 256 "$OUT/$ASSET" | cut -d' ' -f1)" = "$want" ]; then
    say "engine $ID is already built: $OUT/$ASSET"
    exit 0
  fi
fi

[ "$(uname -s)/$(uname -m)" = "Darwin/arm64" ] || die "the engine is built on an Apple-silicon Mac"
command -v uv >/dev/null || die "uv isn't installed (brew install uv)"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
free_kb="$(df -Pk "$WORK" | awk 'NR==2 {print $4}')"
[ "$free_kb" -gt 6000000 ] || die "about 6 GB free is needed in $WORK ($((free_kb / 1024)) MB free)"
T0="$(now)"
say "engine $ID: Python $PYTHON_VERSION (python-build-standalone $PBS_RELEASE), requirements.release.txt, macOS $MACOSX_DEPLOYMENT_TARGET+"

# 1. Python ------------------------------------------------------------------------------------
PBS_FILE="$CACHE/python/$PBS_ASSET"
mkdir -p "$CACHE/python"
if [ -f "$PBS_FILE" ] && [ "$(shasum -a 256 "$PBS_FILE" | cut -d' ' -f1)" = "$PBS_SHA256" ]; then
  say "python-build-standalone: $PBS_FILE (already downloaded)"
else
  say "downloading $PBS_ASSET"
  rm -f "$PBS_FILE" "$PBS_FILE.part"
  /usr/bin/curl -q -fL --proto '=https' --tlsv1.2 --retry 3 --silent --show-error -o "$PBS_FILE.part" "$PBS_URL"
  got="$(shasum -a 256 "$PBS_FILE.part" | cut -d' ' -f1)"
  [ "$got" = "$PBS_SHA256" ] || { rm -f "$PBS_FILE.part"; die "python-build-standalone's SHA-256 is $got, not engine.env's $PBS_SHA256"; }
  mv "$PBS_FILE.part" "$PBS_FILE"
fi

E="$WORK/engine"
rm -rf "$WORK/engine" "$WORK/unpack" "$WORK/selftest" "$WORK/$ASSET.part"
mkdir -p "$WORK/unpack"
/usr/bin/tar -xzf "$PBS_FILE" -C "$WORK/unpack"
[ -x "$WORK/unpack/python/bin/python3" ] || die "the archive has no python/bin/python3"
mv "$WORK/unpack/python" "$E"
rmdir "$WORK/unpack"
PY="$E/bin/python3"
got="$("$PY" -I -c 'import platform; print(platform.python_version())')"
[ "$got" = "$PYTHON_VERSION" ] || die "the archive's Python is $got, not $PYTHON_VERSION"
"$PY" -I -m pip --version >/dev/null 2>&1 || { say "adding pip (ensurepip, offline)"; "$PY" -I -m ensurepip --default-pip >/dev/null; }
# What NEO.app itself runs in the engine (setup.rs: compileall, pip install of the wheel, the smoke test)
"$PY" -I -m pip --version >/dev/null || die "the engine has no pip"

# 2. Packages ----------------------------------------------------------------------------------
say "installing requirements.release.txt (uv $(uv --version | cut -d' ' -f2), macOS $MACOSX_DEPLOYMENT_TARGET wheels)"
T1="$(now)"
# Only what this build says: no uv.toml, no inherited Python or environment settings.
env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONUSERBASE -u VIRTUAL_ENV -u CONDA_PREFIX \
    -u UV_PYTHON -u UV_SYSTEM_PYTHON -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INDEX -u UV_FIND_LINKS \
    -u UV_CONSTRAINT -u UV_OVERRIDE -u UV_LINK_MODE -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
    MACOSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" UV_PYTHON_DOWNLOADS=never UV_NO_CONFIG=1 \
  uv pip install --python "$PY" --python-platform "$PYTHON_PLATFORM" --break-system-packages \
    --require-hashes --no-compile --strict -r "$ROOT/requirements.release.txt" > "$WORK/uv.log" 2>&1 \
  || { sed 's/^/    /' "$WORK/uv.log" >&2; die "uv pip install failed"; }
# what had to be fetched or built (uv's cache had the rest)
grep -E '^ *(Downloading|Downloaded|Building|Built) ' "$WORK/uv.log" | sed 's/^ */    /' || true
say "installed in $(since "$T1") s"

# Exactly the lock: every pin at its version, and nothing else but pip.
"$PY" -I - "$ROOT/requirements.release.txt" <<'PY' || die "the engine isn't exactly requirements.release.txt"
import importlib.metadata as md, re, sys
want = {}
for line in open(sys.argv[1]):
    m = re.match(r"^([A-Za-z0-9_.-]+)(?:==([^\s\\]+)| @ )", line)
    if m:
        want[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
have = {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version for d in md.distributions()}
bad = [f"{n}: {have.get(n)} (lock {v})" for n, v in want.items() if n not in have or (v and have[n] != v)]
extra = sorted(set(have) - set(want) - {"pip"})
for b in bad: print("  wrong:", b)
for x in extra: print("  not in the lock:", x)
sys.exit(1 if bad or extra else 0)
PY
# mlx and mlx-metal are the packages whose wheels differ by macOS: these must be macOS 14's.
for w in mlx mlx_metal; do
  tag="$(grep -h '^Tag:' "$E"/lib/python"$PYMM"/site-packages/"$w"-[0-9]*.dist-info/WHEEL | head -1 | awk '{print $2}')"
  case "$tag" in *macosx_14_0*) ;; *) die "$w's wheel is ${tag:-missing}, not macOS 14's" ;; esac
done

# 3. Console scripts find their Python wherever the engine is: uv wrote this folder's path into
# them; they get the trampoline python-build-standalone's own bin/pip3 starts with instead.
"$PY" -I - "$E" "$PYMM" <<'PY' || die "couldn't make the console scripts relocatable"
import os, sys
engine, mm = sys.argv[1], sys.argv[2]
bindir = os.path.join(engine, "bin")
q = "'" * 3
head = f'#!/bin/sh\n{q}exec\' "$(dirname -- "$(realpath -- "$0")")/python{mm}" "$0" "$@"\n\' {q}\n'
n = 0
for name in sorted(os.listdir(bindir)):
    p = os.path.join(bindir, name)
    if os.path.islink(p) or not os.path.isfile(p):
        continue
    with open(p, "rb") as f:
        text = f.read().decode("utf-8", "surrogateescape")
    lines = text.split("\n")
    if lines[0].startswith("#!" + bindir + "/"):  # #!/build/engine/bin/python3
        rest = lines[1:]
    elif lines[0] == "#!/bin/sh" and len(lines) > 2 and bindir in lines[1] and lines[1].startswith(q + "exec"):
        rest = lines[3:]  # what uv writes when the path has a space
    else:
        continue
    with open(p, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write(head + "\n".join(rest))
    n += 1
print(f"→ {n} console scripts now find their Python wherever the engine is")
PY

# 4. Prune --------------------------------------------------------------------------------------
before="$(du -sk "$E" | cut -f1)"
while read -r kind glob; do
  case "$kind" in ''|\#*) continue ;; esac
  case "$kind" in
    name) find "$E" -name "$glob" -prune -exec rm -rf {} + ;;
    path) case "$glob" in /*|*..*) die "engine.prune: a path outside the engine: $glob" ;; esac
          for p in "$E"/$glob; do [ -e "$p" ] && rm -rf "$p"; done ;;
    *) die "engine.prune: unknown rule $kind" ;;
  esac
done < "$ROOT/scripts/engine.prune"
after="$(du -sk "$E" | cut -f1)"
say "pruned $(((before - after) / 1024)) MB (engine.prune); $((after / 1024)) MB left"

# 5. Every Mach-O file's arm64 code still carries a valid signature (nothing here edits one). Universal
# wheels (pyobjc's, among others) ship an unsigned x86_64 half beside a signed arm64 one: only the
# arm64 half ever runs here, so that is the one checked (codesign --arch arm64).
"$PY" -I - "$E" > "$WORK/macho.txt" <<'PY'
import os, struct, sys
ARM64, X86_64 = 0x0100000C, 0x01000007
def archs(head: bytes) -> list[int] | None:
    magic = head[:4]
    if magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):        # thin, little-endian
        return [struct.unpack("<I", head[4:8])[0]]
    if magic in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):        # fat (32- or 64-bit offsets)
        n = struct.unpack(">I", head[4:8])[0]
        if not 0 < n < 20:                                          # a Java class file starts the same
            return None
        size = 20 if magic == b"\xca\xfe\xba\xbe" else 32
        return [struct.unpack(">I", head[8 + i * size : 12 + i * size])[0] for i in range(n)]
    return None
for d, _, files in os.walk(sys.argv[1]):
    for f in files:
        p = os.path.join(d, f)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        with open(p, "rb") as fh:
            got = archs(fh.read(8 + 20 * 32))
        if got is not None:
            print(("arm64" if ARM64 in got else "other") + "\t" + p)
PY
machos="$(wc -l < "$WORK/macho.txt" | tr -d ' ')"
bad=0
while IFS=$'\t' read -r arch m; do
  if [ "$arch" != arm64 ]; then
    echo "    (no arm64 code, never loaded here: ${m#"$E"/})"
    continue
  fi
  codesign --verify --arch arm64 "$m" 2>/dev/null || { echo "    not validly signed: ${m#"$E"/}" >&2; bad=$((bad + 1)); }
done < "$WORK/macho.txt"
[ "$bad" = 0 ] || die "$bad Mach-O files fail codesign --verify --arch arm64"
say "$machos Mach-O files, every arm64 one validly signed"

# 6. Licences, and the engine's own record
"$PY" -I "$ROOT/scripts/engine_licenses.py" --engine "$E" --texts "$ROOT/scripts/licenses" --pbs "$PBS_RELEASE" \
  --combined "$WORK/$LICS" | sed 's/^/    /'
packages="$("$PY" -I -c 'import importlib.metadata as m; print(len(list(m.distributions())))')"
"$PY" -I - "$E/neo-engine.json" "$ID" "$PYTHON_VERSION" "$MACOSX_DEPLOYMENT_TARGET" "$ENGINE_PLATFORM" "$ENGINE_RECIPE" \
  "$PBS_RELEASE" "$PBS_ASSET" "$PBS_SHA256" "$packages" <<'PY'
import json, sys
f, eid, py, minos, plat, recipe, rel, asset, sha, n = sys.argv[1:]
json.dump({"schema": 1, "engine_id": eid, "python": py, "min_macos": minos, "platform": plat, "recipe": int(recipe),
           "python_build_standalone": {"release": rel, "asset": asset, "sha256": sha}, "packages": int(n),
           "lock": "requirements.release.txt"}, open(f, "w"), indent=2)
open(f, "a").write("\n")
PY

# 7. Pack ---------------------------------------------------------------------------------------
say "packing $ASSET (xz)"
T2="$(now)"
# No owners, ACLs, flags, extended attributes or AppleDouble files: the archive holds only the files.
# xz on every core when the build Mac has the xz tool (its output unpacks with any /usr/bin/tar).
TAR=(/usr/bin/tar --no-mac-metadata --no-xattrs --no-acls --no-fflags --uid 0 --gid 0 --uname root --gname wheel)
if command -v xz >/dev/null; then
  COPYFILE_DISABLE=1 "${TAR[@]}" -cf - -C "$E" . | xz -T0 -6 -c > "$WORK/$ASSET.part"
else
  COPYFILE_DISABLE=1 "${TAR[@]}" --options xz:compression-level=6 -cJf "$WORK/$ASSET.part" -C "$E" .
fi
SHA="$(shasum -a 256 "$WORK/$ASSET.part" | cut -d' ' -f1)"
BYTES="$(stat -f %z "$WORK/$ASSET.part")"
# Read back as Setup will count it (every entry tar lists) and as Setup will look for Python.
/usr/bin/tar -tvJf "$WORK/$ASSET.part" > "$WORK/listing.txt" || die "the packed archive doesn't read back"
read -r FILES UNPACKED HASPY < <(awk '{n++; s+=$5; if ($9 == "./bin/python3") py = 1} END {print n, s, py + 0}' "$WORK/listing.txt")
[ "$HASPY" = 1 ] || die "the packed archive has no ./bin/python3"
rm -f "$WORK/listing.txt"
say "packed in $(since "$T2") s: $((BYTES / 1048576)) MB, $FILES entries, $((UNPACKED / 1048576)) MB unpacked"

# 8. It moves: unpack where a user's would be, with spaces and non-ASCII letters in the path --------
SELF="{}"
if [ "$SELFTEST" = 1 ]; then
  ST="$WORK/selftest/Mac HD ünï/Library/Application Support/NEO/engines/$ID"
  mkdir -p "$ST"
  T3="$(now)"
  /usr/bin/tar -xJf "$WORK/$ASSET.part" -C "$ST" --no-same-owner --no-xattrs --no-acls --no-fflags --no-mac-metadata
  unpack_s="$(since "$T3")"
  # As unpacked, before anything runs in it: no text file names where it was built (or this Mac's
  # home, or uv's cache), and every arm64 Mach-O still verifies.
  for p in "$WORK" "$HOME/" "${UV_CACHE_DIR:-$HOME/.cache/uv}"; do
    named="$(grep -r -I -l -F "$p" "$ST" 2>/dev/null | head -5 || true)"
    [ -z "$named" ] || die "files in the engine name $p: $named"
  done
  bad=0
  while IFS=$'\t' read -r arch m; do
    [ "$arch" = arm64 ] || continue
    codesign --verify --arch arm64 "$ST/${m#"$E"/}" 2>/dev/null || bad=$((bad + 1))
  done < "$WORK/macho.txt"
  [ "$bad" = 0 ] || die "$bad Mach-O files fail codesign --verify after the move"
  # Then what Setup does there (setup.rs): compile, install NEO's wheel offline, the smoke test.
  T3="$(now)"
  "$ST/bin/python3" -I -m compileall -q -j 0 "$ST/lib/python$PYMM" >/dev/null 2>&1 || true
  compile_s="$(since "$T3")"
  wheel_s="0"
  if [ -n "$WHEEL" ]; then
    T3="$(now)"
    "$ST/bin/python3" -I -m pip install --no-index --no-deps --force-reinstall --no-compile --break-system-packages \
      --disable-pip-version-check --no-warn-script-location --quiet "$WHEEL"
    "$ST/bin/python3" -I -m compileall -q -j 0 "$ST/lib/python$PYMM/site-packages/neo2" >/dev/null
    wheel_s="$(since "$T3")"
    SMOKE="import neo2, torch, mlx.core, laya, sounddevice, Quartz, AVFoundation; print('neo2', neo2.__version__)"
  else
    SMOKE="import torch, mlx.core, laya, sounddevice, Quartz, AVFoundation; print('ok')"
  fi
  T3="$(now)"
  said="$(cd "$WORK" && env -i HOME="$WORK/selftest" PATH=/usr/bin:/bin:/usr/sbin:/sbin LANG=en_US.UTF-8 \
      "$ST/bin/python3" -I -c "$SMOKE" 2>&1)" || { printf '%s\n' "$said" | tail -20 >&2; die "the moved engine's smoke test failed"; }
  printf '    imports: %s\n' "$(printf '%s\n' "$said" | tail -1)"
  smoke_s="$(since "$T3")"
  "$ST/bin/pip3" --version >/dev/null || die "the moved engine's pip3 script doesn't run"
  say "it moves: unpacked in $unpack_s s, compiled in $compile_s s, wheel in $wheel_s s, imports in $smoke_s s"
  SELF="{\"unpack_s\": $unpack_s, \"compile_s\": $compile_s, \"wheel_s\": $wheel_s, \"smoke_s\": $smoke_s}"
  rm -rf "$WORK/selftest"
fi

# 9. Publish into --out ---------------------------------------------------------------------------
"$PY" -I - "$WORK/$FACTS" <<PY
import json, sys
json.dump({"schema": 1, "engine_id": "$ID", "asset": "$ASSET", "sha256": "$SHA", "bytes": $BYTES,
           "unpacked_bytes": $UNPACKED, "files": $FILES, "python": "$PYTHON_VERSION", "min_macos": "$MACOSX_DEPLOYMENT_TARGET",
           "platform": "$ENGINE_PLATFORM", "packages": $packages, "macho_files": $machos, "selftest": $SELF},
          open(sys.argv[1], "w"), indent=2)
open(sys.argv[1], "a").write("\n")
PY
mv "$WORK/$ASSET.part" "$OUT/$ASSET"
mv "$WORK/$LICS" "$OUT/$LICS"
mv "$WORK/$FACTS" "$OUT/$FACTS"
[ "$KEEP" = 1 ] || rm -rf "$WORK/engine" "$WORK/macho.txt"
say "engine $ID built in $(since "$T0") s: $OUT/$ASSET"
echo "  sha256 $SHA  ($BYTES bytes)"
