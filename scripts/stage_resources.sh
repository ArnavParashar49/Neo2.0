#!/usr/bin/env bash
# What NEO.app carries besides its binary, staged for `tauri build` (tauri.release.conf.json maps
# ui/src-tauri/resources/ into the bundle's Resources; NEO Sandbox's tauri.sandbox.conf.json maps
# resources-sandbox/):
#
#   engine/manifest.json      which engine this NEO runs, where it's published, its SHA-256 and sizes,
#                             and the neo2 wheel beside it (the format: ui/src-tauri/src/setup.rs)
#   engine/neo2-X.Y.Z-py3-none-any.whl
#   Licenses/                 NEO's licence, THIRD_PARTY_NOTICES.md, the full licence texts they name
#                             (scripts/licenses), rust-crates.txt and npm-packages.txt (what's compiled
#                             into the app: scripts/app_licenses.py) and engine-packages.txt (what's in
#                             the engine: build_engine.sh's neo-engine-<id>-licenses.txt), README.txt
#
# The page itself (ui/dist, with its .build-id) is mapped straight from where Vite builds it.
#
#   scripts/stage_resources.sh --wheel dist/neo2-2.2.0-py3-none-any.whl --engine path/to/neo-engine-<id>-macos14-arm64.tar.xz
#   scripts/stage_resources.sh --wheel … --engine-json neo-engine.json      # the sizes build_engine.sh measured
#   scripts/stage_resources.sh [--wheel …] --no-engine                      # a build that installs nothing
#                                                                            (scripts/install-app.sh's)
#
# Options: --url <engine URL> (default: this version's GitHub release asset), --sandbox (stage for NEO
# Sandbox: resources-sandbox/, and a file:// or http://127.0.0.1 URL is allowed), --out <dir>,
# --smoke <mod,mod> (what Setup's smoke test imports; NEO's main libraries unless a test engine says),
# --engine-licenses <file> (default: neo-engine-<id>-licenses.txt beside the engine), --strict (a
# release: every licence list must be made, or nothing is staged).
# Standard tools only (python3 for the JSON, shasum, tar; cargo's own registry, offline, for the crate
# list). Writes nothing outside --out.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHEEL="" ENGINE="" ENGINE_JSON="" ENGINE_LICS="" URL="" OUT="" SANDBOX=0 NO_ENGINE=0 STRICT=0 SMOKE="neo2,torch,mlx.core,laya,sounddevice,Quartz,AVFoundation"
while [ $# -gt 0 ]; do
  case "$1" in
    --wheel) WHEEL="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --engine-json) ENGINE_JSON="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --sandbox) SANDBOX=1; shift ;;
    --no-engine) NO_ENGINE=1; shift ;;
    --smoke) SMOKE="$2"; shift 2 ;;
    --engine-licenses) ENGINE_LICS="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "stage_resources.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
if [ -z "$OUT" ]; then
  if [ "$SANDBOX" = 1 ]; then OUT="$ROOT/ui/src-tauri/resources-sandbox"; else OUT="$ROOT/ui/src-tauri/resources"; fi
fi
if [ -n "$WHEEL" ] || [ "$NO_ENGINE" = 0 ]; then
  [ -n "$WHEEL" ] && [ -f "$WHEEL" ] || { echo "stage_resources.sh: --wheel <neo2 wheel> is needed" >&2; exit 2; }
  case "$(basename "$WHEEL")" in neo2-*-py3-none-any.whl) ;; *) echo "stage_resources.sh: $WHEEL isn't neo2's wheel" >&2; exit 2 ;; esac
fi
if [ "$NO_ENGINE" = 0 ] && [ -z "$ENGINE" ] && [ -z "$ENGINE_JSON" ]; then
  echo "stage_resources.sh: --engine <archive>, --engine-json <neo-engine.json> or --no-engine" >&2; exit 2
fi

APP_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$ROOT/ui/src-tauri/tauri.conf.json")"
NEO2_VERSION="$(basename "${WHEEL:-neo2-none-py3-none-any.whl}" | sed -e 's/^neo2-//' -e 's/-py3-none-any\.whl$//')"

# Staged beside --out first, then swapped in: a failure leaves the last staging as it was.
NEW="$OUT.new.$$"
rm -rf "$NEW"
mkdir -p "$NEW/engine" "$NEW/Licenses"
trap 'rm -rf "$NEW"' EXIT
if [ -n "$WHEEL" ]; then
  cp "$WHEEL" "$NEW/engine/"
  WHEEL_SHA="$(shasum -a 256 "$WHEEL" | cut -d' ' -f1)"
fi

# Licences: NEO's own, the notices, the full texts they name, and every package compiled into the app.
L="$NEW/Licenses"
cp "$ROOT/LICENSE" "$L/NEO-LICENSE.txt"
cp "$ROOT/THIRD_PARTY_NOTICES.md" "$L/"
cp "$ROOT"/scripts/licenses/*.txt "$L/"
LIC_ARGS=(--out "$L")
[ "$STRICT" = 1 ] && LIC_ARGS+=(--strict)
python3 "$ROOT/scripts/app_licenses.py" "${LIC_ARGS[@]}" || { echo "stage_resources.sh: the app's licence lists couldn't be made" >&2; exit 1; }
if [ "$NO_ENGINE" = 0 ] && [ -z "$ENGINE_LICS" ]; then
  # build_engine.sh leaves neo-engine-<id>-licenses.txt beside the archive and its facts
  for f in "${ENGINE_JSON%.json}-licenses.txt" "$(dirname "${ENGINE:-.}")/$(basename "${ENGINE:-x}" | sed -n 's/^\(neo-engine-[0-9a-f]\{16\}\)-.*\.tar\.xz$/\1/p')-licenses.txt"; do
    [ -f "$f" ] && { ENGINE_LICS="$f"; break; }
  done
fi
if [ -n "$ENGINE_LICS" ]; then
  cp "$ENGINE_LICS" "$L/engine-packages.txt"
elif [ "$NO_ENGINE" = 0 ] && [ "$STRICT" = 1 ]; then
  echo "stage_resources.sh: no engine licence list (neo-engine-<id>-licenses.txt from build_engine.sh)" >&2; exit 1
fi
{
  echo "NEO $APP_VERSION — licences"
  echo
  echo "NEO-LICENSE.txt          NEO itself: MIT License, (c) 2026 Arnav Parashar"
  echo "THIRD_PARTY_NOTICES.md   other people's work NEO uses or downloads: the engine's copyleft parts and"
  echo "                         where their sources are, the models (Parakeet's CC-BY-4.0 attribution among"
  echo "                         them), YOLOE, shazamio"
  echo "rust-crates.txt          the Rust crates compiled into NEO.app, with their licence texts"
  echo "npm-packages.txt         the npm packages in NEO.app's page, with their licence texts"
  [ -f "$L/engine-packages.txt" ] && echo "engine-packages.txt      Python and every package in NEO's engine (the download Setup installs)"
  echo "*.txt                    the full texts of the licences named above (GPL-3.0, LGPL-2.1, MPL-2.0, …)"
  echo
  echo "Source: https://github.com/ArnavParashar49/Neo2.0"
} > "$L/README.txt"

if [ "$NO_ENGINE" = 1 ]; then
  # (tauri needs the folder to exist; with no manifest the app installs nothing: setup.rs NoEngine)
  echo "This build carries no engine manifest: it runs scripts/install-app.sh's engine only." > "$NEW/engine/README.txt"
  rm -rf "$OUT" && mv "$NEW" "$OUT"
  trap - EXIT
  echo "→ staged $OUT (no engine${WHEEL:+; neo2 $NEO2_VERSION})"
  exit 0
fi

# The engine's facts: measured from the archive, or as build_engine.sh wrote them.
if [ -n "$ENGINE_JSON" ]; then
  read -r ID SHA BYTES UNPACKED FILES PYVER MINOS ASSET < <(python3 - "$ENGINE_JSON" <<'PY'
import json, sys
e = json.load(open(sys.argv[1]))
print(e["engine_id"], e["sha256"], e["bytes"], e["unpacked_bytes"], e["files"], e["python"], e.get("min_macos", "14.0"), e.get("asset", "-"))
PY
)
else
  [ -f "$ENGINE" ] || { echo "stage_resources.sh: no engine archive at $ENGINE" >&2; exit 2; }
  ASSET="$(basename "$ENGINE")"
  ID="$(printf '%s' "$ASSET" | sed -n 's/^neo-engine-\([0-9a-f]\{16\}\)-.*\.tar\.xz$/\1/p')"
  [ -n "$ID" ] || { echo "stage_resources.sh: $ASSET isn't named neo-engine-<16 hex>-….tar.xz" >&2; exit 2; }
  echo "→ measuring $ASSET"
  SHA="$(shasum -a 256 "$ENGINE" | cut -d' ' -f1)"
  BYTES="$(stat -f %z "$ENGINE")"
  read -r FILES UNPACKED < <(/usr/bin/tar -tvJf "$ENGINE" | awk '{n++; s+=$5} END {print n, s}')
  PYVER="$(/usr/bin/tar -tJf "$ENGINE" | grep -o -m1 'lib/python3\.[0-9]*/os\.py$' | sed -e 's|^lib/python||' -e 's|/os\.py$||' || true)"
  [ -n "$PYVER" ] || PYVER="3.12"
  # the patch version, from the engine's own record when it has one
  FULL="$( (/usr/bin/tar -xOJf "$ENGINE" ./neo-engine.json 2>/dev/null || /usr/bin/tar -xOJf "$ENGINE" neo-engine.json 2>/dev/null || true) | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("python",""))
except Exception: print("")')"
  case "$FULL" in "$PYVER".*) PYVER="$FULL" ;; *) PYVER="$PYVER.0" ;; esac
  MINOS="14.0"
fi
[ "$ASSET" = "-" ] && ASSET="neo-engine-$ID-macos14-arm64.tar.xz"
[ -n "$URL" ] || URL="https://github.com/ArnavParashar49/Neo2.0/releases/download/v$APP_VERSION/$ASSET"
if [ "$SANDBOX" = 0 ]; then
  case "$URL" in https://*) ;; *) echo "stage_resources.sh: a release engine comes over https only ($URL)" >&2; exit 2 ;; esac
fi

python3 - "$NEW/engine/manifest.json" <<PY
import json, sys
m = {
    "schema": 1,
    "app_version": "$APP_VERSION",
    "neo2_version": "$NEO2_VERSION",
    "wheel": "$(basename "$WHEEL")",
    "wheel_sha256": "$WHEEL_SHA",
    "engine": {
        "id": "$ID",
        "url": "$URL",
        "sha256": "$SHA",
        "bytes": int("$BYTES"),
        "unpacked_bytes": int("$UNPACKED"),
        "files": int("$FILES"),
        "python": "$PYVER",
        "min_macos": "$MINOS",
    },
    "models_bytes": 1700000000,
    "smoke": [m for m in "$SMOKE".split(",") if m],
}
with open(sys.argv[1], "w") as f:
    json.dump(m, f, indent=2)
    f.write("\n")
PY
rm -rf "$OUT" && mv "$NEW" "$OUT"
trap - EXIT
echo "→ staged $OUT: NEO $APP_VERSION, neo2 $NEO2_VERSION, engine $ID ($BYTES bytes) from $URL"
