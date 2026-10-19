#!/usr/bin/env bash
# NEO Sandbox: a test build of NEO.app that runs beside the real NEO without touching it.
#
#   scripts/sandbox_build.sh --out <dir> --engine <neo-engine-<id>-macos14-arm64.tar.xz> [--url <url>] [--dmg]
#   scripts/sandbox_build.sh --out <dir> --fake-engine [--dmg]      # a tiny engine: walks Setup, runs no core
#   scripts/sandbox_build.sh --out <dir> --no-engine                # Setup's "no engine" page only
#
# It builds "<dir>/NEO Sandbox.app" (bundle id com.arnav.neo.sandbox, the cargo `sandbox` feature and
# ui/src-tauri/tauri.sandbox.conf.json): its home is "<dir>/home" — its own ~/.neo, Application
# Support/NEO, LaunchAgents, Hugging Face and Playwright caches, and its core's HOME — and its core
# listens on 18765. It never writes the real LaunchAgent, ~/.neo or Application Support/NEO, and never
# uses port 8765; macOS lists its permissions under "NEO Sandbox". Its engine URL may be file:// (the
# default: the archive where it is) or a server on this Mac (--url http://127.0.0.1:<port>/<asset>).
# --dmg also makes "<dir>/NEO-Sandbox.dmg" (the app and an Applications link), to try the disk-image
# and translocation pages. Updates are checked only if "<dir>/home/.neo/update-url" names a local server.
#
# This script never opens the app. When you're done with it: quit it, `tccutil reset All
# com.arnav.neo.sandbox`, `lsregister -u "<dir>/NEO Sandbox.app"`, and move <dir> to the Trash.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="" ENGINE="" ENGINE_JSON="" URL="" FAKE=0 NONE=0 DMG=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --engine-json) ENGINE_JSON="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --fake-engine) FAKE=1; shift ;;
    --no-engine) NONE=1; shift ;;
    --dmg) DMG=1; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "sandbox_build.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "sandbox_build.sh: --out <dir> is needed (NEO Sandbox and its home live there)" >&2; exit 2; }
if [ -z "$ENGINE" ] && [ "$FAKE" = 0 ] && [ "$NONE" = 0 ]; then
  echo "sandbox_build.sh: --engine <archive>, --fake-engine or --no-engine" >&2; exit 2
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
REAL_SUPPORT="$HOME/Library/Application Support/NEO"
case "$OUT/" in
  "$HOME/.neo/"*|"$REAL_SUPPORT/"*|/Applications/*) echo "sandbox_build.sh: not inside NEO's real folders ($OUT)" >&2; exit 2 ;;
esac
WORK="$OUT/.work"
mkdir -p "$WORK"

for d in "$HOME/.cargo/bin" "$HOME"/.rustup/toolchains/stable-*/bin "$(brew --prefix rustup 2>/dev/null || echo /nonexistent)/bin"; do
  [ -x "$d/cargo" ] && PATH="$d:$PATH" && break
done
command -v cargo >/dev/null || { echo "Rust isn't installed (https://rustup.rs)"; exit 1; }
command -v uv >/dev/null || { echo "uv isn't installed (brew install uv)"; exit 1; }

echo "→ building neo2's wheel"
rm -rf "$WORK/wheel" "$ROOT/build"
(cd "$ROOT" && uv build --wheel --offline --out-dir "$WORK/wheel" >/dev/null)
rm -rf "$ROOT/build"
WHEEL="$(ls "$WORK"/wheel/neo2-*-py3-none-any.whl | head -1)"

SMOKE_ARGS=()
if [ "$FAKE" = 1 ]; then
  # A tiny engine: python-build-standalone as uv installed it (an APFS clone of it; that folder is
  # only read), and nothing else. Setup installs it like the real one; its core can't run (no
  # packages), so it's for Setup's own pages. The smoke test asks for neo2 only.
  BASE="$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12.*-macos-aarch64-none 2>/dev/null | sort -V | tail -1)"
  [ -n "$BASE" ] || { echo "sandbox_build.sh: --fake-engine needs uv's CPython 3.12 (uv python install 3.12)" >&2; exit 1; }
  rm -rf "$WORK/fake"
  mkdir -p "$WORK/fake" "$OUT/engine"
  cp -c -R "$BASE" "$WORK/fake/engine" 2>/dev/null || cp -R "$BASE" "$WORK/fake/engine"
  rm -f "$WORK/fake/engine/lib/python3.12/EXTERNALLY-MANAGED"
  find "$WORK/fake/engine" -name __pycache__ -type d -prune -exec rm -rf {} +
  printf '{"python": "%s", "fake": true}\n' "$("$WORK/fake/engine/bin/python3" -I -c 'import platform; print(platform.python_version())')" > "$WORK/fake/engine/neo-engine.json"
  FID="f$(date +%s | shasum -a 256 | cut -c1-15)"
  ENGINE="$OUT/engine/neo-engine-$FID-macos14-arm64.tar.xz"
  echo "→ packing a test engine ($FID)"
  /usr/bin/tar -cJf "$ENGINE" -C "$WORK/fake/engine" .
  SMOKE_ARGS=(--smoke neo2)
fi

STAGE=(--sandbox --wheel "$WHEEL")
if [ "$NONE" = 1 ]; then
  STAGE+=(--no-engine)
else
  ENGINE="$(cd "$(dirname "$ENGINE")" && pwd)/$(basename "$ENGINE")"
  if [ -z "$URL" ]; then
    URL="$(python3 -c 'import sys, urllib.parse; print("file://" + urllib.parse.quote(sys.argv[1]))' "$ENGINE")"
  fi
  case "$URL" in file:///*|http://127.0.0.1:*|http://localhost:*) ;; *) echo "sandbox_build.sh: NEO Sandbox's engine comes from this Mac (file:// or http://127.0.0.1)" >&2; exit 2 ;; esac
  STAGE+=(--engine "$ENGINE" --url "$URL")
  [ -n "$ENGINE_JSON" ] && STAGE=(--sandbox --wheel "$WHEEL" --engine-json "$ENGINE_JSON" --url "$URL")
fi
"$ROOT/scripts/stage_resources.sh" "${STAGE[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}

echo "→ building NEO Sandbox.app"
cd "$ROOT/ui"
[ -d node_modules ] || npm ci
TARGET="${CARGO_TARGET_DIR:-$WORK/target}"
CARGO_TARGET_DIR="$TARGET" npm run tauri build -- --features sandbox --config src-tauri/tauri.sandbox.conf.json --bundles app
BUILT="$TARGET/release/bundle/macos/NEO Sandbox.app"
[ -d "$BUILT" ] || { echo "sandbox_build.sh: no app at $BUILT" >&2; exit 1; }

APP="$OUT/NEO Sandbox.app"
rm -rf "$APP"
ditto "$BUILT" "$APP"
# Ad-hoc, with an identifier requirement: grants survive a rebuild, and they're NEO Sandbox's alone.
codesign --force --sign - --identifier com.arnav.neo.sandbox -r='designated => identifier "com.arnav.neo.sandbox"' "$APP"
codesign --verify --strict "$APP"
ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP/Contents/Info.plist")"
[ "$ID" = "com.arnav.neo.sandbox" ] || { echo "sandbox_build.sh: the app's bundle id is $ID" >&2; exit 1; }
[ -f "$APP/Contents/Resources/ui/.build-id" ] || { echo "sandbox_build.sh: the bundle has no ui/.build-id" >&2; exit 1; }
if [ "$NONE" = 0 ]; then
  [ -f "$APP/Contents/Resources/engine/manifest.json" ] || { echo "sandbox_build.sh: the bundle has no engine manifest" >&2; exit 1; }
fi

if [ "$DMG" = 1 ]; then
  echo "→ making NEO-Sandbox.dmg"
  rm -rf "$WORK/dmg" "$OUT/NEO-Sandbox.dmg"
  mkdir -p "$WORK/dmg"
  ditto "$APP" "$WORK/dmg/NEO Sandbox.app"
  ln -s /Applications "$WORK/dmg/Applications"
  hdiutil create -quiet -volname "NEO Sandbox" -srcfolder "$WORK/dmg" -ov -format UDZO "$OUT/NEO-Sandbox.dmg"
  rm -rf "$WORK/dmg"
fi

cat <<EOF

NEO Sandbox is built (it was not opened):
  app:     $APP
  home:    $OUT/home   (made when it first runs: its ~/.neo, Application Support, caches)
  engine:  ${URL:-none}
  port:    18765
$( [ "$DMG" = 1 ] && echo "  dmg:     $OUT/NEO-Sandbox.dmg" )
To try the disk-image and Gatekeeper pages, give the dmg a browser's quarantine flag first:
  xattr -w com.apple.quarantine "0083;\$(printf %x \$(date +%s));Safari;" "$OUT/NEO-Sandbox.dmg"
Afterwards: quit it, tccutil reset All com.arnav.neo.sandbox, lsregister -u the app, and trash $OUT.
EOF
