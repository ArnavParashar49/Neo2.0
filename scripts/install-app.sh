#!/usr/bin/env bash
# Build NEO.app, sign it so macOS keeps NEO's permissions across rebuilds, install it to
# /Applications and start it. Safe to re-run: it replaces the installed copy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_ID="com.arnav.neo"
DEST="/Applications/NEO.app"
LSREG=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
AGENT="$HOME/Library/LaunchAgents/$BUNDLE_ID.plist"

# cargo from rustup, even when ~/.cargo/bin isn't on PATH
for d in "$HOME/.cargo/bin" "$HOME"/.rustup/toolchains/stable-*/bin; do
  [ -x "$d/cargo" ] && PATH="$d:$PATH" && break
done
command -v cargo >/dev/null || { echo "Rust isn't installed (https://rustup.rs)"; exit 1; }
[ -x "$ROOT/.venv/bin/python" ] || { echo "No Python environment at $ROOT/.venv — see README (uv sync)"; exit 1; }

RUNTIME="$HOME/Library/Application Support/NEO"
echo "→ installing NEO's runtime in $RUNTIME"
# The app runs the core from here — never from this checkout, which macOS guards when it's in
# ~/Documents. Exact versions from requirements.lock; uv's cache makes re-installs quick.
command -v uv >/dev/null || { echo "uv isn't installed (brew install uv)"; exit 1; }
PYVER="$("$ROOT/.venv/bin/python" -c 'import platform; print(platform.python_version())')"
mkdir -p "$RUNTIME"
uv venv --quiet --allow-existing --python "$PYVER" "$RUNTIME/.venv"
uv pip install --quiet --python "$RUNTIME/.venv/bin/python" -r "$ROOT/requirements.lock"
uv pip install --quiet --python "$RUNTIME/.venv/bin/python" --no-deps --reinstall "$ROOT"
mkdir -p "$HOME/.neo" && chmod 700 "$HOME/.neo"
if [ ! -f "$HOME/.neo/.env" ] && [ -f "$ROOT/.env" ]; then
  install -m 600 "$ROOT/.env" "$HOME/.neo/.env"
  echo "  copied your keys to ~/.neo/.env (edit them there)"
fi
[ -f "$HOME/.neo/.env" ] || echo "  ! no API keys yet: put GEMINI_API_KEY=… in ~/.neo/.env (see README)"

echo "→ building"
cd "$ROOT/ui"
[ -d node_modules ] || npm ci
npm run tauri build >/dev/null
APP="$ROOT/ui/src-tauri/target/release/bundle/macos/NEO.app"

echo "→ signing"
# Ad-hoc signatures identify an app by a hash of this exact build, so every rebuild used to look
# like a new app to macOS and lose Accessibility / Screen Recording. Identify NEO by its bundle
# id instead: permissions granted once stay granted.
codesign --force --sign - --identifier "$BUNDLE_ID" -r="designated => identifier \"$BUNDLE_ID\"" "$APP"
codesign --verify --strict "$APP"

echo "→ installing"
launchctl bootout "gui/$(id -u)/$BUNDLE_ID" 2>/dev/null || true
pkill -TERM -f "$DEST/Contents/MacOS/" 2>/dev/null || true
for _ in $(seq 1 40); do pgrep -f "$DEST/Contents/MacOS/" >/dev/null || break; sleep 0.5; done
rm -rf "$DEST"
ditto "$APP" "$DEST"
# Exactly one NEO: forget and remove the build copy, and keep Spotlight out of the build folder.
"$LSREG" -u "$APP" 2>/dev/null || true
rm -rf "$APP"
touch "$ROOT/ui/src-tauri/target/.metadata_never_index"
"$LSREG" -f "$DEST"

echo "→ starting"
if [ -f "$AGENT" ]; then
  launchctl bootstrap "gui/$(id -u)" "$AGENT"
else
  nohup "$DEST/Contents/MacOS/neo-ui" >/dev/null 2>&1 &
fi
echo "NEO $(defaults read "$DEST/Contents/Info" CFBundleShortVersionString) is installed and running (menu bar: the dotted orb)."
