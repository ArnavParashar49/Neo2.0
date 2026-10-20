#!/usr/bin/env bash
# NEO's pinned dependencies, at the exact versions the dev environment was tested with:
#
#   requirements.lock          NEO.app's runtime: what neo2[app] needs on macOS 14+ (Apple silicon),
#                              nothing more (scripts/install-app.sh, CI)
#   requirements-dev.lock      that, plus the tests and Laya's training rows (README: Install; CI)
#   requirements.release.txt   requirements.lock with every file's SHA-256: the release engine
#                              (scripts/build_engine.sh installs it with --require-hashes)
#
#   scripts/lock-deps.sh                 pins from the checkout's .venv (uv pip freeze)
#   scripts/lock-deps.sh --from FILE     pins from FILE instead (a freeze or an older lock)
#
# Only the pins' versions are used: uv resolves neo2[app] against them, so a package the dev
# environment has for other reasons never reaches the app. Needs the network for PyPI's index
# (the hashes), or uv's cache.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
command -v uv >/dev/null || { echo "uv isn't installed (brew install uv)" >&2; exit 1; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ -n "$FROM" ]; then
  cp "$FROM" "$TMP/freeze.txt"
else
  uv pip freeze --python "$ROOT/.venv/bin/python" > "$TMP/freeze.txt"
fi
# NEO itself (editable, a local path or a version) and the lint tools are never pinned here
grep -v -E "^(#|-e |neo2? @ |neo2?==|ruff==|mypy==)" "$TMP/freeze.txt" | grep -v -E '^\s*$' > "$TMP/pins.txt" || true

# misaki (Kokoro's English) downloads spaCy's small English model at run time when it's missing
# (spacy.cli.download): it is pinned instead, from the release URL the dev environment has.
ENSM="$(grep -E '^en-core-web-sm @ https://' "$TMP/pins.txt" || true)"
[ -n "$ENSM" ] || { echo "no en-core-web-sm @ https://… in the pins (uv pip install it into .venv)" >&2; exit 1; }

# torch 2.14 and mlx 0.32 have no wheel for macOS 13: macOS 14 is NEO's floor, and the release
# engine is built for it (mlx's macosx_14_0 wheels, not the build Mac's newer ones).
export MACOSX_DEPLOYMENT_TARGET=14.0
TARGET=(--python-platform aarch64-apple-darwin --python-version 3.12)

echo "$ENSM" > "$TMP/app.in"
uv pip compile --quiet "$ROOT/pyproject.toml" "$TMP/app.in" --extra app -c "$TMP/pins.txt" "${TARGET[@]}" \
  --generate-hashes --no-header --no-annotate -o "$TMP/release.txt"
# the same resolution without the hashes
sed -e '/^[[:space:]]*--hash=/d' -e 's/[[:space:]]*\\$//' "$TMP/release.txt" > "$TMP/app.txt"

printf '%s\npytest\npytest-asyncio\n' "$ENSM" > "$TMP/dev.in"
uv pip compile --quiet "$ROOT/pyproject.toml" "$TMP/dev.in" --extra app --extra train -c "$TMP/pins.txt" \
  "${TARGET[@]}" --no-header --no-annotate -o "$TMP/dev-all.txt"
grep -v -x -F -f "$TMP/app.txt" "$TMP/dev-all.txt" > "$TMP/dev-only.txt" || true

# every pin must come from the pins given: nothing resolved fresh from PyPI
missing="$(grep -v -x -F -f "$TMP/pins.txt" "$TMP/dev-all.txt" || true)"
[ -z "$missing" ] || { printf 'not in the pins (install it into .venv first):\n%s\n' "$missing" >&2; exit 1; }

{
  echo "# Exact versions NEO is tested with: what NEO.app's runtime needs (neo2[app]) on macOS 14+ on"
  echo "# Apple silicon. scripts/install-app.sh builds the runtime from this file; the tests and"
  echo "# training extras are in requirements-dev.lock. Regenerate after changing dependencies:"
  echo "# scripts/lock-deps.sh"
  cat "$TMP/app.txt"
} > "$TMP/requirements.lock"
{
  echo "# The dev environment: NEO.app's runtime (requirements.lock), plus the tests and Laya's"
  echo "# training rows (the train extra). uv pip install -r requirements-dev.lock && uv pip install"
  echo "# -e \".[app,dev]\". Regenerate: scripts/lock-deps.sh"
  echo "-r requirements.lock"
  cat "$TMP/dev-only.txt"
} > "$TMP/requirements-dev.lock"
{
  echo "# requirements.lock with the SHA-256 of every file PyPI (or the URL) serves for each pin: what"
  echo "# the release engine installs, with uv pip install --require-hashes (scripts/build_engine.sh)."
  echo "# Never edit by hand: scripts/lock-deps.sh"
  cat "$TMP/release.txt"
} > "$TMP/requirements.release.txt"

mv "$TMP/requirements.lock" "$ROOT/requirements.lock"
mv "$TMP/requirements-dev.lock" "$ROOT/requirements-dev.lock"
mv "$TMP/requirements.release.txt" "$ROOT/requirements.release.txt"
n() { grep -c -E '^[a-z0-9]' "$1"; }
echo "wrote requirements.lock ($(n "$ROOT/requirements.lock") packages)," \
  "requirements-dev.lock (+$(n "$TMP/dev-only.txt")), requirements.release.txt (hashed)"
