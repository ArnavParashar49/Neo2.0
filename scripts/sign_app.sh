#!/usr/bin/env bash
# Sign a built NEO.app the way every NEO build is signed, so the permissions macOS granted NEO
# (Accessibility, Screen Recording, the microphone…) stay granted across builds and updates.
#
#   scripts/sign_app.sh <NEO.app> [--identifier com.arnav.neo]
#
# With NEO_SIGN_IDENTITY set (the certificate scripts/make_signing_cert.sh made, by its name or its
# SHA-1): signed with it, and the designated requirement — what macOS checks a later build against —
# is the bundle id and that certificate:
#     identifier "com.arnav.neo" and certificate leaf = H"<the certificate's SHA-1>"
# Without it: ad-hoc, with the bundle id alone (what every build did before 2.2):
#     identifier "com.arnav.neo"
# No hardened runtime and no entitlements, as before. Then `codesign --verify --strict`, and the
# requirement is checked. Used by scripts/install-app.sh and scripts/build_release.sh. Without
# NEO_SIGN_IDENTITY it never looks in the keychain.
set -euo pipefail
APP="" ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --identifier) ID="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    -*) echo "sign_app.sh: unknown option $1" >&2; exit 2 ;;
    *) APP="$1"; shift ;;
  esac
done
[ -n "$APP" ] && [ -d "$APP/Contents" ] || { echo "sign_app.sh: give it the app (…/NEO.app)" >&2; exit 2; }
[ -n "$ID" ] || ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP/Contents/Info.plist")"
[[ "$ID" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "sign_app.sh: not a bundle id: $ID" >&2; exit 2; }

if [ -n "${NEO_SIGN_IDENTITY:-}" ]; then
  # The certificate's SHA-1: given as such, or looked up by name (an untrusted self-signed one is
  # still listed without -v; trust isn't what the requirement checks).
  if [[ "$NEO_SIGN_IDENTITY" =~ ^[0-9A-Fa-f]{40}$ ]]; then
    HASH="$NEO_SIGN_IDENTITY"
  else
    HASHES="$(security find-identity -p codesigning 2>/dev/null | awk -v n="\"$NEO_SIGN_IDENTITY\"" 'index($0, n) && $2 ~ /^[0-9A-F]{40}$/ {print $2}' | sort -u)"
    case "$(printf '%s\n' "$HASHES" | grep -c .)" in
      1) HASH="$HASHES" ;;
      0) echo "sign_app.sh: no code-signing identity named \"$NEO_SIGN_IDENTITY\" in your keychains (scripts/make_signing_cert.sh)" >&2; exit 1 ;;
      *) echo "sign_app.sh: several identities are named \"$NEO_SIGN_IDENTITY\"; set NEO_SIGN_IDENTITY to one SHA-1:" >&2; echo "$HASHES" >&2; exit 1 ;;
    esac
  fi
  HASH="$(printf '%s' "$HASH" | tr 'a-f' 'A-F')"
  REQ="identifier \"$ID\" and certificate leaf = H\"$HASH\""
  echo "→ signing $(basename "$APP") with certificate $HASH"
  codesign --force --sign "$HASH" --identifier "$ID" --timestamp=none -r="designated => $REQ" "$APP"
else
  REQ="identifier \"$ID\""
  echo "→ signing $(basename "$APP") ad-hoc (NEO_SIGN_IDENTITY isn't set)"
  codesign --force --sign - --identifier "$ID" -r="designated => $REQ" "$APP"
fi
codesign --verify --strict "$APP"
got="$(codesign -d -r- "$APP" 2>&1 | sed -n 's/^designated => //p')"
norm() { printf '%s' "$1" | tr 'a-f' 'A-F' | tr -d ' '; }
[ "$(norm "$got")" = "$(norm "$REQ")" ] || { echo "sign_app.sh: the designated requirement is [$got], not [$REQ]" >&2; exit 1; }
echo "  designated => $got"
