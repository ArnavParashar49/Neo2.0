#!/usr/bin/env bash
# Once, by hand, on the Mac that builds NEO's releases: make the certificate NEO is signed with.
# No build runs this; scripts/sign_app.sh (install-app.sh, build_release.sh) uses the certificate
# whenever NEO_SIGN_IDENTITY names it.
#
#   scripts/make_signing_cert.sh [--name "NEO Code Signing"] [--backup DIR] [--keychain FILE] [--trust]
#
# What it does:
#   1. makes a self-signed code-signing certificate with a new RSA-3072 key (openssl; valid 10 years;
#      key usage: digital signature; extended key usage: code signing; not a CA);
#   2. packs the two into a password-protected .p12 (the PKCS#12 encryption macOS's `security` reads:
#      openssl 3 needs -legacy for it, LibreSSL's /usr/bin/openssl writes it by default) and keeps the
#      .p12, its password and the certificate in the backup folder (default ~/NEO-signing, yours
#      only). The plain key file is deleted. BACK THAT FOLDER UP somewhere safe and private: with a
#      new key, every NEO user has to grant NEO's permissions again, once;
#   3. imports the .p12 into your login keychain, letting /usr/bin/codesign use the key;
#   4. --trust only: marks the certificate trusted for code signing in your login keychain (macOS asks
#      for your password). Not needed for NEO's permissions; only if codesign refuses the identity;
#   5. signs a scratch copy of /usr/bin/true with it, and prints the designated requirement every NEO
#      build will carry, and the line for your shell profile:  export NEO_SIGN_IDENTITY="<name>"
#
# Why: an ad-hoc signature identifies NEO by its bundle id alone, so any ad-hoc app calling itself
# com.arnav.neo would inherit NEO's permissions. Signed with this certificate, the designated
# requirement is `identifier "com.arnav.neo" and certificate leaf = H"<its SHA-1>"`: only builds
# signed with this key qualify, and every release qualifies, so permissions survive updates. macOS
# checks the certificate's hash, not whether Apple vouches for it, so it works without a Developer ID;
# Gatekeeper's Open Anyway step for downloaded copies stays as it is. Nothing is uploaded anywhere.
# After you switch, macOS asks once more for NEO's permissions on your own Mac.
set -euo pipefail
NAME="NEO Code Signing"
BACKUP="$HOME/NEO-signing"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"
TRUST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --backup) BACKUP="$2"; shift 2 ;;
    --keychain) KEYCHAIN="$2"; shift 2 ;;
    --trust) TRUST=1; shift ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "make_signing_cert.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
die() { echo "make_signing_cert.sh: $*" >&2; exit 1; }
NAME_RE='^[A-Za-z0-9 ._()-]{3,64}$'
[[ "$NAME" =~ $NAME_RE ]] || die "--name: letters, digits, spaces and ._()- only (3–64)"
[ "$(uname -s)" = Darwin ] || die "this is for macOS"
[ -f "$KEYCHAIN" ] || die "no keychain at $KEYCHAIN"
if security find-identity -p codesigning "$KEYCHAIN" 2>/dev/null | grep -q -F "\"$NAME\""; then
  die "your keychain already has an identity named \"$NAME\": keep using it (export NEO_SIGN_IDENTITY=\"$NAME\"), or pick another --name"
fi
if [ -e "$BACKUP/neo-signing.p12" ]; then
  die "$BACKUP/neo-signing.p12 already exists: that is a key you made before. Import it instead of making a new one: security import \"$BACKUP/neo-signing.p12\" -k \"$KEYCHAIN\" -T /usr/bin/codesign"
fi

OPENSSL="/usr/bin/openssl"
command -v "$OPENSSL" >/dev/null || OPENSSL="$(command -v openssl || true)"
[ -n "$OPENSSL" ] || die "openssl isn't installed"
LEGACY=()
case "$("$OPENSSL" version)" in OpenSSL\ [3-9]*) LEGACY=(-legacy) ;; esac   # LibreSSL: legacy by default

umask 077
WORK="$(mktemp -d "${TMPDIR:-/tmp}/neo-signing.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$BACKUP"
chmod 700 "$BACKUP"

cat > "$WORK/cert.cnf" <<EOF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $NAME
[ext]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
subjectKeyIdentifier = hash
EOF

echo "→ making the certificate \"$NAME\" (RSA-3072, 10 years)"
"$OPENSSL" req -x509 -newkey rsa:3072 -sha256 -days 3650 -nodes -config "$WORK/cert.cnf" \
  -keyout "$WORK/key.pem" -out "$WORK/cert.pem" >/dev/null
"$OPENSSL" rand -base64 24 | tr -d '\n' > "$WORK/p12-password"
"$OPENSSL" pkcs12 -export ${LEGACY[@]+"${LEGACY[@]}"} -inkey "$WORK/key.pem" -in "$WORK/cert.pem" -name "$NAME" \
  -out "$WORK/neo-signing.p12" -passout "file:$WORK/p12-password"
rm -f "$WORK/key.pem"
SHA1="$("$OPENSSL" x509 -in "$WORK/cert.pem" -noout -fingerprint -sha1 | sed 's/^.*=//' | tr -d ':' | tr 'a-f' 'A-F')"
[[ "$SHA1" =~ ^[0-9A-F]{40}$ ]] || die "couldn't read the certificate's SHA-1"

cp "$WORK/neo-signing.p12" "$WORK/p12-password" "$WORK/cert.pem" "$BACKUP/"
chmod 600 "$BACKUP/neo-signing.p12" "$BACKUP/p12-password" "$BACKUP/cert.pem"
cat > "$BACKUP/README.txt" <<EOF
NEO's code-signing identity: "$NAME", SHA-1 $SHA1, made $(date -u +%Y-%m-%d), valid 10 years.
neo-signing.p12 is the certificate and its private key, encrypted with the password in p12-password.
Keep a copy of this folder somewhere safe and private (an encrypted backup, a password manager).
To use it on another Mac:  security import neo-signing.p12 -k ~/Library/Keychains/login.keychain-db -T /usr/bin/codesign
Builds use it when your shell has:  export NEO_SIGN_IDENTITY="$NAME"
Every NEO build signed with it carries the designated requirement:
  identifier "com.arnav.neo" and certificate leaf = H"$SHA1"
EOF

echo "→ importing it into $(basename "$KEYCHAIN") (codesign may use the key)"
security import "$WORK/neo-signing.p12" -k "$KEYCHAIN" -f pkcs12 -P "$(cat "$WORK/p12-password")" -T /usr/bin/codesign >/dev/null
if [ "$TRUST" = 1 ]; then
  echo "→ trusting it for code signing (macOS asks for your password)"
  security add-trusted-cert -r trustRoot -p codeSign -k "$KEYCHAIN" "$WORK/cert.pem"
fi

echo "→ signing a test file with it (macOS may ask to let codesign use the key: Always Allow)"
cp /usr/bin/true "$WORK/neo-signing-test"
if codesign --force --sign "$SHA1" --identifier com.arnav.neo.signing-test --timestamp=none "$WORK/neo-signing-test" 2>"$WORK/codesign.err" \
   && codesign --verify "$WORK/neo-signing-test" 2>>"$WORK/codesign.err"; then
  echo "  signed: $(codesign -d -r- "$WORK/neo-signing-test" 2>&1 | grep '^designated')"
else
  cat "$WORK/codesign.err" >&2
  echo "  ! codesign couldn't use it yet: unlock the login keychain, or run again with --trust (after deleting the identity in Keychain Access, or with another --name)" >&2
  exit 1
fi

cat <<EOF

Done. NEO's signing identity is "$NAME" (SHA-1 $SHA1).
  backup:  $BACKUP  (neo-signing.p12 + p12-password: copy it somewhere safe now)
  builds:  add this to your shell profile (~/.zprofile), then open a new terminal:
             export NEO_SIGN_IDENTITY="$NAME"
           scripts/build_release.sh and scripts/install-app.sh then sign with it, and every build carries
             designated => identifier "com.arnav.neo" and certificate leaf = H"$SHA1"
After the first build signed with it, macOS asks for NEO's permissions once more on this Mac.
EOF
