#!/usr/bin/env bash
# Remove NEO from this Mac the way the app's own menu › Uninstall NEO… does (ui/src-tauri/src/uninstall.rs):
# everything goes to the Trash, so it can still be put back, and nothing but NEO's own.
#
#   scripts/uninstall-app.sh                NEO.app, its login item, caches and web data; its engine
#                                           (Application Support/NEO: NotchHub's NEO page goes with it);
#                                           NEO's models (its own Hugging Face repos, ~/.neo/models)
#   scripts/uninstall-app.sh --purge        …and "My NEO data" too: ~/.neo (keys, memory, logs)
#     --keep-models                         leave the models
#     --chromium                            also Playwright's Chromium (other tools may share it)
#     --reset-permissions                   also `tccutil reset All com.arnav.neo` (NEO's own grants only)
#
# ~/Documents/NEO Notes is never touched, nor the rest of the Hugging Face cache.
set -euo pipefail
BUNDLE_ID="com.arnav.neo"
PURGE=0 MODELS=1 CHROMIUM=0 RESET=0
for a in "$@"; do
  case "$a" in
    --purge) PURGE=1 ;;
    --keep-models) MODELS=0 ;;
    --chromium) CHROMIUM=1 ;;
    --reset-permissions) RESET=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "uninstall-app.sh: unknown option $a" >&2; exit 2 ;;
  esac
done
STAMP="$(date +%Y%m%d-%H%M%S)"
trash() {
  local p="$1"
  [ -e "$p" ] || [ -L "$p" ] || return 0
  mkdir -p "$HOME/.Trash"
  mv "$p" "$HOME/.Trash/$(basename "$p") (NEO $STAMP)"
  echo "  → Trash: ${p/#$HOME/~}"
}

launchctl bootout "gui/$(id -u)/$BUNDLE_ID" 2>/dev/null || true
pkill -TERM -f "/Applications/NEO.app/Contents/MacOS/" 2>/dev/null || true
sleep 2

echo "Moving NEO to the Trash:"
trash /Applications/NEO.app
trash "$HOME/Library/LaunchAgents/$BUNDLE_ID.plist"
trash "$HOME/Library/Caches/$BUNDLE_ID"
trash "$HOME/Library/WebKit/$BUNDLE_ID"
trash "$HOME/Library/Application Support/NEO"
if [ "$MODELS" = 1 ]; then
  HUB="$HOME/.cache/huggingface/hub"
  for repo in convaiinnovations/laya sentence-transformers/all-MiniLM-L6-v2 mlx-community/parakeet-tdt-0.6b-v3 \
              mlx-community/Kokoro-82M-bf16 prince-canuma/Kokoro-82M mlx-community/gemma-4-12B-it-4bit; do
    trash "$HUB/models--${repo//\//--}"
  done
  trash "$HOME/.neo/models"
fi
[ "$CHROMIUM" = 1 ] && trash "$HOME/Library/Caches/ms-playwright"
[ "$PURGE" = 1 ] && trash "$HOME/.neo"
if [ "$RESET" = 1 ]; then
  tccutil reset All "$BUNDLE_ID" && echo "  NEO's permissions are reset."
else
  echo "NEO's permissions stay listed in System Settings › Privacy & Security (or: tccutil reset All $BUNDLE_ID)."
fi
[ "$PURGE" = 1 ] || echo "Your keys, memory and logs in ~/.neo stay (--purge moves them to the Trash too)."
echo "Empty the Trash to free the space."
