#!/usr/bin/env bash
# Remove NEO.app and its login item. Your data in ~/.neo stays unless you pass --purge, which
# moves it to the Trash (so it can still be recovered).
set -euo pipefail
BUNDLE_ID="com.arnav.neo"
launchctl bootout "gui/$(id -u)/$BUNDLE_ID" 2>/dev/null || true
pkill -TERM -f "/Applications/NEO.app/Contents/MacOS/" 2>/dev/null || true
sleep 2
rm -f "$HOME/Library/LaunchAgents/$BUNDLE_ID.plist"
rm -rf /Applications/NEO.app
echo "NEO.app and its login item are removed."
echo "Its permissions: remove NEO under System Settings → Privacy & Security → Accessibility / Screen Recording (or: tccutil reset All $BUNDLE_ID)."
if [ "${1:-}" = "--purge" ] && [ -d "$HOME/.neo" ]; then
  mv "$HOME/.neo" "$HOME/.Trash/neo-data-$(date +%Y%m%d-%H%M%S)"
  echo "~/.neo (memory, logs, models, learned examples) was moved to the Trash."
fi
