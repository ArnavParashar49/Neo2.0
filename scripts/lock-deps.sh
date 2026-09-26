#!/usr/bin/env bash
# Pin the exact versions in the dev environment (the ones the tests just passed with).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
{
  echo "# Exact versions NEO is tested with. The installer (scripts/install-app.sh) builds NEO's"
  echo "# runtime from this file. Regenerate after changing dependencies: scripts/lock-deps.sh"
  uv pip freeze --python "$ROOT/.venv/bin/python" | grep -v -E "^(-e |neo==|pytest|pytest-asyncio|ruff|mypy)"
} > "$ROOT/requirements.lock"
echo "wrote requirements.lock ($(grep -vc '^#' "$ROOT/requirements.lock") packages)"
