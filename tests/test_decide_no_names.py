"""The rule for NEO's decision code: no app names, bundle ids or websites, only evidence.

Scans the new decision modules' whole source (docstrings and comments included), lower-cased,
with word-bounded patterns. ``Task.signals`` and ``_signal`` are engine vocabulary, so the word
"signal" alone is allowed. The allowlist holds exact lines only, per file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_DECIDE = (
    "evidence", "run", "aescript", "axkit", "eventkit", "routes", "people", "workspace", "when", "learned", "sites",
    "shadow",
)  # fmt: skip
SCOPE = sorted(
    {ROOT / "neo" / "decide" / f"{m}.py" for m in _DECIDE}
    | set((ROOT / "neo" / "decide" / "kinds").glob("*.py"))
    | ({ROOT / "neo" / "tools" / "tasks.py"} if (ROOT / "neo" / "tools" / "tasks.py").exists() else set())
)
FORBIDDEN = [
    re.compile(
        r"\b(spotify|itunes|apple music|youtube|safari|chrome|firefox|whatsapp|telegram|slack|discord|gmail|"
        r"outlook|facetime|amazon|flipkart|ebay|google|obsidian|notion|fantastical)\b"
    ),
    re.compile(r"org\.whispersystems"),
    re.compile(r"signal-desktop"),
    re.compile(r"com\.apple\."),
    re.compile(r'tell application "'),
    re.compile(r"\bwww\.[a-z]"),
]
ALLOW: dict[str, frozenset[str]] = {}  # "neo/tools/web.py": frozenset({exact stripped line, ...}) from R6


def _hits(path: Path, root: Path = ROOT) -> list[str]:
    rel = str(path.relative_to(root))
    allowed = ALLOW.get(rel, frozenset())
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip() in allowed:
            continue
        low = line.lower()
        out += [f"{rel}:{n}: {line.strip()}" for rx in FORBIDDEN if rx.search(low)]
    return out


def test_every_module_in_scope_exists():
    missing = [str(p.relative_to(ROOT)) for p in SCOPE if not p.exists()]
    assert not missing and len(SCOPE) >= len(_DECIDE) + 1


@pytest.mark.parametrize("path", SCOPE, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_app_or_site_names(path):
    assert _hits(path) == []


def test_the_guard_catches_names_and_spares_engine_words(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "X = 'com.apple.Notes'\n# prefer Signal-Desktop\nS = 'tell application \"Mail\"'\n"
        "U = 'https://www.example.test'\n# on YouTube\nsignals = ('handles:sms',)\n# chromium web tree\n"
    )
    assert [h.split(":")[1] for h in _hits(bad, tmp_path)] == ["1", "2", "3", "4", "5"]
