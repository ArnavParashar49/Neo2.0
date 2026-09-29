"""NEO.app's song card on the UI side: it never loads a remote image (a cover is a data URL the core
fetched, checked in one place), never opens a link itself (it asks the core, which opens only the
newest result's https links), and its scripted mock core stays out of release builds. The card's
logic has its own tests under ui/tests (Node's test runner: `npm test` in ui/)."""

import re
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "ui"
SONG = UI / "src" / "song"


def _sources(root: Path) -> dict[Path, str]:
    return {p: p.read_text() for p in root.rglob("*") if p.suffix in (".ts", ".tsx")}


def test_covers_are_checked_data_urls_of_raster_images():
    state = (SONG / "state.ts").read_text()
    pattern = re.search(r"const COVER = /(.+)/;", state).group(1)
    assert pattern.startswith(r"^data:image\/(?:") and "svg" not in pattern and pattern.endswith("$")
    # The card's images are the checked cover and nothing else.
    card = (SONG / "SongCard.tsx").read_text()
    assert set(re.findall(r"src=\{([^}]+)\}", card)) == {"r.cover"}


def test_the_card_never_opens_links_or_urls_itself():
    for path, src in _sources(SONG).items():
        if "mock" in path.parts:
            continue
        assert "openUrl" not in src and "window.open" not in src and "fetch(" not in src, path.name
    hub = (SONG / "hub.ts").read_text()
    assert 'cmd: "song_link", sid, which' in hub  # the core opens it, by kind, never by URL


def test_the_mock_core_is_dev_only():
    ws = (UI / "src" / "ws.ts").read_text()
    assert 'import.meta.env.DEV && SONG_MOCK) {\n        const { SongMockSocket } = await import("./song/mock/core")' in ws
    assert re.search(r'const MOCK_KIND = import\.meta\.env\.DEV \?', ws)
    app = (UI / "src" / "App.tsx").read_text()
    assert 'import.meta.env.DEV && SONG_MOCK ? lazy(() => import("./song/mock/Hud"))' in app
    for path, src in _sources(UI / "src").items():
        if "mock" not in path.parts and path.name not in ("ws.ts", "App.tsx"):
            assert "song/mock" not in src, path.name
    for path, src in _sources(SONG).items():
        if "mock" not in path.parts:
            assert "./mock/" not in src, path.name


def test_the_card_logic_has_its_tests():
    tests = (UI / "tests" / "song-state.test.ts").read_text()
    assert 'from "../src/song/state.ts"' in tests
    assert '"test": "node --test' in (UI / "package.json").read_text()
