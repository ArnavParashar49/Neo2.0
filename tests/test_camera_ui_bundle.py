"""NEO.app's camera plumbing on the UI side: macOS asks for the camera once (with NEO's reason),
the pop-out camera window has no powers beyond showing itself, and the card's frame kind stays
distinct from the browser's."""

import json
import plistlib
import re
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "ui"


def test_info_plist_explains_the_camera():
    info = plistlib.loads((UI / "src-tauri" / "Info.plist").read_bytes())
    why = info.get("NSCameraUsageDescription", "")
    assert "camera" in why.lower() and "ask" in why.lower()


def test_camera_window_capability_is_minimal():
    cap = json.loads((UI / "src-tauri" / "capabilities" / "camera.json").read_text())
    assert cap["windows"] == ["camera"]
    perms = set(cap["permissions"])
    assert not any("webview" in p or "opener" in p or "shortcut" in p for p in perms)


def test_camera_frame_kind_is_not_a_browser_kind():
    src = (UI / "src" / "camera" / "frames.ts").read_text()
    kind = int(re.search(r"KIND_CAMERA = (\d+)", src).group(1))
    assert kind not in (1, 2)  # browser: 1 = live page, 2 = still
