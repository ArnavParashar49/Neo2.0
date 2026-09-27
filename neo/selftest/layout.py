"""What the controls test window holds, shared by the window (testwin) and the test (controls).

Positions are the window's own business: it reports every target's frame on screen when it is
ready, and the test clicks where the window says. Everything here is plain data.

No label here may read as a commitment ("Send", "OK", "Delete"…: neo.tools.computer.policy
.commits), or NEO's click gate would ask the user before pressing it.
"""

from __future__ import annotations

APP_NAME = "NEO Self-Test"  # + a short random tag, so the name is this run's alone
TITLE = "NEO controls self-test - please don't touch the mouse or keyboard"

WIDTH, HEIGHT = 1000, 380  # the window's content, points
MARGIN = 40  # from the top-left of the main display's visible area (NEO's own overlay is bottom-right)

BOARD_N = 8  # a Chess-like board: 8×8 cells drawn in one view, nothing in it for Accessibility
CELL = 30  # points
# Pieces on it, as a game starts (white at the bottom); the one at PIECE is the drag check's own.
START_ROWS = {0: "rnbqkbnr", 1: "pppppppp", 6: "PPPPPPPP", 7: "RNBQKBNR"}
PIECE = (1, 6)  # b2
# A second board, tilted back and textured like a 3D board seen at an angle (the default Chess
# board is one): its squares are only in its pixels, in perspective. Top-left x, y and size.
TILT = (740, 40, 240, 240)

BUTTON_A = "Alpha"  # clicked by its ax_tree id
BUTTON_B = "Bravo"  # pressed with ax_press
FIELD_HINT = "Type here"
MENU_TITLE = "Test Action"
MENU_KEYS = "cmd+j"  # the Test menu's shortcut; plain ⌘+letter chords are never global shortcuts
ROUND_MENU = "New Round…"  # Test › New Round… opens a sheet on the window, as a game's New Game does
ROUND_KEYS = "cmd+k"
SHEET_TITLE = "Start a new round?"
SHEET_START, SHEET_CANCEL = "Start", "Cancel"
HINDI_BUTTON = "गुलाबी"  # a button titled in Hindi ("pink"; it is pink)
HINDI_PHRASES = {  # asking for it by meaning: English, Hinglish, Hindi (the resolver's stages)
    "en": "press the pink button",
    "hinglish": "gulabi wala button dabao",
    "hi": "गुलाबी बटन दबाओ",
}
TINY_PT = 12  # a button this small: zoom, then click it
TINY_RGB = (0, 200, 220)
DRAG_PAYLOAD = "neo-selftest-drop"
LIST_ROWS = 60
ROW_H = 24

# Typed into the single-line field: Hindi and emoji, with a run of emoji long enough that a
# 20-character chunk of it is well over 20 UTF-16 units (what one keyboard event can carry).
TYPE_TEXT = "Hi नमस्ते 👋🏽🎉🚀🌟🙂🔥🍕🎈🎯🧪 ok"
AREA_TEXT = "first line\nsecond line"  # typed into the text area: the line break must be a real one
SET_TEXT = "set by Accessibility ✓\nline two"  # ax_set_value on the text area

MARKER_PT = 36  # a magenta square screenshots are searched for
MARKER_RGB = (255, 0, 255)

# Targets the window reports, by name.
TARGETS = (
    "board", "double", "button_a", "button_b", "field", "area", "list", "marker", "tile", "well",
    "hindi", "tiny", "tilt",
)  # fmt: skip
