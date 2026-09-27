"""What the controls test window holds, shared by the window (testwin) and the test (controls).

Positions are the window's own business: it reports every target's frame on screen when it is
ready, and the test clicks where the window says. Everything here is plain data.

No label here may read as a commitment ("Send", "OK", "Delete"…: neo2.core.tools.computer.policy
.commits), or NEO's click gate would ask the user before pressing it.
"""

from __future__ import annotations

APP_NAME = "NEO Self-Test"  # + a short random tag, so the name is this run's alone
TITLE = "NEO controls self-test - please don't touch the mouse or keyboard"

WIDTH, HEIGHT = 720, 330  # the window's content, points
MARGIN = 40  # from the top-left of the main display's visible area (NEO's own overlay is bottom-right)

BOARD_N = 8  # a Chess-like board: 8×8 cells drawn in one view, nothing in it for Accessibility
CELL = 30  # points

BUTTON_A = "Alpha"  # clicked by its ax_tree id
BUTTON_B = "Bravo"  # pressed with ax_press
FIELD_HINT = "Type here"
MENU_TITLE = "Test Action"
MENU_KEYS = "cmd+j"  # the Test menu's shortcut; plain ⌘+letter chords are never global shortcuts
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
)  # fmt: skip
