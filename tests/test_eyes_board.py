"""The board helper (neo/tools/computer/board.py) on synthetic screens: flat boards and grids, boards
seen at an angle (textured wood, 3D pieces, shadows), square names, pieces, changes after a move, and
the rules the pointer tools rely on (a blind brain gets square names only on a board NEO found
itself in a fresh picture; nothing it reads carries coordinates). Nothing here touches the Mac."""

from __future__ import annotations

import ast
import functools
import io
import re
import time
from pathlib import Path

import numpy as np
import pytest
from eyes_board_synth import (
    START,
    board_to_image,
    flat_screen,
    moved,
    thin_grid_screen,
    tilted_screen,
    truth_occupancy,
)
from PIL import Image, ImageDraw

from neo.tools.computer import board as B

CHESS = [f + r for f in "abcdefgh" for r in "12345678"]


@pytest.fixture(autouse=True)
def _fresh_boards():
    B.forget()
    yield
    B.forget()


@functools.cache
def _flat(**kw):
    return flat_screen(**kw)


@functools.cache
def _tilted(**kw):
    return tilted_screen(**kw)


def _truth_centre(tr, name: str, white_at_bottom: bool = True) -> tuple[float, float]:
    f, r = "abcdefgh".index(name[0]), int(name[1])
    col, row = (f, 8 - r) if white_at_bottom else (7 - f, r - 1)
    return tuple(board_to_image(tr["H"], [(col + 0.5, row + 0.5)])[0])


def _worst(b: B.Board, tr, k: float = 1.0) -> float:
    """The largest distance between a square's centre as NEO maps it and the truth (pixels)."""
    H = np.diag([k, k, 1.0]) @ tr["H"]
    worst = 0.0
    for r in range(tr["rows"]):
        for c in range(tr["cols"]):
            tx, ty = board_to_image(H, [(c + 0.5, r + 0.5)])[0]
            x, y = b.centre(c, r)
            worst = max(worst, abs(x - tx), abs(y - ty))
    return worst


def _occ_right(b: B.Board, tr) -> int:
    """How many squares' occupancy matches the truth (by where each square's centre falls)."""
    truth = truth_occupancy(tr["position"])
    tc = board_to_image(tr["H"], [(c + 0.5, r + 0.5) for r in range(8) for c in range(8)])
    right = 0
    for r in range(b.rows):
        for c in range(b.cols):
            k = int(np.argmin(np.linalg.norm(tc - b.centre(c, r), axis=1)))
            right += b.occupancy[r][c] == truth[k // 8][k % 8]
    return right


# ---- flat boards ------------------------------------------------------------------------------

FLAT = [
    dict(board_at=(420, 150), sq_pt=76),
    dict(board_at=(380, 130), sq_pt=90),
    dict(board_at=(500, 200), sq_pt=52),
    dict(board_at=(450, 170), sq_pt=64),
    dict(board_at=(430, 160), sq_pt=70, light=(240, 217, 181), dark=(181, 136, 99)),
    dict(board_at=(420, 150), sq_pt=76, highlight=((4, 6), (4, 4))),
    dict(board_at=(600, 300), sq_pt=30),
    dict(board_at=(420, 150), sq_pt=60, n=10),
]


@pytest.mark.parametrize("kw", FLAT, ids=lambda kw: f"{kw['board_at']}-{kw['sq_pt']}pt-{kw.get('n', 8)}")
def test_every_flat_board_is_found_with_its_squares_exactly_where_they_are(kw):
    img, tr = _flat(**kw)
    s = B.survey(B.Picture.of(img), shot="s1")
    b = s.board
    assert b is not None and (b.rows, b.cols) == (tr["rows"], tr["cols"])
    assert b.flat and not b.lines and b.source == "neo"
    assert _worst(b, tr) <= 0.5


def test_a_resized_noisy_jpeg_screenshot_still_maps_every_square():
    img, tr = _flat(board_at=(420, 150), sq_pt=76)
    k = 1280 / 1512
    small = img.resize((1280, round(982 * k)), Image.LANCZOS)
    b = B.survey(B.Picture.of(small, ppp=1 / k)).board  # (points per pixel: the Shot's scale)
    assert b is not None and _worst(b, tr) <= 1.2  # in points: the picture's pixels are mapped back
    a = np.asarray(img).astype(np.int16) + np.random.default_rng(0).normal(0, 6, (982, 1512, 3)).astype(np.int16)
    buf = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=60)
    b = B.survey(B.Picture.of(Image.open(buf).convert("RGB"))).board
    assert b is not None and _worst(b, tr) <= 1.2


def test_a_board_under_a_sheet_is_not_read_and_the_note_says_why():
    img, _ = _flat(sheet=True)
    assert B.find_grid(B.Picture.of(img).gray) is None
    clear, _ = _flat()
    s = B.survey(B.Picture.of(clear), sheet=True)  # Accessibility says a sheet is open: nothing is read
    assert s.covered and s.board is None
    assert "answer the sheet first" in B.covered_note()


def test_pieces_are_read_on_every_square_and_orientation_follows_them():
    img, tr = _flat()
    b = B.survey(B.Picture.of(img)).board
    assert b.occupancy == truth_occupancy(START)
    assert b.orientation == "bottom" and b.oriented_by == "pieces" and b.chess
    x, y = _truth_centre(tr, "e2")
    assert np.allclose(b.centre(*b.parse("e2")), (x, y), atol=0.5)
    assert b.name(0, 7) == "a1" and b.name(7, 0) == "h8" and b.names() == "a1–h8"


def test_a_board_seen_from_blacks_side_names_its_squares_from_there():
    flipped = tuple(row[::-1] for row in reversed(START))
    img, tr = _flat(position=flipped)
    b = B.survey(B.Picture.of(img)).board
    assert b.orientation == "top"
    assert b.name(3, 0) == "e1" and b.name(7, 0) == "a1" and b.name(0, 7) == "h8"
    assert np.allclose(b.centre(*b.parse("e1")), _truth_centre(tr, "e1", white_at_bottom=False), atol=0.5)


def test_without_a_known_orientation_squares_have_row_and_column_names_only():
    img, _ = _flat(position=("." * 8,) * 8)
    b = B.survey(B.Picture.of(img), shot="s1").board
    assert not b.chess and b.orientation == ""
    assert b.name(0, 0) == "r1c1" and b.parse("r8c8") == (7, 7) and b.parse("R2C3") == (2, 1)
    with pytest.raises(ValueError, match="no chess names"):
        b.parse("e4")
    with pytest.raises(ValueError, match="off the board"):
        b.parse("r9c1")
    # a model that can see says which side White is on: chess names from then on
    text, ok = B.map_board({"shot": "s1", "orientation": "bottom"})
    assert ok and b.chess and b.oriented_by == "model" and "as the model said" in text
    assert b.parse("a1") == (0, 7)


def test_a_move_changes_exactly_its_two_squares_and_the_report_says_what_is_where():
    img, tr = _flat()
    b = B.survey(B.Picture.of(img), shot="s1").board
    after, _ = _flat(position=moved(START, "e2", "e4"))
    m = B.after_move(b, B.Picture.of(after), "e2", "e4")
    assert m.changed is True and sorted(m.cells) == ["e2", "e4"]
    assert m.said == "On the board: e2 is now empty; e4 has a light piece."
    same = B.after_move(b, B.Picture.of(after), "e7", "e5")  # nothing moved since
    assert same.changed is False and "Nothing changed on the board" in same.said
    covered, _ = _flat(sheet=True)
    gone = B.after_move(b, B.Picture.of(covered), "e7", "e5")
    assert gone.changed is None and "isn't where it was" in gone.said


def test_the_other_sides_reply_shows_as_the_squares_changed_since_the_last_look():
    one, _ = _flat(position=moved(START, "e2", "e4"))
    B.survey(B.Picture.of(one), shot="s1")
    two, _ = _flat(position=moved(moved(START, "e2", "e4"), "g8", "f6"))
    s = B.survey(B.Picture.of(two), shot="s2")
    assert sorted(s.since) == ["f6", "g8"]
    assert "Changed since the last look: " in B.note(s.board, s.since)


def test_a_grid_drawn_with_lines_counts_only_on_a_canvas_and_never_gets_chess_names():
    img, tr = thin_grid_screen()
    pic = B.Picture.of(img)
    assert B.survey(pic).board is None  # a table or a calendar drawn with lines is no board by itself
    b = B.survey(pic, canvas=True).board
    assert b is not None and b.lines and (b.rows, b.cols) == (9, 9)
    assert _worst(b, tr) <= 3.5  # (a line's width off, on 50 pt cells)
    assert not b.chess and b.name(4, 4) == "r5c5"


@pytest.mark.parametrize(
    "what",
    ["blank", "noise", "photo", "transparency", "zebra", "screen"],
)
def test_no_board_is_seen_where_there_is_none(what):
    rng = np.random.default_rng(0)
    if what == "blank":
        a = np.asarray(_flat()[0]).copy()
        a[140:770, 410:1040] = 250
    elif what == "noise":
        a = rng.integers(0, 255, (982, 1512, 3), dtype=np.uint8)
    elif what == "photo":
        x = rng.normal(0, 1, (62, 95, 3))
        a = np.asarray(Image.fromarray(((x - x.min()) / np.ptp(x) * 255).astype(np.uint8)).resize((1512, 982), Image.BICUBIC))
    elif what == "transparency":  # an image editor's checkerboard behind a transparent picture
        a = np.full((982, 1512, 3), 236, np.uint8)
        yy, xx = np.mgrid[200:700, 300:1100]
        a[200:700, 300:1100] = np.where((((yy // 8) + (xx // 8)) % 2 == 0)[..., None], 255, 204)
    elif what == "zebra":  # a table with striped rows
        a = np.full((982, 1512, 3), 255, np.uint8)
        for i in range(1, 20, 2):
            a[200 + i * 30 : 230 + i * 30, 300:1100] = 235
        for c in range(300, 1101, 100):
            a[200:800, c : c + 1] = 180
    else:  # the synthetic screen's menus, sidebar and buttons, without the board
        a = np.asarray(_flat()[0]).copy()
        a[140:770, 410:1040] = 250
        img = Image.fromarray(a)
        ImageDraw.Draw(img).rectangle([500, 300, 900, 600], outline=(0, 0, 0))
        a = np.asarray(img)
    s = B.survey(B.Picture.of(a), canvas=True, shot="s9")
    assert s.board is None
    assert s.hint == "s9"  # (a canvas with no grid: a model that can see may give the corners)


# ---- boards seen at an angle ----------------------------------------------------------------------

TILTED = [(55, 0), (60, 0), (65, 20), (45, -30), (50, 45), (40, 10), (35, 0)]


@pytest.mark.parametrize("elev,yaw", TILTED)
def test_neo_finds_a_tilted_textured_board_by_itself(elev, yaw):
    img, tr = _tilted(elev=elev, yaw=yaw)
    pic = B.Picture.of(img)
    assert B.survey(pic).board is None  # not a flat grid: only looked for with a canvas's evidence
    s = B.survey(pic, canvas=True, shot="s1")
    b = s.board
    assert b is not None and (b.rows, b.cols) == (8, 8)
    assert b.source == "neo" and not b.flat and b.agree >= 0.9
    assert _worst(b, tr) <= 3.0  # squares ≥ 30 px: every click lands deep inside its square
    assert b.orientation == "bottom" and b.name(4, 6) == "e2"
    assert np.allclose(b.centre(*b.parse("e2")), _truth_centre(tr, "e2"), atol=3.0)


@pytest.mark.parametrize("elev,yaw", [(55, 0), (60, 0), (65, 20), (45, -30), (50, 45)])
def test_pieces_on_a_tilted_board_are_read_and_a_move_changes_its_two_squares(elev, yaw):
    img, tr = _tilted(elev=elev, yaw=yaw)
    b = B.survey(B.Picture.of(img), canvas=True).board
    assert _occ_right(b, tr) >= 62
    after, _ = _tilted(elev=elev, yaw=yaw, position=moved(START, "e2", "e4"), seed=8)
    m = B.after_move(b, B.Picture.of(after), "e2", "e4")
    assert sorted(m.cells) == ["e2", "e4"] and m.said.startswith("On the board: e2 is now empty; e4 has a light piece")
    knight, _ = _tilted(elev=elev, yaw=yaw, position=moved(moved(START, "e2", "e4"), "g1", "f3"), seed=9)
    assert sorted(B.after_move(b, B.Picture.of(knight), "g1", "f3").cells) == ["f3", "g1"]


def test_from_a_low_angle_tall_pieces_hide_squares_but_the_move_is_still_seen():
    img, tr = _tilted(elev=40, yaw=10)
    b = B.survey(B.Picture.of(img), canvas=True).board
    assert _occ_right(b, tr) >= 58
    after, _ = _tilted(elev=40, yaw=10, position=moved(START, "e2", "e4"), seed=8)
    m = B.after_move(b, B.Picture.of(after), "e2", "e4")
    assert {"e2", "e4"} <= set(m.cells)
    assert "a tall piece can hide the square behind it" in B.note(b)


@pytest.mark.parametrize("elev,yaw", [(55, 0), (60, 0), (65, 20), (45, -30), (50, 45), (40, 10), (35, 0)])
def test_corners_from_a_seeing_model_are_refined_onto_the_squares(elev, yaw):
    img, tr = _tilted(elev=elev, yaw=yaw)
    pic = B.Picture.of(img)
    true = board_to_image(tr["H"], [(0, 0), (8, 0), (8, 8), (0, 8)])
    rng = np.random.default_rng(elev * 100 + yaw)
    noisy = true + rng.normal(0, 10, true.shape)  # ~0.7% of the screen's width, per corner
    b, why = B.from_corners(pic, noisy[[2, 0, 3, 1]])  # (any order)
    assert b is not None, why
    assert b.source == "vision" and (b.rows, b.cols) == (8, 8)
    assert _worst(b, tr) <= 3.0


def test_wrong_corners_are_refused_with_a_reason():
    img, tr = _tilted(elev=55, yaw=0)
    pic = B.Picture.of(img)
    inner = board_to_image(tr["H"], [(1, 1), (7, 1), (7, 7), (1, 7)])  # a square in from every edge
    b, why = B.from_corners(pic, inner, rows=8, cols=8)
    assert b is None and why
    b, why = B.from_corners(pic, [[40, 40], [340, 40], [340, 260], [40, 260]])  # the window's empty corner
    assert b is None and why
    b, why = B.from_corners(pic, [[10, 10], [12, 10], [12, 12], [10, 12]])
    assert b is None and "almost nothing" in why
    b, why = B.from_corners(pic, [[1, 2], [3, 4]])
    assert b is None and "four corners" in why


def test_map_board_takes_pixels_or_gemini_points_on_the_screenshot_neo_kept():
    img, tr = _tilted(elev=55, yaw=0)
    pic = B.Picture.of(img)
    blank = np.asarray(img).copy()
    blank[:] = 60
    s = B.survey(B.Picture.of(blank), canvas=True, shot="s3")  # nothing found: the picture is kept for corners
    assert s.hint == "s3" and "map_board" in B.hint_note("s3")
    B.forget()
    assert B.survey(pic, canvas=True, shot="s4").board is not None  # (NEO finds this one itself)
    B.forget()
    B._pictures["s5"] = pic  # as survey keeps a canvas it found no grid in
    true = board_to_image(tr["H"], [(0, 0), (8, 0), (8, 8), (0, 8)])
    text, ok = B.map_board({"shot": "s5", "corners": true.tolist()})
    assert ok and "NEO mapped (from the corners given, checked by NEO)" in text and B.latest().source == "vision"
    B.forget()
    B._pictures["s5"] = pic
    yx = [[y / 982 * 1000, x / 1512 * 1000] for x, y in true]
    text, ok = B.map_board({"shot": "s5", "corners_norm": yx})
    assert ok and B.latest().chess
    text, ok = B.map_board({"shot": "s5", "corners": [[0, 0], [5000, 0], [5000, 5000], [0, 5000]]})
    assert not ok and "lie in the screenshot" in text
    text, ok = B.map_board({"shot": "gone", "corners": true.tolist()})
    assert not ok and "no longer has screenshot" in text


# ---- the rules the pointer tools rely on ---------------------------------------------------------


def test_a_blind_brain_gets_squares_only_on_a_board_neo_found_itself():
    img, tr = _flat()
    B.survey(B.Picture.of(img), shot="s1")
    pts, b, names = B.points_for({"cell": "e2"}, ("cell",), blind=True)
    assert names == ["e2"] and np.allclose(pts[0], _truth_centre(tr, "e2"), atol=0.5)
    pts, _, names = B.points_for({"from": "g1", "to": "f3"}, ("from", "to"))
    assert names == ["g1", "f3"] and len(pts) == 2
    b.source = "vision"  # mapped from a seeing model's corners
    with pytest.raises(B.BoardError, match="can't see the screen"):
        B.points_for({"cell": "e2"}, ("cell",), blind=True)
    assert B.points_for({"cell": "e2"}, ("cell",))[2] == ["e2"]  # a seeing brain may


def test_a_blind_brain_can_never_give_corners_or_an_orientation():
    for a in ({"shot": "s1", "corners": [[1, 2]] * 4}, {"shot": "s1", "orientation": "bottom"}, {"corners_norm": [[1, 2]] * 4}):
        text, ok = B.map_board(a, blind=True)
        assert not ok and "can't see the screen" in text


def test_a_stale_board_or_a_moved_window_is_looked_at_again_first():
    img, _ = _flat()
    B.survey(B.Picture.of(img), shot="s1", window=(100, 100, 900, 800), now=1000.0)
    with pytest.raises(B.BoardError, match="take a new screenshot"):
        B.points_for({"cell": "e2"}, ("cell",), now=1000.0 + B.FRESH_S + 1)
    with pytest.raises(B.BoardError, match="window moved"):
        B.points_for({"cell": "e2"}, ("cell",), now=1001.0, window=(140, 100, 900, 800))
    assert B.points_for({"cell": "e2"}, ("cell",), now=1001.0, window=(100, 100, 900, 800))[2] == ["e2"]


def test_no_board_known_or_a_bad_name_is_an_error_not_a_click():
    with pytest.raises(B.BoardError, match="doesn't know a board"):
        B.points_for({"cell": "e2"}, ("cell",))
    B.survey(B.Picture.of(_flat()[0]))
    for bad in ("z9", "", None, "e", "r0c1"):
        with pytest.raises(B.BoardError):
            B.points_for({"cell": bad}, ("cell",))


_COORD = re.compile(r"\d{2,}\s*[,×x]\s*\d{2,}|\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+|\b\d{3,}\b")


def test_nothing_a_blind_brain_reads_about_a_board_carries_coordinates():
    img, _ = _tilted(elev=55, yaw=0)
    s = B.survey(B.Picture.of(img), canvas=True, shot="s1")
    after, _ = _tilted(elev=55, yaw=0, position=moved(START, "e2", "e4"), seed=8)
    texts = [
        B.note(s.board, blind=True),
        B.note(s.board, ["e2", "e4"], blind=True),
        B.after_move(s.board, B.Picture.of(after), "e2", "e4").said,
        B.hint_note("s1", blind=True),
        B.covered_note(),
        B.blind_refusal(),
    ]
    for t in texts:
        assert not _COORD.search(t), t
    assert "corners" not in B.hint_note("s1", blind=True)


def test_the_note_names_squares_and_says_how_to_act_on_them():
    img, _ = _flat()
    s = B.survey(B.Picture.of(img), shot="s1")
    text = B.note(s.board)
    assert "NEO found a board of 8×8 squares" in text and "a1–h8" in text and "White (the light pieces) at the bottom" in text
    assert 'click cell="e2" · drag from="e2" to="e4"' in text
    assert "DDDDDDDD/DDDDDDDD/8/8/8/8/LLLLLLLL/LLLLLLLL" in text


def test_the_board_helper_names_no_app_and_loads_nothing_of_the_mac():
    """No app names or bundle ids in its logic, and numpy with the standard library only: no
    capture, no events, no macOS frameworks (library mode may import it)."""
    src = Path(B.__file__).read_text()
    assert not re.search(r"\bChess\b|\.app\b|com\.apple|bundle", src)
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "itertools", "math", "re", "time", "dataclasses", "typing", "numpy"}, mods


def test_it_is_fast_enough_to_run_on_every_screenshot():
    img, _ = _flat()
    pic = B.Picture.of(img)
    g = pic.gray
    best = min(_timed(lambda: B.find_grid(g)) for _ in range(3))
    assert best < 0.035  # ≤ 35 ms per full 1512×982 frame (measured ~5 ms)
    b = B.survey(pic).board
    after, _ = _flat(position=moved(START, "e2", "e4"))
    ap = B.Picture.of(after)
    s1, s2 = B.state(pic, b), B.state(ap, b)
    assert min(_timed(lambda: B.changed_cells(b, s1, s2)) for _ in range(3)) < 0.002
    tp = B.Picture.of(_tilted(elev=55, yaw=0)[0])
    assert min(_timed(lambda: B.find_tilted(tp)) for _ in range(3)) < 0.1  # (measured ~20 ms)


def _timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


# ---- evidence and the one-call helpers the screenshot tool uses ----------------------------------


def test_a_large_view_with_no_accessible_elements_is_the_evidence_for_a_board():
    win = (100.0, 50.0, 1000.0, 800.0)
    toolbar = [(110 + i * 60, 60, 50, 24, "AXButton") for i in range(8)]
    view = (150.0, 120.0, 700.0, 700.0, "AXGroup")  # one element, nothing inside it: drawn in one piece
    side = [(900, 120, 180, 30, "AXButton"), (900, 160, 180, 30, "AXButton")]
    assert B.canvas_area(win, [*toolbar, view, *side]) == view[:4]
    # the view isn't exposed at all: the largest part of the window nothing covers
    x, y, w, h = B.canvas_area(win, [*toolbar, *side])
    assert w * h >= 0.4 * 1000 * 800 and y >= 84 - 800 / 32
    assert B.canvas_area(win, []) == win  # a window with nothing accessible at all
    rows = [(100, 50 + i * 20, 1000, 20, "AXRow") for i in range(40)]  # a list filling the window
    assert B.canvas_area(win, rows) is None
    scroll = (100.0, 50.0, 1000.0, 800.0, "AXScrollArea")  # a container: its rows are the content
    assert B.canvas_area(win, [scroll, *rows]) is None
    assert B.canvas_area(win, [(100, 90, 1000, 760, "AXWebArea")]) is None  # a page: read, not pixels
    assert B.canvas_area(win, [(100, 90, 1000, 760, "AXTextArea"), *toolbar]) is None
    assert B.canvas_area(None, [view]) is None


def test_the_screenshot_note_part_says_what_neo_found_or_why_not():
    img, _ = _flat()
    text = B.look(B.Picture.of(img), shot="s1")
    assert text.startswith("NEO found a board of 8×8 squares") and B.latest().shot == "s1"
    assert B.look(B.Picture.of(img), shot="s2", sheet=True, canvas=True) == B.covered_note()
    a = np.full((982, 1512, 3), 60, np.uint8)
    assert "map_board" in B.look(B.Picture.of(a), shot="s3", canvas=True)
    assert "needs a model that can see" in B.look(B.Picture.of(a), shot="s4", canvas=True, blind=True)
    assert B.look(B.Picture.of(a), shot="s5") == ""  # no evidence, no board: nothing to say
    x, y, w, h = B.region(B.latest())
    assert x < 420 and y < 150 and x + w > 1028 and y + h > 758


def test_a_window_crop_is_mapped_to_the_same_screen_points():
    img, tr = _flat()
    crop = img.crop((300, 100, 1200, 900))  # a window's own picture: its pixels start at its origin
    b = B.survey(B.Picture.of(crop, x0=300, y0=100)).board
    assert b is not None and _worst(b, tr) <= 0.5
    retina = crop.resize((1800, 1600), Image.NEAREST)  # the same window at 2 px per point
    b2 = B.survey(B.Picture.of(retina, x0=300, y0=100, ppp=0.5)).board
    assert b2 is not None and _worst(b2, tr) <= 0.5


def test_orientation_is_kept_for_the_same_board_when_the_pieces_no_longer_say():
    img, _ = _flat()
    assert B.survey(B.Picture.of(img)).board.orientation == "bottom"
    few = ("....k...", "........", "........", "...q....", "....Q...", "........", "........", "....K...")
    later = B.survey(B.Picture.of(_flat(position=few)[0])).board
    assert B.orientation(later.occupancy) == "" and later.orientation == "bottom" and later.chess


def test_squares_map_through_a_screenshots_shot_like_its_pixels_do():
    from neo.tools.computer import pointer

    img, tr = _flat()
    k = 1280 / 1512
    small = img.resize((1280, round(982 * k)), Image.LANCZOS)
    shot = pointer.remember(0, 0, 1 / k, small.width, small.height)
    s = B.survey(B.Picture.from_shot(small, shot), shot=shot.id)
    assert s.board.shot == shot.id and B.for_shot(shot.id) is s.board
    (pt,), _, _ = B.points_for({"cell": "e2", "shot": shot.id}, ("cell",))
    assert np.allclose(pt, _truth_centre(tr, "e2"), atol=1.2)  # screen points, as the Shot maps pixels


def test_a_board_that_left_its_window_is_forgotten():
    img, _ = _flat()
    win = (300.0, 80.0, 950.0, 820.0)
    B.survey(B.Picture.of(img), shot="s1", window=win)
    a = np.asarray(img).copy()
    a[140:770, 410:1040] = 250  # the same window, the board gone
    B.survey(B.Picture.of(a), shot="s2", window=win)
    with pytest.raises(B.BoardError, match="doesn't know a board"):
        B.points_for({"cell": "e2"}, ("cell",))
