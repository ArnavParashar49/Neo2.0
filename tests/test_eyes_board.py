"""The board helper (neo2/core/tools/computer/board.py) on synthetic screens: flat boards and grids, boards
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

from neo2.core.tools.computer import board as B

CHESS = [f + r for f in "abcdefgh" for r in "12345678"]


@pytest.fixture(autouse=True)
def _fresh_boards():
    B.forget()
    yield
    B.forget()


@functools.lru_cache(maxsize=8)  # (bounded: each render is kept as its small picture, a few at a time)
def _flat(**kw):
    return flat_screen(**kw)


@functools.lru_cache(maxsize=8)
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
    covered, _ = _tilted(elev=55, yaw=0, sheet=True)
    texts.append(B.after_move(s.board, B.Picture.of(covered), "e2", "e4").said)  # (and the board is dropped)
    for call in (
        lambda: B.points_for({"cell": "e2"}, ("cell",), blind=True),
        lambda: B.points_for({"cell": "e2", "shot": "s404"}, ("cell",), blind=True),
        lambda: B.before_move(s.board, B.Picture.of(covered)),
    ):
        with pytest.raises(B.BoardError) as e:
            call()
        texts.append(str(e.value))
    B.survey(B.Picture.of(_flat()[0].crop((300, 100, 1200, 900)), x0=-400, y0=100), shot="w1")
    with pytest.raises(B.BoardError) as e:
        B.points_for({"cell": "a2"}, ("cell",), blind=True, displays=[(0, 0, 1512, 982)])
    texts.append(str(e.value))
    texts += [*B.act_words(["e2"], "Double-clicked")[:1], *B.act_words(["e2"])[1], *B.act_words(["e2", "e4"])[:1], *B.act_words(["e2", "e4"])[1]]
    assert B.act_words(["e2", "e4"]) == ("Dragged e2 → e4", ["the start e2", "the end e4"])
    for t in texts:
        assert not _COORD.search(t), t
    assert "corners" not in B.hint_note("s1", blind=True)


def test_the_note_names_squares_and_says_how_to_act_on_them():
    img, _ = _flat()
    s = B.survey(B.Picture.of(img), shot="s1")
    text = B.note(s.board)
    assert "NEO found a board of 8×8 squares" in text and "a1–h8" in text and "White (the light pieces) at the bottom" in text
    assert 'click cell="<square>" · drag from="<square>" to="<square>"' in text
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


def test_an_orientation_isnt_kept_when_the_pieces_are_another_game_altogether():
    img, _ = _flat()
    assert B.survey(B.Picture.of(img)).board.orientation == "bottom"
    few = ("....k...", "........", "........", "...q....", "....Q...", "........", "........", "....K...")
    later = B.survey(B.Picture.of(_flat(position=few)[0]), shot="s2").board
    # (not the last look's game a move or two on: which side is White's is for the pieces or a model to say)
    assert B.orientation(later.occupancy) == "" and later.orientation == "" and not later.chess
    assert 'map_board orientation="bottom"' in B.note(later)


def test_squares_map_through_a_screenshots_shot_like_its_pixels_do():
    from neo2.core.tools.computer import pointer

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


# ---- review fixes: speed on striped screens, a board's true extent, scale-free changes, stale boards ----


def _stripes(kind: str) -> np.ndarray:
    """Screens with many long, equal straight edges and no board: a striped banner, bars, a barcode."""
    a = np.full((982, 1512, 3), 240, np.uint8)
    if kind == "band-900x120-3px":
        for x in range(100, 1000, 6):
            a[300:420, x : x + 3] = 110
    elif kind == "band-1512x400-3px":
        for x in range(0, 1512, 6):
            a[300:700, x : x + 3] = 120
    elif kind == "bars-8px":
        a[:] = 30
        for x in range(0, 1512, 8):
            a[200:800, x : x + 4] = 200
    elif kind == "stripes-5px":
        for x in range(0, 1512, 5):
            a[100:900, x : x + 2] = 90
    elif kind == "grid-of-stripes":  # stripes both ways: a woven texture
        for x in range(0, 1512, 6):
            a[100:900, x : x + 3] = 120
        for y in range(100, 900, 6):
            a[y : y + 3, :] //= 2
    return a


@pytest.mark.parametrize("kind", ["band-900x120-3px", "band-1512x400-3px", "bars-8px", "stripes-5px", "grid-of-stripes"])
def test_the_grid_search_stays_fast_on_stripes_and_bars(kind):
    g = B.Picture.of(_stripes(kind)).gray
    assert B.find_grid(g) is None
    assert min(_timed(lambda: B.find_grid(g)) for _ in range(3)) < 0.035  # (was 0.6–7 s: cubic in the edges)


def test_the_grid_search_stays_fast_on_a_retina_frame_of_stripes():
    a = np.asarray(Image.fromarray(_stripes("band-1512x400-3px")).resize((3024, 1964), Image.NEAREST))
    g = B.Picture.of(a).gray
    assert B.find_grid(g) is None
    assert min(_timed(lambda: B.find_grid(g)) for _ in range(2)) < 0.07  # (4× the pixels)


def test_a_board_next_to_a_striped_banner_is_still_found():
    img, tr = _flat()
    a = np.asarray(img).copy()
    for x in range(40, 400, 6):  # a striped strip beside the board, as tall as the board
        a[150:758, x : x + 3] = 110
    b = B.survey(B.Picture.of(a)).board
    assert b is not None and (b.rows, b.cols) == (8, 8) and _worst(b, tr) <= 0.5


FRAMES = [
    ((78, 50, 30), (52, 54, 60)),
    ((78, 50, 30), (235, 235, 235)),  # the default dark frame in a light-mode window
    ((150, 110, 70), (235, 235, 235)),
    ((200, 165, 115), (52, 54, 60)),
    ((200, 165, 115), (120, 80, 45)),
    ((230, 200, 150), (52, 54, 60)),  # a light wooden frame on a dark window
    ((230, 200, 150), (120, 80, 45)),
    ((245, 240, 230), (52, 54, 60)),
    ((245, 240, 230), (235, 235, 235)),
]


def _true_cell(tr, pt) -> tuple[int, int]:
    u, v = board_to_image(np.linalg.inv(tr["H"]), [pt])[0]
    return int(np.floor(u)), int(np.floor(v))


@pytest.mark.parametrize("frame,bg", FRAMES, ids=lambda v: "-".join(map(str, v)))
def test_a_tilted_board_never_grows_into_its_frame_or_the_window_behind(frame, bg):
    img, tr = _tilted(elev=60, frame=frame, background=bg)
    b = B.survey(B.Picture.of(img), canvas=True, shot="s1").board
    # 8×8 where it is, or nothing: never a ring of frame and background taken for squares
    assert b is None or ((b.rows, b.cols) == (8, 8) and _worst(b, tr) <= 3.0), (b.rows, b.cols, _true_cell(tr, b.centre(0, 0)))
    assert b is not None  # (and these boards are all plain to see)


@pytest.mark.parametrize("frame,bg", [((230, 200, 150), (52, 54, 60)), ((78, 50, 30), (235, 235, 235))])
def test_a_tilted_board_from_the_side_keeps_to_its_squares_too(frame, bg):
    img, tr = _tilted(elev=45, yaw=-30, frame=frame, background=bg)
    b = B.survey(B.Picture.of(img), canvas=True).board
    assert b is not None and (b.rows, b.cols) == (8, 8) and _worst(b, tr) <= 3.0


@pytest.mark.parametrize("k", [1280 / 1512, 1024 / 1512, 1600 / 1728])
def test_a_move_is_judged_the_same_between_pictures_at_different_scales(k):
    img, _ = _flat()
    small = img.resize((round(1512 * k), round(982 * k)), Image.LANCZOS)
    b = B.survey(B.Picture.of(small, ppp=1 / k), shot="s1").board
    same = B.after_move(b, B.Picture.of(img), "", "e4")  # a click that took nothing, seen at 1 px per point
    assert same.changed is False, same.said
    B.forget()
    b = B.survey(B.Picture.of(small, ppp=1 / k), shot="s1").board
    after, _ = _flat(position=moved(START, "e2", "e4"))
    m = B.after_move(b, B.Picture.of(after), "e2", "e4")
    assert sorted(m.cells) == ["e2", "e4"], m.said
    retina = after.resize((3024, 1964), Image.LANCZOS)  # …and back at 2 px per point: nothing more changed
    assert B.after_move(b, B.Picture.of(retina, ppp=0.5), "", "").changed is False


def test_small_squares_are_judged_the_same_at_different_scales():
    img, _ = _flat(board_at=(600, 300), sq_pt=30)
    k = 1280 / 1512
    small = img.resize((1280, round(982 * k)), Image.LANCZOS)
    b = B.survey(B.Picture.of(small, ppp=1 / k)).board
    assert b is not None
    same = B.after_move(b, B.Picture.of(img), "", "")
    assert same.changed is False, same.said
    after, _ = _flat(board_at=(600, 300), sq_pt=30, position=moved(START, "g1", "f3"))
    assert sorted(B.after_move(b, B.Picture.of(after), "g1", "f3").cells) == ["f3", "g1"]


def test_nothing_changed_since_the_last_look_when_only_the_scale_did():
    img, _ = _flat()
    k = 1280 / 1512
    small = img.resize((1280, round(982 * k)), Image.LANCZOS)
    B.survey(B.Picture.of(small, ppp=1 / k), shot="s1")
    assert B.survey(B.Picture.of(img), shot="s2").since == []
    reply, _ = _flat(position=moved(START, "e2", "e4"))
    assert sorted(B.survey(B.Picture.of(reply.resize((1280, round(982 * k)), Image.LANCZOS), ppp=1 / k), shot="s3").since) == ["e2", "e4"]


def test_a_board_seen_covered_after_a_move_is_dropped_and_stops_what_follows():
    img, _ = _flat()
    win = (300.0, 80.0, 950.0, 820.0)
    b = B.survey(B.Picture.of(img), shot="s1", window=win, now=100.0).board
    covered, _ = _flat(sheet=True)
    m = B.after_move(b, B.Picture.of(covered), "e2", "e4", now=101.0)
    assert m.changed is None and "isn't where it was" in m.said
    for a in ({"cell": "e7"}, {"cell": "e7", "shot": "s1"}):
        with pytest.raises(B.BoardError, match="screenshot"):
            B.points_for(a, ("cell",), blind=True, now=102.0, window=win)


def test_a_board_that_left_its_window_is_gone_by_its_shot_id_too():
    img, _ = _flat()
    win = (300.0, 80.0, 950.0, 820.0)
    B.survey(B.Picture.of(img), shot="s1", window=win, now=10.0)
    B.survey(B.Picture.of(img), shot="s0", window=(0.0, 0.0, 200.0, 200.0), now=10.0)  # (another window's)
    a = np.asarray(img).copy()
    a[140:770, 410:1040] = 250
    B.survey(B.Picture.of(a), shot="s2", window=win, now=11.0)
    with pytest.raises(B.BoardError):
        B.points_for({"cell": "e2", "shot": "s1"}, ("cell",), blind=True, now=12.0, window=win)
    assert B.points_for({"cell": "e2", "shot": "s0"}, ("cell",), now=12.0)[2] == ["e2"]


def test_a_shot_with_no_board_is_an_error_not_another_board():
    img, _ = _flat()
    B.survey(B.Picture.of(img), shot="s1", now=10.0)
    with pytest.raises(B.BoardError, match="no board"):
        B.points_for({"cell": "e2", "shot": "no-such-shot"}, ("cell",), now=11.0)
    blank = np.full((982, 1512, 3), 60, np.uint8)
    B.survey(B.Picture.of(blank), shot="s2", canvas=True, now=11.0)  # (a canvas NEO found no grid in)
    with pytest.raises(B.BoardError, match="no board"):
        B.points_for({"cell": "e2", "shot": "s2"}, ("cell",), now=11.0)
    assert B.points_for({"cell": "e2", "shot": "s1"}, ("cell",), now=11.0)[2] == ["e2"]


def test_a_newer_look_at_the_same_window_replaces_its_older_boards():
    win = (360.0, 90.0, 900.0, 800.0)
    img, _ = _flat(board_at=(420, 150))
    B.survey(B.Picture.of(img), shot="s1", window=win, now=10.0)
    again = B.survey(B.Picture.of(img), shot="s2", window=win, now=11.0).board  # the same board again
    assert B.for_shot("s1") is again  # (s1's names still work: they name the same squares, freshly seen)
    scrolled, tr2 = _flat(board_at=(420, 302))
    B.survey(B.Picture.of(scrolled), shot="s3", window=win, now=12.0)
    for old in ("s1", "s2"):
        with pytest.raises(B.BoardError):
            B.points_for({"cell": "e2", "shot": old}, ("cell",), now=13.0, window=win)
    (pt,), _, _ = B.points_for({"cell": "e2"}, ("cell",), now=13.0, window=win)
    assert np.allclose(pt, _truth_centre(tr2, "e2"), atol=0.5)


def test_a_board_scrolled_in_its_window_is_refused_before_the_input():
    win = (360.0, 90.0, 900.0, 800.0)
    img, _ = _flat(board_at=(420, 150))
    b = B.survey(B.Picture.of(img), shot="s1", window=win, now=10.0).board
    shifted, _ = _flat(board_at=(420, 150 + 152))  # the content scrolled two squares, the window didn't move
    (pt,), _, _ = B.points_for({"cell": "e2"}, ("cell",), now=20.0, window=win)
    with pytest.raises(B.BoardError, match="isn't where"):
        B.before_move(b, B.Picture.of(shifted), now=20.0)
    with pytest.raises(B.BoardError):  # …and it is forgotten: no square of it is acted on again
        B.points_for({"cell": "e2"}, ("cell",), now=20.0, window=win)


def test_the_look_just_before_the_input_is_what_the_move_is_judged_against():
    img, _ = _flat()
    k = 1280 / 1512
    small = img.resize((1280, round(982 * k)), Image.LANCZOS)
    b = B.survey(B.Picture.of(small, ppp=1 / k), shot="s1", now=10.0).board
    B.before_move(b, B.Picture.of(img), now=11.0)  # the capture path the after-look uses
    after, _ = _flat(position=moved(START, "e2", "e4"))
    m = B.after_move(b, B.Picture.of(after), "e2", "e4", now=12.0)
    assert sorted(m.cells) == ["e2", "e4"] and m.changed is True


def test_scrolling_or_keys_in_a_window_forget_its_boards():
    img, _ = _flat()
    B.survey(B.Picture.of(img), shot="s1", window=(360.0, 90.0, 900.0, 800.0), now=10.0)
    B.survey(B.Picture.of(img), shot="s2", window=(0.0, 0.0, 300.0, 300.0), now=10.0)
    B.forget_window((360.0, 90.0, 900.0, 800.0))
    with pytest.raises(B.BoardError):
        B.points_for({"cell": "e2", "shot": "s1"}, ("cell",), now=11.0)
    assert B.points_for({"cell": "e2", "shot": "s2"}, ("cell",), now=11.0)[2] == ["e2"]
    B.forget_window(None)  # (the window isn't known: every board)
    with pytest.raises(B.BoardError):
        B.points_for({"cell": "e2", "shot": "s2"}, ("cell",), now=11.0)


class _Display:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


def test_a_square_off_every_display_is_refused_without_coordinates():
    img, _ = _flat()
    crop = img.crop((300, 100, 1200, 900))
    B.survey(B.Picture.of(crop, x0=-400, y0=100), shot="w1", now=5.0)  # a window hanging past the left edge
    shown = [_Display(0, 0, 1512, 982)]
    with pytest.raises(B.BoardError) as e:
        B.points_for({"cell": "a2"}, ("cell",), blind=True, now=6.0, displays=shown)
    assert "off" in str(e.value) and not _COORD.search(str(e.value))
    assert B.points_for({"cell": "h2"}, ("cell",), blind=True, now=6.0, displays=shown)[2] == ["h2"]
    with pytest.raises(B.BoardError):
        B.points_for({"from": "h2", "to": "a2"}, ("from", "to"), now=6.0, displays=shown)


def test_a_picture_that_cant_be_mapped_to_the_screen_registers_no_board():
    img, _ = _flat()
    s = B.survey(B.Picture.of(img), shot="w2", clickable=False)
    assert s.board is None and B.latest() is None and B.for_shot("w2") is None
    assert B.look(B.Picture.of(img), shot="w3", clickable=False) == ""
    with pytest.raises(B.BoardError):
        B.points_for({"cell": "e2"}, ("cell",))


def test_corners_are_refused_when_the_squares_there_are_a_row_short():
    img, tr = _tilted(elev=60, sheet=True)  # something over the far row that isn't an Accessibility sheet
    true = board_to_image(tr["H"], [(0, 0), (8, 0), (8, 8), (0, 8)])
    b, why = B.from_corners(B.Picture.of(img), true)
    assert b is None or ((b.rows, b.cols) == (8, 8) and _worst(b, tr) <= 3.0), (b.rows, b.cols, _true_cell(tr, b.centre(0, 0)))
    assert b is None and why


def _photo(w: int, h: int) -> np.ndarray:
    x = np.random.default_rng(0).normal(0, 1, (h // 16, w // 16, 3))
    return np.asarray(Image.fromarray(((x - x.min()) / np.ptp(x) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC))


def _wavy_tiles(w: int = 1512, h: int = 982) -> np.ndarray:
    """Floor tiles in a photo, a map's warped grid: saddles everywhere, but no board's edge anywhere."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    v = np.sin(xx / 9 + np.sin(yy / 40)) * np.sin(yy / 9 + np.sin(xx / 50))
    return np.repeat(((v > 0) * 120 + 70).astype(np.uint8)[..., None], 3, axis=2)


def test_a_texture_of_tiles_with_no_edge_is_no_board():
    s = B.survey(B.Picture.of(_wavy_tiles()), canvas=True, shot="s1")
    assert s.board is None


def test_a_canvas_with_no_board_is_searched_once_and_hinted_once():
    pic = B.Picture.of(_photo(1512, 982))
    win = (0.0, 0.0, 1512.0, 982.0)
    first = B.look(pic, shot="s1", canvas=True, window=win)
    assert "map_board" in first
    again = B.Picture.of(np.asarray(pic.rgb).copy())  # the same canvas again, a new screenshot
    t = _timed(lambda: B.look(again, shot="s2", canvas=True, window=win))
    assert t < 0.01  # (the search found nothing there a moment ago: not searched again)
    assert B.look(B.Picture.of(np.asarray(pic.rgb).copy()), shot="s3", canvas=True, window=win) == ""  # (the hint once)
    B.forget()
    img, _ = _flat()  # a board there now: found at once
    assert B.look(B.Picture.of(img), shot="s4", canvas=True, window=win).startswith("NEO found a board")


def test_a_transparency_checkerboard_is_given_up_on_quickly():
    a = np.full((982, 1512, 3), 236, np.uint8)
    yy, xx = np.mgrid[0:982, 0:1512]
    a[:] = np.where((((yy // 10) + (xx // 10)) % 2 == 0)[..., None], 255, 204)
    pic = B.Picture.of(a)
    assert B.survey(pic, canvas=True).board is None
    B.forget()
    assert min(_timed(lambda: (B.forget(), B.survey(B.Picture.of(a), canvas=True))) for _ in range(2)) < 0.1  # (was 0.3 s)


def test_an_orientation_kept_from_the_last_look_must_fit_the_pieces_now():
    img, _ = _flat()
    assert B.survey(B.Picture.of(img), now=1.0).board.orientation == "bottom"
    # a mid-game position that doesn't show the sides, on the board turned round: not the same game
    mid = ("........", "..k.....", ".p.P....", "..P.p...", "...p.P..", "....K...", "........", "........")
    turned = tuple(row[::-1] for row in reversed(mid))
    b2 = B.survey(B.Picture.of(_flat(position=turned)[0]), now=2.0).board
    assert B.orientation(b2.occupancy) == ""
    assert not b2.chess and b2.orientation == ""  # row and column names until the pieces or a model say
    assert "judged from the pieces" not in B.note(b2)


def test_an_orientation_is_kept_across_a_move_or_two_and_the_note_says_so():
    one = ("....k...", "...ppp..", "........", "........", "........", "........", "...PPP..", "....K...")
    b1 = B.survey(B.Picture.of(_flat(position=one)[0]), now=1.0).board
    assert b1.orientation == "bottom" and b1.oriented_by == "pieces"
    two = ("....k...", "...pp...", "........", ".....p..", "........", "........", "...PP...", "....K...")  # (3 light: unclear)
    b2 = B.survey(B.Picture.of(_flat(position=two)[0]), now=2.0).board
    assert B.orientation(b2.occupancy) == "" and b2.orientation == "bottom" and b2.oriented_by == "kept"
    text = B.note(b2)
    assert "judged from the pieces" not in text and "last look" in text


def test_the_note_shows_how_to_name_squares_without_suggesting_a_move():
    img, _ = _flat()
    text = B.note(B.survey(B.Picture.of(img), shot="s1").board)
    assert 'cell="<square>"' in text and 'from="<square>" to="<square>"' in text
    assert not re.search(r'(cell|from|to)="[a-h][1-8]"', text)
    assert not re.search(r'(cell|from|to)="[a-h][1-8]"', B.SCREENSHOT_WORDS)


def test_a_board_mapped_from_corners_stays_while_it_is_there_and_goes_when_it_isnt():
    img, tr = _tilted(elev=55, yaw=0)
    win = (0.0, 0.0, 1512.0, 982.0)
    blank = np.asarray(img).copy()
    blank[:] = 60
    B.survey(B.Picture.of(blank), canvas=True, shot="s1", window=win, now=1.0)  # (kept for its corners)
    B._pictures["s1"] = B.Picture.of(img)  # as if NEO had found no grid in this one
    true = board_to_image(tr["H"], [(0, 0), (8, 0), (8, 8), (0, 8)])
    text, ok = B.map_board({"shot": "s1", "corners": true.tolist()}, now=2.0)
    b = B.latest()
    assert ok and b.source == "vision" and b.window == win  # (its window: a moved window means a new look)
    B._empty.clear()
    orig = B.find_tilted
    try:
        B.find_tilted = lambda *a, **k: None  # NEO can't find this board by itself
        s = B.survey(B.Picture.of(img), canvas=True, shot="s2", window=win, now=3.0)
        assert s.board is b and B.for_shot("s2") is b  # still where it was: kept, its squares named
        after, _ = _tilted(elev=55, yaw=0, position=moved(START, "e2", "e4"), seed=8)
        s = B.survey(B.Picture.of(after), canvas=True, shot="s3", window=win, now=4.0)
        assert s.board is b and sorted(s.since) == ["e2", "e4"]
        B.survey(B.Picture.of(blank), canvas=True, shot="s4", window=win, now=5.0)  # gone from its window
        assert B.latest() is None and B.for_shot("s2") is None
    finally:
        B.find_tilted = orig
