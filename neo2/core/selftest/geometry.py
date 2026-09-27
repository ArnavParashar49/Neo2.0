"""Plain geometry for the controls self-test's boards: square names, and the tilted board's
perspective (a homography from the board's own 8×8 grid to where it is drawn).

Shared by the test window (testwin: drawing, and which square a click landed on), the checks
(controls: where a square's centre is on screen) and the tests. Pure Python, no frameworks.
"""

from __future__ import annotations

FILES = "abcdefgh"

Point = tuple[float, float]
Matrix = tuple[float, float, float, float, float, float, float, float, float]


def square(col: int, row: int) -> str:
    """A cell's name as a chess player says it: col 0 is file a, row 0 (the top) is rank 8."""
    return f"{FILES[col]}{8 - row}"


def cell_of(name: str) -> list[int]:
    """[col, row] of a square's name ("c2" → [2, 6])."""
    name = name.strip().lower()
    if len(name) != 2 or name[0] not in FILES or not name[1].isdigit() or not 1 <= int(name[1]) <= 8:
        raise ValueError(f"not a square: {name!r}")
    return [FILES.index(name[0]), 8 - int(name[1])]


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (8×8 here)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("degenerate quad")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def homography(src: list[Point], dst: list[Point]) -> Matrix:
    """The projective map taking the four ``src`` corners to the four ``dst`` corners."""
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = _solve(a, b)
    return (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0)


def apply(h: Matrix, x: float, y: float) -> Point:
    w = h[6] * x + h[7] * y + h[8]
    return ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)


def invert(h: Matrix) -> Matrix:
    a, b, c, d, e, f, g, i, j = h
    det = a * (e * j - f * i) - b * (d * j - f * g) + c * (d * i - e * g)
    if abs(det) < 1e-12:
        raise ValueError("singular")
    inv = (
        (e * j - f * i), -(b * j - c * i), (b * f - c * e),
        -(d * j - f * g), (a * j - c * g), -(a * f - c * d),
        (d * i - e * g), -(a * i - b * g), (a * e - b * d),
    )  # fmt: skip
    return tuple(v / inv[8] for v in inv)  # type: ignore[return-value]


GRID: list[Point] = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]  # the board's own corners: TL, TR, BR, BL


def tilt_quad(w: float, h: float, tilt: float = 0.6) -> list[Point]:
    """Where a board seen from above at an angle is drawn in a w×h view (top-left origin): the far
    edge narrower and higher, like a 3D board tilted back. ``tilt``: how much narrower the far edge."""
    inset = w * tilt / 2 * 0.5
    return [(inset, h * 0.12), (w - inset, h * 0.12), (w - 2.0, h - 2.0), (2.0, h - 2.0)]


def board_map(quad: list[Point]) -> Matrix:
    """Board grid (0..8, 0..8) → view (or screen) points."""
    return homography(GRID, quad)


def cell_at(quad: list[Point], x: float, y: float) -> list[int] | None:
    """[col, row] of the tilted board's cell under a point (in the same space as ``quad``)."""
    try:
        gx, gy = apply(invert(board_map(quad)), x, y)
    except ValueError:
        return None
    if not (0 <= gx < 8 and 0 <= gy < 8):
        return None
    return [int(gx), int(gy)]


def cell_center(quad: list[Point], col: int, row: int) -> Point:
    return apply(board_map(quad), col + 0.5, row + 0.5)


def cell_polygon(quad: list[Point], col: int, row: int) -> list[Point]:
    m = board_map(quad)
    return [apply(m, col + dx, row + dy) for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]
