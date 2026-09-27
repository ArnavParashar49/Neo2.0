"""The controls self-test's own window: a small app that NEO starts, drives and closes again.

neo.selftest.controls runs it as a child process of NEO's core, from NEO's own Python::

    python -m neo.selftest.testwin --log FILE --tag TAG --parent PID

It is a separate app with an identity of its own (its own process and the name "NEO Self-Test
TAG", never NEO's), with a known layout: two buttons (Alpha, Bravo) and one titled in Hindi, a box
that wants a double click, a 12-pt button, a text field, a multi-line text area, a list to scroll, a
tile to drag onto a well (real drag and drop), a magenta marker for screenshots, a Test menu with a
⌘J shortcut and a New Round… (⌘K) that opens a "Start a new round?" sheet, a Chess-like 8×8 board
with its pieces drawn in one view with nothing in it for Accessibility, whose first click on an
inactive window only activates the window (like Chess's board), and a second board tilted back and
textured the way a 3D board looks at an angle. With a second display, a small second window goes
there.

Everything it receives is appended to FILE as JSON lines (``{"t": …, "ev": …}``): each click
with its screen position and click count, drags and drops, typed text, scroll offsets, keys, the
menu, focus changes, plus the raw mouse and key events the app got at all. The first line NEO
waits for is ``ready``: the window's number and every target's frame on screen (points, top-left
origin, like Quartz). Lines on stdin are commands (``{"cmd": "deactivate" | "quit"}``).

It never outlives the test: it quits when NEO says so, when stdin closes, when its parent is gone,
when its window is closed, and after two minutes whatever happens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Any

from neo.selftest import geometry as G
from neo.selftest import layout as L

_LIFETIME_S = 120.0
_TICK_S = 0.1

# ---- plain helpers (tested without a window) ---------------------------------------------------


def rect_tl(x: float, y: float, w: float, h: float, height: float = L.HEIGHT) -> tuple[float, float, float, float]:
    """A frame given from the top-left of a (non-flipped) view `height` points tall, as AppKit
    wants it: from the bottom-left."""
    return (x, height - y - h, w, h)


def to_global(x: float, y: float, w: float, h: float, primary_h: float) -> list[float]:
    """An AppKit screen rect (bottom-left origin) as Quartz's global one (top-left origin of the
    primary display), rounded to a tenth of a point."""
    return [round(x, 1), round(primary_h - y - h, 1), round(w, 1), round(h, 1)]


def point_to_global(x: float, y: float, primary_h: float) -> list[float]:
    return [round(x, 1), round(primary_h - y, 1)]


def cell_at(x: float, y: float, n: int = L.BOARD_N, cell: float = L.CELL) -> list[int] | None:
    """The board cell [col, row] under a point in the board's own (flipped) coordinates."""
    if x < 0 or y < 0:
        return None
    col, row = int(x // cell), int(y // cell)
    return [col, row] if col < n and row < n else None


def start_pieces() -> dict[tuple[int, int], str]:
    """The pieces as a game starts: (col, row) → letter (upper case white, at the bottom)."""
    return {(c, r): p for r, line in L.START_ROWS.items() for c, p in enumerate(line)}


def move_piece(pieces: dict[tuple[int, int], str], start: list[int] | None, end: list[int] | None) -> bool:
    """Move whatever stands on ``start`` to ``end`` (taking what was there). False: nothing moved."""
    if start is None or end is None or start == end or tuple(start) not in pieces:
        return False
    pieces[tuple(end)] = pieces.pop(tuple(start))
    return True


def place(visible: tuple[float, float, float, float], frame_w: float, frame_h: float) -> tuple[float, float]:
    """Where the window goes (AppKit origin): the top-left of the visible area, `MARGIN` in —
    away from NEO's overlay, which lives bottom-right — but always on that screen."""
    vx, vy, vw, vh = visible
    x = vx + min(L.MARGIN, max(0.0, vw - frame_w))
    y = vy + vh - frame_h - min(L.MARGIN, max(0.0, vh - frame_h))
    return x, max(vy, y)


class Log:
    """JSON lines, one per event, flushed as they come (NEO reads the file while it grows)."""

    def __init__(self, path: str) -> None:
        self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115 - open for the app's lifetime
        self._lock = threading.Lock()

    def write(self, ev: str, **data: Any) -> None:
        line = json.dumps({"t": round(time.time(), 4), "ev": ev, **data}, ensure_ascii=False)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m neo.selftest.testwin")
    p.add_argument("--log", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--parent", type=int, default=0)
    p.add_argument("--lifetime", type=float, default=_LIFETIME_S)
    return p.parse_args(argv)


def command(line: str) -> str:
    """The command in a stdin line ("" for anything else)."""
    try:
        cmd = json.loads(line).get("cmd", "")
    except (ValueError, AttributeError):
        return ""
    return cmd if cmd in ("deactivate", "quit") else ""


# ---- the app ---------------------------------------------------------------------------------


class _Ctx:
    """What the views need at run time (the classes are defined once, before any of it exists)."""

    log: Log
    primary_h: float = 0.0
    board: Any = None
    tick: Any = None  # the app's tick (a plain function; the timer calls the controller)
    win: Any = None  # the main window (the sheet opens on it)
    sheet: Any = None  # the open "Start a new round?" alert, while it is open
    round: int = 1
    round_label: Any = None


_ctx = _Ctx()
_classes: dict[str, Any] | None = None

# NSEvent types the raw monitor logs.
_RAW_MOUSE = {1: "ldown", 2: "lup", 3: "rdown", 4: "rup", 6: "ldragged", 7: "rdragged", 22: "scroll"}
_RAW_KEYS = {10: "key_down", 11: "key_up"}
_DRAG_OP_COPY = 1


def classes() -> dict[str, Any]:
    """The window's view and controller classes — defined once per process (Objective-C class
    names are global). Defining them opens nothing."""
    global _classes
    if _classes is not None:
        return _classes
    import objc
    from AppKit import (
        NSAlert,
        NSBezierPath,
        NSColor,
        NSDraggingItem,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSImage,
        NSPasteboardItem,
        NSPasteboardTypeString,
        NSView,
    )
    from Foundation import NSMakeRect, NSMakeSize, NSObject, NSString

    def gpoint(view, event) -> list[float]:
        win = event.window() or view.window()
        p = event.locationInWindow()
        sp = win.convertPointToScreen_(p) if win is not None else p
        return point_to_global(sp.x, sp.y, _ctx.primary_h)

    def local(view, event) -> tuple[float, float]:
        p = view.convertPoint_fromView_(event.locationInWindow(), None)
        return float(p.x), float(p.y)

    def text(s: str, x: float, y: float, size: float = 12.0, color=None) -> None:
        attrs = {
            NSFontAttributeName: NSFont.systemFontOfSize_(size),
            NSForegroundColorAttributeName: color or NSColor.labelColor(),
        }
        NSString.stringWithString_(s).drawAtPoint_withAttributes_((x, y), attrs)

    class NEOSelfTestBoard(NSView):
        """8×8 cells in one view: no Accessibility children, and (like Chess) the first click on
        an inactive window only activates it. A piece can be dragged from cell to cell."""

        def initWithFrame_(self, frame):
            self = objc.super(NEOSelfTestBoard, self).initWithFrame_(frame)
            if self is None:
                return None
            self.piece = list(L.PIECE)  # the drag check's own piece, followed wherever it goes
            self.pieces = start_pieces()
            self.grab = None
            self.drags = 0
            return self

        def isFlipped(self):
            return True

        def acceptsFirstMouse_(self, event):
            return False

        def drawRect_(self, rect):
            light = NSColor.colorWithSRGBRed_green_blue_alpha_(0.93, 0.93, 0.82, 1.0)
            dark = NSColor.colorWithSRGBRed_green_blue_alpha_(0.46, 0.59, 0.34, 1.0)
            for c in range(L.BOARD_N):
                for r in range(L.BOARD_N):
                    (light if (c + r) % 2 == 0 else dark).set()
                    NSBezierPath.fillRect_(NSMakeRect(c * L.CELL, r * L.CELL, L.CELL, L.CELL))
            for (c, r), p in self.pieces.items():
                mine = [c, r] == self.piece
                (NSColor.blackColor() if mine or p.islower() else NSColor.whiteColor()).set()
                oval = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(c * L.CELL + 6, r * L.CELL + 6, L.CELL - 12, L.CELL - 12))
                oval.fill()
                NSColor.darkGrayColor().set()
                oval.stroke()
                if not mine:
                    text(p.upper(), c * L.CELL + 11, r * L.CELL + 7, 10.0, NSColor.whiteColor() if p.islower() else NSColor.blackColor())

        @objc.python_method
        def _down(self, event, button: str) -> None:
            x, y = local(self, event)
            cell = cell_at(x, y)
            self.drags = 0
            self.grab = cell if (button == "left" and cell is not None and tuple(cell) in self.pieces) else None
            _ctx.log.write("board_down", button=button, cell=cell, at=gpoint(self, event), clicks=int(event.clickCount()))

        def mouseDown_(self, event):
            self._down(event, "left")

        def rightMouseDown_(self, event):
            self._down(event, "right")

        def mouseDragged_(self, event):
            self.drags += 1
            if self.drags == 1:
                x, y = local(self, event)
                _ctx.log.write("board_drag", cell=cell_at(x, y), at=gpoint(self, event))

        def mouseUp_(self, event):
            x, y = local(self, event)
            cell = cell_at(x, y)
            _ctx.log.write("board_up", cell=cell, at=gpoint(self, event), drags=self.drags)
            if self.grab is not None and cell is not None and cell != self.grab and self.drags:
                start = self.grab
                if move_piece(self.pieces, start, cell):
                    _ctx.log.write("piece_moved", start=start, end=cell, squares=[G.square(*start), G.square(*cell)])
                    if start == self.piece:
                        self.piece = cell
                    self.setNeedsDisplay_(True)
            self.grab = None

    class NEOSelfTestTilt(NSView):
        """The flat board's twin, tilted back in perspective on a wood-like texture: the squares are
        only in its pixels (no Accessibility children), and like the flat one the first click on an
        inactive window only activates it. Pieces move by dragging."""

        def initWithFrame_(self, frame):
            self = objc.super(NEOSelfTestTilt, self).initWithFrame_(frame)
            if self is None:
                return None
            self.pieces = start_pieces()
            self.grab = None
            self.drags = 0
            return self

        def isFlipped(self):
            return True

        def acceptsFirstMouse_(self, event):
            return False

        @objc.python_method
        def quad(self):
            b = self.bounds().size
            return G.tilt_quad(float(b.width), float(b.height))

        def drawRect_(self, rect):
            b = self.bounds().size
            for i in range(0, int(b.height), 3):  # wood: bands of browns
                shade = 0.35 + 0.08 * ((i * 7919) % 13) / 13
                NSColor.colorWithSRGBRed_green_blue_alpha_(shade + 0.18, shade + 0.05, shade * 0.5, 1.0).set()
                NSBezierPath.fillRect_(NSMakeRect(0, i, b.width, 3))
            q = self.quad()
            for c in range(L.BOARD_N):
                for r in range(L.BOARD_N):
                    poly = G.cell_polygon(q, c, r)
                    path = NSBezierPath.bezierPath()
                    path.moveToPoint_(poly[0])
                    for pt in poly[1:]:
                        path.lineToPoint_(pt)
                    path.closePath()
                    light = (c + r) % 2 == 0
                    grain = 0.04 * ((c * 31 + r * 17) % 5) / 5
                    color = (0.86 - grain, 0.74 - grain, 0.55 - grain) if light else (0.45 - grain, 0.27 - grain, 0.14)
                    NSColor.colorWithSRGBRed_green_blue_alpha_(*color, 1.0).set()
                    path.fill()
            for (c, r), p in self.pieces.items():
                cx, cy = G.cell_center(q, c, r)
                poly = G.cell_polygon(q, c, r)
                h = abs(poly[2][1] - poly[0][1])
                rad = max(3.0, h * 0.34)
                (NSColor.whiteColor() if p.isupper() else NSColor.blackColor()).set()
                NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx - rad, cy - rad * 1.6, rad * 2, rad * 2)).fill()

        @objc.python_method
        def _cell(self, event):
            x, y = local(self, event)
            return G.cell_at(self.quad(), x, y)

        def mouseDown_(self, event):
            cell = self._cell(event)
            self.drags = 0
            self.grab = cell if cell is not None and tuple(cell) in self.pieces else None
            _ctx.log.write("tilt_down", cell=cell, square=G.square(*cell) if cell else None, at=gpoint(self, event))

        def mouseDragged_(self, event):
            self.drags += 1

        def mouseUp_(self, event):
            cell = self._cell(event)
            if self.grab is not None and cell is not None and cell != self.grab and self.drags:
                start = self.grab
                if move_piece(self.pieces, start, cell):
                    _ctx.log.write("tilt_moved", start=start, end=cell, squares=[G.square(*start), G.square(*cell)])
                    self.setNeedsDisplay_(True)
            self.grab = None

    class NEOSelfTestBox(NSView):
        """A labelled box that logs its clicks: the double-click box, the marker, the second
        display's target."""

        def initWithFrame_(self, frame):
            self = objc.super(NEOSelfTestBox, self).initWithFrame_(frame)
            if self is None:
                return None
            self.box_name, self.box_label, self.box_rgb, self.box_first = "box", "", (0.85, 0.85, 0.9), False
            return self

        def isFlipped(self):
            return True

        def acceptsFirstMouse_(self, event):
            return bool(self.box_first)

        def drawRect_(self, rect):
            NSColor.colorWithSRGBRed_green_blue_alpha_(*self.box_rgb, 1.0).set()
            NSBezierPath.fillRect_(self.bounds())
            if self.box_label:
                text(self.box_label, 8, 7)

        def mouseDown_(self, event):
            _ctx.log.write(f"{self.box_name}_down", at=gpoint(self, event), clicks=int(event.clickCount()))

    class NEOSelfTestList(NSView):
        """The list's document: rows drawn in one view (nothing for Accessibility to list)."""

        def isFlipped(self):
            return True

        def drawRect_(self, rect):
            first = max(0, int(rect.origin.y // L.ROW_H))
            last = min(L.LIST_ROWS, int((rect.origin.y + rect.size.height) // L.ROW_H) + 1)
            for i in range(first, last):
                if i % 2:
                    NSColor.colorWithSRGBRed_green_blue_alpha_(0.92, 0.94, 0.98, 1.0).set()
                    NSBezierPath.fillRect_(NSMakeRect(0, i * L.ROW_H, self.bounds().size.width, L.ROW_H))
                text(f"Row {i + 1}", 8, i * L.ROW_H + 4)

    try:
        source = [objc.protocolNamed("NSDraggingSource")]
        destination = [objc.protocolNamed("NSDraggingDestination")]
    except objc.error:  # pragma: no cover - AppKit always has them
        source = destination = []

    class NEOSelfTestTile(NSView, protocols=source):
        """Drag me: starts a real drag-and-drop session (a string on the pasteboard)."""

        def initWithFrame_(self, frame):
            self = objc.super(NEOSelfTestTile, self).initWithFrame_(frame)
            if self is None:
                return None
            self.started = False
            return self

        def isFlipped(self):
            return True

        def acceptsFirstMouse_(self, event):
            return False

        def drawRect_(self, rect):
            NSColor.systemBlueColor().set()
            NSBezierPath.fillRect_(self.bounds())
            text("Drag me", 8, 10, color=NSColor.whiteColor())

        def mouseDown_(self, event):
            self.started = False
            _ctx.log.write("tile_down", at=gpoint(self, event))

        def mouseDragged_(self, event):
            if self.started:
                return
            self.started = True
            size = self.bounds().size
            img = NSImage.alloc().initWithSize_(NSMakeSize(size.width, size.height))
            img.lockFocus()
            NSColor.systemBlueColor().set()
            NSBezierPath.fillRect_(NSMakeRect(0, 0, size.width, size.height))
            img.unlockFocus()
            writer = NSPasteboardItem.alloc().init()
            writer.setString_forType_(L.DRAG_PAYLOAD, NSPasteboardTypeString)
            item = NSDraggingItem.alloc().initWithPasteboardWriter_(writer)
            item.setDraggingFrame_contents_(self.bounds(), img)
            self.beginDraggingSessionWithItems_event_source_([item], event, self)
            _ctx.log.write("dnd_begin", at=gpoint(self, event))

        def mouseUp_(self, event):
            _ctx.log.write("tile_up", at=gpoint(self, event), started=bool(self.started))

        def draggingSession_sourceOperationMaskForDraggingContext_(self, session, context):
            return _DRAG_OP_COPY

        def draggingSession_endedAtPoint_operation_(self, session, point, operation):
            _ctx.log.write("dnd_end", operation=int(operation), at=point_to_global(point.x, point.y, _ctx.primary_h))

    class NEOSelfTestWell(NSView, protocols=destination):
        """Drop here: accepts the tile's string and logs it."""

        def initWithFrame_(self, frame):
            self = objc.super(NEOSelfTestWell, self).initWithFrame_(frame)
            if self is None:
                return None
            self.registerForDraggedTypes_([NSPasteboardTypeString])
            self.hot = False
            self.filled = False
            return self

        def isFlipped(self):
            return True

        def drawRect_(self, rect):
            if self.filled:
                NSColor.systemGreenColor().set()
            elif self.hot:
                NSColor.systemYellowColor().set()
            else:
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.85, 0.85, 0.85, 1.0).set()
            NSBezierPath.fillRect_(self.bounds())
            text("Drop here", 8, 10)

        def draggingEntered_(self, info):
            self.hot = True
            self.setNeedsDisplay_(True)
            _ctx.log.write("dnd_enter")
            return _DRAG_OP_COPY

        def draggingUpdated_(self, info):
            return _DRAG_OP_COPY

        def draggingExited_(self, info):
            self.hot = False
            self.setNeedsDisplay_(True)
            _ctx.log.write("dnd_exit")

        def prepareForDragOperation_(self, info):
            return True

        def performDragOperation_(self, info):
            payload = info.draggingPasteboard().stringForType_(NSPasteboardTypeString)
            self.filled, self.hot = True, False
            self.setNeedsDisplay_(True)
            _ctx.log.write("drop", payload=str(payload or ""))
            return True

    class NEOSelfTestController(NSObject):
        """Buttons, the field, the text area, the menu, notifications and the tick."""

        def buttonA_(self, sender):
            _ctx.log.write("button", name=L.BUTTON_A)

        def buttonB_(self, sender):
            _ctx.log.write("button", name=L.BUTTON_B)

        def buttonHindi_(self, sender):
            _ctx.log.write("button", name=L.HINDI_BUTTON)

        def newRound_(self, sender):
            if _ctx.sheet is not None or _ctx.win is None:
                return
            alert = NSAlert.alloc().init()
            alert.setMessageText_(L.SHEET_TITLE)
            alert.setInformativeText_("The boards stay as they are.")
            alert.addButtonWithTitle_(L.SHEET_START)
            alert.addButtonWithTitle_(L.SHEET_CANCEL)
            _ctx.sheet = alert
            _ctx.log.write("sheet_open", title=L.SHEET_TITLE)

            def answered(code):
                button = L.SHEET_START if int(code) == 1000 else L.SHEET_CANCEL  # NSAlertFirstButtonReturn
                _ctx.sheet = None
                _ctx.log.write("sheet_answer", button=button)
                if button == L.SHEET_START:
                    _ctx.round += 1
                    _ctx.log.write("round", n=_ctx.round)
                    if _ctx.round_label is not None:
                        _ctx.round_label.setStringValue_(f"Round {_ctx.round}")

            alert.beginSheetModalForWindow_completionHandler_(_ctx.win, answered)

        def fieldReturn_(self, sender):
            _ctx.log.write("field_return", value=str(sender.stringValue()))

        def menuAction_(self, sender):
            _ctx.log.write("menu", title=L.MENU_TITLE)

        def controlTextDidChange_(self, note):
            editor = (note.userInfo() or {}).get("NSFieldEditor")
            value = editor.string() if editor is not None else note.object().stringValue()
            _ctx.log.write("text", target="field", value=str(value))

        def textDidChange_(self, note):
            _ctx.log.write("text", target="area", value=str(note.object().string()))

        def appActive_(self, note):
            _ctx.log.write("focus", active=True)

        def appInactive_(self, note):
            _ctx.log.write("focus", active=False)

        def windowKey_(self, note):
            _ctx.log.write("key_window", window=int(note.object().windowNumber()))

        def windowClosing_(self, note):
            _ctx.log.write("closed", window=int(note.object().windowNumber()))
            self.performSelector_withObject_afterDelay_("quit:", None, 0.0)

        def listScrolled_(self, note):
            _ctx.log.write("scroll", target="list", offset=round(float(note.object().bounds().origin.y), 1))

        def tick_(self, timer):
            if _ctx.tick is not None:
                _ctx.tick()

        def quit_(self, sender):
            from AppKit import NSApp

            NSApp.terminate_(None)

    _classes = {
        "Board": NEOSelfTestBoard,
        "Tilt": NEOSelfTestTilt,
        "Box": NEOSelfTestBox,
        "List": NEOSelfTestList,
        "Tile": NEOSelfTestTile,
        "Well": NEOSelfTestWell,
        "Controller": NEOSelfTestController,
    }
    return _classes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ctx.log = Log(args.log)
    try:
        return _run(args)
    except Exception as e:  # noqa: BLE001 - NEO reads the reason from the log
        _ctx.log.write("crash", error=repr(e)[:300])
        return 1


def _run(args: argparse.Namespace) -> int:  # pragma: no cover - needs a real window server
    from AppKit import (
        NSApplication,
        NSApplicationDidBecomeActiveNotification,
        NSApplicationDidResignActiveNotification,
        NSButton,
        NSColor,
        NSEvent,
        NSMenu,
        NSMenuItem,
        NSScreen,
        NSScrollView,
        NSTextField,
        NSTextView,
        NSViewBoundsDidChangeNotification,
        NSWindow,
        NSWindowDidBecomeKeyNotification,
        NSWindowWillCloseNotification,
    )
    from Foundation import (
        NSBundle,
        NSMakeRect,
        NSNotificationCenter,
        NSProcessInfo,
        NSRunLoop,
        NSRunLoopCommonModes,
        NSTimer,
    )
    from PyObjCTools import AppHelper

    log, started = _ctx.log, time.time()
    name = f"{L.APP_NAME} {args.tag}"
    # The name macOS knows this app by (menu bar, Dock, LaunchServices): set before it checks in.
    try:
        NSBundle.mainBundle().infoDictionary()["CFBundleName"] = name
    except Exception:  # noqa: BLE001 - only cosmetic: NEO identifies the window by its pid
        pass
    try:
        NSProcessInfo.processInfo().setProcessName_(name)
    except Exception:  # noqa: BLE001
        pass

    k = classes()
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(0)  # a regular app: a menu bar, a Dock icon, activatable (like Chess)
    screens = list(NSScreen.screens())
    _ctx.primary_h = float(screens[0].frame().size.height)
    ctl = k["Controller"].alloc().init()

    # Menus: the app menu (Quit) and Test › Test Action ⌘J.
    bar = NSMenu.alloc().init()
    app_item, test_item = NSMenuItem.alloc().init(), NSMenuItem.alloc().init()
    bar.addItem_(app_item)
    bar.addItem_(test_item)
    app_menu = NSMenu.alloc().initWithTitle_(name)
    app_menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(f"Quit {name}", "terminate:", "q"))
    app_item.setSubmenu_(app_menu)
    test_menu = NSMenu.alloc().initWithTitle_("Test")
    action = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(L.MENU_TITLE, "menuAction:", L.MENU_KEYS.split("+")[-1])
    action.setTarget_(ctl)
    test_menu.addItem_(action)
    new_round = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(L.ROUND_MENU, "newRound:", L.ROUND_KEYS.split("+")[-1])
    new_round.setTarget_(ctl)
    test_menu.addItem_(new_round)
    test_item.setSubmenu_(test_menu)
    app.setMainMenu_(bar)

    def window(content_w: float, content_h: float, screen, title: str):
        content = NSMakeRect(0, 0, content_w, content_h)
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(content, 1 | 2 | 4, 2, False)
        win.setTitle_(title)
        win.setReleasedWhenClosed_(False)
        win.setMovable_(False)
        win.setCollectionBehavior_((1 << 0) | (1 << 8))  # every Space, over a full-screen app too
        frame = win.frameRectForContentRect_(content)
        v = screen.visibleFrame()
        vis = (v.origin.x, v.origin.y, v.size.width, v.size.height)
        win.setFrameOrigin_(place(vis, frame.size.width, frame.size.height))
        return win

    def add(parent, view, x, y, w, h, height=L.HEIGHT):
        view.setFrame_(NSMakeRect(*rect_tl(x, y, w, h, height)))
        parent.addSubview_(view)
        return view

    def box(name, label, rgb, first_mouse=False):
        b = k["Box"].alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10))
        b.box_name, b.box_label, b.box_rgb, b.box_first = name, label, rgb, first_mouse
        return b

    win = window(L.WIDTH, L.HEIGHT, screens[0], f"{L.TITLE} ({args.tag})")
    _ctx.win = win
    cv = win.contentView()
    add(cv, NSTextField.labelWithString_(f"{L.TITLE}."), 20, 12, 680, 18)
    board = add(cv, k["Board"].alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), 20, 40, L.BOARD_N * L.CELL, L.BOARD_N * L.CELL)
    _ctx.board = board
    button_a = add(cv, NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), 280, 40, 110, 30)
    button_b = add(cv, NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), 400, 40, 110, 30)
    for b, title, sel in ((button_a, L.BUTTON_A, "buttonA:"), (button_b, L.BUTTON_B, "buttonB:")):
        b.setTitle_(title)
        b.setBezelStyle_(1)
        b.setTarget_(ctl)
        b.setAction_(sel)
    double = add(cv, box("double", "Double-click me", (0.85, 0.87, 0.95)), 520, 40, 180, 30)
    field = add(cv, NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), 280, 84, 420, 24)
    field.setPlaceholderString_(L.FIELD_HINT)
    field.setDelegate_(ctl)
    field.setTarget_(ctl)
    field.setAction_("fieldReturn:")
    area_view = add(cv, NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), 280, 120, 260, 110)
    area_view.setHasVerticalScroller_(True)
    area_view.setBorderType_(2)
    size = area_view.contentSize()
    area = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, size.width, size.height))
    for editor in (area, win.fieldEditor_forObject_(True, field)):
        if editor is None:
            continue
        # What NEO types must arrive exactly as typed: no smart quotes, replacements or corrections.
        editor.setAutomaticQuoteSubstitutionEnabled_(False)
        editor.setAutomaticDashSubstitutionEnabled_(False)
        editor.setAutomaticTextReplacementEnabled_(False)
        editor.setAutomaticSpellingCorrectionEnabled_(False)
        editor.setContinuousSpellCheckingEnabled_(False)
    area.setRichText_(False)
    area.setAllowsUndo_(True)
    area.setVerticallyResizable_(True)
    area.setHorizontallyResizable_(False)
    area.setAutoresizingMask_(2)
    area.textContainer().setWidthTracksTextView_(True)
    area.setDelegate_(ctl)
    area_view.setDocumentView_(area)
    list_view = add(cv, NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), 560, 120, 140, 160)
    list_view.setHasVerticalScroller_(True)
    list_view.setBorderType_(2)
    doc = k["List"].alloc().initWithFrame_(NSMakeRect(0, 0, list_view.contentSize().width, L.LIST_ROWS * L.ROW_H))
    list_view.setDocumentView_(doc)
    clip = list_view.contentView()
    clip.setPostsBoundsChangedNotifications_(True)
    marker = add(cv, box("marker", "", tuple(c / 255 for c in L.MARKER_RGB)), 280, 244, L.MARKER_PT, L.MARKER_PT)
    tile = add(cv, k["Tile"].alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), 340, 244, 80, 36)
    well = add(cv, k["Well"].alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), 440, 244, 100, 36)
    tx, ty, tw, th = L.TILT
    tilt = add(cv, k["Tilt"].alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), tx, ty, tw, th)
    hindi = add(cv, NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10)), 280, 300, 110, 30)
    hindi.setTitle_(L.HINDI_BUTTON)
    hindi.setBezelStyle_(1)
    hindi.setTarget_(ctl)
    hindi.setAction_("buttonHindi:")
    try:
        hindi.setBezelColor_(NSColor.systemPinkColor())
    except Exception:  # noqa: BLE001 - only the colour
        pass
    tiny = add(cv, box("tiny", "", tuple(c / 255 for c in L.TINY_RGB), True), 410, 309, L.TINY_PT, L.TINY_PT)
    _ctx.round_label = add(cv, NSTextField.labelWithString_("Round 1"), 440, 306, 200, 18)
    add(cv, NSTextField.labelWithString_(f"Test window {args.tag} - it closes by itself."), 20, 350, 680, 18)

    second_win = second = None
    if len(screens) > 1:
        second_win = window(260, 120, screens[1], f"NEO controls self-test - second display ({args.tag})")
        second = add(second_win.contentView(), box("second", "Second display", (0.9, 0.85, 0.8), True), 20, 20, 220, 80, 120)

    nc = NSNotificationCenter.defaultCenter()
    nc.addObserver_selector_name_object_(ctl, "appActive:", NSApplicationDidBecomeActiveNotification, None)
    nc.addObserver_selector_name_object_(ctl, "appInactive:", NSApplicationDidResignActiveNotification, None)
    nc.addObserver_selector_name_object_(ctl, "windowKey:", NSWindowDidBecomeKeyNotification, None)
    for w in (win, second_win):  # only the test's own windows (a drag image's closing is no reason to quit)
        if w is not None:
            nc.addObserver_selector_name_object_(ctl, "windowClosing:", NSWindowWillCloseNotification, w)
    nc.addObserver_selector_name_object_(ctl, "listScrolled:", NSViewBoundsDidChangeNotification, clip)

    def raw(event):
        try:
            t = int(event.type())
            if t in _RAW_MOUSE:
                w = event.window()
                p = event.locationInWindow()
                sp = w.convertPointToScreen_(p) if w is not None else p
                data: dict[str, Any] = {"at": point_to_global(sp.x, sp.y, _ctx.primary_h)}
                data["window"] = int(w.windowNumber()) if w is not None else 0
                if t in (1, 2, 3, 4):
                    data["clicks"] = int(event.clickCount())
                if t == 22:
                    data["dy"] = float(event.scrollingDeltaY())
                    data["precise"] = bool(event.hasPreciseScrollingDeltas())
                log.write("raw_" + _RAW_MOUSE[t], **data)
            elif t in _RAW_KEYS:
                log.write(
                    "raw_" + _RAW_KEYS[t],
                    chars=str(event.characters() or ""),
                    code=int(event.keyCode()),
                    mods=int(event.modifierFlags()) & 0xFFFF0000,
                )
        except Exception as e:  # noqa: BLE001
            log.write("monitor_error", error=repr(e)[:200])
        return event

    mask = 0
    for t in (*_RAW_MOUSE, *_RAW_KEYS):
        mask |= 1 << t
    monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, raw)

    def field_text() -> str:
        editor = field.currentEditor()
        return str(editor.string() if editor is not None else field.stringValue())

    def frame_of(view) -> list[float]:
        r = view.window().convertRectToScreen_(view.convertRect_toView_(view.bounds(), None))
        return to_global(r.origin.x, r.origin.y, r.size.width, r.size.height, _ctx.primary_h)

    def tilt_quad() -> list[list[float]]:
        """The tilted board's corners on screen (top-left, top-right, bottom-right, bottom-left)."""
        x, y, _w, _h = frame_of(tilt)
        return [[round(x + qx, 1), round(y + qy, 1)] for qx, qy in tilt.quad()]

    last_state: dict[str, Any] = {}
    readied = [False]

    def tick() -> None:
        if not readied[0] and time.time() - started > 0.3:
            readied[0] = True
            f = win.frame()
            log.write(
                "ready",
                pid=os.getpid(),
                tag=args.tag,
                name=name,
                window=int(win.windowNumber()),
                frame=to_global(f.origin.x, f.origin.y, f.size.width, f.size.height, _ctx.primary_h),
                targets={
                    "board": frame_of(board),
                    "double": frame_of(double),
                    "button_a": frame_of(button_a),
                    "button_b": frame_of(button_b),
                    "field": frame_of(field),
                    "area": frame_of(area_view),
                    "list": frame_of(list_view),
                    "marker": frame_of(marker),
                    "tile": frame_of(tile),
                    "well": frame_of(well),
                    "hindi": frame_of(hindi),
                    "tiny": frame_of(tiny),
                    "tilt": frame_of(tilt),
                },
                tilt_quad=tilt_quad(),
                cell=L.CELL,
                board_n=L.BOARD_N,
                piece=list(board.piece),
                round=_ctx.round,
                screens=[
                    to_global(s.frame().origin.x, s.frame().origin.y, s.frame().size.width, s.frame().size.height, _ctx.primary_h)
                    for s in screens
                ],
                second=(
                    {"window": int(second_win.windowNumber()), "target": frame_of(second)}
                    if second_win is not None
                    else None
                ),
                active=bool(app.isActive()),
            )
        state = {
            "field": field_text(),
            "area": str(area.string()),
            "list": round(float(clip.bounds().origin.y), 1),
            "piece": list(board.piece),
            "active": bool(app.isActive()),
            "key": bool(win.isKeyWindow()),
            "round": _ctx.round,
            "sheet": _ctx.sheet is not None,
        }
        if state != last_state:
            last_state.clear()
            last_state.update(state)
            log.write("state", **state)
        gone = args.parent and os.getppid() != args.parent  # NEO's core went away
        if gone or time.time() - started > args.lifetime:
            log.write("quit", reason="parent gone" if gone else "lifetime")
            app.terminate_(None)

    def on_command(line: str) -> None:
        cmd = command(line)
        if cmd == "deactivate":
            log.write("deactivating")
            app.deactivate()
        elif cmd == "quit":
            log.write("quit", reason="asked")
            app.terminate_(None)

    def stdin_reader() -> None:
        for line in sys.stdin:
            AppHelper.callAfter(on_command, line)
        AppHelper.callAfter(on_command, '{"cmd": "quit"}')  # NEO closed the pipe (or went away)

    threading.Thread(target=stdin_reader, daemon=True, name="selftest-stdin").start()
    _ctx.tick = tick
    timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(_TICK_S, ctl, "tick:", None, True)
    NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)  # ticks during drags too

    # On screen without taking the focus: NEO starts from a window that isn't in front.
    win.orderFrontRegardless()
    if second_win is not None:
        second_win.orderFrontRegardless()
    log.write("launched", pid=os.getpid(), name=name)
    _keep = (monitor, raw, timer, ctl, doc)  # noqa: F841 - alive for the app's lifetime
    app.run()  # (AppKit logs an exception in a handler and carries on: no alert panel)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
