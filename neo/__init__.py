"""NEO — a free, local-first AI agent for your Mac.

NEO's own status lines go to the ``neo`` logger. Embedded in another app it says nothing unless
that app configures logging; NEO.app and the CLI print them to stdout (:func:`log_to_stdout`),
which is where NEO.app's core.log reads them from.
"""

from __future__ import annotations

import logging
import sys

logging.getLogger("neo").addHandler(logging.NullHandler())


def log_to_stdout() -> None:
    """NEO.app and the CLI: NEO's log lines on stdout, as plain text (the lines it used to print).
    Only NEO's own loggers — a library's INFO lines (every HTTP request httpx makes) stay out."""
    lg = logging.getLogger("neo")
    if not any(getattr(h, "_neo_stdout", False) for h in lg.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        h._neo_stdout = True  # type: ignore[attr-defined]
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
