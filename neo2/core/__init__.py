"""NEO's own agent: NEO.app's core (``python -m neo2``) and the engine inside the ``neo2`` library.

Until 2.1 this was the top-level package ``neo``. A checkout still answers to that name (its
compatibility shim); the installed ``neo2`` distribution doesn't.

NEO's own status lines go to the ``neo`` logger, the name they have always had: a module here logs
as ``neo.<module>`` (:func:`logger`) and approvals to ``neo.audit``. Embedded in another app it says
nothing unless that app configures logging; NEO.app and the CLI print them to stdout
(:func:`log_to_stdout`), which is where NEO.app's core.log reads them from.
"""

from __future__ import annotations

import logging
import sys

# First, before anything runs: under pytest or with NEO_NO_EFFECTS=1, nothing real happens.
from neo2.core import effects as _effects  # noqa: F401

LOGGER = "neo"  # NEO's logger: NEO.app and hosts configure it (and "neo.audit") by this name
logging.getLogger(LOGGER).addHandler(logging.NullHandler())


def logger(module: str) -> logging.Logger:
    """The logger of one of NEO's modules, from its ``__name__``: under the ``neo`` logger NEO has
    always used, so ``neo2.core.agent.session`` logs to its child ``agent.session``."""
    if module == __name__ or module.startswith(__name__ + "."):
        module = LOGGER + module[len(__name__) :]
    return logging.getLogger(module)


def log_to_stdout() -> None:
    """NEO.app and the CLI: NEO's log lines on stdout, as plain text (the lines it used to print).
    Only NEO's own loggers — a library's INFO lines (every HTTP request httpx makes) stay out."""
    lg = logging.getLogger(LOGGER)
    if not any(getattr(h, "_neo_stdout", False) for h in lg.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        h._neo_stdout = True  # type: ignore[attr-defined]
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
