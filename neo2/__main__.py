"""``python -m neo2``: NEO.app's core (what NEO.app starts), or NEO in a terminal.

    python -m neo2            # voice + the overlay's websocket: what NEO.app runs
    python -m neo2 --text     # a text REPL in the terminal
    python -m neo2 --doctor   # check keys, permissions, models
    python -m neo2 --local    # download and serve the offline model

Nothing is imported until it runs: ``import neo2`` stays the standard library only.
"""

from __future__ import annotations


def _main() -> None:
    from neo2.core.main import main

    main()


if __name__ == "__main__":
    _main()
