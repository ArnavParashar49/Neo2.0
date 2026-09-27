"""``python -m neo2.core``: the same as ``python -m neo2`` (NEO.app's core).

Only when run: importing it (a doc generator, ``--doctest-modules``, a plugin scan of ``neo2``)
starts nothing."""

if __name__ == "__main__":
    from neo2.core.main import main

    main()
