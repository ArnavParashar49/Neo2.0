"""NEO's self-tests: NEO checking its own abilities on this Mac, when the user asks it to.

``controls`` — "Neo, check your computer controls": NEO opens a small test window of its own (a
separate app it starts, ``testwin``), drives it only through its own tools (screenshot, click,
double and right click, drag, scroll, type_text, hotkey, ax_tree, ax_press, ax_set_value), and
compares what the window received with what it sent. See ``neo/selftest/controls.py``.

Nothing heavy is imported here: the reflex asks ``phrases`` whether a sentence is a request for a
self-test on every utterance.
"""
