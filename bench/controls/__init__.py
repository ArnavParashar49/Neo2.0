"""The hermetic computer-controls bench.

NEO's real tool handlers (click, drag, scroll, hotkey, ax_press, type_text, ax_tree, click_text,
screenshot) run against a pretend Mac (``fakes``): Accessibility trees that count every call, a
window server that routes posted events, synthetic Retina frames that count captures, a clock that
counts every sleep, and a scripted brain that counts round trips and tokens. Nothing touches the
real Mac and no model is called (NEO_NO_EFFECTS=1 refuses anything a fake misses).

    python -m bench.controls.run            # the table, and bench/results/<sha>.json
    python -m bench.controls.run --diff bench/baselines/step0.json

``scenarios`` holds the task scenarios (each a fake app state, an utterance and a goal check);
``run`` the per-primitive and per-task measurements and the report.
"""
