"""The wake gate: after a spoken "Neo", did anyone go on to say something?

Most spoken wakes were wasted. Since 10-02, 60% of them ended with nothing said, and in 79 of 119
the local voice detector heard no voice at all after the hit: a TV, a "you know" the spotter forced
onto "neo". Each one opens a Gemini Live session and streams the room to it until `listen_timeout_s`
closes it again. A real "Neo" is followed by a request, so the question is cheap to ask on the
Mac: does the detector NEO already runs (silero, or the energy fallback without it) hear a voice
within a few seconds of the hit?

:class:`WakeGate` answers it for one wake, in **audio time**: what it is fed is the microphone's
frames after the hit, in the order they were captured. During the Live connect they wait in the
microphone's queue and come all at once afterwards, so a wall clock would see seconds of the user
waiting as an instant.

* The first ``tail_ms`` (250 ms) of audio after the hit don't count: that is the end of "Neo"
  itself (the spotter fires on a stable partial, about 60 ms after the word is recognised).
* After that, ``speech_ms`` (300 ms) of voiced audio in all confirms the wake.
* Without that by the deadline, the wake is rejected. The deadline is the later of ``gate_s`` (3 s)
  of audio after the hit and ``gate_s`` of wall clock after ``opened_at`` (the session was open:
  people wait for the rim before they talk, the same anchor `listen_timeout_s` uses), so a slow
  connect never eats the user's time to speak.

The counting stops at the deadline, whatever the verdict, so a passed wake's ``speech_ms`` says how
much voice there was in its first seconds, not in the whole conversation after it.

What NEO does with the verdict is neo2.core.voice.live's (NEO_WAKE_GATE): ``off`` — no gate;
``shadow`` (the default) — the audio goes up exactly as before, and the verdict is only logged and
counted; ``on`` — the audio after the hit is held back until the gate confirms, then sent in full
and in order, and a rejected wake's session is closed without anything of the room having left the
Mac (or, when the session is in use anyway — typed words being answered — it only stops listening).
Only a spoken "Neo" is ever gated: push-to-talk, the summon keys, the orb, a host and a check-in are
the user asking (or NEO, after a check-in), never a guess; one of them while a gate waits settles it.

What is measured is the room, not the request: a TV or someone else talking passes the gate (about
40 of the 119 wakes above had a voice but no words), so ``on`` takes "nothing said" from about 60% of
spoken wakes to about 20%. In shadow, a drop whose session then got words (``wake_gate_drop_words``)
is the gate being wrong; switching it on is for when, over 30 or more spoken wakes, that stays at
2% of the wakes with words or below.

Nothing here keeps audio: it counts numbers. The log line and the daily counters (``wake_voice``,
``wake_gate_pass``/``_drop``, ``wake_gate_drop_words``, ``wake_gate_pass_nowords``; the quality
report's wake line reads them) are numbers and fixed words only.
"""

from __future__ import annotations

import time
from typing import Literal, get_args

Mode = Literal["off", "shadow", "on"]
MODES: tuple[str, ...] = get_args(Mode)
Verdict = Literal["pending", "confirm", "reject"]

RATE = 16000  # the microphone's frames (neo2.core.voice.audio.MIC_RATE): mono, so one sample each
TAIL_MS = 250  # the end of "Neo" itself: voice in it says nothing about a request after it
_clock = time.time  # (a test's hook: the wall clock the deadline's second half is measured by)


class WakeGate:
    """One spoken wake's gate. Feed it every frame NEO would send after the hit (:meth:`feed`)."""

    def __init__(
        self,
        mode: str,
        gate_s: float = 3.0,
        speech_ms: int = 300,
        *,
        opened_at: float = 0.0,
        tail_ms: int = TAIL_MS,
        rate: int = RATE,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"wake gate mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.gate_s = float(gate_s)
        self.opened_at = float(opened_at)  # wall clock: when the user could start talking
        self.rate = int(rate)
        self._tail = int(tail_ms * rate / 1000)
        self._need = max(1, int(speech_ms * rate / 1000))
        self._window = int(self.gate_s * rate)
        self.frames = 0  # audio after the hit, in frames
        self.voiced = 0  # voiced frames counted (after the tail, until the deadline)
        self.first_voice = -1  # frames after the hit where the first counted voice was
        self.verdict: Verdict = "pending"
        self.settled = False  # past the deadline: nothing more is counted
        self.forced = False  # confirmed by something other than speech (a wake by hand, a reconnect)

    def feed(self, voiced: int, frames: int) -> Verdict:
        """The next `frames` of audio after the hit, of which the detector called `voiced` a voice.
        Returns the verdict so far: ``pending`` until it confirms, or until the deadline rejects."""
        if self.settled:
            return self.verdict
        start, self.frames = self.frames, self.frames + max(0, int(frames))
        counted = min(max(0, int(voiced)), self.frames - self._tail) if self.frames > self._tail else 0
        if counted > 0:
            if self.first_voice < 0:
                self.first_voice = max(start, self._tail)
            self.voiced += counted
        if self.verdict == "pending" and self.voiced >= self._need:
            self.verdict = "confirm"
        if self.frames >= self._window and _clock() >= self.opened_at + self.gate_s:
            self.settled = True
            if self.verdict == "pending":
                self.verdict = "reject"
        return self.verdict

    def pending(self) -> bool:
        return self.verdict == "pending"

    def confirm(self) -> None:
        """Someone other than the room settled it: the user summoned NEO by hand meanwhile, or a
        dropped session is coming back mid-turn (words came). Nothing is left to ask."""
        if self.verdict == "pending":
            self.verdict, self.forced = "confirm", True

    @property
    def passed(self) -> bool:
        """The verdict as it stands. One that never settled (the session closed first) is a drop:
        no voice confirmed the wake in the time it had."""
        return self.verdict == "confirm"

    def line(self, *, words: bool) -> str:
        """The one content-free log line for this wake, at its session's close."""
        first = f"{self.first_voice * 1000 // self.rate}" if self.first_voice >= 0 else "-"
        return (
            f"[voice] wake gate ({self.mode}): {'pass' if self.passed else 'drop'}; "
            f"words={'yes' if words else 'no'}; speech_ms={self.voiced * 1000 // self.rate}; "
            f"first_voice_ms={first}"
        )

    def counters(self, *, words: bool) -> list[str]:
        """The daily counters this wake's verdict bumps (neo2.core.reflex.learn)."""
        if self.passed:
            return ["wake_gate_pass"] + ([] if words else ["wake_gate_pass_nowords"])
        return ["wake_gate_drop"] + (["wake_gate_drop_words"] if words else [])


def mode_now() -> str:
    """NEO_WAKE_GATE as the settings have it; settings it can't read are ``off`` (the voice as it
    always was)."""
    try:
        from neo2.core.config import settings

        mode = str(settings().wake_gate)
    except Exception:  # noqa: BLE001
        return "off"
    return mode if mode in MODES else "off"


def new_gate(*, opened_at: float) -> WakeGate | None:
    """The gate for a spoken wake that just opened its session, or None when NEO_WAKE_GATE is off."""
    mode = mode_now()
    if mode == "off":
        return None
    try:
        from neo2.core.config import settings

        s = settings()
        return WakeGate(mode, s.wake_gate_s, s.wake_gate_speech_ms, opened_at=opened_at)
    except Exception:  # noqa: BLE001 — the gate's own trouble never costs a wake
        return None


def tally(name: str) -> None:
    """Bump one of NEO.app's daily counters (content-free: the quality report's wake line)."""
    try:
        from neo2.core.reflex import learn

        learn.count(name)
    except Exception:  # noqa: BLE001 — a counter's trouble never costs a wake
        pass
