"""NEO's eyes on a camera frame ("what am I holding?"), in three layers:

* :mod:`.apple` — Apple Vision, on this Mac, nothing to download: labels for the whole frame, the
  words printed on things, and hands (how many, fingers raised, where);
* :mod:`.detect` — D-FINE small (Apache-2.0, 40 MB, transformers on the CPU): object boxes for the live
  preview, and which object is by the hand;
* :mod:`.answer` — the answer itself, from NEO's cloud brains that see, as the user chose: Gemini
  first, then Cerebras's vision model — one still per question, with the two layers above as short
  hints. Never a text-only brain. Cloud answers off (``vision_cloud_ok``), or both brains down: the
  answer is made from the on-device labels alone, and says so.

:func:`.pipeline.look` runs them for one frame. A frame is only ever held in memory: nothing here
writes it anywhere, and only :mod:`.answer` sends it off the Mac (to Gemini / Cerebras, never
elsewhere). A layer that isn't installed or downloaded raises :class:`Unavailable` with a message for
the user; the pipeline answers with what the others saw.
"""

from __future__ import annotations


class Unavailable(RuntimeError):
    """A layer can't run here (not installed, not downloaded, not macOS). ``str()`` is for the user."""


__all__ = ["Unavailable"]
