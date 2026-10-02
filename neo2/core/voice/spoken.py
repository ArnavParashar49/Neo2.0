"""What NEO says when it needs the user's OK — and what it never reads out.

A voice that reads `/private/tmp/…/9f3c1d2e-…/report.pdf` aloud spends half a minute on a folder
nobody asked about. So, for anything meant to be heard:

* a file path is said as the file's name only — never its folders (:func:`speakable`);
* a long name is cut short (:func:`short_name`);
* a pending confirmation is one short line, "I need your OK to move report.pdf to the Trash."
  (:func:`ok_line`) — said once; the card (NEO's page, the notch's Allow / Cancel buttons) shows
  the rest, and the answer comes from its buttons or the user's own "yes" / "no".

:class:`Asked` keeps what has been said already, so the same card is never read out twice and a
task that needs a second OK doesn't ask aloud again while the buttons are on screen.

Pure text in, text out: nothing here talks to a model, the bus or the screen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A spoken file name is at most this long; a longer one keeps its start and its extension.
NAME_MAX = 32
_NAME_HEAD = 24
_EXT_MAX = 6
# One spoken line: a summary longer than this is cut at a word.
LINE_MAX = 140

# An absolute or home-relative path with at least two components: "/a/b", "~/a/b.txt", "/a/b/".
# Not a URL ("https://x.y/a/b": the slash follows ":" or "/" or a letter), not "and/or", "24/7",
# a date or a fraction (those have no leading slash), and not a lone "/Applications".
_PATH = re.compile(r"(?<![\w:/.~-])(?:~/|/)(?:[^\s/`'\"“”‘’<>|]+/)+[^\s/`'\"“”‘’<>|]*")
# The same inside quotes or backticks, where a path may hold spaces ("/Users/me/My Files/a b.txt").
_QUOTED_PATH = re.compile(r"([`'\"“‘])((?:~/|/)(?:[^/`'\"“”‘’<>|\n]+/)+[^/`'\"“”‘’<>|\n]*)([`'\"”’])")
_TRAILING = ".,;:!?)]}"


def short_name(name: str) -> str:
    """A file name short enough to say: a long one keeps its first characters and its extension."""
    name = name.strip()
    if len(name) <= NAME_MAX:
        return name
    stem, dot, ext = name.rpartition(".")
    if not dot or not stem or len(ext) > _EXT_MAX or " " in ext:
        return name[:_NAME_HEAD].rstrip(" ._-") + "…"
    return stem[:_NAME_HEAD].rstrip(" ._-") + "…" + dot + ext


def _name_of(path: str) -> str:
    """The last component of `path` (a folder's own name for a path ending in a slash)."""
    parts = [p for p in path.split("/") if p and p != "~"]
    return short_name(parts[-1]) if parts else path


def speakable(text: str, limit: int | None = None) -> str:
    """`text` as it may be said aloud: every file path is its file's name only, long names cut
    short. Everything else — web addresses included — is left as it is. `limit`: cut at a word
    when longer."""

    def quoted(m: re.Match[str]) -> str:
        return _name_of(m.group(2))  # the quotes go too: a name is said, not spelled

    def bare(m: re.Match[str]) -> str:
        path = m.group(0)
        tail = ""
        while path and path[-1] in _TRAILING:  # the sentence's own punctuation stays outside
            path, tail = path[:-1], path[-1] + tail
        return _name_of(path) + tail

    out = _PATH.sub(bare, _QUOTED_PATH.sub(quoted, text))
    out = " ".join(out.split())
    if limit is not None and len(out) > limit:
        cut = out[:limit].rsplit(" ", 1)[0].rstrip(" ,;:—-")
        out = (cut or out[:limit]) + "…"
    return out


def ok_line(summary: str) -> str:
    """The one line NEO says for a pending confirmation: "I need your OK to <what>." — the card's
    summary in plain words, file names only."""
    what = speakable(summary, LINE_MAX).strip().rstrip(".?! ")
    return f"I need your OK to {what}." if what else "I need your OK."


@dataclass
class Asked:
    """Which confirmations NEO has asked aloud, so none is asked twice.

    `say(card, task, shown)` answers "is this card's line to be said now?":

    * never for a card whose line was already handed over (the model calling the tool again, a
      result arriving twice) — unless `again`, for the one case that needs it: the user said
      "yes" to a card they have not heard;
    * and, while a UI shows the card's buttons (`shown`), not for a later card of a task NEO
      already asked about aloud: the buttons come up again, the question isn't repeated. With no
      UI the voice is the only way to ask, so each card gets its one line.
    """

    cards: list[str] = field(default_factory=list)  # action ids whose line was handed over, newest last
    task: str = ""  # the task of the card last asked aloud ("" = a direct command: no task)
    _KEEP = 16

    def say(self, card: str, task: str = "", *, shown: bool = False, again: bool = False) -> bool:
        if not card:
            return False
        if card in self.cards and not again:
            return False
        if not again and shown and task and task == self.task:
            self._note(card)  # its buttons are up; the question was asked for this task already
            return False
        self._note(card)
        self.task = task
        return True

    def _note(self, card: str) -> None:
        if card in self.cards:
            self.cards.remove(card)
        self.cards.append(card)
        del self.cards[: -self._KEEP]
