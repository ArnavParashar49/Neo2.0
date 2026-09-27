"""Bare "stop" in English, Hinglish and Hindi cancels; a command that merely contains the word doesn't."""

import pytest

from neo2.core.reflex import is_bare_stop


@pytest.mark.parametrize(
    "text",
    ["stop", "Neo, stop it", "never mind", "bas karo", "Neo, ruk jao", "chup ho jao yaar", "rehne do", "bas",
     "arre bas kar", "ruko na", "jaane do", "चुप", "बस करो", "रुक जाओ"],
)
def test_bare_stop(text):
    assert is_bare_stop(text)


@pytest.mark.parametrize(
    "text",
    ["stop the timer", "quit the app", "bas karo timer", "ruk jao aur music band karo", "band karo music",
     "bus stop kahan hai", "chup chap music chalao",
     "band karo"]  # ambiguous: usually "turn it off" (the music), not "cancel NEO",
)
def test_a_command_containing_the_word_is_not_a_stop(text):
    assert not is_bare_stop(text)
