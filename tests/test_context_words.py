"""The user's words that point at what is in front of them (neo2.core.agent.context_words): "summarize
what I copied", "translate the selected text", "ye copy kiya hua padho". Only those make NEO read the
selection, the clipboard, the front window or the Finder selection (neo2.core.tools.context_now) — a bare
"what is this?" / "ye kya hai" stays the camera's, and an ordinary sentence is nobody's."""

from __future__ import annotations

import pytest

from neo2.core.agent import context_words

ASKS = [
    # the clipboard
    "summarize what I copied",
    "Summarise what I just copied",
    "translate what I've copied into Hindi",
    "what did I copy",
    "read my clipboard",
    "what's in my clipboard?",
    "what's on the clipboard",
    "explain the link I copied",
    "add the address I just copied to my contacts",
    # the selection
    "translate the selected text",
    "explain the highlighted text",
    "proofread the selected paragraph",
    "what I selected, summarize it",
    "summarize the selection",
    "what did I just highlight",
    # "this" with a verb that reads text
    "translate this",
    "explain this",
    "rewrite this",
    "proofread this",
    "summarize this",
    "Neo, translate this please",
    "translate this to Hindi",
    "rewrite this more politely",
    "summarise this email",
    "summarize this document",
    "explain this message",
    # "this" naming text
    "what does this paragraph say",
    "is this sentence correct",
    "fix the grammar in this text",
    # this, into an app
    "add this to my calendar",
    "remind me about this",
    # the Finder selection
    "summarize these files",
    "what are the selected files",
    # Hinglish
    "ye copy kiya hua translate karo",
    "jo maine copy kiya hai uska summary do",
    "maine jo copy kiya wo padho",
    "ye select kiya hua padho",
    "isko translate karo",
    "is paragraph ko summarize karo",
    "clipboard mein kya hai",
]

NOT_ASKS = [
    # the camera's
    "what is this?",
    "what's this",
    "what is this thing",
    "ye kya hai",
    "yeh kya hai?",
    "what am I holding",
    "what does this label say",
    # a bare question that could be about anything in view: never reads the selection
    "what does this mean",
    "read this",
    # the page in the browser, a video, the screen: other tools'
    "summarize this page",
    "explain this video",
    "translate this menu",  # (held up to the camera, as often as not)
    "what's on my screen",
    "what is this on my screen",
    # about the clipboard as a thing, not its contents
    "open the clipboard app",
    "open my clipboard history",
    # "that" is the conversation's, never what is in front of the user
    "can you explain that?",
    "explain that again",
    "is it correct that the meeting is at 4",
    "add that to my calendar",
    "remind me about that tomorrow",
    "translate that into French",
    # commands that mention copying or selecting
    "copy this file to the desktop",
    "select all",
    "highlight the text in yellow",
    "I copied the wrong thing",
    "send this message",
    "reply to this email",
    "close this document",
    # ordinary sentences
    "what time is it",
    "open notes",
    "remind me to call mom",
    "translate hello to Spanish",
    "explain quantum physics",
    "search for this",
    "add milk to my shopping list",
    "this is great",
    "",
]


@pytest.mark.parametrize("text", ASKS)
def test_the_users_words_that_point_at_what_is_in_front(text):
    assert context_words.wants(text), text


@pytest.mark.parametrize("text", NOT_ASKS)
def test_the_camera_the_page_and_ordinary_sentences_are_not(text):
    assert not context_words.wants(text), text


@pytest.mark.parametrize(
    "text, kind",
    [
        ("summarize what I copied", "clipboard"),
        ("read my clipboard", "clipboard"),
        ("ye copy kiya hua translate karo", "clipboard"),
        ("explain the link I copied", "clipboard"),
        ("open the files I copied", "clipboard"),  # files on the clipboard: the clipboard's names
        ("translate the selected text", "selection"),
        ("explain the highlighted text", "selection"),
        ("summarize the selection", "selection"),
        ("ye select kiya hua padho", "selection"),
        ("summarize these files", "files"),
        ("what are the selected files", "files"),
        ("translate this", ""),  # no word for where: the selection, else the window
        ("add this to my calendar", ""),
        ("", ""),
    ],
)
def test_an_explicit_word_says_where(text, kind):
    assert context_words.kind(text) == kind


def test_a_calls_own_arguments_are_no_sentence():
    # Gemini Live hands a direct tool the call's arguments when it has no sentence of the user's
    # (live.py: `asked or str(args)`): they never read as a request.
    assert context_words.wants("{'what': 'clipboard'}") is False
    assert context_words.wants("{'what': 'selection'}") is False
