"""The user's words that point at what is in front of them (neo2.core.agent.context_words): "summarize
what I copied", "translate the selected text", "ye copy kiya hua padho", "what does this say". Only those
make NEO read the selection, the clipboard, the front window or the Finder selection
(neo2.core.tools.context_now) — a bare "what is this?" / "ye kya hai" stays the camera's, a follow-up
about the conversation ("explain this answer", "summarize this chat") stays the conversation's, a
statement ("the selection was great") is no request, and an ordinary sentence is nobody's."""

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
    # a question about the words in front of the user ("what does this say"): the selection, else the window
    "what does this say",
    "what does this mean",
    "what does this error mean",
    "what does this text say?",
    "what's this error",
    "what is this message saying",
    "read this",
    "read this to me",
    "read this out loud",
    "Neo, can you read this for me",
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
    "ye copy kiya hua padho",
    "ye kya likha hai",
    "isko padho",
    # Hindi, as Gemini writes it (Devanagari)
    "ये कॉपी किया हुआ पढ़ो",
    "इसे ट्रांसलेट करो",
    "मैंने जो कॉपी किया वो पढ़ो",
    "इस पैराग्राफ का सारांश बताओ",
    "ये सेलेक्ट किया हुआ समझाओ",
    "क्लिपबोर्ड में क्या है",
    "ये क्या लिखा है",
    # a request that starts politely, or with the thing it is about
    "can you translate this into Hindi",
    "could you please summarize what I copied",
    "help me proofread this paragraph",
    "the selected text, translate it to Hindi",
    "translate this into Brazilian Portuguese",
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
    "read this label",
    "what does this sign say",
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
    # "this" that is the conversation: NEO's answer, the chat, a plan or a step
    "explain this answer",
    "explain this in simpler words",
    "can you explain this more simply",
    "explain this again",
    "explain this to me like I'm five",
    "summarize this conversation",
    "summarize this chat",
    "explain this concept",
    "explain this step",
    "how can I improve this plan",
    "improve this",
    "polish this",
    "simplify this",
    "how do you spell this word",
    "what does this mean for my exam",
    # a statement about the selection, the clipboard or the files, not a request
    "the selection was great",
    "these files are too big",
    "what I copied yesterday was wrong",
    "that's correct this time",
    "my clipboard manager isn't working",
    "the copied files are on the desktop",
    "this paragraph is too long",
    "I wrote this paragraph myself",
    "what's the selection process for IIT",
    "maine jo copy kiya wo galat tha",
    "मैंने जो कॉपी किया वो गलत था",
    # the text to work on is in the sentence itself
    "translate this to French: hello how are you",
    "translate this to French hello how are you doing today",
    "add this to my to-do list: buy milk",
    "is this sentence correct: I has a apple",
    'translate this "hola amigo"',
    # editing the selection in its own app: nothing for NEO to read
    "copy the selected text",
    "paste what I copied",
    "delete these files",
    "copy this text",
    "bold the selected text",
    # Hindi commands and the camera's
    "ये क्या है",
    "ये फ़ाइल कॉपी करो",
    "मैंने गलत चीज़ कॉपी की",
    "copy karo",
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
        ("ये कॉपी किया हुआ पढ़ो", "clipboard"),
        ("क्लिपबोर्ड में क्या है", "clipboard"),
        ("ये सेलेक्ट किया हुआ समझाओ", "selection"),
        ("what does this say", ""),  # a question about what is in front: never the clipboard
        ("read this", ""),
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
