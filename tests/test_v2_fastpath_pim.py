"""The task kinds' fast paths (plan §7.3): every English, Hinglish and Devanagari pattern plans the
right tool and arguments while its kind is on, a check that can't tell hands the words to a model,
and in shadow (the release default) nothing claims anything it didn't claim before. Apps, lists and
learned sites are fixtures; nothing is looked up or opened."""

from __future__ import annotations

import pytest

from neo.agent.session import _match_fast_path
from neo.decide import eventkit, inventory, sites
from neo.decide.inventory import App
from neo.events import EventBus

NOTES = App(bundle_id="test.notes", name="Notes", path="/Applications/Notes.app")
WEBBY = App(bundle_id="test.webby", name="Webby", path="/Applications/Webby.app", handles=("https", "http"))
KINDS = ("write_note", "add_reminder", "add_event", "read_calendar", "read_mail", "open_app", "web_search")


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    from neo import events
    from neo.tools import load_all

    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()
    words = tmp_path / "words"
    words.write_text("notebook\nrestaurant\ndinner\ncheap\nflight\napple\nnews\nweather\n")
    monkeypatch.setattr(inventory, "_WORDS_FILE", str(words))
    monkeypatch.setattr(inventory, "snapshot", lambda: [NOTES, WEBBY])
    monkeypatch.setattr(
        inventory, "_common_words", lambda ws: {w: w in {"notes", "apple", "news"} for w in ws}
    )
    monkeypatch.setattr(sites, "_lower_known", {})
    monkeypatch.setattr(sites, "_country", lambda: "in")
    monkeypatch.setitem(eventkit._titles, "reminder", ("Shopping", "Errands"))


@pytest.fixture
def on(decide_modes):
    decide_modes(**dict.fromkeys(KINDS, "on"))


# (utterance, what it plans while its kind is on)
ON = [
    # reminders
    ("remind me to call mom at 5", ("reminder_add", {"title": "call mom", "due": "at 5"})),
    (
        "remind me tomorrow at 9 to call the bank",
        ("reminder_add", {"title": "call the bank", "due": "tomorrow at 9"}),
    ),
    ("remind me in 20 minutes to stretch", ("reminder_add", {"title": "stretch", "due": "in 20 minutes"})),
    (
        "remind me to water the plants tomorrow",
        ("reminder_add", {"title": "water the plants", "due": "tomorrow"}),
    ),
    ("remind me to go to the gym at 6", ("reminder_add", {"title": "go to the gym", "due": "at 6"})),
    ("add milk to my shopping list", ("reminder_add", {"title": "milk", "list": "shopping"})),
    (
        "kal subah 9 baje doctor ko call ka reminder lagao",
        ("reminder_add", {"title": "doctor ko call", "due": "kal subah 9 baje"}),
    ),
    ("shaam 6 baje dawai lena yaad dilana", ("reminder_add", {"title": "dawai lena", "due": "shaam 6 baje"})),
    ("कल सुबह 9 बजे दवाई लेना याद दिलाना", ("reminder_add", {"title": "दवाई लेना", "due": "कल सुबह 9 बजे"})),
    # events
    (
        "put lunch with sam on my calendar tomorrow at 1",
        ("calendar_add", {"title": "lunch with sam", "start": "tomorrow at 1"}),
    ),
    (
        "kal shaam 5 baje dentist calendar mein daal do",
        ("calendar_add", {"title": "dentist", "start": "kal shaam 5 baje"}),
    ),
    # reading the calendar
    ("what's on my calendar today", ("calendar_today", {})),
    ("what's on my calendar tomorrow", ("calendar_today", {"day": "tomorrow"})),
    ("what's on my schedule for next friday", ("calendar_today", {"day": "next friday"})),
    ("what's on my calendar next week", ("calendar_today", {"day": "next week"})),
    ("kal ka calendar kya hai", ("calendar_today", {"day": "kal"})),
    ("kal mera calendar kya tha", ("calendar_today", {"day": "yesterday"})),
    ("mera schedule kya hai", ("calendar_today", {})),
    ("आज का कैलेंडर क्या है", ("calendar_today", {"day": "आज"})),
    # notes and mail
    (
        "note kar lo ki kal meeting 10 baje hai",
        ("notes_create", {"title": "", "body": "kal meeting 10 baje hai"}),
    ),
    (
        "likh lo ki wifi password fridge pe hai",
        ("notes_create", {"title": "", "body": "wifi password fridge pe hai"}),
    ),
    ("नोट कर लो कि दूध लाना है", ("notes_create", {"title": "", "body": "दूध लाना है"})),
    ("make a note saying buy milk", ("notes_create", {"title": "", "body": "buy milk"})),
    ("mera mail check karo", ("mail_unread", {})),
    ("koi naya email aaya?", ("mail_unread", {})),
    ("check my email", ("mail_unread", {})),
    # opening
    ("safari kholo", ("open_app", {"target": "safari"})),
    ("mera mail khol do", ("open_app", {"target": "mail"})),
    ("open safari", ("open_app", {"target": "safari"})),
    # the web: evidence decides what "in/on <W>" is
    ("look up the weather on monday", ("web_search", {"query": "the weather on monday"})),
    ("search for restaurants in Delhi", ("web_search", {"query": "restaurants in Delhi"})),
    ("search for the best laptops under 1000", ("web_search", {"query": "the best laptops under 1000"})),
    ("search 2tb ssd on shopkart.in", ("search_site", {"site": "shopkart.in", "query": "2tb ssd"})),
    ("new note", ("hotkey", {"keys": "cmd+n", "app": "note"})),
]

# words no check can vouch for: a model decides (None here)
TO_A_MODEL = [
    "remind me to put the kettle on at 5",  # "on at 5": the title's own word
    "add milk to my groceries list",  # a list NEO hasn't seen in the store
    "add dentist to my calendar friday",  # no clock time: the model asks
    "put it on my calendar tomorrow at 5",  # "it": only the conversation knows
    "search 2tb ssd on shopkart",  # a site word NEO hasn't landed on yet
    "search news on apple",  # about apple, or on a site?
    "search milk in notes",  # an app on this Mac
]


@pytest.mark.parametrize("text, expected", ON)
def test_each_pattern_plans_its_tool_while_the_kind_is_on(on, text, expected):
    assert _match_fast_path(text) == expected


@pytest.mark.parametrize("text", TO_A_MODEL)
def test_a_check_that_cant_tell_hands_the_words_to_a_model(on, text):
    assert _match_fast_path(text) is None


def test_a_site_is_claimed_once_a_search_landed_on_it(on):
    for text in (
        "search 2tb ssd on shopkart",
        "search shopkart for 2tb ssd",
        "open shopkart and search for 2tb ssd",
    ):
        assert _match_fast_path(text) is None
    sites.learn_landing("shopkart", "www.shopkart.in", "https://www.shopkart.in/s?q={searchTerms}")
    for text in (
        "search 2tb ssd on shopkart",
        "search shopkart for 2tb ssd",
        "open shopkart and search for 2tb ssd",
    ):
        assert _match_fast_path(text) == ("search_site", {"site": "shopkart", "query": "2tb ssd"})
    assert _match_fast_path("search news on apple") is None  # still the model's


def test_a_list_is_claimed_only_once_neo_has_read_the_lists(on, monkeypatch):
    monkeypatch.delitem(eventkit._titles, "reminder")
    assert _match_fast_path("add milk to my shopping list") is None


# In shadow (the release default) the new patterns claim nothing: today's fast paths answer, and
# words they never covered still go to a model.
SHADOW = {
    "remind me to call mom at 5": None,
    "add milk to my shopping list": None,
    "put lunch with sam on my calendar tomorrow at 1": None,
    "what's on my calendar tomorrow": None,
    "kal ka calendar kya hai": None,
    "mera schedule kya hai": None,
    # today NEO's own memory takes "note …" (a note in the notes app once write_note is on)
    "note kar lo ki kal meeting 10 baje hai": (
        "memory",
        {"action": "remember", "text": "kar lo ki kal meeting 10 baje hai", "kind": "note"},
    ),
    "mera mail check karo": None,
    "safari kholo": None,
    "what's on my calendar today": ("calendar_today", {}),
    "check my email": ("mail_unread", {}),
    "make a note saying buy milk": ("notes_create", {"title": "", "body": "buy milk"}),
    "search 2tb ssd on amazon": ("search_site", {"site": "amazon", "query": "2tb ssd"}),
    "search news on apple": ("web_search", {"query": "news on apple"}),
    "search for restaurants in Delhi": ("web_search", {"query": "restaurants in Delhi"}),
    "search milk in notes": None,
    "new note": ("hotkey", {"keys": "cmd+n", "app": "Notes"}),
}


@pytest.mark.parametrize("text, expected", SHADOW.items())
def test_in_shadow_the_fast_paths_are_todays(text, expected):
    assert _match_fast_path(text) == expected
