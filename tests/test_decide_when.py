"""neo.decide.when: time phrases in English, Hinglish and Hindi against a fixed "now"
(Saturday 26 September 2026, 2 PM). A phrase the grammar doesn't cover is None, never a guess."""

from __future__ import annotations

from datetime import datetime

import pytest

from neo.decide.when import When, parse

NOW = datetime(2026, 9, 26, 14, 0)  # a Saturday afternoon


def at(d: int, h: int = 0, m: int = 0, month: int = 9, year: int = 2026) -> datetime:
    return datetime(year, month, d, h, m)


@pytest.mark.parametrize(
    "phrase, start, inferred",
    [
        # English clock times
        ("5pm", at(26, 17), ""),
        ("at 5 p.m.", at(26, 17), ""),
        ("at 5:30 pm", at(26, 17, 30), ""),
        ("17:30", at(26, 17, 30), ""),
        ("09:15", at(27, 9, 15), "9:15 AM tomorrow"),  # passed today: the next one
        ("11am", at(27, 11), "11 AM tomorrow"),
        ("at 5", at(26, 17), "5 PM today"),  # no AM/PM: the next 5 o'clock
        ("5 o'clock", at(26, 17), "5 PM today"),
        ("at 1", at(27, 1), "1 AM tomorrow"),
        ("tomorrow at 5", at(27, 17), "5 PM tomorrow"),  # another day: a daytime hour
        ("tomorrow at 9", at(27, 9), "9 AM tomorrow"),
        ("tomorrow at 9pm", at(27, 21), ""),
        ("today at 5", at(26, 17), "5 PM today"),
        ("tonight", at(26, 21), "9 PM today"),
        ("tonight at 10", at(26, 22), ""),
        ("this evening", at(26, 18), "6 PM today"),
        ("tomorrow morning", at(27, 9), "9 AM tomorrow"),
        ("tomorrow afternoon at 3", at(27, 15), ""),
        ("in 20 minutes", at(26, 14, 20), ""),
        ("in an hour", at(26, 15), ""),
        ("in half an hour", at(26, 14, 30), ""),
        ("in 2 hours", at(26, 16), ""),
        ("monday at 10", at(28, 10), "10 AM Monday"),
        ("on monday at 10am", at(28, 10), ""),
        ("next friday 6pm", at(2, 18, month=10), ""),
        ("on 3 oct at 10", at(3, 10, month=10), "10 AM Saturday 3 October"),
        ("october 3rd at 4:30pm", at(3, 16, 30, month=10), ""),
        ("midnight", at(27, 0), "12 AM tomorrow"),
        # Hinglish
        ("kal subah 9 baje", at(27, 9), ""),
        ("kal shaam 5 baje", at(27, 17), ""),
        ("aaj raat 10 baje", at(26, 22), ""),
        ("raat 2 baje", at(27, 2), ""),  # the small hours after tonight
        ("saadhe 5", at(26, 17, 30), "5:30 PM today"),
        ("shaam saadhe 6 baje", at(26, 18, 30), ""),
        ("sawa 5 baje", at(26, 17, 15), "5:15 PM today"),
        ("paune 5", at(26, 16, 45), "4:45 PM today"),
        ("kal dopahar dhai baje", at(27, 14, 30), ""),
        ("dedh baje", at(27, 1, 30), "1:30 AM tomorrow"),
        ("parso subah 7 baje", at(28, 7), ""),
        ("20 minute mein", at(26, 14, 20), ""),
        ("2 ghante mein", at(26, 16), ""),
        ("aadhe ghante mein", at(26, 14, 30), ""),
        ("agle somvar 10 baje", at(28, 10), "10 AM Monday"),
        ("shukravar ko shaam 7 baje", at(2, 19, month=10), ""),
        # Hindi
        ("कल सुबह ९ बजे", at(27, 9), ""),
        ("परसों शाम 7 बजे", at(28, 19), ""),
        ("आज रात 10 बजे", at(26, 22), ""),
        ("साढ़े 5 बजे", at(26, 17, 30), "5:30 PM today"),
        ("20 मिनट में", at(26, 14, 20), ""),
    ],
)
def test_times(phrase, start, inferred):
    w = parse(phrase, NOW)
    assert w is not None, phrase
    assert (w.start, w.inferred, w.date_only) == (start, inferred, False)


@pytest.mark.parametrize(
    "phrase, day",
    [
        ("today", at(26)),
        ("aaj", at(26)),
        ("tomorrow", at(27)),
        ("kal", at(27)),  # something to do: tomorrow
        ("कल", at(27)),
        ("day after tomorrow", at(28)),
        ("parso", at(28)),
        ("monday", at(28)),
        ("next monday", at(28)),
        ("somvar", at(28)),
        ("अगले सोमवार", at(28)),
        ("saturday", at(3, month=10)),  # today is Saturday: the next one
        ("next week", at(3, month=10)),
        ("agle hafte", at(3, month=10)),
        ("3 oct", at(3, month=10)),
        ("the 3rd", at(3, month=10)),
        ("on 1 sept", at(1, year=2027)),  # passed this year: next year's
        ("1 sept 2026", at(1)),
    ],
)
def test_days(phrase, day):
    w = parse(phrase, NOW)
    assert w == When(day, None, True, ""), phrase


def test_ranges():
    w = parse("3 to 4 pm", NOW)
    assert (w.start, w.end) == (at(26, 15), at(26, 16))
    w = parse("tomorrow from 3pm to 4", NOW)
    assert (w.start, w.end) == (at(27, 15), at(27, 16))
    w = parse("11 to 1 pm", NOW + (at(26, 9) - NOW))  # said at 9 AM: 11 AM to 1 PM
    assert (w.start, w.end) == (at(26, 11), at(26, 13))
    w = parse("monday 10-11am", NOW)
    assert (w.start, w.end) == (at(28, 10), at(28, 11))
    w = parse("11am to 1", NOW)
    assert (w.start, w.end) == (at(27, 11), at(27, 13))
    w = parse("10pm to 1", NOW)
    assert (w.start, w.end) == (at(26, 22), at(27, 1))


def test_a_weekday_that_is_today():
    monday = datetime(2026, 9, 28, 10, 0)
    assert parse("monday at 5pm", monday).start == datetime(2026, 9, 28, 17)
    assert parse("monday at 9am", monday).start == datetime(2026, 10, 5, 9)  # passed: next week's
    assert parse("monday", monday).start == datetime(2026, 10, 5)
    assert parse("next monday at 5pm", monday).start == datetime(2026, 10, 5, 17)


def test_the_past_for_reads():
    # "kal" is yesterday only in a past-tense read ("kal kya tha" -> "kal tha" reaches the parser)
    assert parse("kal tha", NOW, future=False).start == at(25)
    assert parse("कल था", NOW, future=False).start == at(25)
    assert parse("kal tha", NOW).start == at(27)  # a create: always tomorrow
    assert parse("kal", NOW, future=False).start == at(27)  # no past marker: tomorrow
    assert parse("yesterday", NOW).start == at(25)
    assert parse("monday was", NOW, future=False).start == at(21)
    w = parse("at 5", NOW, future=False)
    assert w.start == at(26, 5) and w.inferred == "5 AM today"


@pytest.mark.parametrize(
    "phrase",
    [
        "", "   ", "banana", "at 5 with john", "remind me at 5", "noon at 5", "at 25", "25:00", "13pm", "the 32nd",
        "morning",  # already over at 2 PM
        "this morning", "today at 1",  # 1 AM and 1 PM have both passed today
        "in 20", "in 0 minutes", "20 minute", "tomorrow kal", "kal tomorrow", "monday tuesday", "in 5 minutes at 6",
        "today tomorrow", "morning evening", "tomorrow tonight",
        "5 at 6 at 7", "30 feb",
    ],
)
def test_anything_unsure_is_none(phrase):
    assert parse(phrase, NOW) is None, phrase


def test_devanagari_digits_and_odd_spacing():
    assert parse("  कल   सुबह   ९:३०  बजे ", NOW).start == at(27, 9, 30)
    assert parse("Tomorrow, 5PM!", NOW).start == at(27, 17)
