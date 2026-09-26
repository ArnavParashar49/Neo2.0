"""Public reflex data (neo/reflex/sources.py, hf_data.py), the Hinglish templates and how
dataset.build() takes them in. Hermetic: every file comes from tests/fixtures/reflex_sources
(invented sentences in each source's format), nothing touches the network or ~/.neo."""

import io
import json
import random
import re
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from neo.reflex import hf_data, templates_hi
from neo.reflex import sources as S

FIX = Path(__file__).parent / "fixtures" / "reflex_sources"


class FixtureFetcher:
    """The Fetcher interface over the fixture directory: <repo name>/<path with / → __>, and URL
    files under urls/ by their path after the pinned revision."""

    def __init__(self, tmp: Path):
        self.tmp = tmp

    def records(self, name, path, columns=None):
        repo = S.PINS[name][0].split("/")[-1]
        stem = re.sub(r"-\d{5}-of-\d{5}.*", "", path).removesuffix(".parquet").replace("/", "__")
        rows = [json.loads(line) for line in (FIX / repo / f"{stem}.jsonl").read_text().splitlines() if line.strip()]
        return [{c: r.get(c) for c in columns} for r in rows] if columns else rows

    def text(self, name, path):
        repo = S.PINS[name][0].split("/")[-1]
        return (FIX / repo / path.removesuffix(".gz").replace("/", "__")).read_text()

    def _url(self, url):
        parts = url.split("/")
        i = next(i for i, p in enumerate(parts) if re.fullmatch(r"[0-9a-f]{40}", p))
        return FIX / "urls" / "__".join(parts[i + 1 :])

    def url_file(self, url):
        p = self._url(url)
        d = p.with_name(p.name + ".d")
        if d.is_dir():  # pack a directory fixture into the archive the loader opens
            out = self.tmp / p.name
            with tarfile.open(out, "w:gz") as tar:
                for f in sorted(x for x in d.rglob("*") if x.is_file()):
                    tar.add(f, arcname=str(f.relative_to(d)))
            return out
        return p

    def url_head(self, url, nbytes):
        return self._url(url).read_bytes()[:nbytes]

    def zip_texts(self, url, want, tag):
        texts = json.loads(Path(str(self._url(url)) + ".json").read_text())
        return {k: v for k, v in texts.items() if want(k)}


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SGD_FILES", ["train/dialogues_001.json"])
    return FixtureFetcher(tmp_path)


def load(name, fx):
    skip = Counter()
    return [r for r in S.BY_NAME[name].load(fx, skip) if r], skip


def by_text(rows):
    return {r["text"]: r for r in rows}


# ---- rows ---------------------------------------------------------------------------------------
def test_row_leaves_unknown_labels_out_and_cleans_text():
    r = S.row("olly  mute the  tv.", "quick_action", "volume", "massive", "audio_volume_mute", lang="en", destructive=False)
    assert r["text"] == "mute the tv" and r["source"] == "massive:audio_volume_mute" and r["hf_intent"] == "audio_volume_mute"
    assert r["destructive"] is False and "screen" not in r and "reply" not in r and "kind" not in r
    assert r["split"] == "train" and r["group"].startswith("massive:")
    assert S.text_key("Pause my timer .") == S.text_key("pause my  timer") == "pausemytimer"
    # re-tokenised copies are one sentence; Devanagari vowel signs still tell words apart
    assert S.text_key("I don ' t like this song") == S.text_key("I don't like this song")
    assert S.text_key("Create an alarm for 5 : 30 am") == S.text_key("create an alarm for 5:30am")
    assert S.text_key("read my to do list") == S.text_key("read my todo list")
    assert S.text_key("रुको।") == S.text_key("रुको") != S.text_key("रुकें")
    assert S.text_key("बंद करो") != S.text_key("बंद करें")
    assert S.group_id("top", "Pause my timer .") == S.group_id("top", "pause my timer!")
    assert S.lang_of("मेरा टाइमर रोको") == "hi" and S.lang_of("timer roko", "hinglish") == "hinglish"


# ---- CLINC + MASSIVE ------------------------------------------------------------------------------
def test_clinc_mapping_kind_destructive_and_eval_split(fx):
    rows, skip = load("clinc", fx)
    t = by_text(rows)
    assert t["never mind forget what i said"]["intent"] == "stop"
    msg = t["text my sister that dinner is at eight"]
    assert (msg["intent"], msg["kind"], msg["destructive"]) == ("agent_task", "send_message", True)
    w = t["what is the forecast for tomorrow in boston"]
    assert (w["intent"], w["tool"], w["kind"], w["destructive"], w["screen"]) == ("quick_action", "weather", "other", False, False)
    assert t["set an alarm for six thirty am"]["kind"] == "add_reminder"
    assert "kind" not in t["how do you say good night in german"]  # chat never reaches neo/decide
    assert t["please wire two hundred dollars to my savings"]["destructive"] is True
    assert t["is it going to snow in denver this weekend"]["split"] == "eval"
    assert skip["clinc:unmapped:made_up_label"] == 1  # listed as unmapped in the build report
    assert (w["reply"], msg["reply"], t["never mind forget what i said"]["reply"]) == (True, False, False)


def test_clinc_rows_are_labelled_by_what_they_say(fx):
    rows, skip = load("clinc", fx)
    t = by_text(rows)
    how = t["how do i report my lost card"]  # advice, not reporting it (report_lost_card is destructive)
    assert (how["intent"], how["tool"], how["destructive"], how["reply"]) == ("chat", "", False, True) and "kind" not in how
    later = t["what's on my calendar for april 15"]  # calendar_today lists today only
    assert (later["intent"], later["tool"], later["reply"]) == ("agent_task", "", True)
    assert t["what's on my calendar today"]["tool"] == "calendar_today"
    clear = t["clear my calendar for tomorrow"]
    assert (clear["intent"], clear["tool"], clear["destructive"]) == ("agent_task", "", True)
    add = t["add dentist to my calendar for friday"]
    assert (add["tool"], add["destructive"]) == ("calendar_add", False)
    off = t["take eggs off my shopping list"]
    assert (off["destructive"], off["kind"]) == (True, "other")
    assert t["what is the apr on my visa card"]["intent"] == "agent_task"  # like balance: my account
    assert "unsync from my phone" not in t and skip["clinc:sync_device"] == 1
    assert t["bye for now"]["intent"] == "stop" and t["it was nice chatting with you"]["intent"] == "chat"


def test_massive_mute_rows_without_audio_are_stop_in_both_scripts(fx):
    rows, skip = load("massive", fx)
    t = by_text(rows)
    assert t["be quiet for a while"]["intent"] == "stop"  # wake word stripped too
    assert t["silence"]["intent"] == "stop"  # one word, but a clear stop: kept
    assert (t["mute the kitchen speakers"]["intent"], t["mute the kitchen speakers"]["tool"]) == ("quick_action", "volume")
    assert t["कुछ देर चुप रहो"]["intent"] == "stop"
    assert t["रसोई के स्पीकर म्यूट करो"]["tool"] == "volume"
    assert "emails" not in t  # a lone word is too ambiguous to learn from
    assert t["email the landlord about the broken heater"]["destructive"] is True
    assert t["add eggs to my shopping list"]["kind"] == "add_reminder"
    assert t["silent mode on for three hours"]["tool"] == "volume"  # a device setting, not "stop"
    assert "clear data" not in t and skip["massive:audio_volume_mute_noise"] == 1


def test_massive_rows_and_their_translations_are_labelled_from_the_english(fx):
    rows, _ = load("massive", fx)
    t = by_text(rows)
    for text in ("remind me about the dentist on friday", "मुझे शुक्रवार को दंत चिकित्सक की याद दिलाना"):
        assert (t[text]["tool"], t[text]["kind"]) == ("reminder_add", "add_reminder"), text
    for text in ("save this song to my favourites", "यह गाना पसंदीदा में सहेजो"):  # the Hindi names no command word
        assert (t[text]["intent"], t[text]["reply"]) == ("quick_action", False), text
    assert t["i really like jazz"]["intent"] == "chat"
    assert t["how do i descale the coffee machine"]["intent"] == "chat"
    assert all(isinstance(r["reply"], bool) for r in rows)


def test_massive_translations_share_a_group_and_script_sets_lang(fx):
    rows, _ = load("massive", fx)
    t = by_text(rows)
    en, hi = t["will it be sunny on sunday"], t["क्या रविवार को धूप निकलेगी"]
    assert en["group"] == hi["group"] == "massive:12"
    assert (en["lang"], hi["lang"]) == ("en", "hi")
    assert {r["split"] for r in rows if r["group"] in ("massive:90", "massive:91")} == {"eval"}


# ---- TOP family ------------------------------------------------------------------------------------
def test_top_labels():
    lab = dict([S.top_labels("[IN:CREATE_ALARM set an alarm ]")])
    assert lab["create_alarm"]["intent"] == "quick_action" and lab["create_alarm"]["tool"] == "reminder_add"
    _, kw = S.top_labels("IN:SEND_MESSAGE")
    assert (kw["intent"], kw["kind"], kw["destructive"], kw["reply"]) == ("agent_task", "send_message", True, False)
    _, kw = S.top_labels("[IN:GET_WEATHER is it cold ]")
    assert (kw["tool"], kw["reply"]) == ("weather", True)
    _, kw = S.top_labels("IN:GET_INFO_TRAFFIC")
    assert (kw["tool"], kw["kind"]) == ("web_search", "web_search")
    _, kw = S.top_labels("IN:PAUSE_TIMER")  # the timer tool only starts timers
    assert (kw["intent"], kw["tool"]) == ("quick_action", "")
    _, kw = S.top_labels("IN:DELETE_REMINDER")
    assert kw["destructive"] is True
    _, kw = S.top_labels("IN:GET_RECIPES")
    assert (kw["intent"], kw["reply"], kw["kind"]) == ("chat", True, None)
    assert S.top_labels("[IN:UNSUPPORTED_WEATHER what's the pollen ]") is None
    assert S.top_labels("IN:PREFER") is None
    assert S.top_labels("IN:GET_AGE") is None and S.top_labels("IN:SET_UNAVAILABLE") is None  # people / presence
    assert S.top_labels("IN:SOMETHING_NEW") is None  # unmapped: skipped, not an agent task
    for name in ("GET_REMINDER", "GET_MESSAGE", "GET_CONTACT_METHOD"):
        _, kw = S.top_labels(f"IN:{name}")
        assert (kw["intent"], kw["reply"]) == ("agent_task", True), name
    for name in ("GET_DATE_TIME_EVENT", "UPDATE_DIRECTIONS", "GET_LOCATION"):
        assert S.top_labels(f"IN:{name}")[1]["tool"] == "web_search", name
    _, kw = S.top_labels("IN:CANCEL_MESSAGE", "Sending ko quit karo")
    assert (kw["intent"], kw["reply"]) == ("stop", False)


@pytest.mark.parametrize("parse, text, en, want", [
    ("IN:SILENCE_ALARM", "Quiet now", "", ("stop", False)),
    ("IN:SILENCE_ALARM", "Please chup rahe", "Please be quiet", ("stop", False)),
    ("IN:SILENCE_ALARM", "stop the alarm", "", ("quick_action", False)),
    ("IN:SILENCE_ALARM", "please band karo", "Stop !", ("quick_action", False)),  # bare "band karo": turn it off
    ("IN:SILENCE_ALARM", "turn off tomorrow's alarm", "", ("agent_task", True)),  # deletes a future alarm
    ("IN:END_CALL", "hang up", "", ("agent_task", False)),
    ("IN:REMOVE_FROM_PLAYLIST_MUSIC", "Fir ye song mat chalana", "Don't play this song again", ("quick_action", False)),
    ("IN:REMOVE_FROM_PLAYLIST_MUSIC", "take this off my playlist", "", ("agent_task", True)),
    ("IN:CREATE_REMINDER", "Darcy ko yaad dilao ki cats ko khana do", "Remind Darcy to feed the cats", ("agent_task", True)),
    ("IN:CREATE_REMINDER", "remind me to call mom", "", ("quick_action", False)),
])
def test_top_labels_follow_what_the_sentence_says(parse, text, en, want):
    _, kw = S.top_labels(parse, text, en)
    assert (kw["intent"], kw["destructive"]) == want
    if kw["intent"] == "agent_task" and "CREATE_REMINDER" in parse:
        assert kw["kind"] == "send_message"  # a reminder for someone else is a message to them


def test_hinglish_top_parallel_rows_and_benchmark_split(fx):
    rows, skip = load("hinglish_top", fx)
    t = by_text(rows)
    en, cs = t["Set an alarm for 5 am tomorrow"], t["Kal subah 5 baje ka alarm set karo"]
    assert en["group"] == cs["group"] and (en["lang"], cs["lang"]) == ("en", "hinglish")
    assert cs["tool"] == "reminder_add" and cs["hf_intent"] == "create_alarm"
    assert t["Ravi ko text karo ki main late ho raha hoon"]["destructive"] is True
    assert t["Aaj Pune me mausam kaisa hai ?"]["split"] == "train"  # validation trains
    assert {r["split"] for r in rows if "plants" in r["text"]} == {"eval"}
    assert skip["hinglish_top:unsupported_event"] == 2  # both languages of the row


def test_synthetic_hinglish_only_and_topv2_without_navigation(fx):
    rows, _ = load("hinglish_top_synth", fx)
    assert all(r["hf_intent"] != "play_music" for r in rows)  # its "human" rows are the benchmark
    assert {r["lang"] for r in rows} == {"hinglish"}
    rows, skip = load("topv2", fx)
    assert skip["topv2:navigation"] == 1 and not any("airport" in r["text"] for r in rows)
    t = by_text(rows)
    assert t["skip this song"]["intent"] == "quick_action" and t["delete my alarm for tomorrow morning"]["destructive"]


def test_mtop_pairs_languages_by_id(fx):
    rows, skip = load("mtop", fx)
    t = by_text(rows)
    en, hi = t["call grandma on video"], t["दादी को वीडियो कॉल करो"]
    assert en["group"] == hi["group"] == "mtop:1001" and hi["lang"] == "hi"
    assert (en["intent"], en["destructive"]) == ("agent_task", True)
    assert t["दाल मखनी कैसे बनाते हैं"]["intent"] == "chat"
    assert {r["split"] for r in rows if r["group"] == "mtop:2001"} == {"eval"}
    assert skip["mtop:prefer"] == 1
    # the people domain isn't NEO's; an intent nobody mapped is reported, not guessed
    assert "how old is lia" not in t and "where does sam live" not in t
    assert skip["mtop:get_age"] == 1 and skip["mtop:get_location"] == 1 and skip["mtop:unmapped:fly_to_moon"] == 1
    assert t["stop sending that message"]["intent"] == "stop"
    for text in ("remind erik that the show moved to friday", "एरिक को याद दिलाओ कि शो शुक्रवार को है"):
        assert (t[text]["intent"], t[text]["kind"], t[text]["destructive"]) == ("agent_task", "send_message", True)
    assert t["बंद होजाओ"]["intent"] == "stop" and t["turn it off"]["intent"] == "quick_action"
    assert t["when does the jazz festival start"]["tool"] == "web_search"


# ---- the other sources ---------------------------------------------------------------------------
def test_hinglish_chat_reads_the_head_and_drops_commands(fx):
    rows, _ = load("hinglish_chat", fx)
    texts = {r["text"] for r in rows}
    assert "kal movie dekhne chalein kya?" in texts and "aaj office mein bahut kaam tha" in texts
    assert not any("gaana" in t for t in texts)  # a command, not chit-chat
    assert not any("weather" in t for t in texts)  # NEO answers that with the weather tool
    assert not any("kat" in t for t in texts)  # the cut-off last line
    assert all(r["intent"] == "chat" and r["reply"] and r["lang"] == "hinglish" for r in rows)


def test_leyzer_mapping_and_it_is_opt_in(fx):
    assert not S.BY_NAME["leyzer"].default and "leyzer" in hf_data.build([], fetcher=fx, log=lambda *a: None)[2]["not_used"]
    rows, skip = load("leyzer", fx)
    t = by_text(rows)
    rm = t["remove the folder old_backups"]
    assert (rm["intent"], rm["tool"], rm["destructive"]) == ("agent_task", "shell", True)
    assert t["do i have any new mail"]["tool"] == "mail_unread"
    assert (t["open my email app"]["tool"], t["open my email app"]["kind"]) == ("open_app", "open_app")
    assert t["scroll a bit further down the article"]["screen"] is True
    assert t["look for cooking videos on youtube"]["tool"] == "search_site"
    assert {r["split"] for r in rows} == {"train"}  # ASR-noised in part: never a benchmark
    assert skip["leyzer:Fitbit.ShowSteps"] == 1


def test_sgd_destructive_reply_and_stop(fx):
    rows, skip = load("sgd", fx)
    t = by_text(rows)
    find = t["Can you find me a sushi place in Oakland?"]
    assert (find["tool"], find["destructive"], find["reply"]) == ("web_search", False, True)
    book = t["Book a table there for 7 pm"]
    assert (book["intent"], book["destructive"], book["reply"]) == ("agent_task", True, False)
    play = t["Play the song Clocks please"]  # SGD calls it transactional; NEO: reversible
    assert (play["tool"], play["kind"], play["destructive"]) == ("play_music", "play_media", False)
    assert t["Send 40 dollars to Kim"]["destructive"] is True
    assert t["No, not right now"]["intent"] == "stop" and t["Thanks, that's all"]["intent"] == "stop"
    assert (t["What's their phone number?"]["intent"], t["What's their phone number?"]["reply"]) == ("chat", True)
    assert "Sounds good" not in t and skip["sgd:LookupSong"] == 1
    assert t["Great, thank you"]["intent"] == "chat"  # thanks alone isn't a stop
    assert "No, I didn't" not in t and skip["sgd:negate_intent_answer"] == 1  # an answer, meaningless alone
    assert (t["Am I free on March 5th?"]["intent"], t["Am I free on March 5th?"]["tool"]) == ("agent_task", "")
    assert all(r["reply"] is False for r in rows if r["intent"] == "stop")


def test_glaive_function_names_to_tools_and_calculators_to_chat(fx):
    assert S.glaive_call("USER: hi\n\n\nASSISTANT: <functioncall> {\"name\": \"get_news\"} <|endoftext|>") == ("hi", "get_news")
    assert S.neo_tool("send_email") == "mail_send" and S.neo_tool("check_email_availability") is None
    assert S.neo_tool("check_email") is None and S.neo_tool("create_todo") == "reminder_add"  # glaive's validator
    assert S.neo_tool("get_unread_emails") == "mail_unread" and S.neo_tool("check_inbox") == "mail_unread"
    rows, skip = load("glaive", fx)
    t = by_text(rows)
    mail = t["Please email Dana that the draft is attached"]
    assert (mail["intent"], mail["tool"], mail["kind"], mail["destructive"]) == ("agent_task", "mail_send", "send_email", True)
    assert t["I need to add something to my to-do list"]["tool"] == "reminder_add"  # called after a clarification
    assert not any("news" in r["text"] for r in rows)  # the call came after the third user message
    assert skip["glaive:long_or_multiline"] == 1
    assert "What events are happening in New York on 25th December?" not in t and skip["glaive:public_events"] == 1
    cal = t["What events do I have on my calendar tomorrow?"]
    assert (cal["intent"], cal["tool"]) == ("agent_task", "")
    chat, _ = load("glaive_chat", fx)
    assert [r["text"] for r in chat] == ["What tip should I leave on a 48 dollar bill?",  # not currency: rates change
                                         "Can you check if my email address is valid?"]


def test_computer_use_rows_mark_the_screen_only_when_the_wording_points_at_it(fx):
    rows, skip = load("agentnet", fx)
    t = by_text(rows)
    assert t["Rename the selected file on the desktop to notes-final"]["screen"] is True
    assert "screen" not in t["Create a new playlist called Focus in Spotify and add three lofi songs"]
    # a click isn't a pointer at the screen: NEO's own "click {label}" rows say False
    assert "screen" not in t["Click on Python on python.org and download the latest installer"]
    total = t["What is the total of the column in the current sheet"]
    assert (total["screen"], total["reply"]) == (True, True)
    assert t["Rename the selected file on the desktop to notes-final"]["reply"] is False
    assert all(r["intent"] == "agent_task" for r in rows) and len(rows) == 4
    assert skip["agentnet:filtered"] == 5  # a bare click, a step list, infeasible, a file name, low alignment
    rows, _ = load("omniact", fx)
    t = by_text(rows)
    assert "Add an event called Standup for tomorrow at 9" in t  # the output script is cut off
    assert "screen" not in t["Use the Add to cart button for the first product"]
    assert t["Use the Add to cart button for the first product"]["hf_intent"] == "web/shop"
    assert t["Close this window and open Downloads"]["screen"] is True
    assert not any(r["text"] == "Scroll down" for r in rows)  # one scroll is a quick action
    assert all(isinstance(r["reply"], bool) for r in rows)
    assert not S.DEICTIC.search("read the front page of the times")


def test_hwu64_stop_and_polite_requests_as_reply_negatives(fx):
    rows, _ = load("hwu64", fx)
    assert {r["text"] for r in rows} == {"that is enough stop now", "please just stop talking"}
    assert all(r["intent"] == "stop" and r["reply"] is False for r in rows)
    assert not S.BY_NAME["sentence_types"].default  # no licence stated
    rows, _ = load("sentence_types", fx)
    t = by_text(rows)
    assert set(t) == {"Could you turn the volume down, please", "Can you book a table for four tonight",
                      "Could you please make me a cup of coffee", "Can you adjust the room temperature to 72 degrees",
                      "Can I set a daily goal reminder"}  # not the weather update (wants an answer), not carrying
    assert (t["Could you turn the volume down, please"]["intent"], t["Could you turn the volume down, please"]["tool"]) == (
        "quick_action", "volume")
    assert t["Can you book a table for four tonight"]["intent"] == "agent_task"
    assert t["Could you please make me a cup of coffee"]["intent"] == "quick_action"  # as MASSIVE iot_coffee
    assert t["Can you adjust the room temperature to 72 degrees"]["intent"] == "quick_action"
    assert t["Can I set a daily goal reminder"]["tool"] == "reminder_add"
    assert all(r["reply"] is False for r in rows)


def test_emitted_kinds_are_decide_task_kinds():
    from neo.decide.engine import TASKS

    assert set(S.KINDS) <= set(TASKS)


# ---- the builder -----------------------------------------------------------------------------------
ALL = [s.name for s in S.SOURCES]


def build(fx, **kw):
    return hf_data.build(kw.pop("names", ALL), fetcher=fx, log=lambda *a: None, **kw)


def test_eval_rows_and_their_translations_never_train(fx):
    train, ev, report = build(fx)
    keys = {S.text_key(r["text"]) for r in ev}
    groups = {r["group"] for r in ev}
    assert not [r for r in train if S.text_key(r["text"]) in keys or r["group"] in groups]
    # the TOPv2 and synthetic copies of a Hinglish-TOP test sentence are held out with it
    assert not any("plants" in r["text"] or "paudhon" in r["text"] for r in train)
    assert report["sources"]["topv2"]["held_out"] == 1 and report["sources"]["hinglish_top_synth"]["held_out"] == 1
    # CLINC's train copy of a test sentence (different case and punctuation) too
    assert not any(r["text"] == "how do you say good night in german" for r in train)
    assert {r["split"] for r in ev} == {"eval"} and {r["split"] for r in train} == {"train"}
    assert Counter(r["source"].split(":")[0] for r in ev) == {"clinc": 3, "massive": 4, "hinglish_top": 4, "mtop": 3}
    # a re-tokenised copy from another source ("read my to do list" / CLINC's "read my todo list")
    assert not any(S.text_key(r["text"]) == S.text_key("read my todo list") for r in train)


def test_translations_of_a_benchmark_sentence_from_another_source_are_held_out(fx):
    # MASSIVE's English "will it rain in pune today" is MTOP's test sentence "Will it rain in Pune today ?":
    # its Hindi sibling (same MASSIVE id, no text in common with the benchmark) must not train either
    train, _, report = build(fx)
    texts = {r["text"] for r in train}
    assert "will it rain in pune today" not in texts and "क्या आज पुणे में बारिश होगी" not in texts
    assert report["sources"]["massive"]["held_out"] >= 2
    held_text, held_group = hf_data.holdout(
        [{"text": "B test", "group": "b:1"}],
        [{"text": "B test", "group": "a:7"}, {"text": "A hindi", "group": "a:7"}, {"text": "A hindi", "group": "c:3"},
         {"text": "C other", "group": "c:3"}, {"text": "unrelated", "group": "a:8"}])
    assert held_group == {"b:1", "a:7", "c:3"} and S.text_key("C other") in held_text  # closed transitively
    assert S.text_key("unrelated") not in held_text


@pytest.mark.parametrize("names", [["topv2"], ["glaive", "sgd"], ["clinc"]])
def test_the_benchmark_and_its_holdout_dont_depend_on_the_sources_chosen(fx, names):
    _, full, _ = build(fx)
    train, ev, report = build(fx, names=names)
    assert sorted(r["text"] for r in ev) == sorted(r["text"] for r in full)
    keys = {S.text_key(r["text"]) for r in full}
    assert not [r for r in train if S.text_key(r["text"]) in keys]
    assert {r["source"].split(":")[0] for r in train} <= set(names)
    assert all(report["sources"][n]["benchmark_only"] for n in S.BENCHMARK if n not in names)
    assert "benchmark only" in hf_data.summary(train, ev, report)


def test_duplicates_across_sources_and_translations_are_dropped(fx):
    train, _, report = build(fx)
    keys = [S.text_key(r["text"]) for r in train]
    assert len(keys) == len(set(keys))
    assert len({(r["group"], r["lang"]) for r in train}) == len(train)
    alarms = [r for r in train if "alarm for six thirty" in r["text"].lower()]
    assert len(alarms) == 1 and alarms[0]["source"].startswith("clinc:")  # CLINC comes first
    # "Set an alarm for 5am tomorrow" (TOPv2) is the same sentence as Hinglish-TOP's English one
    assert len([r for r in train if r["lang"] == "en" and "5" in r["text"] and "alarm" in r["text"].lower()]) == 1
    assert report["sources"]["topv2"]["duplicate"] >= 1 and report["sources"]["hinglish_top_synth"]["duplicate"] >= 1


def test_caps_are_per_key_seeded_and_deterministic(fx):
    a, _, _ = build(fx, seed=3)
    b, _, _ = build(fx, seed=3)
    assert a == b
    tiny, _, report = build(fx, scale=1e-6)  # every cap becomes 1 per key
    per_key = Counter((r["source"].split(":")[0], S.BY_NAME[r["source"].split(":")[0]].key_of(r)) for r in tiny)
    assert max(per_key.values()) == 1
    assert report["sources"]["massive"]["capped"] > 0


def test_sample_round_robin_respects_total_and_cap():
    src = S.Source("x", lambda fx, skip: [], "none", cap=3, total=5)
    rows = [{"hf_intent": k, "lang": "en", "text": f"{k} {i}", "group": f"g{k}{i}"} for k in "abc" for i in range(10)]
    kept, unseen = hf_data._sample(rows, src, random.Random(0), lambda r: True, 1.0)
    assert len(kept) == 5 and max(Counter(r["hf_intent"] for r in kept).values()) == 2  # spread, not 3+2
    assert unseen == 25


def test_main_writes_train_and_eval_files_with_the_row_schema(fx, monkeypatch, tmp_path):
    from neo.config import settings

    monkeypatch.setattr(hf_data, "Fetcher", lambda cache: fx)
    monkeypatch.setattr(sys, "argv", ["hf_data", "--with", "leyzer"])
    hf_data.main()
    data = settings().data_dir
    train = [json.loads(line) for line in (data / "reflex_hf.jsonl").read_text().splitlines()]
    ev = [json.loads(line) for line in (data / "reflex_eval.jsonl").read_text().splitlines()]
    assert train and ev
    for r in train + ev:
        assert {"text", "intent", "tool", "source", "hf_intent", "lang", "split", "group"} <= set(r)
        assert r["intent"] in S.INTENTS and r["lang"] in S.LANGS and r["text"] == r["text"].strip()
        assert r.get("kind", "other") in S.KINDS
        assert all(isinstance(r[k], bool) for k in ("destructive", "screen", "reply") if k in r)
        assert r["source"] == f"{r['source'].split(':')[0]}:{r['hf_intent']}"
    assert {r["split"] for r in train} == {"train"} and {r["split"] for r in ev} == {"eval"}
    assert any(r["source"].startswith("leyzer:") for r in train)
    assert {r["lang"] for r in train} == {"en", "hi", "hinglish"}


def test_unknown_source_names_are_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hf_data", "--only", "clinc,nope"])
    with pytest.raises(SystemExit, match="nope"):
        hf_data.main()


# ---- fetching (no network: the HTTP layer is replaced) ---------------------------------------------
def test_zip_members_are_read_by_range_and_cached(monkeypatch, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("data/tasks/desktop/notes/task_1.txt", "Task: Make a new note")
        z.writestr("data/images/big.png", random.Random(0).randbytes(3_000_000))
    blob = buf.getvalue()
    calls = []

    def http(url, rng=None):
        calls.append(rng)
        return blob if rng is None else blob[rng[0] : rng[1] + 1]

    monkeypatch.setattr(S, "_probe", lambda url: (len(blob), url))
    monkeypatch.setattr(S, "_http", http)
    fx = S.Fetcher(tmp_path)
    want = lambda n: "/tasks/" in n  # noqa: E731
    assert fx.zip_texts("https://x/data.zip", want, "tasks") == {"data/tasks/desktop/notes/task_1.txt": "Task: Make a new note"}
    fetched = sum(hi - lo + 1 for lo, hi in calls)
    assert fetched < len(blob)  # the 3 MB image in between was never downloaded whole
    n = len(calls)
    assert fx.zip_texts("https://x/data.zip", want, "tasks") and len(calls) == n  # cached


def test_url_head_and_file_download_once(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(S, "_http", lambda url, rng=None: calls.append(rng) or b"input,output\na,b\nc,d")
    fx = S.Fetcher(tmp_path)
    assert fx.url_head("https://x/y.csv", 10) == b"input,output\na,b\nc,d"
    fx.url_head("https://x/y.csv", 10)
    fx.url_file("https://x/z.json")
    fx.url_file("https://x/z.json")
    assert calls == [(0, 9), None]


def test_class_label_columns_are_decoded():
    import pyarrow as pa

    meta = {"info": {"features": {"text": {"dtype": "string"}, "intent": {"names": ["weather", "cancel"]}}}}
    table = pa.table({"text": ["a", "b"], "intent": [1, 0]}).replace_schema_metadata({"huggingface": json.dumps(meta)})
    assert S._decode_class_labels(table).to_pylist() == [{"text": "a", "intent": "cancel"}, {"text": "b", "intent": "weather"}]


# ---- Hinglish templates --------------------------------------------------------------------------
def test_hinglish_stop_phrases():
    p = templates_hi.stop_phrases()
    assert len(p) == 250 and len(set(p)) == 250 and p == templates_hi.stop_phrases()
    assert {"bas karo", "ruk jao", "rehne do", "chup ho jao", "bolna band karo"} <= set(p)
    assert all(any(c in x for c in templates_hi.STOP_CORE) for x in p)
    assert "hatao" not in p  # "ye file hatao" deletes
    assert not any(x.split()[0] == x.split()[-1] and len(x.split()) > 2 for x in p)  # no "yaar … yaar"
    # a bare "band karo" usually means "turn it off" (neo/reflex/__init__.py's stop rule agrees)
    assert all("bolna band karo" in x for x in p if "band kar" in x)
    # no honorific on the rude form ("chup kar ji"), no "quickly" on "not now" ("abhi nahi jaldi")
    assert not any(x.endswith(" ji") and (x.startswith("oye ") or x[: -len(" ji")].split(" ", 1)[-1] in templates_hi._TU_FORM
                                          or x[: -len(" ji")] in templates_hi._TU_FORM) for x in p)
    assert not any(x.endswith(("nahi jaldi", "nahi abhi", "chahiye jaldi", "itna hi jaldi")) for x in p)
    hi = templates_hi.stop_phrases_devanagari()
    assert len(hi) == 50 and all(S.lang_of(x) == "hi" for x in hi)
    assert all("बोलना बंद करो" in x for x in hi if "बंद कर" in x)


def test_hinglish_template_rows_cover_every_class_with_valid_labels():
    import neo.tools
    from neo.agent.registry import registry

    neo.tools.load_all()
    tools = {t.name for t in registry().all()}
    rows = templates_hi.rows()
    assert rows == templates_hi.rows()
    by = Counter(r["intent"] for r in rows)
    assert by["chat"] >= 40 and by["quick_action"] >= 80 and by["agent_task"] >= 80 and by["stop"] >= 250
    assert len({r["text"].lower() for r in rows}) == len(rows)
    for r in rows:
        assert r["source"] == "template" and r["lang"] in ("hinglish", "hi")
        assert r["tool"] == "" or r["tool"] in tools, r
        assert r.get("kind") is None or r["kind"] in S.KINDS
        # every label explicit: nothing downstream re-derives them with an English regex
        assert all(isinstance(r[k], bool) for k in ("destructive", "screen", "reply")), r
    texts = {r["text"] for r in rows}
    assert not any(t.startswith(("please shukriya", "zara kya haal")) or t.endswith("shukriya please") for t in texts)
    assert not any("photos wala folder padho" in t or "photos wala folder mein receipts" in t for t in texts)
    agent = [r for r in rows if r["intent"] == "agent_task"]
    risky = [r["text"] for r in agent if r["destructive"]]
    assert len(risky) >= 40
    for word in ("bhejo", "message karo", "delete", "hatao", "install", "pay karo", "kharido"):
        assert any(word in t for t in risky), word
    assert {"send_email", "send_message"} <= {r["kind"] for r in agent if r["destructive"]}
    assert any(r["screen"] for r in agent)
    assert {r["tool"] for r in rows if r["intent"] == "quick_action"} >= {"play_music", "open_app", "volume", "timer",
                                                                         "weather", "reminder_add", "web_search"}


# ---- dataset.build() -------------------------------------------------------------------------------
def test_dataset_build_passes_new_labels_and_never_trains_on_eval():
    from neo.config import settings
    from neo.reflex import dataset

    data = settings().data_dir
    data.mkdir(parents=True, exist_ok=True)
    rows = [
        {"text": "Kal subah 5 baje ka alarm set karo", "intent": "quick_action", "tool": "reminder_add",
         "source": "hinglish_top:create_alarm", "hf_intent": "create_alarm", "kind": "add_reminder", "destructive": False,
         "screen": False, "reply": False, "lang": "hinglish", "split": "train", "group": "top:abc"},
        {"text": "rename the selected file", "intent": "agent_task", "tool": "", "source": "agentnet:os",
         "hf_intent": "os", "lang": "en", "split": "train", "group": "agentnet:1", "screen": True},
        {"text": "a stray benchmark row", "intent": "chat", "tool": "", "source": "clinc:greeting", "hf_intent": "greeting",
         "lang": "en", "split": "eval", "group": "clinc:2"},
        {"text": "an old-format row", "intent": "quick_action", "tool": "", "source": "massive:play_radio"},
    ]
    (data / "reflex_hf.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (data / "reflex_eval.jsonl").write_text(json.dumps({**rows[2], "text": "only in the benchmark"}) + "\n")
    out = dataset.build(n_per_class=5)
    hf = {r["text"]: r for r in out if r["source"] == "reflex_hf.jsonl"}
    assert set(hf) == {"Kal subah 5 baje ka alarm set karo", "rename the selected file", "an old-format row"}
    a = hf["Kal subah 5 baje ka alarm set karo"]
    assert (a["kind"], a["reply"], a["screen"], a["lang"], a["group"], a["split"], a["hf_intent"]) == (
        "add_reminder", False, False, "hinglish", "top:abc", "train", "create_alarm")
    assert a["destructive"] is False and a["tool"] == "reminder_add"
    s = hf["rename the selected file"]
    assert s["screen"] is True and s["destructive"] is None and "reply" not in s  # unknown stays unknown
    old = hf["an old-format row"]
    assert old["tool"] == "play_music" and old["hf_intent"] == "play_radio" and "lang" not in old
    assert not any(r["text"] in ("a stray benchmark row", "only in the benchmark") for r in out)


def test_benchmark_drops_the_sentences_neo_trains_on():
    from neo.config import settings
    from neo.reflex import dataset

    data = settings().data_dir
    data.mkdir(parents=True, exist_ok=True)
    bench = [{"text": "Ruk jao !", "intent": "stop", "tool": "", "source": "hinglish_top:silence_alarm", "lang": "hinglish"},
             {"text": "a sentence nobody trains on", "intent": "chat", "tool": "", "source": "clinc:oos", "lang": "en"}]
    (data / "reflex_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bench))
    rows = [{"text": "ruk jao", "intent": "stop"}, {"text": "next song", "intent": "quick_action"}]
    kept, dropped = dataset.benchmark(rows)
    assert [r["text"] for r in kept] == ["a sentence nobody trains on"] and dropped == 1
    kept, dropped = dataset.benchmark()  # default: what build() trains on — "ruk jao" is a template
    assert dropped == 1 and len(kept) == 1


def test_dataset_build_includes_the_hinglish_templates():
    from neo.reflex import dataset

    out = dataset.build(n_per_class=5)
    hi = [r for r in out if r.get("lang") in ("hinglish", "hi")]
    assert hi and all(r["source"] == "template" for r in hi)
    assert Counter(r["intent"] for r in hi).keys() == {"chat", "quick_action", "agent_task", "stop"}
    assert any(r["destructive"] and "bhejo" in r["text"] for r in hi)
    assert any(r["text"] == "bas karo" and r["intent"] == "stop" for r in hi)
