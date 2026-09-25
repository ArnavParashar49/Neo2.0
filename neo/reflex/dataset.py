"""Synthetic labelled utterances for the reflex head — templates × slots, no API needed.

Run ``python -m neo.reflex.finetune`` to build + train. Add real misroutes to
``~/.neo/reflex_extra.jsonl`` (``{"text": ..., "intent": ...}``) and retrain any time.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from neo.config import settings

APPS = [
    "Safari",
    "Mail",
    "Notes",
    "Slack",
    "Spotify",
    "Finder",
    "Terminal",
    "VS Code",
    "Calendar",
    "Messages",
    "Chrome",
    "Preview",
    "Xcode",
    "Zoom",
    "Figma",
    "Discord",
]
SITES = ["youtube", "github", "gmail", "twitter", "reddit", "amazon", "netflix", "notion", "linkedin"]
FILES = [
    "the report",
    "my resume",
    "the budget spreadsheet",
    "the screenshots on my desktop",
    "the PDF I downloaded",
    "this folder",
    "the notes from yesterday",
    "main.py",
    "the invoice",
    "my downloads folder",
]
PEOPLE = ["Sam", "my manager", "the landlord", "Priya", "the team", "mom", "Alex", "the recruiter"]
TOPICS = [
    "the capital of Peru",
    "how photosynthesis works",
    "the difference between TCP and UDP",
    "a good name for a cat",
    "what a Roth IRA is",
    "why the sky is blue",
    "who won the 2018 World Cup",
    "how to say thank you in Japanese",
    "the plot of Dune",
    "how compound interest works",
    "what causes jet lag",
]
MATH = [
    "17 times 23",
    "the square root of 289",
    "15% of 240",
    "2 to the power of 10",
    "how many days until Friday",
    "340 divided by 8",
]
WRITE = [
    "a haiku about rain",
    "a short toast for a wedding",
    "a polite way to decline an invite",
    "a tweet about Monday mornings",
    "a two-line bio for me",
    "a limerick about coffee",
]

# (template, may_take_polite_prefix)
CHAT = [
    ("what's {topic}", False),
    ("tell me {topic}", True),
    ("explain {topic}", True),
    ("do you know {topic}?", False),
    ("what is {math}", False),
    ("calculate {math}", True),
    ("what's {math}", False),
    ("write {write}", True),
    ("give me {write}", True),
    ("what do you think about {topic}", False),
    ("how are you", False),
    ("thanks", False),
    ("thank you", False),
    ("good morning", False),
    ("who are you", False),
    ("what can you do", False),
    ("what's your favourite colour", False),
    ("why is {topic} important", False),
    ("tell me a joke", True),
    ("summarize {topic} in one sentence", True),
    ("is it true that {topic}", False),
    ("what does {topic} mean", False),
    ("help me think through {topic}", True),
    ("that's funny", False),
    ("nice", False),
    ("okay cool", False),
    ("how do you spell necessary", False),
    ("what's a synonym for happy", False),
    ("translate good morning to French", True),
    ("recommend a book about {topic}", True),
]
QUICK = [
    ("open {app}", True),
    ("launch {app}", True),
    ("open {site}", True),
    ("go to {site}", True),
    ("close {app}", True),
    ("quit {app}", True),
    ("switch to {app}", True),
    ("bring up {app}", True),
    ("turn the volume up", True),
    ("volume down", False),
    ("set volume to {n}", True),
    ("turn it up a bit", True),
    ("mute", False),
    ("unmute", False),
    ("brightness up", False),
    ("make the screen dimmer", True),
    ("dim the screen", True),
    ("what time is it", False),
    ("what's the time", False),
    ("set a timer for {n} minutes", True),
    ("start a {n} minute timer", True),
    ("remind me to call {person} at 5", True),
    ("add a reminder to buy milk", True),
    ("remind me about the dentist tomorrow", True),
    ("play some music", True),
    ("pause", False),
    ("pause the music", True),
    ("next song", False),
    ("skip this track", True),
    ("what's the weather", False),
    ("is it going to rain today", False),
    ("how cold is it outside", False),
    ("take a screenshot", True),
    ("lock the screen", True),
    ("put the display to sleep", True),
    ("what's on my calendar today", False),
    ("do I have any meetings today", False),
    ("show me my unread emails", True),
    ("any new mail", False),
    ("empty the trash", True),
    ("show desktop", True),
]
AGENT = [
    ("read {file} and summarize it", True),
    ("reply to {person}'s email and say I'll be late", True),
    ("email {person} the notes from today", True),
    ("find {file} and move it to a new folder called archive", True),
    ("organize {file}", True),
    ("delete {file}", True),
    ("rename {file} to final version", True),
    ("what does this window say", False),
    ("what's on my screen", False),
    ("what am I looking at", False),
    ("read the article that's open and give me the key points", True),
    ("fill in this form with my details", True),
    ("book a table for two tonight near me", True),
    ("research {topic} and write a summary in Notes", True),
    ("install {app}", True),
    ("uninstall {app}", True),
    ("update all my brew packages", True),
    ("clone the repo and run the tests", True),
    ("fix the failing test in {file}", True),
    ("compare prices for a used MacBook Air and put them in a note", True),
    ("go through my unread emails and flag the urgent ones", True),
    ("download the invoice from the last email and put it in Documents", True),
    ("create a presentation about {topic}", True),
    ("translate the page that's open into Spanish", True),
    ("look at this error and tell me how to fix it", True),
    ("clear my downloads folder", True),
    ("send a message to {person} on Slack saying the build is green", True),
    ("schedule a meeting with {person} tomorrow at 3 and email the invite", True),
    ("convert {file} to PDF", True),
    ("write a script that renames all the photos by date", True),
    ("find the cheapest flight to Tokyo next month", True),
    ("what's the total of the receipts in {file}", False),
    ("open {site} and search for {topic}", True),
    ("log into the wifi portal for me", True),
    ("sort {file} by date and delete the duplicates", True),
    ("check if the website is up and tell me the load time", True),
    ("back up {file} to an external drive", True),
    ("find the last message from {person} and summarize the thread", True),
    ("copy the table on this page into a spreadsheet", True),
    ("set up a python project called scraper", True),
    ("what's in {file}", False),
    ("how many unread emails do I have from {person}", False),
    ("close all the tabs except this one", True),
    ("zip up {file} and email it to {person}", True),
]
STOP = [
    ("stop", False),
    ("cancel", False),
    ("never mind", False),
    ("be quiet", False),
    ("shut up", False),
    ("go to sleep", False),
    ("that's all", False),
    ("quit", False),
    ("exit", False),
    ("stop talking", False),
    ("cancel that", False),
    ("forget it", False),
    ("stop it", False),
    ("that's enough", False),
    ("okay stop", False),
]

DESTRUCTIVE_HINTS = (
    "delete",
    "remove",
    "uninstall",
    "send",
    "email",
    "reply",
    "clear",
    "rename",
    "overwrite",
    "install",
    "pay",
    "buy",
    "book",
    "log into",
    "message",
    "empty",
)
_PREFIX = ["can you ", "could you ", "please ", "hey neo, ", "neo, ", "hey neo ", "neo "]


def _fill(t: str, rnd: random.Random) -> str:
    return t.format(
        app=rnd.choice(APPS),
        site=rnd.choice(SITES),
        file=rnd.choice(FILES),
        person=rnd.choice(PEOPLE),
        topic=rnd.choice(TOPICS),
        math=rnd.choice(MATH),
        write=rnd.choice(WRITE),
        n=rnd.choice([5, 10, 20, 30, 50, 75]),
    )


def build(n_per_class: int = 320, seed: int = 7) -> list[dict]:
    rnd = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()
    for intent, templates in (("chat", CHAT), ("quick_action", QUICK), ("agent_task", AGENT), ("stop", STOP)):
        i = 0
        tries = 0
        while i < n_per_class and tries < n_per_class * 6:
            tries += 1
            tpl, polite = templates[(i + tries) % len(templates)]
            t = _fill(tpl, rnd)
            r = rnd.random()
            if r < 0.25:
                t = rnd.choice(_PREFIX if polite else _PREFIX[3:]) + t
            elif r < 0.35:
                t = t.rstrip("?.") + rnd.choice(["?", " please", " now", ""])
            t = t.strip()
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            i += 1
            rows.append(
                {
                    "text": t,
                    "intent": intent,
                    "destructive": intent == "agent_task" and any(h in t for h in DESTRUCTIVE_HINTS),
                }
            )
    # LLM-generated + real-world extras live outside the repo (see generate.py / README).
    for name in ("reflex_generated.jsonl", "reflex_extra.jsonl"):
        extra = settings().data_dir / name
        if extra.exists():
            for line in extra.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows.append(
                        {"text": r["text"], "intent": r["intent"], "destructive": r.get("destructive", False)}
                    )
    rnd.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build()
    out = Path(settings().data_dir) / "reflex_dataset.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{len(rows)} rows → {out}")
