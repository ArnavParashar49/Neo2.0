"""Apps, pages, shell and files: apps_running, open_app, activate_app, quit_app, close,
applescript, open_url, page_read (safari_read), finder_reveal, shell, read_file, write_file,
edit_file, trash."""

from __future__ import annotations

import asyncio
import re

from neo2.core.agent import confirm, grants
from neo2.core.agent.registry import ToolContext
from neo2.core.config import decide_mode
from neo2.core.security import policy
from neo2.core.tools.computer import apps, ax, gates, phrases, shell
from neo2.core.tools.computer._catalog import _OBJ, _p, tool


@tool(
    "apps_running",
    "List running apps and which one is frontmost.",
    _OBJ,
    parallel_safe=True,
    category="apps",
    fast_path=[
        (
            r"^\s*(?:what(?:'s| is)?\s+(?:apps?\s+(?:are|is)\s+)?(?:open|running)(?:\s+right\s+now)?|"
            r"which\s+apps?\s+(?:are|is)\s+(?:open|running)|list\s+(?:the\s+)?(?:open|running)\s+apps?|"
            r"what\s+apps?\s+do\s+i\s+have\s+open)\s*[?.!]*\s*$",
            {},
        )
    ],
    read_only=True,
)
async def apps_running(a: dict, c: ToolContext) -> str:
    front, _ = ax.frontmost_app()
    return f"Frontmost: {front}\nRunning: " + ", ".join(ax.running_apps())


# ---- apps -----------------------------------------------------------------------------


@tool(
    "open_app",
    "Open an app by name, a URL, or a file/folder path.",
    _p({"target": {"type": "string"}}, ["target"]),
    category="apps",
    # 1–3 word app/site names only ("open safari", "launch vs code", "open github.com");
    # anything with articles, "for", "and", timers etc. is a sentence for the agent.
    fast_path=[
        (
            r"^\s*(?:please\s+)?(?:open|launch|switch\s+to|bring\s+up|focus(?:\s+on)?)\s+(?!(?:a|an|the|my|up|it|this|that)\b)"
            r"(?!.*\b(?:please|now|for|and|with|then|to|in|on)\b)"
            r"(?P<target>[\w.+-]+(?:\s+[\w.+-]+){0,2})[\s.!]*$",
            {"target": "<target>"},
        ),
        (  # "<app> kholo", "mera mail khol do"; like the English one, never "<app> mein <site>"
            r"^\s*(?:(?:mera|meri|mere)\s+)?(?!(?:ye|yeh|isko|usko|woh|wo)\b)"
            r"(?!.*\b(?:mein|me|pe|par|se|ka|ki|ke|aur|phir|wala|wali|naya|nayi|naye)\b)"
            r"(?P<target>[\w.+-]+(?:\s+[\w.+-]+){0,2}?)\s+(?:kholo|khol\s+do|khol\s+dijiye)\s*[.!।]*\s*$",
            {"target": "<target>"},
            lambda f: decide_mode("open_app") == "on",
        ),
    ],
    quiet=True,
    early=True,
    idempotent=True,
)
async def open_app(a: dict, c: ToolContext):
    target = str(a["target"]).strip()
    why = policy.open_risk(target)
    if not why and c.tainted() and ("://" in target or "." in target) and policy.url_carries_data(target):
        why = f"the link carries data, after reading {c.tainted()}"
    if why:
        gate = confirm.gated("open_app", a, f"open {target[:80]} — {why}", detail=target)
        if isinstance(gate, str):
            return gate
    m = decide_mode("open_app")
    if m == "on":
        from neo2.core.decide import run
        from neo2.core.decide.kinds import launch

        return run.to_output(await launch.open_target(target))
    out = await apps.open_app(target)
    if m == "shadow":
        from neo2.core.decide import shadow

        with shadow.starting("open_app"):
            from neo2.core.decide.kinds import launch

            ok = not out.startswith(("Error", "NEEDS_"))
            shadow.later("open_app", launch.shadow(target, out), legacy_ok=ok)
    return out


@tool(
    "activate_app",
    "Bring a running app to the front.",
    _p({"name": {"type": "string"}}, ["name"]),
    category="apps",
    quiet=True,
    idempotent=True,
)
async def activate_app(a: dict, c: ToolContext) -> str:
    return await apps.activate(a["name"])


@tool(
    "quit_app",
    "Quit an app gracefully.",
    _p({"name": {"type": "string"}}, ["name"]),
    category="apps",
    quiet=True,
)
async def quit_app(a: dict, c: ToolContext) -> str:
    return await apps.quit_app(a["name"])


@tool(
    "close",
    "Close what the user names, no questions: a running app (quits it) or browser tabs by site or "
    "title ('close YouTube' closes every YouTube tab). Use this for any 'close / quit X' - never "
    "write AppleScript to close something.",
    _p({"target": {"type": "string"}}, ["target"]),
    category="apps",
    quiet=True,
    idempotent=True,
    # "close youtube", "quit spotify", "close the netflix tab" — not "close this tab/window" or
    # "close the app" (a key press on what's in front: see hotkey)
    fast_path=[
        (
            r"^\s*(?:please\s+)?(?:quit|close|kill|exit)\s+(?!(?:this|that|it|all|every)\b)"
            r"(?!(?:the|the\s+current)\s+(?:window|tab|note|document|app)\s*[.!]*\s*$)"
            r"(?P<target>[\w.+'-]+(?:\s+[\w.+'-]+){0,3}?)\s*[.!]*\s*$",
            {"target": "<target>"},
        )
    ],
)
async def close(a: dict, c: ToolContext) -> str:
    from neo2.core.decide.close import close_target

    return await close_target(str(a.get("target") or ""))


_SCRIPT_TRIES = 2  # failed scripts on one app in one task before NEO stops trying that way


@tool(
    "applescript",
    "Run AppleScript and return its result. Only for what an app exposes to scripting that no "
    "dedicated tool covers - not for seeing or clicking in an app's window (ax_tree, screenshot, "
    "click, type_text and hotkey do that). Scripts that only read (get/count/exists/set a camelCase variable) "
    "run straight away; the user approves anything else. After two failures on an app, stop "
    "scripting it.",
    _p({"script": {"type": "string"}}, ["script"]),
    risk="destructive",
    category="apps",
    slow=True,
    untrusted=True,  # file contents and app data it prints are content, never orders
)
async def applescript(a: dict, c: ToolContext) -> str:
    script = str(a.get("script") or "")
    facts = policy.script_facts(script)
    if facts.ui and (err := gates._needs_ax()):
        return err  # System Events' UI scripting takes Accessibility too
    failed = f"applescript-failed:{(facts.app or '?').casefold()}"
    if grants.count(failed) >= _SCRIPT_TRIES:
        return (
            f"Error: AppleScript already failed {_SCRIPT_TRIES} times on {facts.app or 'this'} in this task, so I "
            "didn't run another. Stop scripting it: look with screenshot and act with click, type_text or "
            "hotkey, or tell the user what's in the way."
        )
    # A script that only reads runs straight away — unless this request has read something
    # untrusted (a page, a mail, the screen), which could be what asked for it.
    if not facts.read_only or c.tainted():
        where = f" in {facts.app}" if facts.app else ""
        what = f"read {facts.app} with an AppleScript" if facts.read_only and facts.app else f"run an AppleScript{where}"
        target = grants.Target(facts.app, facts.danger, tuple(c.taint))
        gate = confirm.gated("applescript", a, what, detail=script, target=target)
        if isinstance(gate, str):
            return gate
    out = await apps.osascript(script, timeout=60)
    if out.startswith("Error"):
        if gates._AX_DENIED.search(out):
            return gates._ax_off_reply()
        grants.note(failed)
    return out


def _named_browser(name: str):
    """The installed browser the user named ("chrome", "the arc browser"), by the inventory's
    evidence (it opens web pages); None for anything else ("google", "the internet")."""
    try:
        from neo2.core.decide.inventory import inventory, named_in

        app = named_in(name, inventory())
    except Exception:  # noqa: BLE001
        return None
    if app is None:
        return None
    if "https" in app.handles or (app.media_evidence[:1] == ("web browser",)):
        return app
    return None


@tool(
    "open_url",
    "Open a web page in the user's default browser, or in the browser they named (browser: the "
    "name they said). Use this for any website - never assume a particular browser.",
    _p({"url": {"type": "string"}, "browser": {"type": "string"}}, ["url"]),
    category="web",
    quiet=True,
    idempotent=True,
)
async def open_url(a: dict, c: ToolContext) -> str:
    url = str(a["url"]).strip()
    scheme = url.split(":", 1)[0].lower() if re.match(r"^[a-z][\w+.-]*:", url, re.I) else "https"
    why = "" if scheme in ("http", "https") else f"{scheme}: links run code or open other apps"
    if not why and c.tainted() and policy.url_carries_data(url):
        why = f"the link carries data, after reading {c.tainted()}"
    if why:
        gate = confirm.gated("open_url", a, f"open {url[:80]} — {why}", detail=url)
        if isinstance(gate, str):
            return gate
    browser = str(a.get("browser") or "").strip()
    app = await asyncio.to_thread(_named_browser, browser) if browser else None
    return await apps.open_url(url, app.bundle_id if app else "", app.name if app else "")


# ---- the page the user is looking at -----------------------------------------------------------------
# "summarize this page", "what's on this page", "read me the comments", "is page pe kya hai", "ye page
# summarize karo": the page in the user's own browser, which NEO then answers them from (page_read).
_Q_TAIL = r"(?:\s+(?:for\s+me|please|pls|now))*\s*[.!?]*\s*$"
_PAGE_NOUN = r"(?:web\s*)?(?:page|site|website|tab|article|post|blog(?:\s+post)?)"
# "this page", and "the page" / "the tab" (on screen, the one in front): the page the user is looking
# at. "the article", "the post", "the site" alone may be one NEO fetched or the conversation named,
# which only a model can tell.
_THIS_PAGE = rf"(?:(?:this|that|current|the\s+current)\s+{_PAGE_NOUN}|the\s+(?:web\s*)?(?:page|tab))"
# The comments and replies are a page's (the browser in front: page_read reads no other app).
_COMMENTS = r"(?:comments?|replies)"  # a page's, wherever they are said
_REPLIES = r"(?:comments?|replies|reviews?|responses)"  # a page's when the page is named
_ON_PAGE = rf"\s+(?:on|in|under|of|below|from)\s+(?:this|that|the)\s+(?:{_PAGE_NOUN}|video)"
_HI_PAGE = r"(?:ye|yeh|is|iss|isi)\s+(?:web\s*)?(?:page|site|website|tab|article|post)"
_HI_DO = r"(?:karo|kar\s+do|kardo|kijiye|kar\s+dijiye|do|dijiye|batao|bata\s+do|sunao|suna\s+do)"
_PAGE_PATHS = [
    # summarize / sum up / recap / tl;dr this page, give me a summary of it, what's the gist of it
    (
        rf"{phrases._LEAD}(?:summari[sz]e|sum\s+up|recap|tl;?\s*dr|give\s+me\s+(?:a\s+|the\s+)?(?:quick\s+|short\s+|brief\s+)?"
        rf"(?:summary|gist|tl;?\s*dr|rundown|recap)\s+of|what(?:'s|\s+is)\s+the\s+gist\s+of)\s+"
        rf"(?:{_THIS_PAGE}|(?:the\s+)?{_COMMENTS}(?:{_ON_PAGE})?|(?:the\s+)?{_REPLIES}{_ON_PAGE}){_Q_TAIL}",
        {},
    ),
    # what's on / in this page, what's this page about, what does this article say
    (
        rf"{phrases._LEAD}(?:what(?:'s|\s+is)\s+(?:on|in)\s+{_THIS_PAGE}|"
        rf"what(?:'s|\s+is)\s+(?:this|that|current|the\s+current)\s+{_PAGE_NOUN}(?:\s+(?:all\s+)?about)?|"
        rf"what(?:'s|\s+is)\s+{_THIS_PAGE}\s+(?:all\s+)?about|what\s+does\s+{_THIS_PAGE}\s+say|"
        rf"tell\s+me\s+(?:about|what(?:'s|\s+is)\s+(?:on|in))\s+{_THIS_PAGE}){_Q_TAIL}",
        {},
    ),
    # read me this page, read this article out loud, read me the comments
    (
        rf"{phrases._LEAD}read\s+(?:me\s+|out\s+|aloud\s+)?"
        rf"(?:{_THIS_PAGE}|(?:the\s+)?(?:top\s+)?{_COMMENTS}(?:{_ON_PAGE})?|(?:the\s+)?(?:top\s+)?{_REPLIES}{_ON_PAGE})"
        rf"(?:\s+(?:to\s+me|out(?:\s+loud)?|aloud))?{_Q_TAIL}",
        {},
    ),
    # what are people saying in the comments, what do the reviews on this page say
    (
        rf"{phrases._LEAD}(?:what\s+are\s+(?:people|they)\s+saying\s+in\s+the\s+{_COMMENTS}(?:{_ON_PAGE})?|"
        rf"what\s+(?:are|do)\s+the\s+{_COMMENTS}(?:{_ON_PAGE})?\s+(?:saying|say)|"
        rf"what\s+(?:are|do)\s+the\s+{_REPLIES}{_ON_PAGE}\s+(?:saying|say)){_Q_TAIL}",
        {},
    ),
    # what's the price here / on this page
    (rf"{phrases._LEAD}what(?:'s|\s+is)\s+the\s+(?:price|cost)\s+(?:here|on\s+{_THIS_PAGE}){_Q_TAIL}", {}),
    # Hinglish: "is page pe kya hai", "is page mein kya likha hai", "ye page summarize karo", "is page
    # ka summary do", "ye article padh ke sunao", "comments padh ke sunao", "comments mein kya likha hai"
    (
        rf"{phrases._LEAD}{_HI_PAGE}\s+(?:pe|par|mein|me|main)\s+kya\s+(?:hai|likha\s+(?:hua\s+)?hai|chal\s+raha\s+hai|bola\s+hai)"
        rf"{_Q_TAIL}",
        {},
    ),
    (rf"{phrases._LEAD}{_HI_PAGE}\s+(?:ko\s+|ka\s+|ki\s+)?(?:summari[sz]e|summary|saar|saransh)\s+{_HI_DO}{_Q_TAIL}", {}),
    (
        rf"{phrases._LEAD}(?:{_HI_PAGE}|(?:is\s+page\s+(?:ke|ki)\s+)?{_COMMENTS})\s+(?:ko\s+)?"
        rf"(?:padh\s+ke\s+sunao|padhkar\s+sunao|padh\s+kar\s+sunao|padho|padh\s+do|sunao){_Q_TAIL}",
        {},
    ),
    (
        rf"{phrases._LEAD}(?:log\s+)?{_COMMENTS}\s+(?:mein|me|main)\s+kya\s+(?:likha\s+hai|hai|keh\s+rahe\s+hain|bol\s+rahe\s+hain)"
        rf"{_Q_TAIL}",
        {},
    ),
    (rf"{phrases._LEAD}(?:yahan|yaha|{_HI_PAGE}\s+(?:pe|par))\s+(?:price|daam|keemat|kimat)\s+kya\s+hai{_Q_TAIL}", {}),
    # Devanagari: "इस पेज पर क्या है", "ये पेज समराइज़ करो"
    (r"^\s*(?:इस|ये|यह)\s+पेज\s+(?:पे|पर|में)\s+क्या\s+(?:है|लिखा\s+है)\s*[?।.!]*\s*$", {}),
    (r"^\s*(?:इस|ये|यह)\s+पेज\s+(?:को\s+|का\s+)?(?:समराइज़|समराइज|सारांश)\s+(?:करो|कर\s+दो|बताओ)\s*[?।.!]*\s*$", {}),
]


@tool(
    "page_read",
    "Read the web page the user is looking at in their own browser (the front tab of whichever "
    "browser is in front): its URL, title and main text, or, when it has little text (a video, a "
    "picture, a PDF) or Accessibility is off, what a screenshot of it shows. Use it for 'summarize "
    "this page', 'what's on this page', 'read me the comments', 'what's the price here', then answer "
    "from it. With another app in front it reads nothing and says so: look at that app instead. Not "
    "for NEO's own browser window, and not for a link the user isn't looking at (web_fetch).",
    _OBJ,
    parallel_safe=True,
    category="web",
    untrusted=True,
    slow=True,  # a wait for the page's tree, a walk of it, maybe a vision call: Live talks meanwhile
    chain=False,  # "open X and summarize it" would read the page before it loads
    fast_path=_PAGE_PATHS,
    read_only=True,
)
async def page_read(a: dict, c: ToolContext):
    from neo2.core.decide import run
    from neo2.core.decide.kinds import site

    # The user's own words chose it (a fast path, or Laya's pick): no model is waiting to read the
    # page, so NEO answers them from it. A model's call gets the page itself.
    ask = " ".join(c.user_text.split()) if c.origin == "user" else ""
    return run.to_output(await site.page_read(ask=ask))


@tool(
    "safari_read",
    "Read the web page the user is looking at in their browser (the old name of page_read).",
    _OBJ,
    parallel_safe=True,
    category="web",
    untrusted=True,
    hidden=True,  # page_read reads whichever browser is in front; this name stays for old playbooks
    read_only=True,
)
async def safari_read(a: dict, c: ToolContext):
    return await page_read(a, c)


@tool(
    "finder_reveal",
    "Reveal a file or folder in Finder.",
    _p({"path": {"type": "string"}}, ["path"]),
    category="apps",
    quiet=True,
)
async def finder_reveal(a: dict, c: ToolContext) -> str:
    return await apps.finder_reveal(a["path"])


# ---- shell & files ----------------------------------------------------------------------


@tool(
    "shell",
    "Run a zsh command (cwd defaults to home). Read-only commands (ls, cat, grep, git status…) run "
    "straight away; anything that changes something is shown to the user to approve first.",
    _p(
        {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout": {"type": "number"},
        },
        ["command"],
    ),
    risk="destructive",
    category="files",
    slow=True,
    untrusted=True,  # file contents and app data it prints are content, never orders
)
async def shell_tool(a: dict, c: ToolContext) -> str:
    return await shell.run(a["command"], cwd=a.get("cwd"), timeout=float(a.get("timeout", 60)), args=a)


@tool(
    "read_file",
    "Read a text file (numbered lines), a PDF, or list a directory.",
    _p({"path": {"type": "string"}, "start": {"type": "integer"}, "lines": {"type": "integer"}}, ["path"]),
    parallel_safe=True,
    category="files",
    untrusted=True,
    read_only=True,
)
async def read_file(a: dict, c: ToolContext) -> str:
    return shell.read_file(a["path"], start=int(a.get("start") or 1), lines=int(a.get("lines") or 400))


@tool(
    "write_file",
    "Create a file with content. New files in Desktop/Documents/Downloads are written straight away; "
    "overwrites and anything elsewhere are shown to the user to approve first.",
    _p({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    risk="destructive",
    category="files",
)
async def write_file(a: dict, c: ToolContext) -> str:
    return shell.write_file(a["path"], a["content"], a)


@tool(
    "edit_file",
    "Replace one exact occurrence of old_string with new_string in a file (the user approves the change).",
    _p(
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"],
    ),
    risk="destructive",
    category="files",
)
async def edit_file(a: dict, c: ToolContext) -> str:
    return shell.edit_file(a["path"], a["old_string"], a["new_string"], a)


@tool(
    "trash",
    "Move a file or folder to the Trash (the user approves first).",
    _p({"path": {"type": "string"}}, ["path"]),
    risk="destructive",
    category="files",
)
async def trash(a: dict, c: ToolContext) -> str:
    return shell.trash(a["path"], a)
