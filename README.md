# NEO 2.0

**An AI agent for your Mac that does what you'd do — sees the screen, clicks, types, runs commands, handles mail. Voice-first. Runs on free tiers or fully local.**

Say *"Neo"* (or press **⌥⌘**), ask for something, and NEO works your Mac the way a person would: reads the Accessibility tree, clicks and types, opens apps, runs shell commands, edits files, browses, sends mail, sets reminders — and asks before doing anything it can't undo.

There is no subscription. The brains are Gemini and Groq free tiers, or Gemma 4 running on your own GPU — or bring your own key for Claude, OpenAI, OpenRouter or any OpenAI-compatible endpoint, and pick which one does what in **Brains & Keys** (see [Choose your brain](#choose-your-brain)). The decision layer, wake word, speech-to-text and text-to-speech all run on-device.

NEO's agent is also a Python library, `neo2`: other apps and AI frameworks (OpenAI, Anthropic, Gemini, LangChain, any MCP client) can embed it, with its safety rules and its confirmation gate — see [Use NEO from your own app](#use-neo-from-your-own-app).

---

## Why it feels fast

Most assistants send every word to a large model and wait. NEO routes first.

```
 "turn the volume up"        → regex fast path            0 LLM calls,   ~0 ms
 "what's the capital of Peru" → Laya (local) → Groq stream               ~150 ms + first token
 "reply to Sam's email…"      → Laya (local) → agent loop with 52 tools  ~150 ms + Gemini
```

The router is **Laya**, a 421M-parameter local decision model. One encoder pass (~20 ms on an M3 Pro) feeds five small calibrated heads trained on ~16k utterances — NEO's own templates plus 12k real crowd-sourced phrasings from CLINC150 and MASSIVE mapped onto NEO's labels: *what kind of request is this* (0.90 held-out on real phrasings, where Laya's zero-shot alone scores 0.44), *which single tool does it* (0.85), *does it want an answer or just doing*, *is it destructive*, *does it need the screen*. When the intent head is unsure it falls back to Laya's zero-shot answers, locally. Only then does a language model get involved — and only the one the task needs.

## What it can do

| Area | Tools |
|---|---|
| **See** | `ax_tree`, `ax_find` (Accessibility tree as text — fast, exact, private), `screenshot` (vision fallback for canvas/Electron apps) |
| **Act** | `click`, `drag`, `scroll`, `type_text`, `hotkey`, `ax_press`, `ax_set_value` |
| **Apps** | `open_app`, `activate_app`, `quit_app`, `apps_running`, `applescript`, `finder_reveal` |
| **Mail · Calendar · Notes** | `mail_unread`, `mail_send`, `calendar_today`, `calendar_add`, `notes_create`, `reminder_add` |
| **Web** | `web_search`, `web_fetch` (renders JavaScript-built pages invisibly), `open_url` (default or named browser), `page_read` (the tab you are looking at) |
| **Browser** | `browser_open`, `browser_read` (ARIA snapshot + text), `browser_click`, `browser_type`, `browser_press`, `browser_scroll`, `browser_back`, `browser_screenshot` — NEO's own Chromium with a persistent profile, so logins stick. It runs headless and shows live in the chat's browser card, where you can take it over, hand it back or pop it out (`NEO_BROWSER_VIEW=window` gives it a separate window instead) |
| **Files & shell** | `shell`, `read_file`, `write_file`, `edit_file`, `trash` |
| **System** | `volume`, `brightness`, `timer`, `clock`, `sleep_display` |
| **Memory** | `memory_search`, `remember`, `forget` (and the older all-in-one `memory`) — facts about you, a journal of what was said, conversation summaries; searched together (keywords + meaning) and told to every model on every route. See [Memory](#memory) |
| **Routines & quality** | Teach NEO “when I say *I want to study*, open …” for an exact spoken app/site setup; ask “how's your reliability?” for local interaction signals. |

Every tool is a plain Python function registered with a decorator; adding one is a few lines (see below).

## Architecture

```
 mic ─► wake word (Vosk, local) ─► voice backend
                                     ├─ Gemini Live: native speech-to-speech, delegates work via agent_task
                                     └─ local cascade: silero VAD → Parakeet STT (live partials) → brain → Kokoro TTS
                                               │
                                               ▼
                       Reflex — Laya + calibrated heads (local, ~20 ms)
                       intent: chat | quick_action | agent_task | stop · destructive? · needs screen?
                                               │
              ┌────────────────────────────────┼───────────────────────────────┐
              ▼                                ▼                               ▼
     stream a reply                    run one tool                   agent loop
     Groq → Gemini                     (no LLM at all)                Gemini 3.x Flash → Groq → local Gemma 4
                                                                      tools ⇄ results until done
                                                                      confirm gate · audit log · thrash guard
                                                                               │
                                                                               ▼
                                              Overlay (Tauri 2 + React, thinking-orbs) ⇄ websocket ⇄ Python core
```

**Every call is kept small.** A request only ships the tools it plausibly needs — picked by embedding the request against each tool's description, then expanding families (`browser_*`, `mail_*`) and adding the screen tools when the reflex says so — typically 3–7 of its 52 plus a `more_tools` escape hatch the model can call mid-run. Tool results older than two turns are stubbed out of the history. Simple one-tool actions that miss the regex fast path go to Gemini Flash-Lite instead of the full model. Net effect: a day of use fits the free tiers, and Groq's 8K-TPM tier can run a whole agent turn as the fallback.

**Brains are pluggable and fail over.** Gemini walks 3.8 → 3.7 → 3.5 Flash across every API key you give it (free-tier quota is per key *and* per model), backs off on overload, and hands text-only turns to Groq's `gpt-oss-120b` at ~700 tokens/s. `python -m neo2 --local` serves Gemma 4 12B on-device for offline use. Claude, OpenAI, OpenRouter and OpenAI-compatible endpoints (xAI, DeepSeek, Mistral, Together, Ollama, LM Studio…) take any role you give them: see [Choose your brain](#choose-your-brain).

**It acts while you're still talking.** The voice layer feeds the live transcript — words as you say them — through Laya and the fast-path matcher every few hundred milliseconds. A quick command that has clearly finished ("open YouTube and…") runs *before the sentence ends*; what already ran is remembered so the model's own later call for it is a no-op. Slow tools and agent tasks are non-blocking in the Live session: NEO acknowledges, keeps listening, and reports when the result is in and you're not mid-sentence. Several requests can run at once — ask for the time while it's still reading your mail — and the orb shows the most important thing going on.

**Commands are done quietly; questions get answers.** "open notes", "type Laptops", "set a timer" finish with no speech and no chat bubble — just the tool chip and the orb going back to listening (the spoken acknowledgement was ~2 s per step, with the mic muted). Anything whose result *is* the answer (time, mail, calendar, search, "what's on my screen") is spoken, as is every failure, confirmation and clarifying question. Over Gemini Live, action results go back with `SILENT` scheduling, so the model doesn't narrate them either.

**Single-step control runs in about a second, with no model at all.** "click play", "close this tab", "next tab", "go back", "zoom in", "scroll down", "take a screenshot", "switch to Safari", "quit Spotify", "check my email", "what's on my calendar", "search for …", "remember that …", "lock the screen" are regex fast paths straight to the tool; Laya's tool head covers the phrasings the regexes miss. Multi-step tasks still need a model.

**Typing never waits for a model.** "type Laptops in the heading", "new line", "select all", "press enter" are regex fast paths — the words *are* the payload, so they go straight to the keyboard (`type_text`, `hotkey`), with the caret put into the document's editor first (a fresh note if the app shows none). "type some points about laptops" is a *brief*: one small chat call writes the lines, then they're typed (`dictate`). None of it touches the agent loop, so it's the same speed on a bad Gemini day.

**It checks its own work.** After any side-effect tool, the loop makes the model re-observe (Accessibility tree, file, page, inbox) and confirm the goal before it reports — one extra call, far fewer "done!" replies that weren't.

**It remembers how it did things.** A successful multi-step task is saved as a *playbook* (goal + the tools it used, never the text it typed or sent). A similar request later gets that path in its prompt, so repeats are shorter and cheaper. The conversation is summarised every 20 idle minutes and at shutdown, so "what did we do yesterday" works and a crash loses little; recall is hybrid — FTS5 keyword search fused with local embeddings from Laya's encoder, already resident on the GPU.

**It suggests shortcuts from repeated success.** After the same app-and-website setup succeeds three times, NEO can offer a spoken shortcut. It stores only a hash and count for this suggestion signal; it does not monitor the screen in the background. Suggestions require `NEO_LEARN` to be on. Create a routine by saying, for example, “When I say ‘I want to study’, open Notion, Safari, and https://example.com.” This clear “when I say…, open…” form is parsed locally; more open-ended definitions may be interpreted by NEO's configured brain (Gemini when the cloud brain is selected). Routine matching, saved definitions and execution stay on the Mac. Actions can only open apps or ordinary web pages, and launches still pass through the regular safety checks. Routine definitions live in `~/.neo/routines.json`. Ask “list my routines” or “delete the study routine” to manage them. `NEO_WORKFLOW_SUGGESTIONS=false` disables suggestions.

**It reports how it is doing.** Ask “how's your reliability?” for the last seven days of local counts: settled requests, verified outcomes, corrections, quick interruptions and unresolved outcomes. These are interaction signals, not a factual-accuracy score. Laya's corrections and held-out no-regression evaluation already feed the learning loop; this report makes the recent signals visible.

**Safety is structural, not a prompt.** Anything consequential *stages* the action and returns `NEEDS_CONFIRM`; the run stops, the overlay shows exactly what will run (the full command, script, file path and contents, or mail body), and nothing happens until *you* approve it:

- **What runs without asking is an allowlist.** Shell commands only when every part is read-only (`ls`, `cat`, `git status`…, no redirects or substitutions); AppleScript only when it just reads (`get`, `count`); new files only in Desktop, Documents and Downloads. Everything else — deleting, overwriting, editing, installing, sending mail, Return in a terminal, buttons like *Send*, *Buy* or *Delete* (or an *OK* in a dialog that says it deletes), submitting a web form that isn't a search box — asks first.
- **Only you can approve.** Your typed or spoken "yes", or the card's button (⌘⏎) for that exact action. A model can't approve anything, a stale card can't approve a later action, and the approval covers exactly what the card showed — if anything changed, NEO asks again. By voice, "yes" only counts once NEO has read the question out.
- **Web pages, emails, files and the screen are data, never instructions.** They reach the model fenced as untrusted; they can't fake a confirmation, and after reading them NEO asks before saving memories or opening links that could carry your data out. Background fetches never reach your Mac or home network.
- **Secrets are off limits.** NEO doesn't read or write SSH/GPG keys, `.env` files, cloud credentials, keychains, browser profiles or its own data, and commands it runs don't inherit API keys.

Every staged, confirmed or cancelled action is appended to `~/.neo/audit.log` (the action, not the words of a message; about 1 MB, 180 days).

## Install

Requirements: macOS 14+ on Apple Silicon (built and tested on an M3 Pro, 18 GB), Python 3.12, and — for the overlay — Node 22.12+ (22.18+ for `npm test`) and Rust.

```bash
brew install uv rustup && rustup default stable
git clone https://github.com/ArnavParashar49/Neo2.0 && cd Neo2.0
uv venv --python 3.12 && uv pip install -r requirements-dev.lock && uv pip install -e ".[app,dev]"
cp .env.example .env        # keys here, or paste them later in Brains & Keys (menu)
.venv/bin/python -m neo2 --doctor
scripts/install-app.sh      # build NEO.app, install it to /Applications and start it
```

Keys: NEO needs at least one model. The simplest is a free Gemini key; NEO.app's **Brains & Keys…** (in the menu) takes it, or any other provider's, and tests it before saving — see [Choose your brain](#choose-your-brain). In a `.env` file instead:

- `GEMINI_API_KEY` — [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Add more in `GEMINI_API_KEYS=key2,key3` for more free quota.
- `GROQ_API_KEY` — [console.groq.com/keys](https://console.groq.com/keys) (optional; instant chat replies).
- `CEREBRAS_API_KEY` — [cloud.cerebras.ai](https://cloud.cerebras.ai) (optional, **paid**: a very fast brain for quick answers and text-only steps, tried ahead of Groq, and a seeing fallback when Gemini is overloaded). NEO spends at most `NEO_CEREBRAS_DAILY_BUDGET` a day on it.

The browser tools need Chromium once: `.venv/bin/python -m playwright install chromium`.

**NEO.app** (recommended): run `scripts/install-app.sh` from the project folder. It installs NEO's Python runtime (exact versions from `requirements.lock`: what the app needs; `requirements-dev.lock` adds the tests and Laya's training rows) in `~/Library/Application Support/NEO`, copies your `.env` to `~/.neo/.env`, builds and signs the app, puts it in `/Applications` (replacing any older copy) and starts it. Re-run it after pulling changes; `scripts/uninstall-app.sh` removes the app (`--purge` also moves `~/.neo` to the Trash).

It lives in the menu bar (the dotted orb), with no Dock icon: it runs the core, restarts it if it crashes or hangs, stops it on quit, and opens at login (menu → *Open at Login*). **⌥⌘** — press Option and Command together and let go — shows NEO; ⌘⇧Space works too. The first launch asks for **Accessibility** and **Screen Recording** (and the microphone when you first talk to it); the menu shows what's still missing, with shortcuts to the right settings pane. The app is signed by its bundle id, so what you grant once stays granted across updates. The menu also has *Brains & Keys…* (pick each role's brain, paste and test keys, set daily caps), *Edit settings file…* (opens `~/.neo/.env` in TextEdit), *Open Logs*, *Restart NEO* and *Erase NEO's Memory…*. With no model set up yet, NEO's chat shows a *Give NEO a brain* card, and a request that needs a model opens Brains & Keys.

Without the app: grant **Accessibility** and **Screen Recording** to the terminal you run `python -m neo2` from, in *System Settings → Privacy & Security*. macOS asks for the **Microphone** on the first voice run. Models (Laya, Parakeet, Kokoro, Vosk) download on first run; the wake-word model is checked against a pinned checksum.

## Run

```bash
.venv/bin/python -m neo2                 # voice + core; the overlay connects to it
.venv/bin/python -m neo2 --text -v       # terminal REPL that shows routing and every tool call
cd ui && npm install && npm run tauri dev   # the overlay (npm run tauri build → NEO.app)
```

The overlay lives in the bottom-right corner as a small pill (orb + status + last reply) and expands into a card on
**⌥⌘** (press both, let go), a click, or whenever NEO needs a confirmation; it folds back to the pill after a quiet stretch unless you
pin it. The card shows the conversation as a timeline — your messages, the tools NEO ran (as chips), rendered
markdown replies, and under each reply which brain answered and how long it took (*Groq gpt-oss-120b · 1.4 s*,
*quick action · no model*). Anything that needs your OK surfaces a card showing exactly what will run; ⌘⏎ approves, Esc declines. Setup problems (no API key, microphone blocked, voice down) show at the top of the card. While a turn is in
flight the header shows what's happening ("Gemini 3.8 Flash overloaded, trying next"). The thinking-orb mirrors
NEO's state:

| NEO state | Orb animation |
|---|---|
| idle | breathing |
| listening (mic open) | listening |
| thinking | solving |
| working (running tools) | solving |
| searching (web / search) | searching |
| speaking | composing |
| connecting (opening a voice session) | connecting |
| confirming (waiting for your OK) | shaping |

The mapping lives in `ui/src/App.tsx` (`ORB`).

### Checking NEO's computer controls

If clicking, typing or dragging doesn't seem to work, say **"Neo, check your computer controls"** (or type it
in the overlay; Hinglish works too: *"apne computer controls check karo"*). NEO then spends under a minute on
its own **test window** — a small separate app it starts, top-left on your main display — and drives it only
through its own tools, exactly as a model would: click at a point, right and double click, a screenshot turned
into coordinates (a region and the whole screen), `ax_tree` → click by id, `ax_press`, `ax_set_value`,
`type_text` with Hindi and emoji and with two lines, `hotkey` (Return and a menu shortcut), scrolling down and
up, dragging a piece on a Chess-like board, drag and drop, bringing the window forward with `activate_app`, and
a click on a second display if you have one. The window starts *behind* your app, so the first click shows
whether a click on a window that isn't in front gets through (Chess's board ignores that click).

While it runs, keep your hands off the mouse and keyboard: the first move, click or key press of yours stops it
at once. It never clicks, scrolls or types anywhere but its test window — every event is checked before it is
sent — and it won't start if something covers that window, while another NEO job is still at work, or while a
mouse button is held down. Afterwards it closes the window, puts the pointer back and brings back the app you
were in — unless you've taken over the mouse or keyboard (even as it finishes) or gone to another app: then
it leaves both as they are. The reply lists what worked and what didn't, with what NEO sent, what its tool
answered and what actually arrived; the full report is saved in `~/.neo/selftest/`.

Only your own words start it ("your computer controls", "NEO's", *apne* — "test the input controls" could be
about a game), once per request: no model can, and a model handing your earlier words back doesn't run it
again. For developers, `python -m neo2.core.selftest controls` runs the same test from your own Terminal window, with
that terminal's permissions (Accessibility; Screen Recording for the screenshot checks); it refuses to run as a
command NEO itself runs (its shell tool) or without an interactive terminal.

## The camera

Ask NEO *"what am I holding?"*, *"how many fingers?"* or *"what does this say?"* and it turns on the
Mac's camera. It does this only when you ask it yourself. The camera shows live in the chat, and NEO
answers from one still. On the Mac, Apple Vision reads the picture's labels, the words printed on
things, and your hands and raised fingers. The object detector draws its boxes on the live card, the
way NEO 1 did: everything it's sure of, people as "person", at most ten, each box following its thing.
The answer comes from Gemini,
or from Cerebras's vision model when Gemini is slow or down, with what the Mac found as hints.
`NEO_VISION_CLOUD_OK=false` keeps the still on the Mac and answers from the on-device labels alone.
The camera turns off 20 s after an answer (`NEO_CAMERA_FOLLOW_S`), on the stop button, or on the
card's own stop.

**The object detector** is YOLOE-26s prompt-free by Ultralytics, from NEO 1's YOLO family. Its
vocabulary is open, 4,585 words, but most of them are photo tags rather than things ("fairy",
"share", "jazz artist"), and on camera frames its confident answers in those words were mostly
wrong. So by default it may name only everyday things: about 230 you hold up to a camera or keep on
a desk, COCO's 80 among them, with people as "person". `NEO_VISION_WORDS=all` gives it the whole
vocabulary. NEO runs it through Core ML on the Neural Engine: about 9 ms a frame on an M3 Pro, with
no GPU and no PyTorch.

`scripts/install-app.sh` makes its model the first time, from Ultralytics' weights (40 MB, checked by
size and SHA-256), in a throwaway environment of its own: a minute or two and about 1 GB of memory.
By hand:

```bash
uv run --no-project --python 3.12 --with ultralytics==8.4.165 --with coremltools==9.0 \
    --with numpy==2.3.5 --with torch==2.14.0 --with torchvision==0.29.0 \
    scripts/get_yoloe.py --out ~/.neo/models --yes
```

That writes `~/.neo/models/yoloe-26s-pf.mlpackage` (23 MB). The first time the camera turns on, NEO
compiles it for the Neural Engine (about 2 s, once). Without it, or if it won't load, NEO uses
D-FINE small instead, which knows COCO's 80 objects and runs on the CPU, if its weights are in
`~/.neo/models/dfine-small-coco`. Without either, the card shows no boxes and the answer comes from
Apple Vision and the cloud brain. `NEO_VISION_DETECTOR` picks the detector: `auto` (the default),
`yoloe`, or a D-FINE repo id or folder. YOLOE's weights and the model made from them are AGPL-3.0:
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Naming a song

Ask NEO *"what song is this?"*, *"which song is playing?"* or *"ye kaunsa gaana hai"* and it listens
for a few seconds and names it: *"That's Blinding Lights by The Weeknd, from After Hours, 2020."* It
listens only when you ask it yourself. While it listens, the chat shows a card with a meter of how loud
the sound is. Then the card shows the song, its cover, and links to it on Apple Music and Shazam.

NEO listens to the room through the microphone. While your Mac is playing something, it also listens
to the Mac's own sound, so it can name a song you hear in headphones. For that it uses the Screen
Recording permission it already asks for; without it, only the microphone listens. It tries to match
after about 3.5 seconds, again at 7 and 11, and stops at the first match. After 12 seconds
(`NEO_SONG_LISTEN_S`) with no match it says it couldn't recognise the song. The stop button, saying
"stop" or the card's own stop end it at once.

What NEO hears stays in memory: it is never saved or logged. Only Shazam's fingerprint of it goes to
Shazam, which is a summary of the sound's strongest frequencies that can't be played back. NEO fetches
the cover art itself. `NEO_SONG_ID=false` turns this off. It uses
[shazamio](https://github.com/dotX12/ShazamIO) (MIT), an unofficial client for Shazam: see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Memory

NEO remembers three kinds of things, all in one SQLite file on this Mac (`~/.neo/memory.sqlite3`):

- **Facts** about you — your name, your people, what you like, how you want NEO to behave. You can tell it ("remember that I park on level 3"), and it also picks them up from what you say: sentences that state a fact in so many words ("my sister's name is Priya", "I live in Dubai", "mujhe adrak wali chai pasand hai") are read by rules on this Mac the moment you say them, and while NEO is idle the fast brain reads the new lines for the rest. Each fact knows which lines it came from. A newer value replaces the old one (the old one is kept, marked superseded): never two that contradict.
- **The journal** — every line you say or type and every line NEO says, written as it happens (within a quarter of a second, by a background writer: nothing waits for it). A voice session that closes, a quit or a crash loses nothing. Lines of NEO's that quote a page, a mail or a file are marked and never fed back into a prompt; tool outputs are never journaled, only which tools ran.
- **Summaries** — one per conversation, written from the journal once it goes quiet. A conversation that never got one (NEO was killed) gets it at the next start.

**Recall.** Every request — typed, spoken to the on-device voice, or spoken to Gemini Live — starts with a memory block: your profile and lessons, the last three conversation summaries, the last lines said today, and the best search hits for what you just asked. Gemini Live gets it fresh every time a session opens, so "Neo" ten seconds later continues the same conversation. For anything older, the models call `memory_search` (and are told never to say "I don't remember" without it). You can ask directly: "what do you remember about my sister", "what did we talk about yesterday / last week / on Monday", "what do you know about me", "kal humne kya baat ki thi".

**Search** is keyword and meaning together: SQLite FTS5 (Unicode words — Devanagari included — with diacritics folded and English endings stemmed, plus a trigram index for substrings) and MiniLM embeddings held in RAM, fused, then ordered by how much of the question a row covers (rare words count for more), how recent and important it is, and what it is (a fact over a summary over a line). Time phrases ("yesterday", "last week", "kal", "pichle hafte") become a time window. A warm search takes about a millisecond on 20,000 journal lines; the embedding model is never loaded on a request's path (keywords only until it is ready).

**Secrets are never stored.** Passwords, PINs, one-time codes, API keys, tokens and card numbers are masked before anything is written, and "remember my password is …" is refused out loud.

**Forgetting.** "Forget what I said about the Goa trip" finds the facts, the journal lines and the summaries about it, shows you exactly what will go, and deletes them when you say yes; the conversations that lost a line are summarised again without it. *Erase NEO's Memory…* removes everything (see [Privacy](#privacy)).

When an older database is first opened by this version it is upgraded in place, after a copy is kept next to it (`memory.sqlite3.bak-v2`); every memory it held is still there.

| Setting | Default | |
|---|---|---|
| `NEO_MEMORY_JOURNAL` | `true` | keep the journal (off: only what NEO is told to remember is kept) |
| `NEO_MEMORY_RETENTION_DAYS` | `365` | journal lines older than this are deleted; facts and summaries stay (`0` keeps everything) |
| `NEO_MEMORY_EXTRACT` | `rules+llm` | `rules+llm` · `rules` (on this Mac only) · `off` — how facts are taken from what you say |
| `NEO_MEMORY_RECALL_K` | `6` | search hits a request's memory block may carry |
| `NEO_MEMORY_LIVE_INJECT` | `false` | Gemini Live: also send what memory has about a sentence as a note before the model answers. Off until measured on your connection: the log line `[memory] live recall: …` says, for each sentence, whether the note would have arrived in time — a late one would cut the answer short |
| `NEO_CLOUD_MEMORY` | `full` | what of all this goes to cloud models: `full` · `profile` (only your profile facts ride along with requests) · `off` (nothing: no memory block, no search results; summaries are made locally, and no model reads the journal) |

## Choose your brain

NEO thinks with the provider you pick, on your own key. In NEO.app open **Brains & Keys…** from the menu (or NotchHub's Settings → NEO → *Brains & Keys…*): paste a key, test it (no tokens spent), and choose which brain does what. With no model set up yet it opens on a quick setup — pick a provider, paste its key, done — with Gemini preselected (free, no card).

| Provider | Cost | Can be | Key | Defaults when set up in the window |
|---|---|---|---|---|
| **Google Gemini** | free tier, no card | Brain, Quick replies, Eyes, camera, and the Gemini Live voice | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Brain and Eyes: Gemini 3.8 Flash (then 3.7, 3.5); Quick replies: 3.5 Flash-Lite |
| **Anthropic (Claude)** | paid | Brain, Quick replies, Eyes, camera | [console.anthropic.com](https://console.anthropic.com/settings/keys) | Brain and Eyes: Claude Opus 5.5 ($4 / $20 per million tokens in / out); Quick replies: Claude Haiku 5.5 ($0.10 / $0.50). Also Sonnet 5.5, Fable 5.1 (needs 30-day data retention on the account), Opus 5 |
| **OpenAI** | paid | Brain, Quick replies, Eyes, camera | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Brain and Eyes: GPT-6.1 Sol ($2 / $10); Quick replies: GPT-6 Luna ($0.10 / $0.50); GPT-6 Astra is the most capable. Through the Responses API |
| **Groq** | free | Brain (text only), Quick replies, Eyes (its vision model) | [console.groq.com/keys](https://console.groq.com/keys) | `gpt-oss-120b` for everything but pictures; `qwen/qwen3.8-27b` sees (a preview, with a small per-minute budget) |
| **Cerebras** | paid, $0.50 a day unless you set a cap | Brain (text), Quick replies, Eyes, camera | [cloud.cerebras.ai](https://cloud.cerebras.ai) | `gpt-oss-120b`, and `qwen-3.8-27b` that sees; tried ahead of Groq |
| **OpenRouter** | pay as you go (a few models are free) | whatever the model you pick can do | [openrouter.ai/keys](https://openrouter.ai/keys) | none: one key, hundreds of models, and you pick one (`deepseek/deepseek-v4-pro`, `anthropic/claude-opus-5.5`…) |
| **Other endpoint** (OpenAI-compatible) | the endpoint's | Brain if it calls tools, Quick replies, Eyes if it sees | the provider's | templates for xAI, DeepSeek, Mistral, Together, Ollama and LM Studio (both on this Mac, no key), or any URL |
| **NEO's model on this Mac** (Gemma 4) | free, offline | Brain (text only), Quick replies | — | runs while `python -m neo2 --local` runs |

Prices are the catalogue's as of October 2026: check them at the provider. The model menus read each key's own live list; the defaults are only pre-selections.

**The roles.** *Brain* does tasks with tools. *Quick replies* answers chat turns, writes conversation summaries and memory facts, and cleans up dictation. *Eyes* is the first brain asked for any step with a screenshot or a photo, and for camera questions (empty: the Brain if it sees, then the other brains that see). *Backups* (Automatic by default) are tried when those fail. *Screenshots* — *Only the brains I picked* (`NEO_CLOUD_EYES=off`) or *Other brains with a key may help* — decides whether a configured brain you didn't name may look at a picture when yours can't. *Voice*: with a Gemini key, Gemini Live stays the voice whatever the brain (it answers simple spoken questions itself and hands tasks to your brain); *On this Mac* lets your brain answer everything. NEO checks every choice again: a brain that can't call tools can't be the Brain or a Backup, and Eyes has to see.

What a brain that lacks something means:

- **No tool calling** (some endpoints and local models): Quick replies only; a step that offers tools passes it over.
- **Text only** (Groq's or Cerebras's `gpt-oss`, DeepSeek, most local models): NEO works through the Accessibility tree; a step that must look at the screen goes to Eyes, and with no Eyes it stops and says no brain can see the screen.
- **No Gemini key**: no Gemini Live — the on-device voice (Parakeet, your brain, Kokoro) takes over — and no Laya *lite* mode.
- **A paid Quick replies brain** costs money on every summary and memory extraction: the window suggests the cheap tier (Haiku 5.5, GPT-6 Luna) and marks it *Paid*.

**Where keys live.** A key you paste in Brains & Keys (or `python -m neo2 keys set`) goes to `~/.neo/api_keys.json`; your choices of brain, the endpoints you add and the daily caps go to `~/.neo/brains.json` (no keys in it). Both are mode 0600 in the 0700 `~/.neo`, written whole or not at all, and made only when you save something: NEO never copies a key from a `.env` file or your environment into them, and never writes a brain choice into `.env`. That's *readable only by your Mac account* — any program running as you can read it, just like `~/.neo/.env`; it isn't the Keychain. NEO's own tools never read either file, and a key goes only to the provider (or the endpoint's own host) it was saved for. The window shows a key's last four characters at most; every other client hears only whether one is set.

**Which one wins**, setting by setting: an environment variable, then what was saved in NEO, then a checkout's `.env` (running from source), then `~/.neo/.env`, then the default. So a setup made with `.env` lines keeps working as it was, and the window says when an environment variable wins over what you save there (*Set by environment variable ANTHROPIC_API_KEY*). *Replace* on a key from `~/.neo/.env` saves the new one in NEO, which wins; the `.env` line stays. A brain setting NEO can't read (`NEO_BRAIN=claud`) never stops it from starting: that role is skipped, and the problem list says so. If `api_keys.json` can't be read, NEO treats it as empty and sets it aside as `api_keys.json.bad-<date>` when you next save; the window and `--doctor` say while such copies are there — delete them yourself, they may hold keys. An agent embedded in another app (the `neo2` library, the MCP server) never reads these files.

**In a terminal** (a core you started yourself has no window):

```bash
python -m neo2 keys list                  # each provider: where its key comes from, its last 4
python -m neo2 keys set anthropic         # typed without an echo, never an argument; tested first
python -m neo2 keys test openai           # does it work? (no tokens spent)
python -m neo2 keys remove anthropic
python -m neo2 brain show
python -m neo2 brain set --brain claude@claude-opus-5-5 --quick claude@claude-haiku-5-5 --eyes claude@claude-opus-5-5
python -m neo2 brain set --budget anthropic=5
```

A running NEO picks the change up at once. `python -m neo2 --doctor` lists every provider with where its key comes from, the roles and whether each can answer; `--doctor --test` asks each provider.

**Brain references.** A role holds `<brain>[@<model>]`: `gemini`, `gemini_lite`, `groq`, `groq_vision`, `cerebras`, `cerebras_vision`, `claude` (or `anthropic`), `openai`, `openrouter`, `local` or `custom:<name>`, and after `@` a model id — `claude@claude-haiku-5-5`, `openai@gpt-6-luna`, `openrouter@deepseek/deepseek-v4-pro`, `custom:ollama@qwen3:14b`. A bare name is exactly the brain it always was.

**Daily caps.** Each paid provider can have one (the window's *Spending*, `brain set --budget`, or `NEO_DAILY_BUDGETS='{"anthropic": 2}'`). A paid key saved in Brains & Keys gets $2 a day on its own (Anthropic, OpenAI, OpenRouter); a key from an environment variable or a `.env` file stays uncapped. Each answer's token usage is counted at the catalogue's prices — OpenRouter's at the price its list gives each model — and a model with no known price at the dearest. At 80% NEO says so; once it's spent NEO skips that provider until midnight and says why. An endpoint you add has no price list, so its cap is only suggested (and counts at the dearest price NEO knows). Cerebras keeps its own budget (`NEO_CEREBRAS_DAILY_BUDGET`).

**Other endpoints.** The base URL must be `https://` (`http://` only on this Mac: `localhost`, `127.0.0.1`, `[::1]`), with no user, password or query in it. Set whether it calls tools and whether it sees pictures; *Check tools* sends one tiny request to prove it (it uses a few tokens). A template's own variable (`XAI_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `TOGETHER_API_KEY`) wins over a key saved for it, but only at that template's own host; moving an endpoint to another host asks for its key again. A key whose shape names another provider (an `sk-ant-` key in OpenRouter's card) isn't sent at all. From a terminal: `NEO_CUSTOM_PROVIDERS` takes the same endpoints as JSON.

**Claude as the brain.**

1. Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys).
2. Brains & Keys → *Claude* (or *Providers* → Anthropic → *Add key*) → paste → *Connect*. NEO tests the key, saves it, and makes Claude Opus 5.5 the Brain and Eyes and Claude Haiku 5.5 Quick replies, with a $2 a day cap.
3. With a Gemini key too, Gemini Live stays the voice and Gemini can still look at a screenshot Claude can't take; *Voice → On this Mac* lets Claude answer everything, and *Screenshots → Only the brains I picked* keeps every picture with Claude.

With `.env` lines instead: `ANTHROPIC_API_KEY=…` and `NEO_BRAIN=claude` (Claude Opus 5, or `NEO_CLAUDE_MODEL`; `NEO_FAST_BRAIN=claude@claude-haiku-5-5` and `NEO_EYES_BRAIN=claude@claude-opus-5-5` for the rest). As before, an Anthropic key alone only makes Claude a screenshot fallback.

**OpenAI as the brain.**

1. Get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. Brains & Keys → *OpenAI* → paste → *Connect*: GPT-6.1 Sol becomes the Brain and Eyes and GPT-6 Luna Quick replies, with a $2 a day cap.
3. The voice and screenshots work as for Claude above.

With `.env` lines: `OPENAI_API_KEY=…`, `NEO_BRAIN=openai@gpt-6.1-sol`, `NEO_FAST_BRAIN=openai@gpt-6-luna`, `NEO_EYES_BRAIN=openai@gpt-6.1-sol`. An OpenAI (or OpenRouter, or endpoint) key alone sends nothing anywhere: only the roles you give it.

## Configuration

Everything is an environment variable with the `NEO_` prefix (or a line in `.env`). The useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `NEO_BRAIN` | `gemini` | the Brain, as a [brain reference](#choose-your-brain): `gemini` · `groq` · `local` · `claude` · `openai` · `openrouter` · `custom:<name>`, each with an optional `@<model>` · `cerebras` (its text model `gpt-oss-120b` for text steps, chat and single actions, ahead of Groq; steps that carry a screenshot go to Cerebras' own `qwen` first, then the other brains that see) · `cerebras_vision` |
| `NEO_FAST_BRAIN` | `groq` | Quick replies: chat-only turns, summaries, memory facts, dictation |
| `NEO_EYES_BRAIN` | — | Eyes: the first brain asked for a step with a screenshot or photo, and for the camera (empty: the Brain if it sees, then the other brains that see) |
| `NEO_BACKUP_BRAINS` | automatic | up to four brains tried after the Brain (and Quick replies): `claude,groq`; `none` for none; automatic: Quick replies backs up the Brain |
| `NEO_LIGHT_BRAIN` | `gemini_lite` | one-step commands that missed the regex fast path; empty: Quick replies' brain (the window empties it when Quick replies isn't Gemini) |
| `OPENAI_API_KEY` · `OPENROUTER_API_KEY` | — | used only where a role names `openai` or `openrouter` (a key alone sends nothing) |
| `NEO_CLAUDE_MODEL` · `NEO_OPENAI_MODEL` · `NEO_OPENROUTER_MODEL` | `claude-opus-5` · `gpt-6.1-sol` · — | the model a bare `claude`, `openai` or `openrouter` means |
| `NEO_DAILY_BUDGETS` | — | dollars a day per provider (JSON or `anthropic=2,openai=1`): see [daily caps](#choose-your-brain) |
| `NEO_CUSTOM_PROVIDERS` | — | OpenAI-compatible endpoints as JSON: `[{"id": "custom:ollama", "label": "Ollama", "template": "ollama", "base_url": "http://localhost:11434/v1", "model": "qwen3:14b"}]` (Brains & Keys saves them in `~/.neo/brains.json` instead) |
| `NEO_GROQ_VISION_MODEL` | `qwen/qwen3.8-27b` | Groq's model that sees: with a Groq key, the last model a screenshot goes to when Gemini (and Flash-Lite, Claude) can't answer; empty turns it off |
| `NEO_CLOUD_EYES` | `auto` | whether a configured brain you didn't name may look at a picture yours can't (Gemini, Flash-Lite, Claude, Groq's vision model). `auto`: yes, except with `NEO_BRAIN=local` and no cloud brain that sees named (the step stops and says so); `on`: yes; `off`: never — only the brains you picked see a picture (before Brains & Keys, `off` read as `auto`) |
| `CEREBRAS_API_KEY` | — | a paid [Cerebras](https://cloud.cerebras.ai) key: `gpt-oss-120b` takes chat and text-only steps wherever Groq would (tried just ahead of it); `qwen-3.8-27b` sees the screen when Gemini is overloaded (ahead of Groq). Never ahead of the local brain when you put it first; with only the local brain named (`NEO_BRAIN`/`NEO_FAST_BRAIN=local`) Cerebras isn't used, and your screenshots go to it only if you also named a cloud brain that sees. No key: nothing changes |
| `NEO_CEREBRAS_DAILY_BUDGET` | `0.50` | dollars a day NEO may spend on Cerebras (usage × price, counted per answer). At 80% it says so; once spent it skips Cerebras until midnight and says why. `0` or `off` turns Cerebras off (no chain names it); so does a value that isn't a dollar amount (a negative number, a typo), with a warning in the log |
| `NEO_CEREBRAS_MODEL` / `NEO_CEREBRAS_VISION_MODEL` | `gpt-oss-120b` / `qwen-3.8-27b` | the text and seeing models (one that `/v1/models` doesn't list for your key is skipped) |
| `NEO_CEREBRAS_PRICES` | published prices | JSON, $ per million tokens: `{"gpt-oss-120b": [0.35, 0.75], "qwen-3.8-27b": [0.99, 1.49]}` (entries override or add models; an unknown model counts at the dearest price) |
| `NEO_VOICE` | `auto` | `auto` (Live if the key allows it, else local) · `live` · `local` · `off` |
| `NEO_REFLEX` | `laya` | `laya` · `lite` (Gemini Flash-Lite) · `off` |
| `NEO_LISTEN_TIMEOUT_S` | `5` | stop listening after this many seconds without speech from you (silero VAD; noise doesn't count) |
| `NEO_WAKE_WORD` | `true` | listen for “Neo”. `true`: the microphone stays open so the wake word can be heard (macOS shows its orange microphone dot all the time). `false`: the microphone is closed while NEO is idle and opens only when you summon NEO by hand — the summon keys (⌥⌘), the orb, push-to-talk — until NEO is idle again (`NEO_LISTEN_TIMEOUT_S`); typed requests never open it, and “what song is this?” opens it just for the listen. Switched while NEO runs from NotchHub's Settings → NEO (“Listen for “Neo””), which writes this line itself (`{"cmd": "set", "name": "wake_word"}`) |
| `NEO_WELLBEING_CHECKINS` | `false` | gentle check-ins: a couple of coughs or sneezes, sniffles, or a few seconds of crying make NEO ask a short, kind question and listen about 5 s for your answer. Detection stays on-device (Apple's sound classifier hears only what NEO's microphone captures while it is open anyway: it never opens it, and with `NEO_WAKE_WORD=false` it hears only the sessions you start, and NEO doesn't listen for a reply). With Gemini Live, the check-in line and ~5 s of your reply audio go to Gemini; neither goes to NEO's journal. The same kind waits 10 min, then 1 h, then 4 h; at most 3 a day; none for the rest of the day after a “no” or two unanswered ones; only while NEO is idle and no other app is playing sound. Switched from NotchHub's Settings → NEO (“Gentle check-ins”) |
| `NEO_WAKE_GATE` | `shadow` | after “Neo”, does anyone actually speak? `shadow` only measures it (shown in “how's NEO doing?”); `on` sends no audio to Gemini until you start speaking (within `NEO_WAKE_GATE_S`, default `3` s; at least `NEO_WAKE_GATE_SPEECH_MS`, default `300` ms of speech), so a false wake costs nothing; `off` turns the check off. Switch to `on` once the report shows few false drops |
| `NEO_MIC_ESCALATE` | `true` | when the microphone goes silent, NEO checks that a reopened stream really hears, retries with growing gaps, tells you once (a note and a notification) if it stays deaf, and says when it is back |
| `NEO_STUCK_BREAKER` | `true` | in computer-control tasks, a step that failed the same way twice is not tried a third time; NEO is told to try something else |
| `NEO_CONTEXT_CAPTURE` | `true` | “this”: when your own words ask about what you selected, copied or are looking at (“summarize what I copied”, “translate this”, “what does this say”), NEO reads the selection, the clipboard or the front window for that one request. It never reads password fields or password-manager copies, keeps nothing, and the text is handled as untrusted content |
| `NEO_CONTEXT_COPY_FALLBACK` | `true` | when a web page doesn't share its selection through Accessibility (Google Chrome, other Chromium browsers, Electron apps), NEO copies it: ⌘C sent to that app alone, the copy's plain text read, then your clipboard put back exactly as it was (every item, every format; an empty one stays empty), even if something fails on the way — and put back again over a copy that lands up to a second late. Only on web pages, never in a text view, a terminal or a password manager's list (where ⌘C may do something else). It never does this while your clipboard holds a private copy (a password manager's), a copy from another device, more than 32 MB or anything it can't keep whole, never on a Mac set to ask before apps read the clipboard, never in a password field, and never reads a copy marked private; if your clipboard couldn't be put back, NEO says so. An app behind NEO's window (you typed to NEO there) may not copy at all — NEO then says it isn't in front; ask by voice or from the notch panel with the app in front. A page may copy something of its own for ⌘C, and a clipboard-history app may keep the copy. `false`: only what Accessibility gives |
| `NEO_SHORTCUTS` | `true` | your macOS Shortcuts as commands: “run my <name> shortcut”. A shortcut the model picks by itself asks first |
| `NEO_SHORTCUTS_NAMES_IN_PROMPT` | `false` | also tell the model your shortcuts' names, so it can suggest them |
| `NEO_LOCAL_MODEL` | `mlx-community/gemma-4-12B-it-4bit` | offline model served by `--local` |
| `NEO_CONFIRM_DESTRUCTIVE` | `true` | set `false` only if you enjoy surprises |
| `NEO_CLOUD_MEMORY` | `full` | what NEO's memory sends to cloud models: `full` · `profile` (stable facts only) · `off` |
| `NEO_LEARN` | `on` | `off` stops Laya recording your requests to learn from (a reflex with no head for its encoder still gets one, trained on the public data alone) |
| `NEO_WORKFLOW_SUGGESTIONS` | `true` | offer a spoken shortcut after the same successful app/site setup repeats three times; stores only a local hash and count |
| `NEO_LEARN_TEACHER` | `off` | `local` · `cloud`: while NEO is idle, a model re-checks up to 50 a day of the requests Laya was least sure about — `local` only with the local model, `cloud` only with the provider that already handled that request |
| `NEO_HOME_LOCATION` | — | where "the weather" means if NEO doesn't know your city (else it guesses from your IP) |
| `NEO_DATA_DIR` | `~/.neo` | memory DB, audit log, models, trained reflex head |
| `NEO_UI_HOST` | `window` | where NEO's UI lives: `window` (NEO.app's own orb and card) · `notch` (NotchHub hosts the same page in the notch; NEO.app keeps its window hidden and its hotkeys; its menu-bar item goes while NotchHub is connected, whose Settings carry the menu). Written by the core itself when NotchHub asks (`{"cmd": "ui_host"}`) and put back to `window` when NotchHub quits — nothing to set by hand |

With NEO.app, keys and brains are set in the menu's *Brains & Keys…* (they apply at once; a change to the Gemini key or the voice asks for a restart). The rest of the settings live in `~/.neo/.env` (menu → *Edit settings file…*, then *Restart NEO*).

### NEO in the notch

[NotchHub](../../NotchHub) can host NEO's whole UI in the notch: its lane shows NEO's orb and a line
while NEO listens or works, and its panel opens down from the notch with this same chat page — the
transcript, the confirm buttons, the browser, camera and song cards — loaded in a WKWebView from
its own `neo://app/` scheme. For that the page has a `host=notch` mode (`ui/src/host.ts`,
`ui/src/shell.ts`: every Tauri call shimmed, no window of its own, the host's black theme), the
core's websocket admits exactly the origin `neo://app`, and `scripts/install-app.sh` puts the built
page in `~/Library/Application Support/NEO/ui/` where NotchHub reads it (Tauri embeds its copy in
the binary). NotchHub tells the core `ui_host: notch` whenever it connects and `window` as it quits.

While NotchHub is connected NEO.app has no menu-bar item: NotchHub's Settings carry its menu, and
ask the core for an action (`{"cmd": "shell"}`, a host's alone). The core hands it to NEO.app over a
socket the two share — an anonymous pair NEO.app makes when it starts the core
(`ui/src-tauri/src/core.rs`, `neo2/core/server/shell_link.py`) — and hears what the menu shows, and
the summon keys, the same way. Not through NEO.app's own page: its window is hidden in the notch,
and macOS suspends the process of a web page nobody can see.

## Privacy

NEO keeps its data on this Mac, in `~/.neo` (readable only by you). What leaves it, and where:

| Goes to | What | When |
|---|---|---|
| **Google** (Gemini API, Gemini Live) | what you say and type, the recent conversation, NEO's memory about you (see `NEO_CLOUD_MEMORY`), tool results, and screenshots when a task needs to see the screen | every request that needs a model; voice sessions stream your microphone audio while listening; with gentle check-ins on (`NEO_WELLBEING_CHECKINS`), the check-in line and ~5 s of your reply audio (the sound detection itself stays on the Mac) |
| **Groq** (optional key) | chat turns, and the conversation text when NEO writes a conversation summary or reads it for facts while idle (not with `NEO_CLOUD_MEMORY=off`); a screenshot only when no other model that sees can answer (its vision model, `NEO_GROQ_VISION_MODEL`) | quick answers and summaries; a screen step while Gemini is overloaded or out of quota |
| **Cerebras** (optional paid key) | the same as a Gemini request (what you say and type, the conversation, NEO's memory per `NEO_CLOUD_MEMORY`, tool results); screenshots only to its seeing model, at most two per request | steps and chat turns it answers (ahead of Groq), and screen steps when Gemini is overloaded |
| **Anthropic** (Claude, optional paid key) | the same as a Gemini request, and screenshots and camera stills when it has a role that sees | the roles you gave it; an Anthropic key alone (kept from before) makes Claude a screenshot fallback when no brain you named that sees can answer — *Screenshots → Only the brains I picked* turns that off |
| **OpenAI** (optional paid key) | the same as a Gemini request, and screenshots and camera stills when it has a role that sees | only the roles you gave it |
| **OpenRouter** (optional key) and the company behind the model you picked | the same as a Gemini request, and screenshots when that model sees; OpenRouter passes each request on to the model's own provider | only the roles you gave it |
| **an endpoint you added** (xAI, DeepSeek, Mistral, Together, any URL) | the same as a Gemini request; screenshots only when you said it sees | only the roles you gave it; one on this Mac (Ollama, LM Studio) keeps it all on the Mac |
| **the provider of a key you test** | the key (in a header), for its list of models — no conversation, no tokens; *Check tools* sends one tiny request | when you press *Test*, save a key, or open Brains & Keys (it lists each key's models, kept 10 minutes) |
| **DuckDuckGo, Brave, Bing** | your search queries (raced for speed) | web searches |
| **Open-Meteo**, and **ipinfo.io** if NEO doesn't know your city | the place, or your IP address to guess it | weather |
| any website NEO opens or fetches for you | normal web requests | browsing tasks |

The free tiers of Gemini and Groq allow those companies to use what you send to improve their products; use paid keys, `NEO_CLOUD_MEMORY=profile|off`, or the local brain (`--local`) if that matters for you. Each paid provider's own terms say what it keeps; OpenRouter's, and the provider's behind the model you pick, both apply there. The wake word, speech-to-text and text-to-speech of the on-device voice, and Laya's decisions all run locally.

What NEO keeps, and for how long:

- **Memory**: facts and preferences until you delete them ("forget …"); **the journal** of what was said — `NEO_MEMORY_RETENTION_DAYS` (a year), never passwords, codes, keys or card numbers; **conversation summaries** until you erase; **playbooks** (the path of past tasks, without message text) — the newest 300; **spoken routines** (the trigger phrase and app/site launch list) until you delete them. `NEO_MEMORY_JOURNAL=false` keeps no journal at all.
- **Requests Laya learns from** (`reflex_extra.jsonl`: no dictated text, notes, memories, mail, or anything that looks like a code or password) — the latest 2,000 within 180 days; `NEO_LEARN=off` stops it.
- **Logs** (`~/.neo/logs`) record which tools ran and whether they worked — never what you typed, said, sent or read. The **audit log** records confirmed actions (not the words of messages).
- **Your keys**: in `~/.neo/api_keys.json` (see [where keys live](#choose-your-brain)), never in a log — a loaded key that turns up in an error message or a provider's answer is written as `••••` and its last 4 — and never in memory: text you type that starts with a key is sent to no model ("That looks like an API key, so I didn't send it anywhere").
- The overlay and the Brains & Keys window are hidden from screen sharing, recordings and screenshots.

**Erase everything:** menu → *Erase NEO's Memory…* (or `python -m neo2 --forget-all`) deletes memories, the journal, conversation summaries (and the copy of the database kept from before an upgrade), playbooks, spoken routines, workflow-suggestion counters, learned requests and their cached features, logs, the audit log and the logins in NEO's own browser. Your settings and keys stay. No model or web page can trigger it.

## Training the reflex

The heads are trained locally in a few minutes (plus feature extraction the first time):

```bash
.venv/bin/python -m neo2.core.reflex.hf_data    # optional: ~45k public phrasings (English, Hindi, Hinglish) + a benchmark
.venv/bin/python -m neo2.core.reflex.generate   # optional: grow the dataset with Gemini + Groq, cross-labelled
.venv/bin/python -m neo2.core.reflex.generate --focus chat:quick_action   # or --from-confusion FILE, --lang hinglish: ≤300 rows at one weak boundary
.venv/bin/python -m neo2.core.reflex.finetune   # embed (cached per text), fit the heads, gate, save
.venv/bin/python -m neo2.core.reflex.evaluate   # the head on the benchmark: by language and source, recalls, tool/kind, ECE, latency
```

`hf_data` maps open intent, dialogue, function-calling and computer-use sets (CLINC150, MASSIVE en/hi, Hinglish-TOP, TOPv2, MTOP en/hi, Schema-Guided Dialogue, glaive, AgentNet, OmniACT, …) onto NEO's labels — route, tool, task kind, destructive, needs the screen, wants a reply — and writes `~/.neo/reflex_hf.jsonl`. Every source is pinned to a revision and only the columns or byte ranges it needs are downloaded; `neo2/core/reflex/sources.py` lists each one's licence. Some are CC-BY-SA (Hinglish-TOP too: it translates TOPv2), so the rows are built on your Mac and never committed. The Hinglish-TOP, MTOP, CLINC and MASSIVE test splits go to `reflex_eval.jsonl`, a benchmark whose sentences — and their translations and re-punctuated copies, from any source — are never trained on, whichever sources a build uses. NEO's own templates share a few short sentences with it ("next song", "ruk jao"): evaluate on `dataset.benchmark(rows)`, which drops what the head trained on. Leyzer is opt-in (`--with leyzer`): the repository hosting its data is licensed non-commercial, no-derivatives. So is `sentence_types` (no licence stated). Romanised-Hindi templates for every class, including destructive requests and stop phrases ("bas karo", "ruk jao"), are part of NEO's own data (`neo2/core/reflex/templates_hi.py`).

`finetune` embeds every sentence once with Laya's multilingual checkpoint (`NEO_REFLEX_ENCODER=english` keeps the older English one; a head only ever runs on the encoder it was trained for) and fits six small MLP heads on it — intent, destructive, needs the screen, wants a reply, tool and task kind — each a 3-seed ensemble with its loss and temperature chosen on held-out rows. Rows are split by hashes, so every run agrees: a fixed base slice of the public and NEO rows (the anchors), 20% of your verified requests (the user slice), and a calibration slice (10% of your requests plus 1 in 20 public groups). A new head replaces the current one only if it clears absolute floors and ceilings — every time, not just the first: how many destructive, stop and screen requests it catches (judged only on rows whose label a rule didn't write), and how often it goes off by mistake (a short command it would cancel every job on, conversation it would send to the vision agent) — and, against the current head, nothing gets worse by more than a row or two (or a couple of points) or significantly (paired McNemar): your requests, Hinglish and Devanagari, and each of those signals; nor may any of them drift from the first head's numbers (the anchors), so losses too small to reject one at a time can't pile up. Calibration (measured on rows that chose no temperature) alone never keeps NEO without a head. Every run also fits the head a run without your requests would fit — on none of your rows or labels — and "forget everything" puts that one back. With the public data (~50k rows), the multilingual head scores 0.86 on the benchmark (English 0.87, Hinglish 0.89, Devanagari 0.81) at ~12 ms a decision; the English checkpoint's 0.84 (Devanagari 0.70) at ~22 ms.

The best training data is your own usage. When NEO misroutes something, add a line to `~/.neo/reflex_extra.jsonl`:

```json
{"text": "turn off wifi", "intent": "quick_action"}
```

and retrain. Intents are `chat`, `quick_action`, `agent_task`, `stop`.

## Adding a tool

```python
# neo2/core/tools/mine.py
from neo2.core.agent.registry import ToolContext, tool

@tool(
    "wifi",
    "Turn Wi-Fi on or off.",
    {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]},
    risk="reversible", category="system",
    fast_path=[(r"^\s*turn (?P<on>on|off) wi-?fi\s*$", {"on": "<on>"})],
)
async def wifi(a: dict, c: ToolContext) -> str:
    ...
    return "Wi-Fi off."
```

Register the module in `neo2/core/tools/__init__.py` and it is available to every brain, to Gemini Live, and to the zero-LLM fast path. Return `confirm.gated(...)` from a handler to put it behind the confirmation gate.

## Use NEO from your own app

NEO's agent — its tools, its safety rules and its confirmation gate — is also a Python package, `neo2`, that other apps and AI frameworks embed. An embedded agent is not NEO.app: it never reads or writes `~/.neo` or NEO.app's settings, and each agent has its own folder (a private temp folder unless you give it one), brains, tools and memory, and a confirmation gate per conversation.

### Install

Python 3.12. The distribution is `neo2`; install it with the extras you need:

```bash
pip install "neo2[web,openai]"
pip install "neo2[web,openai] @ git+https://github.com/ArnavParashar49/Neo2.0"   # or the latest from GitHub
uv pip install -e ".[web,openai]"      # or from a checkout
```

| Extra | What it adds |
|---|---|
| *(none)* | the core: `neo2.AsyncAgent` and `Agent`, NEO's agent loop, host tools, the framework adapters, and the `weather` and `clock` tools. No model, macOS framework or browser. |
| `web` | `web_search` and `web_fetch`. Without it the `"web"` preset (the default) leaves them out, with a warning, and asking for one by name is refused. |
| `browser` | NEO's own headless Chromium (`browser_*`). Then run `playwright install chromium`. |
| `mac` | the macOS tools: Accessibility, click and type, apps, files, shell, mail and calendar |
| `gemini`, `openai`, `claude` | NEO's provider classes for Gemini, any OpenAI-compatible server (OpenAI, Groq, Ollama, LM Studio, vLLM…) and Claude. `brains` is all three. |
| `mcp` | the MCP server, `neo2-mcp`. Add `web` for its default `--tools web`: `neo2[mcp,web]`. |
| `langchain` | `neo2.adapters.langchain` |
| `embeddings` | local embeddings for memory recall |
| `app` | everything NEO.app runs. `all` is `app` plus `mcp` and `langchain`. |

NEO.app's own package comes inside it, as `neo2.core`: nothing but `neo2` is installed at the top level. (Before 2.1 it was a top-level `neo`, the folder NeuralEnsemble's `neo` uses too.)

Installed NEO from GitHub before 2.1, when its distribution was called `neo`? Run `pip uninstall neo` first, then install `neo2`. Both own the files in `neo2/`, so removing `neo` afterwards takes `neo2`'s files with it: if that happened, install `neo2` again. (NeuralEnsemble's `neo` is another project: this is only about NEO's.)

NEO logs to the `neo` logger, the name it always had: a module logs as `neo.<module>`, and the audit lines go to `neo.audit`. The library's own lines go to `neo2` (the MCP server's to `neo2.mcp`). Nothing is printed unless your app configures logging.

The snippets below use top-level `await`: run them inside an `async def`, or in `python -m asyncio`.

### An agent in your app

<!-- snippet: quickstart -->
```python
import neo2
from neo2.providers import OpenAICompatible

brain = OpenAICompatible(model="gpt-5.1", api_key="sk-...")  # or Gemini(key), Claude(key), Groq(key), Cerebras(key), your own
async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain) as neo:
    r = await neo.ask("What's the weather in Lisbon this afternoon?")
    print(r.status, r.text)  # done, and the answer

    async for e in neo.stream("And in Porto?"):  # the same, as events while it runs
        if e.type == "text_delta":
            print(e.text, end="")
```

`AgentConfig` takes the tools as presets, globs and names: `"web"`, `"browser"`, `"mac_read"`, `"mac"`, `"memory"`, `"all"`, `"none"`, `("web", "browser_*")`. It also takes `data_dir`, `memory=True` or `MemoryConfig(path=...)`, `persona`, `instructions`, `protected_apps`, `extra_secret_paths` and the step and time limits. `AgentConfig.from_env()` reads keys and settings from the environment, but only when you call it. Each `conversation=` is its own thread with its own gate. The agent also has `pending(conversation)`, `cancel()`, `history(conversation)`, `reset(conversation)` and `forget_all()`, which deletes only files NEO made in the agent's folder.

Each agent's brains keep what they learn to themselves: a Gemini key's quota window (in `<data_dir>/brain_cooldowns.json`), a model that was overloaded, a server's real request size. A provider object you build with `neo2.providers` keeps that in memory, in the object, and writes no file. Give each agent its own object if they must not share it. `OpenAICompatible(token_budget=...)` is the most one request may carry (prompt and answer): longer requests are trimmed to fit before they are sent.

An agent's browser (`tools="browser"`) is its own Chromium with its profile in the agent's folder. It runs headless unless `browser_headless=False`, and it is never NEO.app's chat card.

The sync twin needs no event loop:

<!-- snippet: sync -->
```python
with neo2.Agent(neo2.AgentConfig(tools="web"), provider=brain) as neo:
    r = neo.ask("What time is it in Tokyo?")
    print(r.text)
```

### Approvals: only your code says yes

Anything consequential is staged, not run, and `ask()` returns `status="needs_approval"`. That covers NEO's own tools that change something and your host tools marked `risk="destructive"` or given `confirm=`. `r.approval` is the card: show its `summary` and `detail` (exactly what will run) to your user as they are.

<!-- snippet: approvals -->
```python
@neo.tool(risk="destructive", confirm=lambda a: f"Refund {a['amount']} on order {a['order_id']}")
async def refund(order_id: str, amount: float, ctx: neo2.ToolContext) -> str:
    """Refund an order, in part or in full."""
    return await billing.refund(order_id, amount, tenant=ctx.extras["tenant"])

r = await neo.ask("Refund order 812 in full", tool_context={"tenant": "acme"})
if r.status == "needs_approval":
    card = r.approval  # card.summary, card.detail (the exact arguments), card.expires_at
    if await my_ui.confirm(card.summary, card.detail):  # your UI, your signed-in user
        r = await neo.approve(card.action_id, via=neo2.Approval("web-ui", "user:42"))
    else:
        r = await neo.deny(card.action_id)
print(r.status, r.text)
```

Or decide inline with an approver, which is asked before `ask()` returns:

<!-- snippet: approver -->
```python
async def ask_my_user(req: neo2.ApprovalRequest) -> neo2.Approval | None:
    if await my_ui.confirm(req.summary, req.detail):
        return neo2.Approval("desktop-dialog", "user:42")
    return None  # declines

neo = neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain, approver=ask_my_user)
```

The rules:

- **Only host code approves.** Either `approve(action_id, via=neo2.Approval(channel, approver))` from your own route, or an `approver` that returns an `Approval`. Anything else declines: `None`, `True`, `"yes"`, an exception, or no answer within `approval_timeout_s` (110 s). `approver="deny"` declines everything at once; `"defer"`, the default, waits for `approve()` or `deny()`.
- **Text never approves.** A "yes" passed to `ask()` is a new request, not an answer to the card. No tool, tool argument, exported spec, adapter or MCP tool can approve, and no action id ever appears in text a model reads.
- **An approval runs exactly what the card showed, once.** It must come within 120 s, in the conversation that staged it. An unknown, superseded, expired or other agent's id gives `status="expired"` and runs nothing. A run that stages something new after an approval comes back as a new `needs_approval`.
- **No "yes for this task".** NEO.app's user can answer a card for the rest of the task, so later calls of the same tool on the same app run without a card. An embedded agent never grants that: its cards offer no scope (`scopes: []`), `approve()` always covers one action, and the next one asks again, even in the same task. The task bookkeeping behind it, including the count of untrusted pages and mails a task has read, belongs to one conversation and never crosses to another conversation or agent.
- **An approver is a person's answer.** Calling `ask()` or `stream()` from an approver raises `ApprovalReentryError`.
- **An approval covers the approved call and nothing it starts.** A request an approved host tool starts (`ask()`, `stream()`, `call_tool()`), or a task it leaves running, is asked afresh. Going back into the approved call's own conversation while that call runs raises `ApprovalReentryError`, so send a follow-up to another conversation.
- **Say where a request came from.** Pass `source="model"` when another model wrote the task; the adapters and the MCP server always do. Such a request, like one with `context=` or one that read a web page, asks first before saving to memory, opening links that could carry data out, or submitting in a chat app.
- **Everything is audited.** Each staged, approved, denied or expired action is recorded with the agent's name and `via`. The record goes to `on_audit=` if you pass one, and to `<data_dir>/audit.log` (mode 0600), or to the `neo.audit` logger when the agent has no `data_dir`.

Host tools are plain functions. The schema comes from the typed signature (or `parameters=`, a JSON schema or a pydantic model). A parameter typed `neo2.ToolContext` gets the request's `tool_context` as `ctx.extras`. Output counts as third-party content unless you pass `untrusted=False`: the model sees it fenced, and the request then asks before memory writes and data-carrying links. Parameters named `confirm`, `cancel` or `_…` are refused, because NEO drops those from model calls. Tool names NEO uses itself are refused too: the names of its own tools; `approve`, `confirm`, `cancel`, `deny` and `more_tools`; and `caller` and `context`, which label the model's own words in `Result.read_untrusted`.

### Plug NEO into another AI framework

The recommended way is the whole agent as one tool. NEO keeps its own prompt, safety rules and gate. The model passes `{"task": ..., "context": ...}` and gets text back:

- the answer
- `NEEDS_CONFIRM: <summary> — waiting for the user's decision in the app. Don't retry or work around it.`
- `The user's OK is required and was not given: <summary>. Nothing was done; don't retry.`
- `NEEDS_USER: <question>`

Anything NEO wrote after reading third-party content comes back inside `<untrusted source="...">`. Every call is a request from a model (`source="model"`), and `agent_tool(..., on_result=...)` hands your code each full `Result`, card included.

<!-- snippet: openai -->
```python
from openai import AsyncOpenAI

import neo2.adapters.openai

client = AsyncOpenAI()
spec, run_neo = neo2.adapters.openai.agent_tool(neo)  # api="responses" for the Responses API
messages = [{"role": "user", "content": "Refund order 812 in full"}]
reply = (await client.chat.completions.create(model="gpt-5.1", messages=messages, tools=[spec])).choices[0].message
messages.append(reply.model_dump(exclude_none=True))
for call in reply.tool_calls or []:
    if call.function.name == spec["function"]["name"]:
        messages.append({"role": "tool", "tool_call_id": call.id, "content": await run_neo(call.function.arguments)})

# NEEDS_CONFIRM: the card waits for your user, not for the model
if (card := neo.pending()) and await my_ui.confirm(card.summary, card.detail):
    r = await neo.approve(card.action_id, via=neo2.Approval("web-ui", "user:42"))
    messages.append({"role": "user", "content": neo2.adapters.result_text(r)})  # how it went
```

<!-- snippet: anthropic -->
```python
from anthropic import AsyncAnthropic

import neo2.adapters.anthropic

client = AsyncAnthropic()
spec, run_neo = neo2.adapters.anthropic.agent_tool(neo)
msg = await client.messages.create(
    model="claude-opus-5", max_tokens=1024, tools=[spec], messages=[{"role": "user", "content": "Refund order 812"}]
)
results = [
    {"type": "tool_result", "tool_use_id": b.id, "content": await run_neo(b.input)}
    for b in msg.content
    if b.type == "tool_use" and b.name == spec["name"]
]
```

<!-- snippet: gemini -->
```python
from google import genai
from google.genai import types

import neo2.adapters.gemini

client = genai.Client()
decl, run_neo = neo2.adapters.gemini.agent_tool(neo)
config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=[decl])])
resp = await client.aio.models.generate_content(model="gemini-3.8-flash", contents="Refund order 812", config=config)
parts = [
    types.Part.from_function_response(name=fc.name, response={"output": await run_neo(fc.args)})
    for fc in resp.function_calls or []
    if fc.name == decl["name"]
]
```

<!-- snippet: langchain -->
```python
import neo2.adapters.langchain

neo_tool = neo2.adapters.langchain.agent_tool(neo)  # a StructuredTool: llm.bind_tools([neo_tool]), create_agent(…)
msg = await neo_tool.ainvoke({"name": "neo", "args": {"task": "Refund order 812"}, "id": "call_1", "type": "tool_call"})
print(msg.content)  # NEEDS_CONFIRM: Refund 49.5 on order 812 — waiting for the user's decision in the app. …
```

Any other framework (the OpenAI Agents SDK, LlamaIndex, CrewAI…) can wrap the same `(spec, handler)` pair, or use the MCP server. A framework's own approval step, such as `needs_approval` or a LangGraph interrupt, only approves handing the task to NEO. It never approves NEO's own staged action.

For finer control, run NEO's tools one by one in your own model loop. Put `neo2.SAFETY_RULES` into your system prompt; untrusted output comes back already fenced:

<!-- snippet: tools -->
```python
tools = neo.export_tools("openai_chat")  # or "openai_responses", "anthropic", "gemini", "mcp", "jsonschema"
async with neo.turn() as t:  # one turn of your loop: what one call read counts for the next
    out = await neo.call_tool("clock", {}, turn=t)  # a gated call returns status="needs_approval"
    messages += neo2.adapters.openai.tool_message(out, "call_1")
```

The other helpers are `neo2.adapters.openai.tool_messages` (a whole turn of Chat calls, tool messages first), `neo2.adapters.openai.function_call_output`, `neo2.adapters.anthropic.tool_result`, `neo2.adapters.gemini.function_response` and `neo2.adapters.langchain.tool_message`. A Chat tool message can't hold images, so `tool_message` sends a tool's screenshots in a user message after it, labelled as that call's output and never the user's request. The model still reads them in the user's role, so for tools that see third-party content (the screen, pages) prefer the Responses API, whose `function_call_output` keeps them in the tool's output, or pass `images="omit"`.

### Use NEO from an MCP client

Install NEO with its `mcp` extra, and `web` for the web tools: `pip install "neo2[mcp,web] @ git+https://github.com/ArnavParashar49/Neo2.0"`. Without `web`, `--tools web` gives only `weather` and `clock`.

<!-- snippet: mcp-command -->
```bash
neo2-mcp --tools web --expose task --approve deny --env      # or: python -m neo2.mcp …
```

For Claude Desktop (`claude_desktop_config.json`), Claude Code (`.mcp.json`) or Cursor (`.cursor/mcp.json`), with the full path to `neo2-mcp` if it isn't on the client's `PATH`:

<!-- snippet: mcp-config -->
```json
{
  "mcpServers": {
    "neo": {
      "command": "neo2-mcp",
      "args": ["--tools", "web", "--approve", "deny", "--env"],
      "env": { "GEMINI_API_KEY": "your-key" }
    }
  }
}
```

- **`--tools`** takes comma-separated presets, globs and names, like `AgentConfig`.
- **`--expose task`** gives one tool, `neo_task(goal, context?, conversation?)`, which hands the whole task to NEO. `tools` lists NEO's tools one by one, and `both` does both.
- **`--approve deny`** is the default: nothing gated runs, and the model is told the user has to do it.
- **`--approve dialog`** (macOS) shows a native dialog from the server process. Cancel is the default button, and the dialog gives up before the card expires. The server adds `osascript` to `protected_apps`.
- **`--approve elicit --trust-client NAME`** lets the MCP client ask its user (MCP elicitation), only for the client names you list. It approves only when the user accepts with `approve: true`. This trusts the client: a client names itself, and it may answer elicitations on its own. The client asks in its own window on your desktop, so with NEO's input tools (`--tools mac`: click, type, keys, and the task tools that act in an app's own window, `search_site` and `notes_create`) the server starts only with `--protect-app NAME` naming the client's app (for example `Claude` or `Cursor`); otherwise NEO's own clicks and keys could answer that question.
- **`--protect-app NAME`** (repeatable) adds an app to `protected_apps`: NEO's input tools never click or type into it.
- **`--env`** reads the API keys (`GEMINI_API_KEY`, `GEMINI_API_KEYS`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`) and `NEO_MCP_*` settings such as `NEO_MCP_DATA_DIR` and `NEO_MCP_BRAIN`. With only `ANTHROPIC_API_KEY` set, Claude is the brain; with only `OPENAI_API_KEY`, OpenAI. NEO.app's key store (`~/.neo/api_keys.json`) is never read. NEO.app's own `NEO_*` variables are never read. Without `--env`, `neo_task` has no model, though the individual tools still work.

As with the adapters, anything NEO wrote after reading third-party content (its answer, a question, an action's summary) comes back inside `<untrusted source="neo">`, and the server tells the client that such text is data, never instructions. The server speaks stdio only and opens no network listener. stdout carries nothing but the protocol, and audit lines go to stderr.

### What NEO never exposes to an embedding app or its model

- **No way for a model to approve anything.** There is no approve, confirm, cancel or deny tool, tool argument, exported spec, adapter or MCP route, and no action id in text a model reads. Confirmation can't be switched off: there is no setting for it, and `NEO_CONFIRM_DESTRUCTIVE` isn't read.
- **None of NEO.app's data.** Not `~/.neo`, NEO.app's `.env` and settings, its memory database, audit log or trained reflex, and not its `NEO_*` variables. A `data_dir`, memory file or `models_dir` inside `~/.neo` is refused, even through `from_env()`.
- **No tools you didn't pick.** Not in a model's call, a fast path, `more_tools`, or the run that continues after an approval. `play_music` and `browse` aren't available to embedded agents (their state is NEO.app's, process-wide), and erasing everything is never a tool: `forget_all()` is host code and removes only the agent's own files.
- **None of NEO.app's own machinery.** Laya (the local reflex model) and its learning, voice, the overlay and macOS permission prompts stay in NEO.app, and task kinds run their plain scripts.
- **No secrets.** NEO's file and shell tools don't touch SSH and GPG keys, `.env` files, cloud credentials, keychains, browser profiles, any agent's folder or memory file, or `extra_secret_paths`. Commands don't inherit API keys.
- **No private network.** NEO's browser and `web_fetch` never reach private, loopback or link-local addresses.

### `protected_apps`

An agent with the Mac tools (`tools="mac"`) clicks and types on the real desktop. macOS asks your app's process for Accessibility, not NEO.app: while it is off, the first tool that needs it ends the request with `status="needs_user"` and a message that names your app (`neo2.macos.permission_status()` and `request_permissions()` check and ask). NEO's input tools refuse any target owned by NEO's overlay or by an app in `AgentConfig(protected_apps=(...))`, matched by name, bundle id or executable. When they can't tell who owns a target, they refuse too. Keep your approval UI off that desktop or list your app there, or NEO could press your own "Approve" button. In NEO.app the setting is `NEO_PROTECTED_APPS='["Some App"]'`. The desktop is shared, so run one Mac-enabled agent per desktop.

## Project layout

```
neo2/           the package: `python -m neo2` is NEO.app's core
  core/         NEO itself (the top-level `neo` before 2.1)
    agent/      loop.py (the one agent loop) · session.py (routing) · confirm.py · registry.py · prompt.py
    providers/  catalog.py (every provider, what each brain can do) · gemini.py · claude.py · openai_responses.py · openai_compat.py (Groq, OpenRouter, endpoints, local) · cerebras.py + spend.py (daily caps) · keycheck.py (key tests, model lists) · base.py (neutral message format)
    keystore.py · brains.py · brainref.py · cli_keys.py    Brains & Keys: the key store, the roles, brain references, `neo2 keys` / `neo2 brain`
    reflex/     laya_reflex.py · features.py · finetune.py · generate.py · dataset.py · schema.py · templates_hi.py · hf_data.py + sources.py (public data)
    tools/      computer/ (gates: every check, frozen · tools_ax/pointer/keys/label/apps · ax, input, pointer, screen, apps, shell) · browser.py · browser_live.py (the chat's live card) · system.py · web.py · memory_tool.py · music/ (naming a song: capture, recognize, the song card)
    voice/      wake.py · local.py · live.py · audio.py · pipeline.py
    vision/     the camera's eyes: apple.py (Apple Vision) · yolo.py + detect.py (object boxes) · pipeline.py · answer.py
    memory/     store.py (SQLite: facts, journal, summaries, playbooks; hybrid search) · journal.py (the background writer) · extract.py (facts from what is said) · summarize.py · redact.py (secrets never stored) · timephrase.py · embed.py
    server/     ws.py (websocket to the overlay)
    selftest/   controls.py ("check your computer controls") · testwin.py (its test window) · watch.py (its safety)
  sdk/ · adapters/ · mcp/ · providers.py    the library other apps embed (AsyncAgent, Agent, host tools)
neo             a checkout's compatibility shim for the old name (not in the wheel)
ui/             Tauri 2 + React overlay (src/App.tsx, src/ws.ts, src-tauri/src/lib.rs)
tests/          2,100+ tests, no network or real model (tests/sdk: the library)
```

## Development

```bash
.venv/bin/python -m pytest -q
ruff check . && ruff format .
cd ui && npx tsc --noEmit && (cd src-tauri && cargo check)
```

The suite can't act on your Mac: in NEO's own test runs (`tests/conftest.py` turns the switch on first thing, and so does any pytest run of NEO from its source tree), `neo2/core/effects.py` refuses every real effect (opening apps or pages, AppleScript, keystrokes and clicks, Accessibility actions, sound, permission prompts, model and web requests including web searches and loopback servers the suite didn't start, loading Laya or the on-device voice…) with `EffectRefused`, so a test fakes the step it needs; a test that means the real thing opts in for itself with `with neo2.core.effects.allow(kind):`. A scratch script gets the same with `NEO_NO_EFFECTS=1` in its environment (plus `NEO_DATA_DIR` pointing somewhere temporary), and never reads `~/.neo/.env` or the checkout's `.env`; NEO.app never sets it.

NEO's package moved from `neo/` to `neo2/core/` in 2.1 (the distribution from `neo` to `neo2`). A checkout keeps a small `neo` shim, so an old import or `python -m neo` still works, with a DeprecationWarning, as the very same modules; the wheel has no `neo`. A branch from before the move: if it is older than `c475799` (main just before the move), merge that into it first, then main, so git follows the move and leaves only ordinary conflicts (merged straight in, files the branch changed a lot come back whole under `neo/`). Then run `python scripts/codemod_neo2.py`, `git add` what it moved and any "file location" conflict, run `ruff check --fix neo neo2 tests scripts` (a rewritten import can sort differently) and review the diff. The codemod rewrites imports, module paths in strings and docs, `neo/…` paths and, in NEO's package, `logging.getLogger(__name__)` (to `logger(__name__)`, so the module still logs as `neo.<module>`), and lists anything it can't decide (`--check` changes nothing). A checkout's venv from before 2.1: `uv pip uninstall neo`, then install again; `scripts/install-app.sh` does the same for NEO.app's runtime. After pulling 2.1, run it before `npm run tauri dev` too: a dev build uses that runtime when there is one (or set `NEO_HOME` to a checkout with `neo2` installed), and one from before 2.1 can't start `-m neo2`. <!-- neo2-codemod: keep (names the old paths on purpose) -->

An app with NEO installed isn't one of NEO's test runs, even under its own pytest: its programs and connections, and the actions its users approve for its agents, happen for real. To refuse them in its tests, it starts the test process with `NEO_NO_EFFECTS=1`. That covers the whole process, the app's own programs and connections too, and it takes effect when NEO is first imported. Wrap each step a test really means to make in `with neo2.core.effects.allow("network"):` (or another kind). The allowance is process-wide while the block runs and ends with it.

## Status and roadmap

Working end to end today: routing, quick actions, chat, web research, screen reading via the Accessibility tree, screenshots + vision, mail/calendar/notes, files and shell with confirmation, memory, the overlay, both voice backends (with automatic fallback when Gemini Live is out of quota or offline), and NEO.app with its supervised core. See [ROADMAP.md](ROADMAP.md) for what's verified and what's next.

NEO 2.0 is a from-scratch successor to [NEO v1](https://github.com/ArnavParashar49/NEO).

## Author

Built by **Arnav Parashar** — [GitHub](https://github.com/ArnavParashar49) · [LinkedIn](https://www.linkedin.com/in/arnavparashar49/)

## License

[MIT](LICENSE). The models and datasets NEO downloads keep their own licences: training
rows derived from public datasets are generated on your Mac and never committed (see
`neo2/core/reflex/sources.py` for each source's licence). The camera's object detector,
YOLOE-26s by Ultralytics, is AGPL-3.0: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
