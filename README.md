# NEO 2.0

**An AI agent for your Mac that does what you'd do — sees the screen, clicks, types, runs commands, handles mail. Voice-first. Runs on free tiers or fully local.**

Say *"Neo"* (or press **⌥⌘**), ask for something, and NEO works your Mac the way a person would: reads the Accessibility tree, clicks and types, opens apps, runs shell commands, edits files, browses, sends mail, sets reminders — and asks before doing anything it can't undo.

There is no subscription. The brains are Gemini and Groq free tiers, or Gemma 4 running on your own GPU; the decision layer, wake word, speech-to-text and text-to-speech all run on-device.

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
| **Memory** | `memory` — profile facts, lessons, notes; hybrid keyword + embedding recall into every prompt |

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

**Brains are pluggable and fail over.** Gemini walks 3.8 → 3.7 → 3.5 Flash across every API key you give it (free-tier quota is per key *and* per model), backs off on overload, and hands text-only turns to Groq's `gpt-oss-120b` at ~700 tokens/s. `python -m neo --local` serves Gemma 4 12B on-device for offline use. An Anthropic key turns on Claude as an optional brain.

**It acts while you're still talking.** The voice layer feeds the live transcript — words as you say them — through Laya and the fast-path matcher every few hundred milliseconds. A quick command that has clearly finished ("open YouTube and…") runs *before the sentence ends*; what already ran is remembered so the model's own later call for it is a no-op. Slow tools and agent tasks are non-blocking in the Live session: NEO acknowledges, keeps listening, and reports when the result is in and you're not mid-sentence. Several requests can run at once — ask for the time while it's still reading your mail — and the orb shows the most important thing going on.

**Commands are done quietly; questions get answers.** "open notes", "type Laptops", "set a timer" finish with no speech and no chat bubble — just the tool chip and the orb going back to listening (the spoken acknowledgement was ~2 s per step, with the mic muted). Anything whose result *is* the answer (time, mail, calendar, search, "what's on my screen") is spoken, as is every failure, confirmation and clarifying question. Over Gemini Live, action results go back with `SILENT` scheduling, so the model doesn't narrate them either.

**Single-step control runs in about a second, with no model at all.** "click play", "close this tab", "next tab", "go back", "zoom in", "scroll down", "take a screenshot", "switch to Safari", "quit Spotify", "check my email", "what's on my calendar", "search for …", "remember that …", "lock the screen" are regex fast paths straight to the tool; Laya's tool head covers the phrasings the regexes miss. Multi-step tasks still need a model.

**Typing never waits for a model.** "type Laptops in the heading", "new line", "select all", "press enter" are regex fast paths — the words *are* the payload, so they go straight to the keyboard (`type_text`, `hotkey`), with the caret put into the document's editor first (a fresh note if the app shows none). "type some points about laptops" is a *brief*: one small chat call writes the lines, then they're typed (`dictate`). None of it touches the agent loop, so it's the same speed on a bad Gemini day.

**It checks its own work.** After any side-effect tool, the loop makes the model re-observe (Accessibility tree, file, page, inbox) and confirm the goal before it reports — one extra call, far fewer "done!" replies that weren't.

**It remembers how it did things.** A successful multi-step task is saved as a *playbook* (goal + the tools it used, never the text it typed or sent). A similar request later gets that path in its prompt, so repeats are shorter and cheaper. The conversation is summarised every 20 idle minutes and at shutdown, so "what did we do yesterday" works and a crash loses little; recall is hybrid — FTS5 keyword search fused with local embeddings from Laya's encoder, already resident on the GPU.

**Safety is structural, not a prompt.** Anything consequential *stages* the action and returns `NEEDS_CONFIRM`; the run stops, the overlay shows exactly what will run (the full command, script, file path and contents, or mail body), and nothing happens until *you* approve it:

- **What runs without asking is an allowlist.** Shell commands only when every part is read-only (`ls`, `cat`, `git status`…, no redirects or substitutions); AppleScript only when it just reads (`get`, `count`); new files only in Desktop, Documents and Downloads. Everything else — deleting, overwriting, editing, installing, sending mail, Return in a terminal, buttons like *Send*, *Buy* or *Delete* (or an *OK* in a dialog that says it deletes), submitting a web form that isn't a search box — asks first.
- **Only you can approve.** Your typed or spoken "yes", or the card's button (⌘⏎) for that exact action. A model can't approve anything, a stale card can't approve a later action, and the approval covers exactly what the card showed — if anything changed, NEO asks again. By voice, "yes" only counts once NEO has read the question out.
- **Web pages, emails, files and the screen are data, never instructions.** They reach the model fenced as untrusted; they can't fake a confirmation, and after reading them NEO asks before saving memories or opening links that could carry your data out. Background fetches never reach your Mac or home network.
- **Secrets are off limits.** NEO doesn't read or write SSH/GPG keys, `.env` files, cloud credentials, keychains, browser profiles or its own data, and commands it runs don't inherit API keys.

Every staged, confirmed or cancelled action is appended to `~/.neo/audit.log` (the action, not the words of a message; about 1 MB, 180 days).

## Install

Requirements: macOS 13+ on Apple Silicon (built and tested on an M3 Pro, 18 GB), Python 3.12, and — for the overlay — Node 20+ and Rust.

```bash
brew install uv rustup && rustup default stable
git clone https://github.com/ArnavParashar49/Neo2.0 && cd Neo2.0
uv venv --python 3.12 && uv pip install -r requirements.lock && uv pip install -e ".[app,dev]"
cp .env.example .env        # paste your keys (both free, no card)
.venv/bin/python -m neo --doctor
scripts/install-app.sh      # build NEO.app, install it to /Applications and start it
```

- `GEMINI_API_KEY` — [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Add more in `GEMINI_API_KEYS=key2,key3` for more free quota.
- `GROQ_API_KEY` — [console.groq.com/keys](https://console.groq.com/keys) (optional; instant chat replies).

The browser tools need Chromium once: `.venv/bin/python -m playwright install chromium`.

**NEO.app** (recommended): run `scripts/install-app.sh` from the project folder. It installs NEO's Python runtime (exact versions from `requirements.lock`) in `~/Library/Application Support/NEO`, copies your `.env` to `~/.neo/.env`, builds and signs the app, puts it in `/Applications` (replacing any older copy) and starts it. Re-run it after pulling changes; `scripts/uninstall-app.sh` removes the app (`--purge` also moves `~/.neo` to the Trash).

It lives in the menu bar (the dotted orb), with no Dock icon: it runs the core, restarts it if it crashes or hangs, stops it on quit, and opens at login (menu → *Open at Login*). **⌥⌘** — press Option and Command together and let go — shows NEO; ⌘⇧Space works too. The first launch asks for **Accessibility** and **Screen Recording** (and the microphone when you first talk to it); the menu shows what's still missing, with shortcuts to the right settings pane. The app is signed by its bundle id, so what you grant once stays granted across updates. The menu also has *Settings & API Keys…* (opens `~/.neo/.env`), *Open Logs*, *Restart NEO* and *Erase NEO's Memory…*.

Without the app: grant **Accessibility** and **Screen Recording** to the terminal you run `python -m neo` from, in *System Settings → Privacy & Security*. macOS asks for the **Microphone** on the first voice run. Models (Laya, Parakeet, Kokoro, Vosk) download on first run; the wake-word model is checked against a pinned checksum.

## Run

```bash
.venv/bin/python -m neo                 # voice + core; the overlay connects to it
.venv/bin/python -m neo --text -v       # terminal REPL that shows routing and every tool call
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
again. For developers, `python -m neo.selftest controls` runs the same test from your own Terminal window, with
that terminal's permissions (Accessibility; Screen Recording for the screenshot checks); it refuses to run as a
command NEO itself runs (its shell tool) or without an interactive terminal.

## Configuration

Everything is an environment variable with the `NEO_` prefix (or a line in `.env`). The useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `NEO_BRAIN` | `gemini` | primary agentic brain: `gemini` · `groq` · `local` · `claude` |
| `NEO_FAST_BRAIN` | `groq` | brain for chat-only turns |
| `NEO_VOICE` | `auto` | `auto` (Live if the key allows it, else local) · `live` · `local` · `off` |
| `NEO_REFLEX` | `laya` | `laya` · `lite` (Gemini Flash-Lite) · `off` |
| `NEO_LISTEN_TIMEOUT_S` | `5` | stop listening after this many seconds without speech from you (silero VAD; noise doesn't count) |
| `NEO_LOCAL_MODEL` | `mlx-community/gemma-4-12B-it-4bit` | offline model served by `--local` |
| `NEO_CONFIRM_DESTRUCTIVE` | `true` | set `false` only if you enjoy surprises |
| `NEO_CLOUD_MEMORY` | `full` | what NEO's memory sends to cloud models: `full` · `profile` (stable facts only) · `off` |
| `NEO_LEARN` | `on` | `off` stops Laya recording your requests to learn from (a reflex with no head for its encoder still gets one, trained on the public data alone) |
| `NEO_LEARN_TEACHER` | `off` | `local` · `cloud`: while NEO is idle, a model re-checks up to 50 a day of the requests Laya was least sure about — `local` only with the local model, `cloud` only with the provider that already handled that request |
| `NEO_HOME_LOCATION` | — | where "the weather" means if NEO doesn't know your city (else it guesses from your IP) |
| `NEO_DATA_DIR` | `~/.neo` | memory DB, audit log, models, trained reflex head |

With NEO.app, settings live in `~/.neo/.env` (menu → *Settings & API Keys…*, then *Restart NEO*).

## Privacy

NEO keeps its data on this Mac, in `~/.neo` (readable only by you). What leaves it, and where:

| Goes to | What | When |
|---|---|---|
| **Google** (Gemini API, Gemini Live) | what you say and type, the recent conversation, NEO's memory about you (see `NEO_CLOUD_MEMORY`), tool results, and screenshots when a task needs to see the screen | every request that needs a model; voice sessions stream your microphone audio while listening |
| **Groq** (optional key) | chat turns, and the conversation text when NEO writes a session summary | quick answers and summaries |
| **DuckDuckGo, Brave, Bing** | your search queries (raced for speed) | web searches |
| **Open-Meteo**, and **ipinfo.io** if NEO doesn't know your city | the place, or your IP address to guess it | weather |
| any website NEO opens or fetches for you | normal web requests | browsing tasks |

The free tiers of Gemini and Groq allow those companies to use what you send to improve their products; use paid keys, `NEO_CLOUD_MEMORY=profile|off`, or the local brain (`--local`) if that matters for you. The wake word, speech-to-text and text-to-speech of the on-device voice, and Laya's decisions all run locally.

What NEO keeps, and for how long:

- **Memory** (facts and preferences you tell it) until you delete it; **playbooks** (the path of past tasks, without message text) — the newest 300; **session summaries** — 90 days.
- **Requests Laya learns from** (`reflex_extra.jsonl`: no dictated text, notes, memories, mail, or anything that looks like a code or password) — the latest 2,000 within 180 days; `NEO_LEARN=off` stops it.
- **Logs** (`~/.neo/logs`) record which tools ran and whether they worked — never what you typed, said, sent or read. The **audit log** records confirmed actions (not the words of messages).
- The overlay is hidden from screen sharing, recordings and screenshots.

**Erase everything:** menu → *Erase NEO's Memory…* (or `python -m neo --forget-all`) deletes memories, playbooks, session summaries, learned requests and their cached features, logs, the audit log and the logins in NEO's own browser. Your settings and keys stay. No model or web page can trigger it.

## Training the reflex

The heads are trained locally in a few minutes (plus feature extraction the first time):

```bash
.venv/bin/python -m neo.reflex.hf_data    # optional: ~45k public phrasings (English, Hindi, Hinglish) + a benchmark
.venv/bin/python -m neo.reflex.generate   # optional: grow the dataset with Gemini + Groq, cross-labelled
.venv/bin/python -m neo.reflex.generate --focus chat:quick_action   # or --from-confusion FILE, --lang hinglish: ≤300 rows at one weak boundary
.venv/bin/python -m neo.reflex.finetune   # embed (cached per text), fit the heads, gate, save
.venv/bin/python -m neo.reflex.evaluate   # the head on the benchmark: by language and source, recalls, tool/kind, ECE, latency
```

`hf_data` maps open intent, dialogue, function-calling and computer-use sets (CLINC150, MASSIVE en/hi, Hinglish-TOP, TOPv2, MTOP en/hi, Schema-Guided Dialogue, glaive, AgentNet, OmniACT, …) onto NEO's labels — route, tool, task kind, destructive, needs the screen, wants a reply — and writes `~/.neo/reflex_hf.jsonl`. Every source is pinned to a revision and only the columns or byte ranges it needs are downloaded; `neo/reflex/sources.py` lists each one's licence. Some are CC-BY-SA (Hinglish-TOP too: it translates TOPv2), so the rows are built on your Mac and never committed. The Hinglish-TOP, MTOP, CLINC and MASSIVE test splits go to `reflex_eval.jsonl`, a benchmark whose sentences — and their translations and re-punctuated copies, from any source — are never trained on, whichever sources a build uses. NEO's own templates share a few short sentences with it ("next song", "ruk jao"): evaluate on `dataset.benchmark(rows)`, which drops what the head trained on. Leyzer is opt-in (`--with leyzer`): the repository hosting its data is licensed non-commercial, no-derivatives. So is `sentence_types` (no licence stated). Romanised-Hindi templates for every class, including destructive requests and stop phrases ("bas karo", "ruk jao"), are part of NEO's own data (`neo/reflex/templates_hi.py`).

`finetune` embeds every sentence once with Laya's multilingual checkpoint (`NEO_REFLEX_ENCODER=english` keeps the older English one; a head only ever runs on the encoder it was trained for) and fits six small MLP heads on it — intent, destructive, needs the screen, wants a reply, tool and task kind — each a 3-seed ensemble with its loss and temperature chosen on held-out rows. Rows are split by hashes, so every run agrees: a fixed base slice of the public and NEO rows (the anchors), 20% of your verified requests (the user slice), and a calibration slice (10% of your requests plus 1 in 20 public groups). A new head replaces the current one only if it clears absolute floors and ceilings — every time, not just the first: how many destructive, stop and screen requests it catches (judged only on rows whose label a rule didn't write), and how often it goes off by mistake (a short command it would cancel every job on, conversation it would send to the vision agent) — and, against the current head, nothing gets worse by more than a row or two (or a couple of points) or significantly (paired McNemar): your requests, Hinglish and Devanagari, and each of those signals; nor may any of them drift from the first head's numbers (the anchors), so losses too small to reject one at a time can't pile up. Calibration (measured on rows that chose no temperature) alone never keeps NEO without a head. Every run also fits the head a run without your requests would fit — on none of your rows or labels — and "forget everything" puts that one back. With the public data (~50k rows), the multilingual head scores 0.86 on the benchmark (English 0.87, Hinglish 0.89, Devanagari 0.81) at ~12 ms a decision; the English checkpoint's 0.84 (Devanagari 0.70) at ~22 ms.

The best training data is your own usage. When NEO misroutes something, add a line to `~/.neo/reflex_extra.jsonl`:

```json
{"text": "turn off wifi", "intent": "quick_action"}
```

and retrain. Intents are `chat`, `quick_action`, `agent_task`, `stop`.

## Adding a tool

```python
# neo/tools/mine.py
from neo.agent.registry import ToolContext, tool

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

Register the module in `neo/tools/__init__.py` and it is available to every brain, to Gemini Live, and to the zero-LLM fast path. Return `confirm.gated(...)` from a handler to put it behind the confirmation gate.

## Use NEO from your own app

NEO's agent — its tools, its safety rules and its confirmation gate — is also a Python package, `neo2`, that other apps and AI frameworks embed. An embedded agent is not NEO.app: it never reads or writes `~/.neo` or NEO.app's settings, and each agent has its own folder (a private temp folder unless you give it one), brains, tools and memory, and a confirmation gate per conversation.

### Install

Python 3.12. The distribution is still named `neo` (the move to `neo2` on PyPI hasn't happened yet), so install it from GitHub or a checkout, with the extras you need:

```bash
pip install "neo[web,openai] @ git+https://github.com/ArnavParashar49/Neo2.0"
uv pip install -e ".[web,openai]"      # or from a checkout
```

| Extra | What it adds |
|---|---|
| *(none)* | the core: `neo2.AsyncAgent` and `Agent`, NEO's agent loop, host tools, the framework adapters, and the `weather` and `clock` tools. No model, macOS framework or browser. |
| `web` | `web_search` and `web_fetch`. Without it the `"web"` preset (the default) leaves them out, with a warning, and asking for one by name is refused. |
| `browser` | NEO's own headless Chromium (`browser_*`). Then run `playwright install chromium`. |
| `mac` | the macOS tools: Accessibility, click and type, apps, files, shell, mail and calendar |
| `gemini`, `openai`, `claude` | NEO's provider classes for Gemini, any OpenAI-compatible server (OpenAI, Groq, Ollama, LM Studio, vLLM…) and Claude. `brains` is all three. |
| `mcp` | the MCP server, `neo2-mcp`. Add `web` for its default `--tools web`: `neo[mcp,web]`. |
| `langchain` | `neo2.adapters.langchain` |
| `embeddings` | local embeddings for memory recall |
| `app` | everything NEO.app runs. `all` is `app` plus `mcp` and `langchain`. |

It also installs `neo`, the package NEO.app runs, as a top-level package. Until the rename, don't install it next to NeuralEnsemble's `neo`, which uses the same folder name.

The snippets below use top-level `await`: run them inside an `async def`, or in `python -m asyncio`.

### An agent in your app

<!-- snippet: quickstart -->
```python
import neo2
from neo2.providers import OpenAICompatible

brain = OpenAICompatible(model="gpt-5.1", api_key="sk-...")  # or Gemini(key), Claude(key), Groq(key), your own
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

Install NEO with its `mcp` extra, and `web` for the web tools: `pip install "neo[mcp,web] @ git+https://github.com/ArnavParashar49/Neo2.0"`. Without `web`, `--tools web` gives only `weather` and `clock`.

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
- **`--env`** reads the API keys (`GEMINI_API_KEY`, `GEMINI_API_KEYS`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`) and `NEO_MCP_*` settings such as `NEO_MCP_DATA_DIR` and `NEO_MCP_BRAIN`. With only `ANTHROPIC_API_KEY` set, Claude is the brain. NEO.app's own `NEO_*` variables are never read. Without `--env`, `neo_task` has no model, though the individual tools still work.

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
neo/
  agent/      loop.py (the one agent loop) · session.py (routing) · confirm.py · registry.py · prompt.py
  providers/  gemini.py · openai_compat.py (Groq, local) · claude.py · base.py (neutral message format)
  reflex/     laya_reflex.py · features.py · finetune.py · generate.py · dataset.py · schema.py · templates_hi.py · hf_data.py + sources.py (public data)
  tools/      computer/ (gates: every check, frozen · tools_ax/pointer/keys/label/apps · ax, input, pointer, screen, apps, shell) · browser.py · browser_live.py (the chat's live card) · system.py · web.py · memory_tool.py
  voice/      wake.py · local.py · live.py · audio.py · pipeline.py
  memory/     store.py (SQLite + FTS5 + vectors, playbooks, sessions) · embed.py · summarize.py
  server/     ws.py (websocket to the overlay)
  selftest/   controls.py ("check your computer controls") · testwin.py (its test window) · watch.py (its safety)
neo2/         the library other apps embed: sdk/ (AsyncAgent, Agent, host tools) · adapters/ · mcp/ · providers.py
ui/           Tauri 2 + React overlay (src/App.tsx, src/ws.ts, src-tauri/src/lib.rs)
tests/        2,100+ tests, no network or real model (tests/sdk: the library)
```

## Development

```bash
.venv/bin/python -m pytest -q
ruff check . && ruff format .
cd ui && npx tsc --noEmit && (cd src-tauri && cargo check)
```

The suite can't act on your Mac: in NEO's own test runs (`tests/conftest.py` turns the switch on first thing, and so does any pytest run of NEO from its source tree), `neo/effects.py` refuses every real effect (opening apps or pages, AppleScript, keystrokes and clicks, Accessibility actions, sound, permission prompts, model and web requests including web searches and loopback servers the suite didn't start, loading Laya or the on-device voice…) with `EffectRefused`, so a test fakes the step it needs; a test that means the real thing opts in for itself with `with neo.effects.allow(kind):`. A scratch script gets the same with `NEO_NO_EFFECTS=1` in its environment (plus `NEO_DATA_DIR` pointing somewhere temporary), and never reads `~/.neo/.env` or the checkout's `.env`; NEO.app never sets it.

An app with NEO installed isn't one of NEO's test runs, even under its own pytest: its programs and connections, and the actions its users approve for its agents, happen for real. To refuse them in its tests, it starts the test process with `NEO_NO_EFFECTS=1`. That covers the whole process, the app's own programs and connections too, and it takes effect when NEO is first imported. Wrap each step a test really means to make in `with neo.effects.allow("network"):` (or another kind). The allowance is process-wide while the block runs and ends with it.

## Status and roadmap

Working end to end today: routing, quick actions, chat, web research, screen reading via the Accessibility tree, screenshots + vision, mail/calendar/notes, files and shell with confirmation, memory, the overlay, both voice backends (with automatic fallback when Gemini Live is out of quota or offline), and NEO.app with its supervised core. See [ROADMAP.md](ROADMAP.md) for what's verified and what's next.

NEO 2.0 is a from-scratch successor to [NEO v1](https://github.com/ArnavParashar49/NEO).

## Author

Built by **Arnav Parashar** — [GitHub](https://github.com/ArnavParashar49) · [LinkedIn](https://www.linkedin.com/in/arnavparashar49/)

## License

[MIT](LICENSE). The models and datasets NEO downloads keep their own licences: training
rows derived from public datasets are generated on your Mac and never committed (see
`neo/reflex/sources.py` for each source's licence).
