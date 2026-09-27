# NEO 2.0

**An AI agent for your Mac that does what you'd do — sees the screen, clicks, types, runs commands, handles mail. Voice-first. Runs on free tiers or fully local.**

Say *"Neo"* (or press **⌥⌘**), ask for something, and NEO works your Mac the way a person would: reads the Accessibility tree, clicks and types, opens apps, runs shell commands, edits files, browses, sends mail, sets reminders — and asks before doing anything it can't undo.

There is no subscription. The brains are Gemini and Groq free tiers, or Gemma 4 running on your own GPU; the decision layer, wake word, speech-to-text and text-to-speech all run on-device.

---

## Why it feels fast

Most assistants send every word to a large model and wait. NEO routes first.

```
 "turn the volume up"        → regex fast path            0 LLM calls,   ~0 ms
 "what's the capital of Peru" → Laya (local) → Groq stream               ~150 ms + first token
 "reply to Sam's email…"      → Laya (local) → agent loop with 37 tools  ~150 ms + Gemini
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

**Every call is kept small.** A request only ships the tools it plausibly needs — picked by embedding the request against each tool's description, then expanding families (`browser_*`, `mail_*`) and adding the screen tools when the reflex says so — typically 3–7 of the 46 (9–19 % of the schema tokens) plus a `more_tools` escape hatch the model can call mid-run. Tool results older than two turns are stubbed out of the history. Simple one-tool actions that miss the regex fast path go to Gemini Flash-Lite instead of the full model. Net effect: a day of use fits the free tiers, and Groq's 8K-TPM tier can run a whole agent turn as the fallback.

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
uv venv --python 3.12 && uv pip install -r requirements.lock && uv pip install -e ".[dev]"
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

## Project layout

```
neo/
  agent/      loop.py (the one agent loop) · session.py (routing) · confirm.py · registry.py · prompt.py
  providers/  gemini.py · openai_compat.py (Groq, local) · claude.py · base.py (neutral message format)
  reflex/     laya_reflex.py · features.py · finetune.py · generate.py · dataset.py · schema.py · templates_hi.py · hf_data.py + sources.py (public data)
  tools/      computer/ (ax, input, screen, apps, shell) · browser.py · browser_live.py (the chat's live card) · system.py · web.py · memory_tool.py
  voice/      wake.py · local.py · live.py · audio.py · pipeline.py
  memory/     store.py (SQLite + FTS5 + vectors, playbooks, sessions) · embed.py · summarize.py
  server/     ws.py (websocket to the overlay)
ui/           Tauri 2 + React overlay (src/App.tsx, src/ws.ts, src-tauri/src/lib.rs)
tests/        72 tests, no network
```

## Development

```bash
.venv/bin/python -m pytest -q
ruff check . && ruff format .
cd ui && npx tsc --noEmit && (cd src-tauri && cargo check)
```

The suite can't act on your Mac: under pytest, `neo/effects.py` refuses every real effect (opening apps or pages, AppleScript, keystrokes and clicks, Accessibility actions, sound, permission prompts, model and web requests including web searches and loopback servers the suite didn't start, loading Laya or the on-device voice…) with `EffectRefused`, so a test fakes the step it needs; a test that means the real thing opts in for itself with `with neo.effects.allow(kind):`. A scratch script gets the same with `NEO_NO_EFFECTS=1` (plus `NEO_DATA_DIR` pointing somewhere temporary), and never reads `~/.neo/.env` or the checkout's `.env`; NEO.app never sets it.

## Status and roadmap

Working end to end today: routing, quick actions, chat, web research, screen reading via the Accessibility tree, screenshots + vision, mail/calendar/notes, files and shell with confirmation, memory, the overlay, both voice backends (with automatic fallback when Gemini Live is out of quota or offline), and NEO.app with its supervised core. See [ROADMAP.md](ROADMAP.md) for what's verified and what's next.

NEO 2.0 is a from-scratch successor to [NEO v1](https://github.com/ArnavParashar49/NEO).

## Author

Built by **Arnav Parashar** — [GitHub](https://github.com/ArnavParashar49) · [LinkedIn](https://www.linkedin.com/in/arnavparashar49/)

## License

[MIT](LICENSE). The models and datasets NEO downloads keep their own licences: training
rows derived from public datasets are generated on your Mac and never committed (see
`neo/reflex/sources.py` for each source's licence).
