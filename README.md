# NEO 2.0

**An AI agent for your Mac that does what you'd do — sees the screen, clicks, types, runs commands, handles mail. Voice-first. Runs on free tiers or fully local.**

Say *"Hey Neo"* (or press **⌘⇧Space**), ask for something, and NEO works your Mac the way a person would: reads the Accessibility tree, clicks and types, opens apps, runs shell commands, edits files, browses, sends mail, sets reminders — and asks before doing anything it can't undo.

There is no subscription. The brains are Gemini and Groq free tiers, or Gemma 4 running on your own GPU; the decision layer, wake word, speech-to-text and text-to-speech all run on-device.

---

## Why it feels fast

Most assistants send every word to a large model and wait. NEO routes first.

```
 "turn the volume up"        → regex fast path            0 LLM calls,   ~0 ms
 "what's the capital of Peru" → Laya (local) → Groq stream               ~150 ms + first token
 "reply to Sam's email…"      → Laya (local) → agent loop with 37 tools  ~150 ms + Gemini
```

The router is **Laya**, a 421M-parameter local decision model, with a small head trained on ~3k NEO utterances. It answers three typed questions per utterance in ~150 ms on an M3 Pro: *what kind of request is this*, *is it destructive*, *does it need the screen*. Only then does a language model get involved — and only the one the task needs.

## What it can do

| Area | Tools |
|---|---|
| **See** | `ax_tree`, `ax_find` (Accessibility tree as text — fast, exact, private), `screenshot` (vision fallback for canvas/Electron apps) |
| **Act** | `click`, `drag`, `scroll`, `type_text`, `hotkey`, `ax_press`, `ax_set_value` |
| **Apps** | `open_app`, `activate_app`, `quit_app`, `apps_running`, `applescript`, `finder_reveal` |
| **Mail · Calendar · Notes** | `mail_unread`, `mail_send`, `calendar_today`, `calendar_add`, `notes_create`, `reminder_add` |
| **Web** | `web_search`, `web_fetch`, `safari_open`, `safari_read` |
| **Browser** | `browser_open`, `browser_read` (ARIA snapshot + text), `browser_click`, `browser_type`, `browser_press`, `browser_scroll`, `browser_back`, `browser_screenshot` — NEO's own Chromium with a persistent profile, so logins stick |
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
                       Reflex — Laya + trained head (local, ~150 ms)
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

**It checks its own work.** After any side-effect tool, the loop makes the model re-observe (Accessibility tree, file, page, inbox) and confirm the goal before it reports — one extra call, far fewer "done!" replies that weren't.

**It remembers how it did things.** A successful multi-step task is saved as a *playbook* (goal + tool sequence). A similar request later gets that path in its prompt, so repeats are shorter and cheaper. Sessions are summarised at shutdown into memory, so "what did we do yesterday" works; recall is hybrid — FTS5 keyword search fused with local embeddings from Laya's encoder, already resident on the GPU.

**Safety is structural, not a prompt.** Destructive tools (send mail, delete, overwrite, risky shell or AppleScript) *stage* the action and return `NEEDS_CONFIRM`; the loop stops, the orb turns to *shaping*, and nothing runs until you say yes. A confirmation is only accepted if it matches the exact action that was asked about, and any retraction ("okay, cancel that") wins over a leading "okay". Everything staged, confirmed or cancelled is appended to `~/.neo/audit.log`.

## Install

Requirements: macOS 13+ on Apple Silicon (built and tested on an M3 Pro, 18 GB), Python 3.12, and — for the overlay — Node 20+ and Rust.

```bash
brew install uv rustup && rustup default stable
git clone https://github.com/ArnavParashar49/Neo2.0 && cd Neo2.0
uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env        # paste your keys (both free, no card)
.venv/bin/python -m neo --doctor
```

- `GEMINI_API_KEY` — [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Add more in `GEMINI_API_KEYS=key2,key3` for more free quota.
- `GROQ_API_KEY` — [console.groq.com/keys](https://console.groq.com/keys) (optional; instant chat replies).

The browser tools need Chromium once: `.venv/bin/python -m playwright install chromium`.

Grant **Accessibility** and **Screen Recording** to the app you launch NEO from (Terminal, or NEO.app) in *System Settings → Privacy & Security*. macOS asks for the **Microphone** on the first voice run. Models (Laya, Parakeet, Kokoro, Vosk) download on first run into `~/.neo`.

## Run

```bash
.venv/bin/python -m neo                 # voice + core; the overlay connects to it
.venv/bin/python -m neo --text -v       # terminal REPL that shows routing and every tool call
cd ui && npm install && npm run tauri dev   # the overlay (npm run tauri build → NEO.app)
```

The overlay lives in the bottom-right corner as a small pill (orb + status + last reply) and expands into a card on
**⌘⇧Space**, a click, or whenever NEO needs a confirmation; it folds back to the pill after a quiet stretch unless you
pin it. The card shows the conversation as a timeline — your messages, the tools NEO ran (as chips), rendered
markdown replies, and under each reply which brain answered and how long it took (*Groq gpt-oss-120b · 1.4 s*,
*quick action · no model*). Destructive actions surface a confirm card you can answer with ⏎ / Esc. While a turn is in
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
| `NEO_DATA_DIR` | `~/.neo` | memory DB, audit log, models, trained reflex head |

## Training the reflex

The head is trained locally in about a minute (plus feature extraction):

```bash
.venv/bin/python -m neo.reflex.generate   # optional: grow the dataset with Gemini + Groq, cross-labelled
.venv/bin/python -m neo.reflex.finetune   # extract features, compare variants, save the fastest ≥94%
```

`finetune` fits several variants (embedding only, embedding + *n* sub-questions) and picks the fastest one above the accuracy bar. On the current ~3k-item set the embedding-only head scores ~95% held-out at ~150 ms.

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
  reflex/     laya_reflex.py · features.py · finetune.py · generate.py · dataset.py · schema.py
  tools/      computer/ (ax, input, screen, apps, shell) · browser.py · system.py · web.py · memory_tool.py
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

## Status and roadmap

Working end to end today: routing, quick actions, chat, web research, screen reading via the Accessibility tree, screenshots + vision, mail/calendar/notes, files and shell with confirmation, memory, the overlay, both voice backends. See [ROADMAP.md](ROADMAP.md) for what's verified and what's next (a signed NEO.app that launches the core itself, tool search for the offline model, cross-platform shims).

NEO 2.0 is a from-scratch successor to [NEO v1](https://github.com/ArnavParashar49/NEO).

## License

CC BY-NC 4.0.
