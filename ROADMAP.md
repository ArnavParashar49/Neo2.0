# NEO 2.0 — Roadmap

NEO 2.0 is a rewrite of NEO as a **free / local-first AI agent for your Mac**: it listens,
thinks, and then does things on the computer the way you would — with a real agent loop,
a fast local reflex layer, and an overlay UI built on thinking-orbs.

## Target hardware

MacBook Pro, Apple M3 Pro, 18 GB unified memory. macOS-first; thin cross-platform shims only.

## Architecture

```
mic ─► wake word (local) ─► voice backend
                              ├─ Gemini Live (native speech-to-speech, if free-tier quota allows)
                              └─ local cascade: Parakeet STT → brain → Kokoro TTS   (mlx-audio)
                                        │
                                        ▼
                     Reflex (Laya, local, ~33 ms)  intent → {chat | fast_tool | agent}
                                                   risk   → {read_only | reversible | destructive}
                                        │
                                        ▼
                     Brain (pluggable providers, picked per turn)
                       • Gemini 3 Flash   — agentic + vision (free tier: 10 RPM / 250K TPM / 1500 RPD)
                       • Groq gpt-oss-120b — fast text (free tier: 30 RPM / 8K TPM)
                       • Gemma 4 12B Q4 (MLX) — offline / private
                       • Claude Opus 5 — optional paid slot, off by default
                                        │
                                        ▼
                     Agent loop (one loop, no planner): tool use, step budget, thrash guard,
                     confirm-gate on destructive actions, append-only audit log
                                        │
                                        ▼
                     Tools: macOS Accessibility tree, input injection, screenshots + vision
                            fallback, AppleScript app tools (Mail/Calendar/Notes/Finder/Safari),
                            bash, file editor, web search/fetch, system controls, memory
                                        │
                                        ▼
                     Overlay UI: Tauri 2 + Vue 3, thinking-orbs states driven by agent events
```

### Orb ↔ state mapping

| NEO state   | Orb animation (`ui/src/App.tsx`) |
|-------------|----------------------------------|
| idle        | breathing   |
| listening   | listening   |
| thinking    | solving     |
| working     | solving     |
| searching   | searching   |
| speaking    | composing   |
| connecting  | connecting  |
| confirming  | shaping     |

## Phases

- [x] **0. Toolchain** — uv + Python 3.12, Rust, Tauri CLI, branch `v2`
- [x] **1. Core** — `neo/` package: providers (Gemini / Groq / local MLX / Claude), agent loop, registry, confirm gate, Laya reflex + synthetic dataset + local head training
- [x] **2. Computer control** — AX tree observation, CGEvent input, screenshot fallback, AppleScript app tools, shell/files
- [x] **3. Voice** — wake word (Vosk) → Parakeet → brain → Kokoro with barge-in; Gemini Live backend with `agent_task` delegation
- [x] **4. UI** — Tauri 2 + React overlay with thinking-orbs, transcript, confirm buttons, tray, ⌘⇧Space
- [x] **5. Memory** — SQLite FTS5 store (profile / lesson / note), memory tool, prompt context
- [x] **6. Purge** — v1 code removed, README rewritten, tests green

### Verified so far (2026-09-25)

- Overlay ⇄ core over websocket: transcript replay, orb state, quick actions with 0 LLM calls, timer, confirm flow.
- Native Tauri window: transparent, always-on-top, bottom-right; tray + global hotkey compile and run.
- Local voice models: load 19 s, Kokoro TTS 0.57 s / 2 s speech, Parakeet STT 0.33 s / 2 s utterance (M3 Pro).
- Laya reflex on MPS: ~150 ms per decision; destructive / needs-screen well calibrated zero-shot.

### Verified with real keys (2026-09-26)

- Chat via Groq, web research via tools, **screenshot → Gemini vision**, memory store + recall — all end to end.
- Gemini free tier overloads (503) at times: the provider now walks 3.8 → 3.7 → 3.5 Flash with backoff, and
  the chain falls over to Groq for text. Gemini 3 needs `thought_signature` echoed on every replayed function
  call; calls from other brains use Google's documented bypass value.
- Reflex head: embedding-only variant, **95.8% held-out at ~165 ms**; sub-questions added nothing.
- Adversarial review (5 finders × 3 verifiers, 116 agents) confirmed 37 defects — all fixed, each with a
  regression test in `tests/test_v2_review_fixes.py` (61 tests total).

### Still needs a human

- `GEMINI_API_KEY` (and optionally `GROQ_API_KEY`) in `.env` to exercise the agent path end to end.
- macOS **Accessibility** permission for whatever launches NEO (Terminal / NEO.app) — screen recording is already granted.
- Microphone permission on first voice run.
- First real-world tuning pass: add misroutes to `~/.neo/reflex_extra.jsonl`, retrain (`python -m neo.reflex.finetune`).

### Reliability + memory (2026-09-26)

- Verification turn: after side-effect tools the loop makes the model re-observe before reporting.
- Playbooks: successful multi-step tool paths saved and recalled by goal similarity into the prompt.
- Browser tools on Playwright/Chromium with a persistent profile (ARIA snapshot, click/type/press/scroll/back/screenshot).
- silero VAD replaces the energy gate; live partial transcripts stream to the overlay while you speak.
- Memory: hybrid FTS5 + embedding recall (Laya's encoder), session summaries at shutdown.
- Reflex head retrained on ~3k items (templates + LLM-generated, cross-labelled): 96.0% held-out at ~165 ms.

### Hardening after first real voice use (2026-09-26)

- `session.receive()` yields one turn: re-enter it for the session's lifetime (the "answers once" bug).
- Half-duplex mic while NEO speaks; speaker hysteresis; single voice state machine; 350 ms UI dwell.
- Per-tool timeouts (registry), 180 s wall-clock budget per agent run, Live connect timeout + auto-reconnect.
- `mail_unread` scans the newest messages instead of a `whose` filter (10k-unread inboxes drop the connection).
- Core log now prints routing, tool calls and per-turn brain/latency.
- Voice session closes after 5 s without user speech (local silero VAD); barge-in needs a committed "stop" so NEO's own voice can't cut it off.

### Token diet (2026-09-26)

- Per-request tool selection (embedding match + families + screen rule + playbook/history tools), capped at 18, with a `more_tools` meta-tool.
- History context editing: tool results older than two turns are stubbed.
- Flash-Lite lane for one-tool actions; Groq now viable as the agent fallback (verified live while Gemini was out of quota).

### Mid-sentence acting + concurrency (2026-09-26)

- EarlyActor: fast-path clauses run as soon as they stabilise in the live transcript (Live interim transcription / local STT partials); 30 s dedupe so the model's later call is a no-op.
- Live: slow tools and agent_task are NON_BLOCKING with WHEN_IDLE result scheduling; the session keeps listening while work runs.
- Session runs requests as concurrent jobs (agent jobs capped at 2); history appended in complete blocks; per-job events; "stop" cancels all.
- EventBus aggregates per-job states by priority so parallel work never flickers the orb.

### Dictation lane + Laya v2 (2026-09-26)

- Sequential test "open notes → type Laptops in the heading → type some points about laptops" over the Live session: 18 s end to end (was minutes / stalled). Fixes on the way: Live session kept alive by typed text and model turns, empty text areas visible to `ax_tree`, `ax_find` keeps ids, app resolution ignores helper processes, persisted Gemini cooldowns + no retry on slow 503s + sticky brain demotion, superseded screen dumps stubbed within a run, agent_task dedupe in Live, `activate_app` waits for focus (AppleScript fallback).
- Dictation lane: `type_text` / `hotkey` / `dictate` fast paths (`neo/tools/computer/__init__.py`), early actor waits for the sentence end on dictation, fast paths win even when the reflex says agent_task, agent prompt lists what already ran.
- Laya v2 (`neo/reflex/finetune.py`): every decision from one encoder pass (~20 ms vs ~126 ms) — intent, destructive, needs_screen (distilled from Laya's zero-shot answers) and a tool head; zero-shot consulted only under 0.6 confidence. Training data +12.3k real utterances from CLINC150 and MASSIVE mapped onto NEO's labels (`neo/reflex/hf_data.py`). The session acts on Laya's tool choice: no-arg tools run with no model at all; others get a single-tool light call.

### Silent commands + control layer (2026-09-26)

- Commands finish silently (Reply.silent, tool `quiet` flag, Live `SILENT` scheduling); questions, failures and confirmations are spoken. Measured: "open notes → type → dictate → what time is it" in 13 s over Live; Safari "open → new tab → type URL → enter → list apps → close tab" with ~1 s per control step.
- Zero-LLM control layer: click by label (`click_text`), tab/window/zoom/scroll/screenshot keys, and command fast paths for mail/calendar/apps/search/memory/notes/lock. `chain=False` keeps context-bound commands ("… and search for X") out of the early actor; `remaining()` hands the whole utterance to the agent with a note of what already ran.
- Listening window counts Gemini's own transcription as speech; session-close reasons are logged.
- Laya v2: MLP heads with temperature calibration, reply head, HF data (`hf_data.py`), per-text feature cache; v2 decisions never escalate to the cloud lite model.

### Review fixes: control layer + early actor (2026-09-26)

Two adversarial reviews (≈235 agents) confirmed 44 defects in the silent-command / control / dictation batch; all fixed with regression tests (199 green):

- Early actor rewritten on a planner (`neo/agent/fastpath.py`): a plan covers the whole utterance or nothing; payload tools (type/dictate/click/note/memory/search) take the rest of the sentence ("type salt and pepper" is never split), a trailing command is split off ("… and press enter"); mid-sentence only opt-in tools run (open_app, volume, brightness); one-shot claims replace the 30 s dedupe (a repeated command runs again; the model's own call for an early action doesn't); Live never runs tools inside the receive loop and never re-runs the whole utterance at turn end.
- Dictation: pronouns / messages / things go to the model ("write that down", "write an email to Sam", "write the report"); only UI places are stripped as targets; compose cues need count nouns; line breaks never send (single-line fields join, chat apps use shift+return, shells never get a multi-line enter); never types into password fields.
- Control: click only on whole-word labels in the focused window, visible, unambiguous; vague labels ("click it") go to the model; browser-only chords (reload, top/bottom) only in browsers; "undo" only right after NEO typed; "new note" goes to Notes; a failed quick action hands the request to the agent.
- Live: keystrokes run in the order asked; a question in the same breath as a command is answered; server-cancelled calls get no response; no stale turn across sessions.
- Session: confirmation crash fixed; only a bare "stop" cancels everything; Laya's no-arg lane only for read-only tools without qualifiers; tool head trained with a real "none" class.

### App focus (2026-09-26)

- Root causes of "typed into Claude": NEO didn't check which app was in front; macOS cooperative activation blocks a background process's `activate`/`open -a` while the user is busy elsewhere; and NSWorkspace's `frontmostApplication` / `runningApplications` never refresh without a Cocoa run loop (stale front app, blind to apps launched after start).
- Fixes: WindowServer focus switch (`neo/tools/computer/focus.py`, the SkyLight calls AltTab/yabai use) — 0.15–0.2 s from the background, verified while the user was in Claude; front app and running apps read fresh from LaunchServices (`lsappinfo`); a Cocoa run-loop pump in the core; every keystroke/typing/click goes only to NEO's target app, confirmed in front, else NEEDS_USER and nothing is sent.
- Live: a plain command the model wraps in agent_task runs through the fast paths on the user's own words.

### Faster answers + Laya learns from use (2026-09-26)

- Weather tool on Open-Meteo (free, no key; "here" = remembered city → `NEO_HOME_LOCATION` → IP location): first spoken word ~2.5 s, was ~40 s through search + agent.
- Web search races DuckDuckGo, Brave and Bing (first with results wins, 4 s cap): ~1 s per search, was 2–6 s. Live: at most 2 searches + 1 page read per question, identical calls answered once, news=true for "latest" questions, simple questions never go to the agent. "Who won the last F1 race": first word at 3.9 s, was 18.7 s.
- Laya learns from use (`neo/reflex/learn.py`): what actually happened labels each request — Live's tool choice (none → chat, one tool → quick action, agent_task), or the agent needing no tool / one fast tool. Rows in `~/.neo/reflex_extra.jsonl` (weighted 3×, always trained on); after ≥30 new rows and 3 idle minutes the heads retrain in-process on the loaded Laya and hot-swap — only if they score at least as well on a fixed hash-based held-out set.
- Tests are isolated from the real `~/.neo` (a leaked row and earlier test artefacts in memory were removed; backup kept).

### Next

- `npm run tauri build` → signed NEO.app that spawns the Python core itself.
- Skills: a small curated set loaded on demand, replacing the v1 271-folder dump.

## Removed from v1 (and why)

| Removed | Replaced by |
|---|---|
| planner, goal dispatcher, agent swarm, sub_agents, hybrid/agents | one agent loop with provider-native parallel tool calls |
| 271-folder `skills/` | ~10 curated skills loaded on demand |
| ~7k lines of PySide6 UI | Tauri + Vue overlay (~10 MB runtime) |
| litellm, model_pool, chromadb | `neo/providers`, SQLite + local embeddings |
| YOLO / DeepFace local vision | Gemini vision on request only |
| flight_finder, youtube, etc. | browser + computer use |
| Windows/Linux deep paths, ARIA leftovers | macOS-first shims |
