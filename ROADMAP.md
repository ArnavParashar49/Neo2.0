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
