# Tauri + React + Typescript

This template should help get you started developing with Tauri, React and Typescript in Vite.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## NEO's in-chat browser — dev harness

`npm run dev`, then open `http://localhost:1420/?mock=browser` in a normal browser tab. A scripted
in-page core (`src/browser/mock/`) speaks the real websocket protocol — binary NBF1 frames with
credits and acks, `browser_*` state/actions/steps, confirm cards — and drives a fictional store
drawn to canvas, so every state of the browser card can be seen without the Python core or any
network. The panel top-left runs the full task or jumps to a state; `?mock=browser&state=login`
(also `starting`, `loading`, `acting`, `thinking`, `waiting`, `user`, `captcha`, `2fa`, `other`,
`dialog`, `select`, `error`, `offline`, `ended`) opens straight into one. Take control and the
page responds to your clicks, scrolling and typing. None of this ships: it is gated on
`import.meta.env.DEV` and stripped from `npm run build`.

## NEO's song card — dev harness

`npm run dev`, then open `http://localhost:1420/?mock=song`. A scripted in-page core
(`src/song/mock/`) speaks the card's side of the protocol — `song_open`, `song_state` with a
synthesised level ten times a second, `song_close`, `song_hello` after a reconnect — and loops
found → no match → found → error. The panel picks a run (`?mock=song&state=found`, also
`not_found`, `error`, `stopped`, `listening`, `matching`), changes speed and theme, drops the
connection ("Reconnect": the card asks `song_hello` and carries on) or restarts the core (the card
gets no answer and ends). No microphone and no network: covers are drawn on a canvas, links point
at example.com and are never opened. Gated on `import.meta.env.DEV` like the browser mock.

The card's logic is one pure reducer (`src/song/state.ts`); `npm test` runs its tests with Node's
own test runner (`tests/*.test.ts`, no packages needed). `tests/fixtures/song-wire.json` is what the
real core sends (recorded by NEO's end-to-end test, `tests/test_song_e2e.py`, which also checks the
core still sends it): `tests/song-wire.test.ts` replays it through the reducer. If the core's messages
change on purpose, rewrite it with `NEO_SONG_WIRE_WRITE=1` on that Python test.
