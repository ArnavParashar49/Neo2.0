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
