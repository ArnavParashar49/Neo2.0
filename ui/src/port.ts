// The core's websocket, on this Mac: NEO's port 8765 — or NEO Sandbox's 18765, for a page built with
// `vite build --mode sandbox` (scripts/sandbox_build.sh). The CSP of each app allows only its own.
// (Read through `?.`: node's test runner loads these modules with no `import.meta.env`.)
const mode = (import.meta as { env?: { MODE?: string } }).env?.MODE;
export const WS_PORT = mode === "sandbox" ? 18765 : 8765;
export const CORE_WS = `ws://127.0.0.1:${WS_PORT}`;
