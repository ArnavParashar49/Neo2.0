// Hosted in the notch (NotchHub's WKWebView), the UI must never reach for Tauri: the shell shim
// answers every call from what the host injected, and talks back through the host's message
// handler. Run with a bare `window` standing in for the page, before the modules load.
import { test } from "node:test";
import assert from "node:assert/strict";

const posted: unknown[] = [];
let tauriTouched = 0;
const win = {
  __NEO_HOST__: "notch",
  __NEO_WS_TOKEN__: "t".repeat(43),
  __NEO_HOST_AREA__: { w: 760, h: 640 },
  webkit: { messageHandlers: { neo: { postMessage: (m: unknown) => { posted.push(m); } } } },
  location: { search: "" },
} as Record<string, unknown>;
// Any look at Tauri's door counts: the window-mode shim checks for it, the notch one must not.
Object.defineProperty(win, "__TAURI_INTERNALS__", { get() { tauriTouched++; return undefined; }, enumerable: false });
(globalThis as Record<string, unknown>).window = win;

const { HOST, inNotch } = await import("../src/host.ts");
const shell = await import("../src/shell.ts");

test("the host is the notch, and the shell never looks for Tauri", () => {
  assert.equal(HOST, "notch");
  assert.equal(inNotch, true);
  assert.equal(shell.hasShell, true);
  assert.equal(tauriTouched, 0);
});

test("the token is the host's, and nothing is invoked for it", async () => {
  assert.equal(await shell.wsToken(), "t".repeat(43));
  assert.equal(posted.length, 0);
});

test("show, hide, focus, links, sizes, pins and the clipboard all go to the host", async () => {
  posted.length = 0;
  await shell.showQuiet();
  await shell.hideWindow();
  await shell.focusWindow();
  await shell.openExternal("https://example.com/");
  shell.requestSize(420, 520);
  shell.pinHost(true);
  assert.equal(await shell.clipboardWrite("hi"), true);
  assert.deepEqual(posted, [
    { type: "show" }, { type: "collapse" }, { type: "focus" },
    { type: "open_url", url: "https://example.com/" },
    { type: "size", w: 420, h: 520 }, { type: "pin", on: true },
    { type: "clipboard", text: "hi" },
  ]);
});

test("the host's events reach listeners through __neoHostEmit, until they unlisten", async () => {
  const emit = (win as { __neoHostEmit?: (e: string) => void }).__neoHostEmit;
  assert.ok(emit, "the shim installed the host's entry point");
  let woke = 0;
  const un = await shell.listen("neo://wake", () => { woke++; });
  emit!("neo://wake");
  emit!("neo://erase"); // someone else's
  assert.equal(woke, 1);
  un();
  emit!("neo://wake");
  assert.equal(woke, 1);
});

test("a ui_host change is NEO.app's business, not the host's", async () => {
  posted.length = 0;
  await shell.uiHostChanged("window");
  assert.equal(posted.length, 0);
  assert.equal(tauriTouched, 0);
});

test("without a host handler, posting reports it rather than throwing", async () => {
  const saved = (win as { webkit?: unknown }).webkit;
  (win as { webkit?: unknown }).webkit = undefined;
  assert.equal(shell.post({ type: "show" }), false);
  assert.equal(await shell.clipboardWrite("x"), false);
  (win as { webkit?: unknown }).webkit = saved;
});

test("the page in the notch never speaks for NEO.app's shell", async () => {
  // The menu's actions and what it shows are NEO.app's own page's to relay (it has the shell);
  // hosted in the notch the page has none, and asks neither Tauri nor the host.
  posted.length = 0;
  await shell.shellAction("quit");
  await shell.shellAction("open_at_login", true);
  assert.equal(await shell.shellStatus(), null);
  await shell.hostAttached(true);
  assert.equal(posted.length, 0);
  assert.equal(tauriTouched, 0);
});
