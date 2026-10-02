// In NEO's own window the page relays the menu for a host: the core's `shell` event becomes the
// shell's `shell_action`, the shell's status is read with `shell_status`, and whether a host is
// there goes to `host_attached`. Run with a stand-in for Tauri's door (`__TAURI_INTERNALS__`).
import { test } from "node:test";
import assert from "node:assert/strict";

const invoked: { cmd: string; args: unknown }[] = [];
let fail = false;
const status = { accessibility: true, screen_recording: false, open_at_login: true, version: "2.1.0", hotkey: "⌥⌘" };
const win = {
  location: { search: "" },
  __TAURI_INTERNALS__: {
    invoke: async (cmd: string, args: unknown) => {
      invoked.push({ cmd, args });
      if (fail) throw new Error("no such command"); // an older NEO.app
      return cmd === "shell_status" ? status : null;
    },
  },
} as Record<string, unknown>;
(globalThis as Record<string, unknown>).window = win;

const { inNotch } = await import("../src/host.ts");
const shell = await import("../src/shell.ts");

test("NEO's own window: the menu's actions go to the shell, with the switch's side when it has one", async () => {
  assert.equal(inNotch, false);
  invoked.length = 0;
  await shell.shellAction("restart");
  await shell.shellAction("open_at_login", false);
  await shell.shellAction("quit", "yes" as unknown as boolean); // not a side: none is passed on
  assert.deepEqual(invoked, [
    { cmd: "shell_action", args: { action: "restart", on: null } },
    { cmd: "shell_action", args: { action: "open_at_login", on: false } },
    { cmd: "shell_action", args: { action: "quit", on: null } },
  ]);
});

test("the shell's status is read from the shell, and the host's presence told to it", async () => {
  invoked.length = 0;
  assert.deepEqual(await shell.shellStatus(), status);
  await shell.hostAttached(true);
  await shell.hostAttached(null);
  assert.deepEqual(invoked, [
    { cmd: "shell_status", args: {} },
    { cmd: "host_attached", args: { attached: true } },
    { cmd: "host_attached", args: { attached: null } },
  ]);
});

test("an older shell without these commands is not an error", async () => {
  fail = true;
  await shell.shellAction("quit");
  await shell.hostAttached(false);
  assert.equal(await shell.shellStatus(), null);
  fail = false;
});
