// Search Screen's doors (src/lens/door.ts, src/lens/socket.ts), with a fake Tauri and a fake
// socket: the shell is asked exactly its five commands and its phase event; outside NEO.app (the
// notch, a plain tab) there is no door at all; the line to the core carries the ticket, reads the
// still and the messages, tries again at most three times in five seconds, and keeps nothing.
import { test } from "node:test";
import assert from "node:assert/strict";
import { COMMANDS, PHASE_EVENT, makeDoor, tauriDoor, type PhaseEvent } from "../src/lens/door.ts";
import { LensSocket, type SocketLike } from "../src/lens/socket.ts";

const T = "cd".repeat(32);

test("the door invokes exactly the shell's five commands, and hears only well-formed phases", async () => {
  const calls: [string, unknown][] = [];
  let heard: ((p: unknown) => void) | null = null;
  const door = makeDoor({
    invoke: async (cmd, args) => { calls.push([cmd, args]); if (cmd === "lens_hello") return { g: 3, phase: "open", ticket: T }; return null; },
    listen: async (event, cb) => { assert.equal(event, PHASE_EVENT); heard = cb; return () => { heard = null; }; },
  });
  assert.deepEqual(await door.hello(), { g: 3, phase: "open", ticket: T });
  await door.painted();
  await door.close("esc");
  await door.peek(true);
  await door.allowSc();
  assert.deepEqual(calls.map((c) => c[0]), [...COMMANDS]);
  assert.deepEqual(calls[2][1], { why: "esc" });
  assert.deepEqual(calls[3][1], { hidden: true });
  const got: PhaseEvent[] = [];
  const off = await door.onPhase((e) => got.push(e));
  heard!({ g: 1, phase: "warm" });
  heard!({ phase: "warm" });
  heard!("open");
  heard!(null);
  assert.deepEqual(got, [{ g: 1, phase: "warm" }]);
  off();
  assert.equal(heard, null);
});

test("a refused command is quiet; a refused hello means nothing to show", async () => {
  const door = makeDoor({ invoke: async () => { throw "forbidden"; }, listen: async () => () => {} });
  assert.equal(await door.hello(), null);
  await door.close("esc"); // no throw
  await door.peek(false);
});

test("outside NEO.app there is no door (the notch hosts the chat, never this page)", async () => {
  assert.equal(await tauriDoor(), null); // no window, no Tauri: node has neither
});

class FakeSocket implements SocketLike {
  readyState = 0;
  binaryType = "";
  sent: string[] = [];
  onopen: SocketLike["onopen"] = null;
  onmessage: SocketLike["onmessage"] = null;
  onclose: SocketLike["onclose"] = null;
  onerror: SocketLike["onerror"] = null;
  url: string;
  constructor(url: string) { this.url = url; }
  open() { this.readyState = 1; this.onopen?.(); }
  send(d: string) { this.sent.push(d); }
  close() { if (this.readyState === 3) return; this.readyState = 3; this.onclose?.(); }
}

function harness() {
  const sockets: FakeSocket[] = [];
  const timers: { fn: () => void; live: boolean }[] = [];
  let clock = 0;
  const events: string[] = [];
  const msgs: unknown[] = [];
  const stills: unknown[] = [];
  const s = new LensSocket(
    {
      token: async () => "tok/en",
      socket: (url) => { const f = new FakeSocket(url); sockets.push(f); return f; },
      later: (fn) => { const t = { fn, live: true }; timers.push(t); return t; },
      cancel: (h) => { (h as { live: boolean }).live = false; },
      now: () => clock,
    },
    { msg: (m) => msgs.push(m), still: (st) => stills.push(st), status: (x) => events.push(x) },
  );
  const fire = () => { const t = timers.filter((x) => x.live); timers.length = 0; t.forEach((x) => x.fn()); };
  return { s, sockets, events, msgs, stills, fire, tick: (ms: number) => { clock += ms; } };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

test("the line carries the token and the ticket, reads messages and the still", async () => {
  const h = harness();
  h.s.connect(T);
  await flush();
  assert.equal(h.sockets[0].url, `ws://127.0.0.1:8765/?token=tok%2Fen&lens=${T}`);
  assert.equal(h.sockets[0].binaryType, "arraybuffer");
  assert.equal(h.s.send({ cmd: "lens_hello" }), false); // not open yet: nothing is kept for later
  h.sockets[0].open();
  assert.deepEqual(h.events, ["connecting", "open"]);
  assert.equal(h.s.send({ cmd: "lens_hello", v: 1 }), true);
  assert.deepEqual(h.sockets[0].sent, ['{"cmd":"lens_hello","v":1}']);
  h.sockets[0].onmessage?.({ data: JSON.stringify({ type: "lens_frozen", data: { ok: true, ms: 5 } }) });
  h.sockets[0].onmessage?.({ data: JSON.stringify({ type: "transcript", data: { text: "not ours" } }) });
  h.sockets[0].onmessage?.({ data: new ArrayBuffer(8) });
  assert.equal(h.msgs.length, 1);
  assert.equal(h.stills.length, 0);
  assert.equal(h.s.send({ cmd: "x", pad: "y".repeat(10_000) }), false);
});

test("a dropped line is tried again three times in five seconds, then it says it failed", async () => {
  const h = harness();
  h.s.connect(T);
  await flush();
  for (let i = 0; i < 2; i++) {
    h.sockets[i].close();
    h.tick(500);
    h.fire();
    await flush();
  }
  assert.equal(h.sockets.length, 3);
  h.sockets[2].close();
  h.fire();
  await flush();
  assert.equal(h.sockets.length, 3);
  assert.equal(h.events.at(-1), "failed");
});

test("closing stops it for good; a bad ticket never dials", async () => {
  const h = harness();
  h.s.connect(T);
  await flush();
  h.s.close();
  h.fire();
  await flush();
  assert.equal(h.sockets.length, 1);
  const k = harness();
  k.s.connect("not-a-ticket");
  await flush();
  assert.equal(k.sockets.length, 0);
  assert.deepEqual(k.events, ["failed"]);
});
