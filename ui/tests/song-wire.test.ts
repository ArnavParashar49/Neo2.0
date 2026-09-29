// The core's real song messages (tests/fixtures/song-wire.json, recorded by NEO's end-to-end test
// tests/test_song_e2e.py: the user's words, the real song_identify, a fake microphone and a fake
// Shazam) replayed through the card's reducer, the way ws.ts feeds it: both sides test one wire.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  EMPTY, announce, fmtSecs, hearing, linkFor, safeCover, viewOf, reduce, type SongAction, type SongSnap,
} from "../src/song/state.ts";

interface Msg { type: string; data: Record<string, unknown>; ts: number }
const WIRE = JSON.parse(readFileSync(new URL("./fixtures/song-wire.json", import.meta.url), "utf8")) as Record<string, Msg[]>;

const on = (m: Msg): SongAction => ({ t: "event", type: m.type, data: m.data });
function play(msgs: Msg[], s: SongSnap = { ...EMPTY, connected: true }): SongSnap {
  return msgs.reduce((acc, m) => reduce(acc, on(m), 1_000 + m.ts * 1000), s);
}

test("a song found: the card listens with a moving meter, shows the song at once, then its cover", () => {
  const msgs = WIRE.found;
  const firstFound = msgs.findIndex((m) => m.data.status === "found");
  assert.ok(firstFound > 2, "listening states come first");
  const listening = play(msgs.slice(0, firstFound));
  const x = listening.live!;
  assert.equal(viewOf(x), "listening");
  assert.ok(hearing(x) && x.level > 0.3 && x.level <= 1, `the meter: ${x.level}`);
  assert.equal(x.source, "mic");
  assert.equal(announce(x), "Listening to the room");

  const found = play(msgs.slice(0, firstFound + 1));
  assert.equal(viewOf(found.live!), "found");
  assert.equal(found.live!.result!.cover, null, "the song shows before its cover");
  assert.equal(linkFor(found, found.live!.sid, "apple_music")?.startsWith("https://music.apple.com/"), true);

  const done = play(msgs);
  assert.equal(done.live, null);
  const e = done.ended[msgs[0].data.sid as string];
  assert.equal(e.closed, "done");
  assert.equal(viewOf(e), "found");
  assert.equal(announce(e), "Found: Blinding Lights by The Weeknd");
  assert.ok(e.result!.cover && safeCover(e.result!.cover) === e.result!.cover, "the core's cover passes the card's check");
  assert.equal(e.result!.year, "2020");
  assert.equal(fmtSecs(e.elapsedS, true), `${e.elapsedS.toFixed(1)} s`);
  assert.ok(linkFor(done, e.sid, "shazam")?.startsWith("https://www.shazam.com/"));
});

test("a reload mid-listen: the replay's opening, then song_hello's answer keeps the card live", () => {
  const [open, answer] = WIRE.rejoin;
  assert.equal(open.type, "song_open");
  assert.equal(answer.type, "song_state");
  let s = play([open]);
  s = reduce(s, { t: "hello_end" }, 2_000);
  assert.equal(s.live?.unconfirmed, true);
  s = reduce(s, on(answer), 2_100);
  assert.equal(s.live?.unconfirmed, false);
  s = reduce(s, { t: "gone", sid: open.data.sid as string }, 4_100); // the 2 s timer, too late to matter
  assert.equal(viewOf(s.live!), "listening");
  assert.equal(s.live!.source, "mic");
});

test("a late overlay: the replay alone shows the song with its cover and links", () => {
  let s = play(WIRE.replay, { ...EMPTY, connected: true });
  s = reduce(s, { t: "hello_end" }, 9_000);
  const sid = WIRE.replay[0].data.sid as string;
  assert.equal(s.live, null);
  assert.equal(viewOf(s.ended[sid]), "found");
  assert.ok(s.ended[sid].result!.cover);
  assert.ok(linkFor(s, sid, "apple_music"));
});

test("stopped: the card's Stop, then the core's close", () => {
  const msgs = WIRE.stopped;
  const close = msgs.length - 1;
  let s = play(msgs.slice(0, close));
  s = reduce(s, { t: "stop" }, 5_000);
  assert.equal(s.live?.stopping, true);
  s = play(msgs.slice(close), s);
  const e = s.ended[msgs[0].data.sid as string];
  assert.equal(viewOf(e), "stopped");
  assert.equal(e.stopping, false);
});

test("an erase while listening: the card goes with the chat", () => {
  const s = play(WIRE.erased);
  assert.equal(s.live, null);
  assert.deepEqual(s.ended, {});
  assert.equal(s.connected, true);
});
