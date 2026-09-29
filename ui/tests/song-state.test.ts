// The song card's reducer and checks (ui/src/song/state.ts), with Node's own test runner and type
// stripping: `npm test` in ui/. No React, no socket, no microphone.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  EMPTY, EMPTY_SHAPE, MAX_ENDED, announce, clockAt, eqBars, extraLine, fmtSecs, hearing, linkFor, normResult, reduce, safeCover, safeLink,
  sessionOf, shapeOf, shownLevel, viewOf,
  type SongAction, type SongSnap,
} from "../src/song/state.ts";

const JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDA==";
const RESULT = {
  title: "Northern Lights", artist: "The Midnight Owls", album: "Glass City", year: 2021, genre: "Synthpop", cover: JPEG,
  links: { apple_music: "https://example.com/am/northern-lights", shazam: "https://example.com/sz/northern-lights" },
};

const ev = (type: string, data: unknown): SongAction => ({ t: "event", type, data });
function run(actions: SongAction[], start: SongSnap = { ...EMPTY, connected: true }, t0 = 1_000): SongSnap {
  return actions.reduce((s, a, i) => reduce(s, a, t0 + i * 100), start);
}
const open = (sid = "a", source = "both", listen_s = 12) => ev("song_open", { sid, source, listen_s });
const state = (sid: string, status: string, extra: Record<string, unknown> = {}) =>
  ev("song_state", { sid, status, level: 0.5, elapsed_s: 4, message: null, result: null, ...extra });

test("a song found: listening, matching, found, then the card stays with its result", () => {
  let s = run([open(), state("a", "listening", { level: 0.8 })]);
  assert.equal(s.live?.sid, "a");
  assert.equal(s.live?.source, "both");
  assert.equal(s.live?.level, 0.8);
  assert.equal(viewOf(s.live!), "listening");
  assert.equal(s.latest, "a");
  s = run([state("a", "matching")], s);
  assert.equal(viewOf(s.live!), "matching");
  s = run([state("a", "found", { level: 0.9, elapsed_s: 7.6, result: RESULT })], s);
  assert.equal(viewOf(s.live!), "found");
  assert.equal(s.live?.level, 0, "no level once it has stopped listening");
  assert.equal(s.live?.result?.year, "2021", "a numeric year becomes text");
  s = run([ev("song_close", { sid: "a", reason: "done" })], s);
  assert.equal(s.live, null);
  const e = s.ended.a;
  assert.equal(e.closed, "done");
  assert.equal(viewOf(e), "found");
  assert.equal(e.result?.title, "Northern Lights");
  assert.equal(e.elapsedS, 7.6);
  assert.equal(announce(e), "Found: Northern Lights by The Midnight Owls");
  assert.equal(extraLine(e.result!), "Glass City · 2021 · Synthpop");
});

test("no match, an error and a stop each leave the right face", () => {
  const nf = run([open(), state("a", "not_found", { message: "I couldn't recognise it." }), ev("song_close", { sid: "a", reason: "done" })]);
  assert.equal(viewOf(nf.ended.a), "not_found");
  assert.equal(announce(nf.ended.a), "I couldn't recognise it.");
  const err = run([open(), ev("song_close", { sid: "a", reason: "error" })]);
  assert.equal(viewOf(err.ended.a), "error");
  const stopped = run([open(), state("a", "listening"), { t: "stop" }, ev("song_close", { sid: "a", reason: "stopped" })]);
  assert.equal(viewOf(stopped.ended.a), "stopped");
  assert.equal(stopped.ended.a.stopping, false);
  // Closed "done" while still listening (no final state came): over without a match.
  const quiet = run([open(), ev("song_close", { sid: "a", reason: "done" })]);
  assert.equal(viewOf(quiet.ended.a), "not_found");
  // "found" without a readable result is not shown as a song.
  const bad = run([open(), state("a", "found", { result: { title: "  " } })]);
  assert.equal(viewOf(bad.live!), "not_found");
});

test("Stop marks the card stopping only while it is listening; a disconnect clears it", () => {
  let s = run([open(), { t: "stop" }]);
  assert.equal(s.live?.stopping, true);
  s = reduce(s, { t: "connected", on: false }, 2_000);
  assert.equal(s.live?.stopping, false);
  const found = run([open(), state("a", "found", { result: RESULT }), { t: "stop" }]);
  assert.equal(found.live?.stopping, false);
  assert.equal(hearing(found.live!), false);
});

test("covers are data URLs of raster images only — never a remote URL, never SVG", () => {
  assert.equal(safeCover(JPEG), JPEG);
  assert.equal(safeCover("data:image/png;base64,iVBORw0KGgo="), "data:image/png;base64,iVBORw0KGgo=");
  for (const bad of [
    "https://is1-ssl.example.com/cover.jpg",
    "http://example.com/a.png",
    "//example.com/a.png",
    "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
    "data:text/html;base64,PGgxPmhpPC9oMT4=",
    'data:image/jpeg;base64,abc") url(https://example.com/x',
    "data:image/jpeg;base64,",
    "data:image/jpeg,rawbytes",
    "data:image/jpeg;base64," + "A".repeat(1_600_000),
    null, 42, {},
  ]) assert.equal(safeCover(bad), null, String(bad).slice(0, 60));
  const r = normResult({ ...RESULT, cover: "https://example.com/cover.jpg" });
  assert.equal(r?.cover, null);
});

test("links are https only, without credentials", () => {
  assert.equal(safeLink("https://example.com/x?y=1"), "https://example.com/x?y=1");
  for (const bad of ["http://example.com", "javascript:alert(1)", "file:///etc/passwd", "shazam://track/1", "https://user:pw@example.com/", "not a url", "", null]) {
    assert.equal(safeLink(bad), null, String(bad));
  }
  const r = normResult({ ...RESULT, links: { apple_music: "javascript:alert(1)", shazam: "https://example.com/sz" } });
  assert.deepEqual(r?.links, { apple_music: null, shazam: "https://example.com/sz" });
});

test("text is trimmed and bounded, and a result needs a title", () => {
  assert.equal(normResult(null), null);
  assert.equal(normResult({ artist: "x" }), null);
  const r = normResult({ title: "  Long\n  title ", artist: null, album: "", year: null, genre: "g".repeat(500), cover: null, links: null });
  assert.equal(r?.title, "Long title");
  assert.equal(r?.artist, "");
  assert.equal(r?.album, null);
  assert.equal(r?.genre?.length, 60);
  assert.deepEqual(r?.links, { apple_music: null, shazam: null });
});

test("level and seconds are clamped; junk statuses are ignored", () => {
  const s = run([open(), state("a", "listening", { level: 7, elapsed_s: -3 }), state("a", "weird", { level: "NaN", elapsed_s: null })]);
  assert.equal(s.live?.status, "listening");
  assert.equal(s.live?.level, 1);
  assert.equal(s.live?.elapsedS, 0);
  const odd = run([ev("song_open", { sid: "b", source: "radio", listen_s: "lots" })]);
  assert.equal(odd.live?.source, null);
  assert.equal(odd.live?.listenS, 12);
});

test("links open only for the newest card", () => {
  let s = run([open("a"), state("a", "found", { result: RESULT }), ev("song_close", { sid: "a", reason: "done" })]);
  assert.equal(linkFor(s, "a", "shazam"), RESULT.links.shazam);
  s = run([open("b")], s);
  assert.equal(linkFor(s, "a", "shazam"), null, "an older card's links are no longer the core's to open");
  assert.equal(linkFor(s, "b", "shazam"), null, "no result yet");
});

test("one session at a time: a new open ends the one still showing", () => {
  const s = run([open("a"), open("b")]);
  assert.equal(s.live?.sid, "b");
  assert.equal(s.ended.a.closed, "gone");
  assert.equal(viewOf(s.ended.a), "stopped");
});

test("a reconnect: the replay can't reopen an ended card, and song_hello's answer keeps a live one", () => {
  const done = run([open("a"), state("a", "found", { result: RESULT }), ev("song_close", { sid: "a", reason: "done" })]);
  const replayed = run([open("a"), ev("song_close", { sid: "a", reason: "done" })], done);
  assert.equal(replayed.live, null);
  assert.equal(replayed.ended.a.result?.title, "Northern Lights", "the replay doesn't wipe a known result");

  let s = run([open("b"), state("b", "listening"), { t: "hello_end" }]);
  assert.equal(s.live?.unconfirmed, true);
  s = run([state("b", "listening", { elapsed_s: 6 })], s);
  assert.equal(s.live?.unconfirmed, false);
  assert.equal(reduce(s, { t: "gone", sid: "b" }, 9_000), s, "a confirmed card isn't gone");
});

test("a reconnect: no answer means it ended meanwhile; a late state brings it back", () => {
  let s = run([open("b"), { t: "hello_end" }, { t: "gone", sid: "b" }]);
  assert.equal(s.live, null);
  assert.equal(s.ended.b.closed, "gone");
  s = run([state("b", "matching")], s);
  assert.equal(s.live?.sid, "b");
  assert.equal(s.live?.closed, null);
  assert.equal(s.ended.b, undefined);
  s = run([ev("song_close", { sid: "b", reason: "done" })], s);
  assert.equal(s.ended.b.closed, "done");
});

test("joining late: a state for a card never seen open makes it live", () => {
  const s = run([state("z", "listening", { elapsed_s: 5 })]);
  assert.equal(s.live?.sid, "z");
  assert.equal(s.live?.source, null);
  assert.equal(s.latest, "z");
  const busy = run([open("a"), state("z", "listening")]);
  assert.equal(busy.live?.sid, "a", "a confirmed live card isn't replaced by a stray state");
});

test("erase and clear forget ended cards; a live one keeps its place", () => {
  const base = run([open("a"), ev("song_close", { sid: "a", reason: "done" }), open("b")]);
  const cleared = reduce(base, { t: "clear" }, 5_000);
  assert.deepEqual(cleared.ended, {});
  assert.equal(cleared.live?.sid, "b");
  assert.equal(cleared.latest, "b");
  const erased = reduce(base, ev("song_close", { sid: "b", reason: "erased" }), 5_000);
  assert.equal(erased.live, null);
  assert.deepEqual(erased.ended, {});
  assert.equal(erased.connected, true);
});

test("ended cards are bounded", () => {
  let s: SongSnap = { ...EMPTY, connected: true };
  for (let i = 0; i < MAX_ENDED + 5; i++) s = run([open(`s${i}`), ev("song_close", { sid: `s${i}`, reason: "done" })], s);
  assert.equal(Object.keys(s.ended).length, MAX_ENDED);
  assert.equal(s.order.length, MAX_ENDED);
  assert.equal(s.ended.s0, undefined);
  assert.ok(s.ended[`s${MAX_ENDED + 4}`]);
});

test("unchanged input returns the same snapshot (no re-render)", () => {
  const s = run([open("a")]);
  assert.equal(reduce(s, open("a"), 9_000), s);
  assert.equal(reduce(s, ev("song_state", { status: "listening" }), 9_000), s, "no sid");
  assert.equal(reduce(s, ev("song_close", { sid: "other", reason: "done" }), 9_000), s);
  assert.equal(reduce(s, { t: "connected", on: true }, 9_000), s);
});

test("the clock runs on between updates and stops at the window", () => {
  const s = run([open("a", "mic", 12), state("a", "listening", { elapsed_s: 4 })], undefined, 0);
  const x = s.live!;
  assert.equal(clockAt(x, x.at), 4);
  assert.equal(clockAt(x, x.at + 1500), 5.5);
  assert.equal(clockAt(x, x.at + 60_000), 12);
  const done = run([ev("song_close", { sid: "a", reason: "stopped" })], s).ended.a;
  assert.equal(clockAt(done, done.at + 60_000), 4);
});

test("small helpers", () => {
  assert.equal(fmtSecs(4.24), "4 s");
  assert.equal(fmtSecs(4.24, true), "4.2 s");
  assert.equal(fmtSecs(12.4, true), "12 s");
  assert.equal(fmtSecs(NaN), "0 s");
  for (const lv of [0, 0.3, 1, 5, NaN]) {
    const bars = eqBars(lv, 1234);
    assert.equal(bars.length, 5);
    assert.ok(bars.every((h) => h >= 0.12 && h <= 1), `${lv}: ${bars}`);
  }
  assert.deepEqual(eqBars(0.6, 42), eqBars(0.6, 42), "deterministic for a seed");
  assert.ok(eqBars(0.9, 7).reduce((a, b) => a + b) > eqBars(0.1, 7).reduce((a, b) => a + b), "louder is taller");
});

test("the overlay's shape ignores the ticking level and changes with the card's face", () => {
  let s = run([open("a"), state("a", "listening", { level: 0.2 })]);
  const sh = shapeOf(s, EMPTY_SHAPE);
  assert.deepEqual({ ...sh, ended: undefined }, { liveSid: "a", liveView: "listening", hearing: true, ended: undefined });
  s = run([state("a", "listening", { level: 0.9, elapsed_s: 5 }), state("a", "listening", { level: 0.4, elapsed_s: 5.1 })], s);
  assert.equal(shapeOf(s, sh), sh, "level and seconds alone never change the shape");
  s = run([state("a", "matching")], s);
  const matching = shapeOf(s, sh);
  assert.notEqual(matching, sh);
  assert.equal(matching.liveView, "matching");
  s = run([state("a", "found", { result: RESULT })], s);
  assert.equal(shapeOf(s, matching).hearing, false);
  s = run([ev("song_close", { sid: "a", reason: "done" })], s);
  const done = shapeOf(s, matching);
  assert.equal(done.liveSid, null);
  assert.equal(done.ended.a.result?.title, "Northern Lights");
  assert.equal(sessionOf(s, "a"), s.ended.a);
  assert.equal(sessionOf(s, "nope"), null);
});

test("the meter's curve lifts ordinary loudness and stays in 0..1", () => {
  assert.equal(shownLevel(0.25), 0.5);
  assert.equal(shownLevel(4), 1);
  assert.equal(shownLevel(-1), 0);
  assert.equal(shownLevel(NaN), 0);
});
