// NEO Setup's page (src/setup): each screen says what the shell's snapshot says, in fixed words; its
// buttons are only the shell's own Setup commands; the mock exists only in dev builds; and NEO Sandbox's
// page talks to its own core's port.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  agoText, checklistRows, defaultChosen, enginePercent, fetchLine, fetchPercent, fixesFor, optionalRows, planRows, rateText, roomLine,
  stageText, title, uninstallTotal, updateLine,
} from "../src/setup/model.ts";
import { MOCK_VIEWS, mockSnapshot, playMock } from "../src/setup/mock.ts";
import { mockKind, mockWire } from "../src/setup/wire.ts";
import { CORE_WS, WS_PORT } from "../src/port.ts";
import type { EngineProgress, Snapshot } from "../src/setup/types.ts";

const ROOT = new URL("../../", import.meta.url);
const read = (rel: string) => readFileSync(new URL(rel, ROOT), "utf8");
const p = (stage: EngineProgress["stage"], done = 0, total = 0, extra: Partial<EngineProgress> = {}): EngineProgress => ({
  stage, done, total, rate: 0, eta_s: null, error: null, ...extra,
});

test("the install's line says what's happening, how far and how long", () => {
  const plan = mockSnapshot("welcome").plan;
  assert.equal(stageText(p("downloading", 84_000_000, 225_000_000, { rate: 6_200_000, eta_s: 23 }), plan),
    "Downloading NEO's engine — 84 MB of 225 MB · 6.2 MB/s · less than a minute left");
  assert.equal(stageText(p("downloading", 0, 225_000_000), plan), "Downloading NEO's engine — 0 MB of 225 MB");
  assert.equal(stageText(p("unpacking", 12_000, 31_412), plan), "Unpacking — 12,000 of 31,412 files");
  assert.equal(stageText(p("unpacking"), plan), "Unpacking…");
  assert.equal(stageText(p("verifying", 50, 200), plan), "Checking the download… 25%");
  assert.equal(stageText(p("installing"), plan), "Installing NEO 2.2.0…");
  assert.equal(stageText(p("failed", 0, 0, { error: { code: "offline", message: "Can't reach github.com — check your connection, then Retry." } }), plan),
    "Can't reach github.com — check your connection, then Retry.");
  assert.equal(rateText(450_000), "450 KB/s");
  assert.equal(rateText(0), "");
});

test("the bar moves forward through the stages and is busy where nothing can be counted", () => {
  const order = [
    enginePercent(p("preparing")), enginePercent(p("downloading", 0, 10)), enginePercent(p("downloading", 10, 10)),
    enginePercent(p("verifying", 5, 10)), enginePercent(p("unpacking", 10, 10)), enginePercent(p("finishing")), enginePercent(p("done")),
  ];
  for (let i = 1; i < order.length; i++) assert.ok((order[i] ?? 0) >= (order[i - 1] ?? 0), `${order}`);
  assert.equal(enginePercent(p("done")), 100);
  assert.equal(enginePercent(p("optimizing")), null);
  assert.equal(enginePercent(p("testing")), null);
  assert.equal(enginePercent(p("failed")), null);
});

test("each failure offers what fixes it", () => {
  assert.deepEqual(fixesFor("offline"), ["retry", "use_file"]);
  assert.deepEqual(fixesFor("not_found"), ["releases", "use_file"]);
  assert.deepEqual(fixesFor("disk_full"), ["retry"]);
  assert.deepEqual(fixesFor("smoke"), ["retry", "logs"]);
  assert.deepEqual(fixesFor("busy"), []);
  assert.deepEqual(fixesFor("anything"), ["retry", "use_file", "logs"]);
});

test("the welcome page says what is downloaded, from where, and whether it fits", () => {
  const plan = mockSnapshot("welcome").plan!;
  const rows = planRows(plan);
  assert.equal(rows[0].from, "github.com");
  assert.equal(rows[0].size, "225 MB");
  assert.equal(rows[0].disk, "1.5 GB");
  assert.equal(rows[1].size, "1.7 GB");
  assert.equal(planRows({ ...plan, downloaded: 100_000_000 })[0].size, "125 MB left of 225 MB");
  assert.deepEqual(roomLine(plan), { ok: true, text: "NEO needs about 4.4 GB of free space; this Mac has 182 GB." });
  const tight = roomLine({ ...plan, free_bytes: 2_100_000_000 });
  assert.equal(tight.ok, false);
  assert.match(tight.text, /Make some room first\.$/);
  assert.equal(roomLine({ ...plan, free_bytes: null }).ok, true);
});

test("the checklist's rows follow the permissions, and the camera waits for the core", () => {
  const s = mockSnapshot("checklist");
  const rows = checklistRows(s);
  assert.deepEqual(rows.map((r) => r.id), ["brain", "accessibility", "screen_recording", "microphone", "camera", "later"]);
  const by = (id: string) => rows.find((r) => r.id === id)!;
  assert.equal(by("accessibility").state, "ok");
  assert.equal(by("screen_recording").state, "todo");
  assert.match(by("screen_recording").detail, /Quit & Reopen/);
  assert.deepEqual(by("microphone").action, { label: "Allow…", cmd: "setup_permission", arg: "microphone" });
  assert.deepEqual(by("camera").action, { label: "Open Settings…", cmd: "setup_open", arg: "camera_settings" }); // denied: only Settings can
  assert.equal(by("later").state, "note");
  const early = checklistRows({ ...s, core_ready: false });
  assert.equal(early.find((r) => r.id === "microphone")!.state, "wait");
  assert.equal(early.find((r) => r.id === "microphone")!.action, undefined);
  const brainless = checklistRows({ ...s, brain_opened: false }).find((r) => r.id === "brain")!;
  assert.equal(brainless.state, "todo");
  assert.deepEqual(brainless.action, { label: "Open Brains & Keys…", cmd: "setup_open", arg: "keys" });
});

test("the models' line and the optional downloads say how far they are", () => {
  const s = mockSnapshot("checklist");
  assert.equal(fetchLine(s.essential), "Downloading NEO's models — 610 MB of 1.6 GB (37%) · about 3 minutes left");
  assert.equal(fetchPercent(s.essential), 37);
  assert.equal(fetchLine({ ...s.essential, state: "done" }), "NEO's models are on this Mac.");
  assert.equal(fetchLine({ ...s.essential, state: "failed", error: "Can't reach the server — check the internet connection." }),
    "Can't reach the server — check the internet connection.");
  const rows = optionalRows(s);
  assert.deepEqual(rows.map((r) => r.group), ["local_voice", "browser", "vision"]);
  assert.equal(rows[0].label, "Download (2.9 GB)");
  assert.equal(rows[2].label, "Download (42 MB)");
  const busy = optionalRows({ ...s, optional: { browser: { ...s.essential, state: "running" } } });
  assert.equal(busy[1].label, "Cancel");
  assert.equal(optionalRows({ ...s, here: { vision: true } })[2].label, "On this Mac");
});

test("headings, updates and uninstall in words", () => {
  assert.equal(title("move", "NEO"), "Move NEO to Applications");
  assert.equal(title("welcome", "NEO Sandbox"), "Set up NEO Sandbox");
  const u = mockSnapshot("updates");
  assert.equal(updateLine(u.update, "2.2.0", "NEO"), "NEO 2.2.1 is out (this is 2.2.0).");
  assert.equal(updateLine({ ...u.update, offer: null }, "2.2.1", "NEO"), "NEO 2.2.1 is the newest version.");
  assert.equal(updateLine({ ...u.update, checking: true }, "2.2.0", "NEO"), "Checking…");
  assert.equal(agoText(0), "never");
  assert.equal(agoText(1000, 1000 + 3600), "today");
  assert.equal(agoText(1000, 1000 + 3 * 86400), "3 days ago");
  const items = mockSnapshot("uninstall").uninstall.items;
  const chosen = defaultChosen(items);
  assert.deepEqual([...chosen].sort(), ["app", "engine", "model_files", "models"]);
  assert.equal(uninstallTotal(items, chosen), 6_100_000 + 1_520_000_000 + 1_480_000_000 + 160_000_000);
  assert.equal(uninstallTotal(items, new Set()), 6_100_000, "the app itself always goes");
});

test("the page's buttons are the shell's own Setup commands, each one registered", () => {
  const page = read("ui/src/setup/SetupApp.tsx") + read("ui/src/setup/model.ts");
  const rust = read("ui/src-tauri/src/setupwin.rs");
  const lib = read("ui/src-tauri/src/lib.rs");
  const used = new Set([...page.matchAll(/"(setup_[a-z_]+)"/g)].map((m) => m[1]));
  assert.ok(used.size >= 10, `${[...used]}`);
  for (const cmd of used) {
    assert.ok(rust.includes(`pub fn ${cmd}(`), `${cmd} is a command`);
    assert.ok(lib.includes(`setupwin::${cmd}`), `${cmd} is registered`);
  }
  // The page talks to no websocket: Setup runs before there is a core.
  assert.ok(!page.includes("WebSocket") && !page.includes("ws://"));
  // The mock is a dev build's only (and only for the setup route).
  assert.match(read("ui/src/setup/SetupApp.tsx"), /mockKind\(location\.search, import\.meta\.env\.DEV\)/);
  assert.equal(mockKind("?mock=welcome", false), null);
  assert.equal(mockKind("?mock=welcome", true), "welcome");
});

test("every mock view starts from a snapshot the page can show, and Set up plays through to the checklist", async () => {
  for (const v of MOCK_VIEWS) {
    const s: Snapshot = mockSnapshot(v);
    assert.ok(typeof s.view === "string" && s.engine && s.essential, v);
  }
  const w = mockWire("welcome");
  const seen: string[] = [];
  w.on((s) => { if (seen.at(-1) !== s.engine.stage) seen.push(s.engine.stage); });
  assert.equal((await w.state()).view, "welcome");
  // (the clock runs fast: each timer at once)
  let snap = await w.state();
  const timers: (() => void)[] = [];
  const realSet = globalThis.setTimeout;
  (globalThis as { setTimeout: unknown }).setTimeout = ((f: () => void) => { timers.push(f); return 0; }) as unknown;
  try {
    playMock("setup_start", {}, () => snap, (s) => { snap = s; });
    for (let i = 0; i < timers.length; i++) timers[i](); // (timers set by timers run too)
  } finally {
    (globalThis as { setTimeout: unknown }).setTimeout = realSet;
  }
  assert.equal(snap.view, "checklist");
  assert.equal(snap.engine.stage, "done");
  assert.equal(snap.essential.state, "done");
});

test("NEO's page talks to port 8765, and only a sandbox build's to 18765", () => {
  assert.equal(WS_PORT, 8765); // (node: no vite mode)
  assert.equal(CORE_WS, "ws://127.0.0.1:8765");
  assert.match(read("ui/src/port.ts"), /mode === "sandbox" \? 18765 : 8765/);
  for (const f of ["ui/src/ws.ts", "ui/src/keys/wire.ts"]) assert.ok(!read(f).includes(":8765"), `${f} takes the port from port.ts`);
  const release = JSON.parse(read("ui/src-tauri/tauri.conf.json"));
  const sandbox = JSON.parse(read("ui/src-tauri/tauri.sandbox.conf.json"));
  assert.match(release.app.security.csp, /ws:\/\/127\.0\.0\.1:8765/);
  assert.doesNotMatch(release.app.security.csp, /18765/);
  assert.match(sandbox.app.security.csp, /ws:\/\/127\.0\.0\.1:18765/);
  assert.doesNotMatch(sandbox.app.security.csp, /:8765/);
  assert.equal(sandbox.identifier, "com.arnav.neo.sandbox");
  assert.equal(sandbox.productName, "NEO Sandbox");
  assert.equal(sandbox.build.beforeBuildCommand, "npm run build:sandbox");
  assert.match(JSON.parse(read("ui/package.json")).scripts["build:sandbox"], /vite build --mode sandbox --outDir dist-sandbox/);
  assert.equal(sandbox.build.frontendDist, "../dist-sandbox");
});
