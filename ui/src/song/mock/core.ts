// The song mock core (dev only, `?mock=song`): speaks the song card's side of NEO's websocket —
// hello with a transcript replay, state, transcripts, tool chips, notes and song_* — and plays
// scripted runs: listening → matching → found, no match, an error, a stop. Nothing real happens:
// no microphone (the level is synthesised), no network (covers are drawn on a canvas, links point
// at example.com), and song_link only says which link NEO would have opened.
import type { SongLinkKind, SongResult, SongSource, SongStatus } from "../types";

type Handler = ((ev: any) => void) | null;
export type SongRun = "found" | "not_found" | "error" | "stopped" | "listening" | "matching";

const CANCEL = Symbol("cancelled");
const RECENT = new Set(["transcript", "tool_start", "tool_end", "turn", "song_open", "song_close"]);
const LISTEN_S = 12;
const nowS = () => Date.now() / 1000;

/** A WebSocket look-alike wired to the in-page song core; messages arrive asynchronously, in order. */
export class SongMockSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  readyState = 0;
  binaryType = "arraybuffer";
  onopen: Handler = null;
  onmessage: Handler = null;
  onclose: Handler = null;
  onerror: Handler = null;

  constructor() {
    setTimeout(() => {
      const core = songMock();
      if (!core.accepting) { this.readyState = 3; this.onerror?.({}); this.onclose?.({}); return; }
      this.readyState = 1;
      this.onopen?.({});
      core.connect(this);
    }, 40);
  }

  send(data: string) {
    if (this.readyState !== 1) return;
    queueMicrotask(() => songMock().receive(this, data));
  }

  close() {
    if (this.readyState >= 2) return;
    this.readyState = 3;
    songMock().drop(this);
    queueMicrotask(() => this.onclose?.({}));
  }

  deliver(data: string) {
    if (this.readyState !== 1) return;
    queueMicrotask(() => { if (this.readyState === 1) this.onmessage?.({ data }); });
  }

  /** The core went away (simulated restart). */
  lose() {
    if (this.readyState !== 1) return;
    this.readyState = 3;
    queueMicrotask(() => this.onclose?.({}));
  }
}

interface Tune { title: string; artist: string; album: string; year: string; genre: string | null; hues: [number, number]; cover: boolean; links: SongLinkKind[] }

// Fictional songs: one with everything, one with no cover and a single link, one with long names.
const TUNES: Tune[] = [
  { title: "Northern Lights", artist: "The Midnight Owls", album: "Glass City", year: "2021", genre: "Synthpop", hues: [262, 328], cover: true, links: ["apple_music", "shazam"] },
  { title: "Paper Boats", artist: "Juniper & the Tides", album: "Low Tide", year: "2019", genre: "Indie Folk", hues: [196, 150], cover: false, links: ["shazam"] },
  {
    title: "Slow Motion Summer (Extended Sunset Version, Remastered)", artist: "Aurelia Vance feat. The Kestrel Quartet",
    album: "Heat Haze: Songs for Long Evenings", year: "2023", genre: "Alternative", hues: [14, 42], cover: true, links: ["apple_music", "shazam"],
  },
];

const covers = new Map<string, string>();
function drawCover(t: Tune): string | null {
  if (!t.cover) return null;
  const hit = covers.get(t.title);
  if (hit) return hit;
  const c = document.createElement("canvas");
  c.width = c.height = 360;
  const g = c.getContext("2d");
  if (!g) return null;
  const [h1, h2] = t.hues;
  const bg = g.createLinearGradient(0, 0, 360, 360);
  bg.addColorStop(0, `hsl(${h1} 62% 18%)`);
  bg.addColorStop(1, `hsl(${h2} 70% 40%)`);
  g.fillStyle = bg;
  g.fillRect(0, 0, 360, 360);
  const sun = g.createLinearGradient(0, 96, 0, 270);
  sun.addColorStop(0, `hsl(${h2 + 24} 95% 72%)`);
  sun.addColorStop(1, `hsl(${h1 + 40} 88% 56%)`);
  g.fillStyle = sun;
  g.beginPath();
  g.arc(180, 186, 92, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = `hsl(${h1} 62% 18%)`;
  for (let i = 0; i < 6; i++) g.fillRect(70, 196 + i * 14, 220, 2 + i * 1.4);
  g.strokeStyle = `hsl(${h2} 90% 76% / 0.45)`;
  g.lineWidth = 1.5;
  for (let i = 0; i < 7; i++) { const y = 286 + i * i * 2.2; g.beginPath(); g.moveTo(0, y); g.lineTo(360, y); g.stroke(); }
  g.fillStyle = "rgba(255, 255, 255, 0.92)";
  g.font = "600 20px -apple-system, system-ui, sans-serif";
  g.textAlign = "center";
  g.fillText(t.album.split(":")[0].toUpperCase(), 180, 52);
  const url = c.toDataURL("image/jpeg", 0.86);
  covers.set(t.title, url);
  return url;
}

function resultOf(t: Tune): SongResult {
  const slug = t.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return {
    title: t.title, artist: t.artist, album: t.album, year: t.year, genre: t.genre, cover: drawCover(t),
    links: {
      apple_music: t.links.includes("apple_music") ? `https://example.com/apple-music/${slug}` : null,
      shazam: t.links.includes("shazam") ? `https://example.com/shazam/${slug}` : null,
    },
  };
}

interface Live { sid: string; source: SongSource; t0: number; status: SongStatus; level: number; message: string | null; result: SongResult | null; hold: boolean }

export class SongMockCore {
  accepting = true;
  speed = 1;
  looping = false;
  /** What the panel shows as running. */
  current: SongRun | "loop" | "" = "";
  private socks = new Set<SongMockSocket>();
  private recent: { type: string; data: unknown; ts: number }[] = [];
  private neoState = "idle";
  private live: Live | null = null;
  /** The newest session's sid and result (song_link opens only these). */
  private last: { sid: string; result: SongResult | null } | null = null;
  private gen = 0;
  private ticker: number | undefined;
  private tuneIx = 0;
  private sourceIx = 0;
  listeners = new Set<() => void>();

  // ---- connections ---------------------------------------------------------------------------------
  connect(sock: SongMockSocket) {
    this.socks.add(sock);
    sock.deliver(JSON.stringify({ type: "hello", data: { state: this.neoState, recent: this.recent, confirm: null, problems: [] }, ts: nowS() }));
  }
  drop(sock: SongMockSocket) { this.socks.delete(sock); }

  /** Every socket closes and reconnects are refused for `ms`; a live run carries on meanwhile.
   *  With `restart`, the core forgets everything too (a live card then gets no answer: "gone"). */
  kill(ms = 1500, restart = false) {
    this.accepting = false;
    for (const s of [...this.socks]) { s.lose(); this.socks.delete(s); }
    if (restart) { this.halt(); this.recent = []; this.neoState = "idle"; this.last = null; }
    window.setTimeout(() => { this.accepting = true; }, ms);
  }

  private bcast(type: string, data: unknown) {
    const ts = nowS();
    if (RECENT.has(type) || (type === "song_state" && this.isFinal(data))) {
      this.recent.push({ type, data, ts });
      if (this.recent.length > 200) this.recent.shift();
    }
    const msg = JSON.stringify({ type, data, ts });
    for (const s of this.socks) s.deliver(msg);
  }
  private isFinal(d: unknown) { const st = (d as { status?: string }).status; return st === "found" || st === "not_found" || st === "error"; }
  private setNeo(state: string) { this.neoState = state; this.bcast("state", { state }); }
  private say(role: "user" | "assistant", text: string, job: string) { this.bcast("transcript", { role, text, final: true, job }); }
  private note(text: string) { this.bcast("note", { text }); }
  private changed() { for (const l of this.listeners) l(); }

  // ---- commands --------------------------------------------------------------------------------------
  receive(sock: SongMockSocket, raw: string) {
    let m: any;
    try { m = JSON.parse(raw); } catch { return; }
    switch (m.cmd) {
      case "song_hello":
        if (this.live) sock.deliver(JSON.stringify({ type: "song_state", data: this.stateMsg(), ts: nowS() }));
        return;
      case "song_stop":
        if (this.live && m.sid === this.live.sid) void this.stopNow();
        return;
      case "song_link": {
        const url = this.last && m.sid === this.last.sid ? this.last.result?.links[m.which as SongLinkKind] : null;
        this.note(url ? `Mock core: NEO would open the ${m.which === "shazam" ? "Shazam" : "Apple Music"} link now (nothing is opened here).` : "Mock core: that link isn't this card's to open.");
        return;
      }
      case "text": {
        const t = String(m.text ?? "");
        if (/song|music|playing|track|shazam/i.test(t)) void this.run("found", t);
        else if (/^\s*stop\b/i.test(t)) void this.stopNow();
        else this.say("assistant", "This is the song mock: ask “what song is this?”, or use the panel.", "chat");
        return;
      }
      case "stop":
        void this.stopNow();
        return;
      case "forget_all":
        this.halt();
        this.recent = [];
        this.last = null;
        this.bcast("erased", {});
        return;
    }
  }

  // ---- the scripted runs --------------------------------------------------------------------------------
  private stateMsg() {
    const l = this.live!;
    const t = ((Date.now() - l.t0) / 1000) * this.speed;
    // A held state (the panel's Listening / Matching) goes round its window instead of stopping at the end.
    const elapsed = l.hold ? t % LISTEN_S : Math.min(LISTEN_S, t);
    return { sid: l.sid, status: l.status, level: l.level, elapsed_s: Math.round(elapsed * 10) / 10, message: l.message, result: l.result };
  }

  private emit() { if (this.live) this.bcast("song_state", this.stateMsg()); }

  /** A lively, made-up loudness: a kick on every beat at 120 bpm, a little wobble and noise. */
  private levelAt(t: number) {
    const beat = Math.pow(Math.max(0, Math.cos(2 * Math.PI * t * 2)), 3);
    return Math.min(1, Math.max(0, 0.24 + 0.42 * beat + 0.16 * Math.random() + 0.08 * Math.sin(t * 0.9)));
  }

  private startTicker() {
    clearInterval(this.ticker);
    this.ticker = window.setInterval(() => {
      const l = this.live;
      if (!l || (l.status !== "listening" && l.status !== "matching")) return;
      l.level = this.levelAt(((Date.now() - l.t0) / 1000) * this.speed);
      this.emit();
    }, 100);
  }

  /** Stop whatever is running, without a word to the UI (a core restart). */
  private halt() {
    this.gen++;
    clearInterval(this.ticker);
    this.live = null;
    this.looping = false;
    this.current = "";
    this.changed();
  }

  private close(reason: "done" | "stopped" | "error") {
    const l = this.live;
    if (!l) return;
    clearInterval(this.ticker);
    this.live = null;
    this.bcast("song_close", { sid: l.sid, reason });
  }

  /** The stop button, a spoken "stop": listening ends at once. */
  async stopNow() {
    if (!this.live) return;
    this.gen++; // the script stops where it is
    this.looping = false;
    this.current = "";
    this.close("stopped");
    this.bcast("tool_end", { name: "recognise_song", ok: true, summary: "stopped", job: "song" });
    this.setNeo("idle");
    this.changed();
  }

  async run(kind: SongRun, words = "Neo, what song is this?") {
    if (this.live) this.close("stopped");
    const my = ++this.gen;
    clearInterval(this.ticker);
    if (!this.looping) { this.current = kind; this.changed(); }
    const sleep = (s: number) => new Promise<void>((ok, no) => window.setTimeout(() => (my === this.gen ? ok() : no(CANCEL)), (s * 1000) / this.speed));
    const job = "song";
    try {
      this.say("user", words, job);
      this.setNeo("thinking");
      await sleep(0.5);
      this.bcast("tool_start", { name: "recognise_song", args: {}, job });
      this.setNeo("working");
      const sources: SongSource[] = ["both", "mic", "mac"];
      const source = sources[this.sourceIx++ % sources.length];
      const sid = `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
      const t0 = Date.now();
      this.live = { sid, source, t0, status: "listening", level: 0, message: null, result: null, hold: kind === "listening" || kind === "matching" };
      this.last = { sid, result: null };
      this.bcast("song_open", { sid, source, listen_s: LISTEN_S, job });
      this.emit();
      this.startTicker();
      const at = (s: number) => sleep(Math.max(0, s - ((Date.now() - t0) / 1000) * this.speed));
      const blip = async (s: number, len = 0.6) => { await at(s); this.live!.status = "matching"; this.emit(); await sleep(len); this.live!.status = "listening"; this.emit(); };
      const final = (status: SongStatus, message: string | null, result: SongResult | null) => {
        const l = this.live!;
        Object.assign(l, { status, message, result, level: 0 });
        this.last = { sid: l.sid, result };
        this.emit();
      };
      const finish = async (reason: "done" | "error", reply: string, ok = true) => {
        await sleep(0.35);
        this.close(reason);
        this.bcast("tool_end", { name: "recognise_song", ok, summary: reply, job });
        this.say("assistant", reply, job);
        this.bcast("turn", { route: "quick", brain: "", model: "", ms: Date.now() - t0 + 500, tools: 1, job });
        this.setNeo("idle");
      };
      switch (kind) {
        case "found": {
          const tune = TUNES[this.tuneIx++ % TUNES.length];
          await blip(3.5);
          await at(7);
          this.live!.status = "matching";
          this.emit();
          await sleep(0.8);
          const r = resultOf(tune);
          final("found", null, r);
          await finish("done", `That's **${r.title}** by ${r.artist}, from *${r.album}* (${r.year}).`);
          break;
        }
        case "not_found": {
          await blip(3.5);
          await blip(7);
          await blip(11, 0.5);
          await at(LISTEN_S);
          final("not_found", "Couldn't recognise it — maybe try again closer to the music", null);
          await finish("done", "I couldn't recognise it. Try again a little closer to the music?");
          break;
        }
        case "error":
          await sleep(1.4);
          final("error", "Shazam didn't answer — check the internet connection and try again", null);
          await finish("error", "I couldn't reach Shazam just now.", false);
          break;
        case "stopped":
          await sleep(4);
          this.say("user", "stop", job);
          await this.stopNow();
          break;
        case "listening":
        case "matching":
          if (kind === "matching") { await sleep(1.2); this.live!.status = "matching"; this.emit(); }
          await new Promise<void>((_, no) => { const w = window.setInterval(() => { if (my !== this.gen) { clearInterval(w); no(CANCEL); } }, 200); });
          break;
      }
    } catch (err) {
      if (err !== CANCEL) console.error("[song mock]", err);
    } finally {
      if (my === this.gen && !this.looping) { this.current = ""; this.changed(); }
    }
  }

  /** found → no match → found (the next song) → …, a pause between, until another run is picked. */
  async loop() {
    this.looping = true;
    this.current = "loop";
    this.changed();
    const kinds: SongRun[] = ["found", "not_found", "found", "error", "found"];
    for (let i = 0; this.looping; i++) {
      const my = this.gen + 1;
      await this.run(kinds[i % kinds.length]);
      if (!this.looping || this.gen !== my) break;
      await new Promise((r) => window.setTimeout(r, 3500 / this.speed));
      if (this.gen !== my) break;
    }
  }

  /** Pick a run from the panel (ends the loop). */
  pick(kind: SongRun | "loop") {
    this.looping = false;
    if (kind === "loop") { void this.loop(); return; }
    void this.run(kind);
  }

  setSpeed(x: number) { this.speed = x; }
}

let core: SongMockCore | null = null;
export function songMock(): SongMockCore {
  if (!core) {
    core = new SongMockCore();
    const st = new URLSearchParams(location.search).get("state") as SongRun | null;
    window.setTimeout(() => core!.pick(st ?? "loop"), 900);
    (window as unknown as { neoSongMock: SongMockCore }).neoSongMock = core;
  }
  return core;
}
