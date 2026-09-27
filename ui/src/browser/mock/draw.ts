// Canvas drawing helpers for the mock pages (dev only).
export type Ctx = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

export const FONT = `-apple-system, "SF Pro Text", "Segoe UI", system-ui, sans-serif`;
export const font = (size: number, weight = 400) => `${weight} ${size}px ${FONT}`;

export function rr(ctx: Ctx, x: number, y: number, w: number, h: number, r: number) {
  const k = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + k, y);
  ctx.arcTo(x + w, y, x + w, y + h, k);
  ctx.arcTo(x + w, y + h, x, y + h, k);
  ctx.arcTo(x, y + h, x, y, k);
  ctx.arcTo(x, y, x + w, y, k);
  ctx.closePath();
}

export function fillRR(ctx: Ctx, x: number, y: number, w: number, h: number, r: number, fill: string | CanvasGradient, stroke?: string, lw = 1) {
  rr(ctx, x, y, w, h, r);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) { ctx.lineWidth = lw; ctx.strokeStyle = stroke; ctx.stroke(); }
}

export function text(ctx: Ctx, s: string, x: number, y: number, f: string, color: string, align: CanvasTextAlign = "left", maxW?: number) {
  ctx.font = f;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = "alphabetic";
  if (maxW && ctx.measureText(s).width > maxW) {
    let t = s;
    while (t.length > 1 && ctx.measureText(t + "…").width > maxW) t = t.slice(0, -1);
    s = t.trimEnd() + "…";
  }
  ctx.fillText(s, x, y);
  return ctx.measureText(s).width;
}

/** Wrap to at most `lines` lines; returns the y after the last line. */
export function wrap(ctx: Ctx, s: string, x: number, y: number, maxW: number, lh: number, f: string, color: string, lines = 3) {
  ctx.font = f;
  const words = s.split(/\s+/);
  let line = "";
  let n = 0;
  for (let i = 0; i < words.length; i++) {
    const test = line ? line + " " + words[i] : words[i];
    if (ctx.measureText(test).width > maxW && line) {
      n++;
      if (n === lines) { text(ctx, line + " " + words.slice(i).join(" "), x, y, f, color, "left", maxW); return y + lh; }
      text(ctx, line, x, y, f, color);
      y += lh;
      line = words[i];
    } else line = test;
  }
  if (line) { text(ctx, line, x, y, f, color, "left", maxW); y += lh; }
  return y;
}

export function stars(ctx: Ctx, x: number, y: number, rating: number, size = 13) {
  for (let i = 0; i < 5; i++) {
    const cx = x + i * (size + 1) + size / 2, cy = y - size / 2 + 1;
    const fillAmt = Math.max(0, Math.min(1, rating - i));
    starPath(ctx, cx, cy, size / 2);
    ctx.fillStyle = "#e2e8f0";
    ctx.fill();
    if (fillAmt > 0) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(cx - size / 2, cy - size / 2, size * fillAmt, size);
      ctx.clip();
      starPath(ctx, cx, cy, size / 2);
      ctx.fillStyle = "#f59e0b";
      ctx.fill();
      ctx.restore();
    }
  }
  return 5 * (size + 1);
}

function starPath(ctx: Ctx, cx: number, cy: number, r: number) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rad = i % 2 === 0 ? r : r * 0.45;
    const px = cx + Math.cos(a) * rad, py = cy + Math.sin(a) * rad;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

function shade(hex: string, amt: number): string {
  const n = parseInt(hex.slice(1), 16);
  let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  const t = amt < 0 ? 0 : 255, p = Math.abs(amt);
  r = Math.round((t - r) * p + r); g = Math.round((t - g) * p + g); b = Math.round((t - b) * p + b);
  return `rgb(${r},${g},${b})`;
}

/** A pair of true-wireless earbuds over their charging case, on a soft pastel ground. */
export function earbuds(ctx: Ctx, x: number, y: number, w: number, h: number, color: string, bg: [string, string], radius = 12) {
  ctx.save();
  rr(ctx, x, y, w, h, radius);
  ctx.clip();
  const g = ctx.createLinearGradient(x, y, x + w, y + h);
  g.addColorStop(0, bg[0]);
  g.addColorStop(1, bg[1]);
  ctx.fillStyle = g;
  ctx.fillRect(x, y, w, h);
  const glow = ctx.createRadialGradient(x + w * 0.5, y + h * 0.45, 0, x + w * 0.5, y + h * 0.45, w * 0.55);
  glow.addColorStop(0, "rgba(255,255,255,0.75)");
  glow.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(x, y, w, h);

  const s = Math.min(w, h * 1.15);
  const cx = x + w / 2, base = y + h * 0.78;
  // shadow
  ctx.fillStyle = "rgba(15,23,42,0.10)";
  ctx.beginPath();
  ctx.ellipse(cx, base + s * 0.035, s * 0.3, s * 0.035, 0, 0, Math.PI * 2);
  ctx.fill();
  // case
  const cw = s * 0.5, ch = s * 0.3;
  const cg = ctx.createLinearGradient(cx, base - ch, cx, base);
  cg.addColorStop(0, shade(color, 0.22));
  cg.addColorStop(1, shade(color, -0.12));
  fillRR(ctx, cx - cw / 2, base - ch, cw, ch, ch * 0.45, cg);
  ctx.strokeStyle = "rgba(0,0,0,0.12)";
  ctx.lineWidth = Math.max(1, s * 0.004);
  ctx.beginPath();
  ctx.moveTo(cx - cw / 2 + ch * 0.2, base - ch * 0.62);
  ctx.lineTo(cx + cw / 2 - ch * 0.2, base - ch * 0.62);
  ctx.stroke();
  ctx.fillStyle = "#34d399";
  ctx.beginPath();
  ctx.arc(cx, base - ch * 0.28, s * 0.008, 0, Math.PI * 2);
  ctx.fill();
  // buds
  const bud = (bx: number, by: number, tilt: number) => {
    ctx.save();
    ctx.translate(bx, by);
    ctx.rotate(tilt);
    const r = s * 0.085;
    const sg = ctx.createLinearGradient(-r * 0.4, 0, r * 0.4, 0);
    sg.addColorStop(0, shade(color, 0.12));
    sg.addColorStop(1, shade(color, -0.18));
    fillRR(ctx, -r * 0.34, 0, r * 0.68, r * 2.4, r * 0.34, sg);
    const hg = ctx.createRadialGradient(-r * 0.35, -r * 0.35, r * 0.1, 0, 0, r * 1.1);
    hg.addColorStop(0, shade(color, 0.45));
    hg.addColorStop(1, shade(color, -0.1));
    ctx.fillStyle = hg;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(15,23,42,0.35)";
    ctx.beginPath();
    ctx.ellipse(r * 0.55, -r * 0.1, r * 0.3, r * 0.42, 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };
  bud(cx - s * 0.14, base - ch - s * 0.26, -0.28);
  bud(cx + s * 0.14, base - ch - s * 0.26, 0.28);
  ctx.restore();
}

export function searchIcon(ctx: Ctx, cx: number, cy: number, color: string) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx - 2, cy - 2, 6, 0, Math.PI * 2);
  ctx.moveTo(cx + 2.5, cy + 2.5);
  ctx.lineTo(cx + 7, cy + 7);
  ctx.stroke();
}

export function cartIcon(ctx: Ctx, x: number, y: number, color: string) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 4, y);
  ctx.lineTo(x + 7, y + 13);
  ctx.lineTo(x + 20, y + 13);
  ctx.lineTo(x + 23, y + 4);
  ctx.lineTo(x + 6, y + 4);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x + 9, y + 18, 1.8, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(x + 18, y + 18, 1.8, 0, Math.PI * 2); ctx.fill();
}

export function check(ctx: Ctx, cx: number, cy: number, r: number, bg: string, fg = "#fff") {
  ctx.fillStyle = bg;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = fg;
  ctx.lineWidth = Math.max(2, r * 0.18);
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(cx - r * 0.4, cy + r * 0.02);
  ctx.lineTo(cx - r * 0.1, cy + r * 0.32);
  ctx.lineTo(cx + r * 0.45, cy - r * 0.3);
  ctx.stroke();
}

export function lockIcon(ctx: Ctx, x: number, y: number, color: string) {
  ctx.strokeStyle = color; ctx.lineWidth = 1.6;
  rr(ctx, x, y + 6, 12, 9, 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(x + 6, y + 6, 3.6, Math.PI, 0); ctx.stroke();
}

/** The store's favicon as a data URL. */
export function makeFavicon(): string {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const ctx = c.getContext("2d")!;
  const g = ctx.createLinearGradient(0, 0, 64, 64);
  g.addColorStop(0, "#1e293b");
  g.addColorStop(1, "#0f172a");
  fillRR(ctx, 0, 0, 64, 64, 14, g);
  ctx.strokeStyle = "#ffb020";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(16, 46); ctx.lineTo(16, 22); ctx.lineTo(48, 46); ctx.lineTo(48, 18);
  ctx.stroke();
  return c.toDataURL("image/png");
}
