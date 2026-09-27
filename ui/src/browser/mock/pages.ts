// The mock's web pages (dev only): a small fictional store, "Northwind", drawn to canvas so the
// live view has something real-looking to show — search results, a product, a cart, a sign-in
// form, checkout. Layout and paint come from one pass, so hit-testing matches the pixels.
import { type Ctx, FONT, font, rr, fillRR, text, wrap, stars, earbuds, searchIcon, cartIcon, check, lockIcon } from "./draw";

export type PageKey = "home" | "search" | "product" | "cart" | "signin" | "checkout" | "confirmed" | "generic";

export interface El {
  id: string;
  kind: "button" | "link" | "input" | "select" | "check";
  x: number; y: number; w: number; h: number;
  label: string; role: string;
  field?: string; secret?: boolean; placeholder?: string;
  href?: PageKey; act?: string;
  options?: string[];
}

export interface PState {
  key: PageKey;
  vw: number; vh: number; scrollY: number;
  hover: string | null; pressed: string | null; focus: string | null;
  values: Record<string, string>;
  caret: boolean;
  loaded: number; // 0 = skeleton, 1 = done
  cart: number;
  colour: string;
  qty: string;
  banner: boolean;
  signedIn: boolean;
  keepSigned: boolean;
  genericUrl: string;
  signinError: boolean;
}

export interface Built { els: El[]; docH: number; bg: string; paint: (ctx: Ctx) => void }

export const HOST = "www.northwind.store";
export const COLOURS: { name: string; hex: string }[] = [
  { name: "Midnight", hex: "#1f2937" }, { name: "Sand", hex: "#d6c7ae" }, { name: "Sage", hex: "#9fb8a1" },
];

export interface Product { id: string; name: string; price: number; was?: number; rating: number; reviews: number; color: string; bg: [string, string]; badge?: string; sponsored?: boolean }
export const PRODUCTS: Product[] = [
  { id: "pulse", name: "Pulse Air 2 Wireless Earbuds with Charging Case, IPX5 Water Resistant", price: 49.99, rating: 4.4, reviews: 8931, color: "#f1f5f9", bg: ["#e0f2fe", "#f0f9ff"] },
  { id: "studio", name: "Studio Sport Earbuds with Ear Hooks, Sweatproof, 36h Playtime", price: 79.0, rating: 4.5, reviews: 5402, color: "#0ea5e9", bg: ["#dcfce7", "#f0fdf4"], sponsored: true },
  { id: "aero", name: "Aero Buds Pro — Adaptive Noise Cancelling Wireless Earbuds, 40h Battery", price: 69.99, was: 99.99, rating: 4.7, reviews: 12408, color: "#1f2937", bg: ["#e0e7ff", "#f5f3ff"], badge: "Best seller" },
  { id: "halo", name: "Halo Mini True Wireless Earbuds — Compact Fit, 24h Playtime", price: 39.0, rating: 4.2, reviews: 3120, color: "#fda4af", bg: ["#ffe4e6", "#fff1f2"] },
  { id: "nimbus", name: "Nimbus ANC Earbuds — Hi-Res Audio, Wireless Charging Case", price: 89.99, rating: 4.6, reviews: 2210, color: "#a78bfa", bg: ["#ede9fe", "#faf5ff"] },
  { id: "echo", name: "Echo Lite Wireless Earbuds, Bluetooth 5.3, USB-C", price: 29.99, rating: 4.0, reviews: 15877, color: "#e2e8f0", bg: ["#f1f5f9", "#f8fafc"] },
  { id: "drift", name: "Drift Open-Ear Wireless Earbuds for Running and Cycling", price: 59.99, rating: 4.3, reviews: 1893, color: "#f59e0b", bg: ["#fef3c7", "#fffbeb"] },
  { id: "vibe", name: "Vibe Pods Kids Wireless Earbuds, Volume Limited to 85 dB", price: 24.99, rating: 3.9, reviews: 744, color: "#34d399", bg: ["#d1fae5", "#ecfdf5"] },
];
export const AERO = PRODUCTS[2];

export function urlOf(st: Pick<PState, "key" | "genericUrl">): string {
  switch (st.key) {
    case "home": return `https://${HOST}/`;
    case "search": return `https://${HOST}/s?k=wireless+earbuds`;
    case "product": return `https://${HOST}/p/aero-buds-pro`;
    case "cart": return `https://${HOST}/cart`;
    case "signin": return "https://accounts.northwind.store/signin?return=%2Fcheckout";
    case "checkout": return `https://${HOST}/checkout`;
    case "confirmed": return `https://${HOST}/orders/NW-208-4417`;
    case "generic": return st.genericUrl;
  }
}

export function titleOf(st: Pick<PState, "key" | "genericUrl">): string {
  switch (st.key) {
    case "home": return "Northwind — Everyday tech, thoughtfully picked";
    case "search": return "Northwind.store : wireless earbuds";
    case "product": return "Aero Buds Pro — Adaptive Noise Cancelling Wireless Earbuds : Northwind";
    case "cart": return "Northwind Shopping Cart";
    case "signin": return "Northwind Sign-In";
    case "checkout": return "Northwind Checkout";
    case "confirmed": return "Thank you, your order has been placed";
    case "generic": { try { return new URL(st.genericUrl).hostname; } catch { return st.genericUrl; } }
  }
}

export function keyForUrl(url: string): PageKey | null {
  try {
    const u = new URL(url);
    if (u.hostname === "accounts.northwind.store") return "signin";
    if (u.hostname !== HOST && u.hostname !== "northwind.store") return null;
    if (u.pathname === "/" || u.pathname === "") return "home";
    if (u.pathname.startsWith("/s")) return "search";
    if (u.pathname.startsWith("/p/")) return "product";
    if (u.pathname.startsWith("/cart")) return "cart";
    if (u.pathname.startsWith("/checkout")) return "checkout";
    if (u.pathname.startsWith("/orders")) return "confirmed";
    return "home";
  } catch { return null; }
}

// ---- builder -----------------------------------------------------------------------------------
class B {
  els: El[] = [];
  ops: ((ctx: Ctx) => void)[] = [];
  constructor(public st: PState) {}
  el(e: El) { this.els.push(e); return e; }
  p(fn: (ctx: Ctx) => void) { this.ops.push(fn); }
  hot(id: string) { return this.st.hover === id; }
  down(id: string) { return this.st.pressed === id; }
}

const INK = "#0f172a", MUTED = "#64748b", LINE = "#e2e8f0", LINK = "#2563eb", AMBER = "#ffb020", ORANGE = "#ff8a3d";

function container(vw: number) { const cw = Math.min(vw - 48, 1240); return { cx: Math.round((vw - cw) / 2), cw }; }

function inputBox(b: B, e: El, opts: { radius?: number; big?: boolean } = {}) {
  const st = b.st;
  const focused = st.focus === e.id;
  const v = st.values[e.id] ?? "";
  b.p((ctx) => {
    const r = opts.radius ?? 8;
    if (focused) fillRR(ctx, e.x - 3, e.y - 3, e.w + 6, e.h + 6, r + 3, "rgba(37,99,235,0.16)");
    fillRR(ctx, e.x, e.y, e.w, e.h, r, "#ffffff", focused ? LINK : b.hot(e.id) ? "#94a3b8" : "#cbd5e1", focused ? 1.5 : 1);
    const f = font(opts.big ? 15 : 14);
    const tx = e.x + 12, ty = e.y + e.h / 2 + 5;
    ctx.save();
    ctx.beginPath(); ctx.rect(e.x + 4, e.y, e.w - 8, e.h); ctx.clip();
    let w = 0;
    if (v) w = text(ctx, e.secret ? "•".repeat(v.length) : v, tx, ty, e.secret ? font(16, 600) : f, INK);
    else if (e.placeholder) text(ctx, e.placeholder, tx, ty, f, "#94a3b8");
    if (focused && st.caret) { ctx.fillStyle = LINK; ctx.fillRect(tx + w + 1, e.y + e.h / 2 - 9, 1.5, 18); }
    ctx.restore();
  });
}

function button(b: B, e: El, style: "amber" | "orange" | "dark" | "ghost" | "blue", r = 999) {
  const hot = b.hot(e.id), down = b.down(e.id);
  const fills = { amber: ["#ffb020", "#f5a30b", "#e8950a"], orange: ["#ff8a3d", "#f57a2c", "#e86d20"], dark: ["#0f172a", "#1e293b", "#334155"], ghost: ["#ffffff", "#f8fafc", "#f1f5f9"], blue: ["#2563eb", "#1d4ed8", "#1e40af"] }[style];
  const fg = style === "dark" || style === "blue" ? "#ffffff" : "#1a1206";
  b.p((ctx) => {
    const y = e.y + (down ? 1 : 0);
    if (style !== "ghost") { fillRR(ctx, e.x, y + 1.5, e.w, e.h, r, "rgba(15,23,42,0.10)"); }
    fillRR(ctx, e.x, y, e.w, e.h, r, fills[down ? 2 : hot ? 1 : 0], style === "ghost" ? "#cbd5e1" : undefined);
    text(ctx, e.label, e.x + e.w / 2, y + e.h / 2 + 5, font(14, 600), style === "ghost" ? INK : fg, "center");
  });
}

function header(b: B): number {
  const { vw } = b.st;
  const { cx, cw } = container(vw);
  const H = 60;
  b.p((ctx) => { ctx.fillStyle = "#0f172a"; ctx.fillRect(0, 0, vw, H); });
  const logo = b.el({ id: "logo", kind: "link", x: cx, y: 12, w: 132, h: 36, label: "Northwind", role: "link", href: "home" });
  b.p((ctx) => {
    if (b.hot("logo")) fillRR(ctx, logo.x - 6, logo.y, logo.w + 12, logo.h, 6, "rgba(255,255,255,0.08)");
    ctx.strokeStyle = AMBER; ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(cx + 2, 40); ctx.lineTo(cx + 2, 22); ctx.lineTo(cx + 18, 40); ctx.lineTo(cx + 18, 20); ctx.stroke();
    text(ctx, "northwind", cx + 28, 37, font(20, 700), "#ffffff");
    text(ctx, ".store", cx + 28 + 94, 37, font(13, 500), AMBER);
  });
  const sx = cx + 180, sw = Math.max(260, cw - 180 - 250);
  const q = b.el({ id: "q", kind: "input", x: sx, y: 10, w: sw - 48, h: 40, label: "Search Northwind", role: "searchbox", field: "Search", placeholder: "Search Northwind" });
  b.p((ctx) => {
    const focused = b.st.focus === "q";
    if (focused) fillRR(ctx, q.x - 3, q.y - 3, sw + 6, q.h + 6, 11, "rgba(255,176,32,0.35)");
    fillRR(ctx, q.x, q.y, sw, q.h, 8, "#ffffff");
  });
  b.p((ctx) => {
    const v = b.st.values.q ?? "";
    const tx = q.x + 14, ty = q.y + 25;
    let w = 0;
    ctx.save(); ctx.beginPath(); ctx.rect(q.x, q.y, q.w, q.h); ctx.clip();
    if (v) w = text(ctx, v, tx, ty, font(15), INK); else text(ctx, "Search Northwind", tx, ty, font(15), "#94a3b8");
    if (b.st.focus === "q" && b.st.caret) { ctx.fillStyle = INK; ctx.fillRect(tx + w + 1, q.y + 11, 1.5, 18); }
    ctx.restore();
  });
  const go = b.el({ id: "q-go", kind: "button", x: sx + sw - 48, y: 10, w: 48, h: 40, label: "Go", role: "button", act: "search" });
  b.p((ctx) => {
    ctx.save(); rr(ctx, go.x, go.y, go.w, go.h, 8); ctx.clip();
    ctx.fillStyle = b.down("q-go") ? "#e8950a" : b.hot("q-go") ? "#f5a30b" : AMBER; ctx.fillRect(go.x, go.y, go.w + 8, go.h);
    ctx.restore();
    searchIcon(ctx, go.x + 24, go.y + 20, "#1a1206");
  });
  const acct = b.el({ id: "acct", kind: "link", x: sx + sw + 18, y: 12, w: 130, h: 36, label: b.st.signedIn ? "Account" : "Hello, sign in", role: "link", href: "signin" });
  b.p((ctx) => {
    if (b.hot("acct")) fillRR(ctx, acct.x - 6, acct.y, acct.w, acct.h, 6, "rgba(255,255,255,0.08)");
    text(ctx, b.st.signedIn ? "Hello, Alex" : "Hello, sign in", acct.x, acct.y + 14, font(12), "#cbd5e1");
    text(ctx, "Account & Lists", acct.x, acct.y + 31, font(14, 600), "#ffffff");
  });
  const cart = b.el({ id: "cart", kind: "link", x: cx + cw - 86, y: 12, w: 86, h: 36, label: "Cart", role: "link", href: "cart" });
  b.p((ctx) => {
    if (b.hot("cart")) fillRR(ctx, cart.x - 6, cart.y, cart.w + 6, cart.h, 6, "rgba(255,255,255,0.08)");
    cartIcon(ctx, cart.x + 4, cart.y + 8, "#ffffff");
    const n = b.st.cart;
    fillRR(ctx, cart.x + 16, cart.y + 1, 18, 18, 9, n ? ORANGE : "#475569");
    text(ctx, String(n), cart.x + 25, cart.y + 14, font(11, 700), n ? "#1a1206" : "#e2e8f0", "center");
    text(ctx, "Cart", cart.x + 40, cart.y + 26, font(14, 600), "#ffffff");
  });
  // sub-nav
  b.p((ctx) => {
    ctx.fillStyle = "#1e293b"; ctx.fillRect(0, H, vw, 38);
    let x = cx;
    for (const s of ["All", "Today's Deals", "Audio", "Computers", "Smart Home", "Gift Cards", "Customer Service"]) {
      const w = text(ctx, s, x, H + 24, font(13, s === "All" ? 700 : 500), "rgba(255,255,255,0.88)");
      x += w + 22;
    }
    text(ctx, "Free delivery on orders over $35", cx + cw, H + 24, font(13, 600), AMBER, "right");
  });
  return H + 38;
}

function skeleton(b: B, top: number) {
  const { vw } = b.st;
  const { cx, cw } = container(vw);
  b.p((ctx) => {
    ctx.fillStyle = "#e9edf3";
    for (let i = 0; i < 4; i++) fillRR(ctx, cx + i * (cw / 4) + 8, top + 90, cw / 4 - 16, 260, 12, "#eef1f6");
    fillRR(ctx, cx, top + 30, cw * 0.4, 18, 6, "#e9edf3");
    fillRR(ctx, cx, top + 390, cw * 0.7, 14, 6, "#eef1f6");
    fillRR(ctx, cx, top + 416, cw * 0.5, 14, 6, "#eef1f6");
  });
}

// ---- pages -----------------------------------------------------------------------------------------
function home(b: B): Built {
  const { vw } = b.st;
  const { cx, cw } = container(vw);
  const top = header(b);
  if (b.st.loaded < 0.6) { skeleton(b, top); return done(b, top + 600, "#f1f5f9"); }
  const heroH = 360;
  b.p((ctx) => {
    const g = ctx.createLinearGradient(0, top, vw, top + heroH);
    g.addColorStop(0, "#312e81"); g.addColorStop(0.55, "#4c1d95"); g.addColorStop(1, "#7c3aed");
    ctx.fillStyle = g; ctx.fillRect(0, top, vw, heroH);
    text(ctx, "NEW · AERO BUDS PRO", cx + 8, top + 96, font(13, 700), "#c4b5fd");
    text(ctx, "Sound that moves", cx + 8, top + 150, font(46, 700), "#ffffff");
    text(ctx, "with you.", cx + 8, top + 204, font(46, 700), "#ffffff");
    text(ctx, "Adaptive noise cancelling. 40 hours of battery. From $69.99.", cx + 8, top + 244, font(17), "#e9d5ff");
    earbuds(ctx, cx + cw - 460, top + 30, 420, 300, "#1f2937", ["#a78bfa", "#6d28d9"], 24);
  });
  const shop = b.el({ id: "hero-shop", kind: "button", x: cx + 8, y: top + 270, w: 150, h: 44, label: "Shop earbuds", role: "button", href: "search" });
  button(b, shop, "amber");
  const tiles = ["Audio", "Laptops", "Smart home", "Accessories"];
  const tw = (cw - 3 * 20) / 4;
  const ty = top + heroH + 32;
  b.p((ctx) => text(ctx, "Shop by category", cx, ty, font(20, 700), INK));
  tiles.forEach((t, i) => {
    const x = cx + i * (tw + 20);
    const e = b.el({ id: `tile-${i}`, kind: "link", x, y: ty + 18, w: tw, h: 220, label: t, role: "link", href: "search" });
    b.p((ctx) => {
      fillRR(ctx, e.x, e.y, e.w, e.h, 14, "#ffffff", b.hot(e.id) ? "#94a3b8" : LINE);
      const p = PRODUCTS[(i * 3) % PRODUCTS.length];
      earbuds(ctx, e.x + 12, e.y + 12, e.w - 24, 150, p.color, p.bg, 10);
      text(ctx, t, e.x + 16, e.y + 196, font(16, 600), INK);
    });
  });
  return done(b, ty + 280, "#f1f5f9");
}

function search(b: B): Built {
  const { vw } = b.st;
  const { cx, cw } = container(vw);
  const top = header(b);
  if (b.st.loaded < 0.6) { skeleton(b, top); return done(b, top + 700, "#ffffff"); }
  b.p((ctx) => {
    ctx.fillStyle = "#ffffff"; ctx.fillRect(0, top, vw, 44);
    ctx.fillStyle = LINE; ctx.fillRect(0, top + 43, vw, 1);
    const w = text(ctx, "1-24 of over 2,000 results for ", cx, top + 28, font(14), INK);
    text(ctx, "“wireless earbuds”", cx + w, top + 28, font(14, 700), "#c2410c");
    fillRR(ctx, cx + cw - 190, top + 8, 190, 28, 8, "#f8fafc", "#cbd5e1");
    text(ctx, "Sort by: Featured  ▾", cx + cw - 178, top + 27, font(13), INK);
  });
  // filters
  const fx = cx, fy = top + 64, fw = 210;
  b.p((ctx) => {
    let y = fy;
    const section = (h: string, items: string[]) => {
      text(ctx, h, fx, y + 14, font(14, 700), INK);
      y += 28;
      for (const it of items) {
        fillRR(ctx, fx, y - 12, 14, 14, 3, "#ffffff", "#94a3b8");
        text(ctx, it, fx + 22, y, font(13), INK, "left", fw - 24);
        y += 24;
      }
      y += 14;
    };
    section("Delivery", ["Free delivery", "Get it by tomorrow"]);
    text(ctx, "Customer reviews", fx, y + 14, font(14, 700), INK); y += 32;
    stars(ctx, fx, y + 2, 4, 14); text(ctx, "& Up", fx + 80, y + 2, font(13), INK); y += 38;
    section("Price", ["Under $25", "$25 to $50", "$50 to $100", "$100 & above"]);
    section("Brand", ["Aero", "Pulse", "Halo", "Nimbus", "Studio"]);
    section("Features", ["Active noise cancelling", "Wireless charging", "Water resistant"]);
  });
  // grid
  const gx = cx + fw + 28, gw = cw - fw - 28;
  const cols = gw >= 900 ? 4 : 3;
  const gap = 18;
  const cardW = (gw - gap * (cols - 1)) / cols;
  const imgH = Math.round(cardW * 0.82);
  const cardH = imgH + 168;
  PRODUCTS.forEach((p, i) => {
    const x = gx + (i % cols) * (cardW + gap), y = fy + Math.floor(i / cols) * (cardH + gap);
    const e = b.el({ id: `p-${p.id}`, kind: "link", x: x + 12, y: y + imgH + 16, w: cardW - 24, h: 42, label: p.name.split(" — ")[0].split(" Wireless")[0], role: "link", href: p.id === "aero" ? "product" : undefined });
    const img = b.el({ id: `img-${p.id}`, kind: "link", x, y, w: cardW, h: imgH, label: p.name.split(" — ")[0], role: "link", href: p.id === "aero" ? "product" : undefined });
    b.p((ctx) => {
      const hot = b.hot(e.id) || b.hot(img.id);
      fillRR(ctx, x, y, cardW, cardH, 12, "#ffffff", hot ? "#cbd5e1" : LINE);
      if (hot) { ctx.save(); rr(ctx, x, y, cardW, cardH, 12); ctx.shadowColor = "rgba(15,23,42,0.10)"; ctx.shadowBlur = 16; ctx.shadowOffsetY = 4; ctx.strokeStyle = "rgba(0,0,0,0)"; ctx.stroke(); ctx.restore(); }
      earbuds(ctx, x + 1, y + 1, cardW - 2, imgH, p.color, p.bg, 11);
      if (p.badge) { fillRR(ctx, x + 12, y + 12, 86, 22, 6, "#c2410c"); text(ctx, p.badge, x + 55, y + 27, font(11, 700), "#ffffff", "center"); }
      if (p.sponsored) text(ctx, "Sponsored", x + 12, y + imgH + 14, font(11), MUTED);
      const ty = wrap(ctx, p.name, x + 12, y + imgH + (p.sponsored ? 32 : 28), cardW - 24, 19, font(14, 500), hot && b.hot(e.id) ? "#c2410c" : INK, 2);
      stars(ctx, x + 12, ty + 8, p.rating, 13);
      text(ctx, p.reviews.toLocaleString("en-US"), x + 88, ty + 8, font(12.5), LINK);
      const dollars = Math.floor(p.price), cents = Math.round((p.price - dollars) * 100);
      text(ctx, "$", x + 12, ty + 36, font(13, 500), INK);
      const dw = text(ctx, String(dollars), x + 21, ty + 42, font(26, 600), INK);
      text(ctx, String(cents).padStart(2, "0"), x + 23 + dw, ty + 30, font(13, 600), INK);
      if (p.was) text(ctx, `List: $${p.was.toFixed(2)}`, x + 50 + dw, ty + 40, font(12.5), MUTED);
      text(ctx, "FREE delivery ", x + 12, ty + 64, font(12.5), INK);
      text(ctx, "Tomorrow", x + 98, ty + 64, font(12.5, 700), INK);
    });
  });
  const rows = Math.ceil(PRODUCTS.length / cols);
  const end = fy + rows * (cardH + gap) + 40;
  b.p((ctx) => {
    ctx.fillStyle = "#f8fafc"; ctx.fillRect(0, end, vw, 120);
    text(ctx, "Need help? Visit the help section or contact us", vw / 2, end + 64, font(13), MUTED, "center");
  });
  return done(b, Math.max(end + 120, fy + 1160), "#ffffff");
}

function product(b: B): Built {
  const { vw, colour } = b.st;
  const { cx, cw } = container(vw);
  const top = header(b);
  if (b.st.loaded < 0.6) { skeleton(b, top); return done(b, top + 700, "#ffffff"); }
  const hex = COLOURS.find((c) => c.name === colour)?.hex ?? AERO.color;
  const y0 = top + 20;
  b.p((ctx) => text(ctx, "Electronics  ›  Headphones  ›  Earbuds", cx, y0 + 12, font(12.5), MUTED));
  const imgW = Math.min(520, cw * 0.42);
  const ix = cx + 72, iy = y0 + 32;
  b.p((ctx) => {
    for (let i = 0; i < 4; i++) {
      fillRR(ctx, cx, iy + i * 70, 56, 56, 8, "#ffffff", i === 0 ? ORANGE : LINE, i === 0 ? 2 : 1);
      earbuds(ctx, cx + 3, iy + i * 70 + 3, 50, 50, hex, i % 2 ? ["#f1f5f9", "#ffffff"] : AERO.bg, 6);
    }
    earbuds(ctx, ix, iy, imgW, imgW, hex, AERO.bg, 16);
  });
  const rx = ix + imgW + 36, rw = cx + cw - rx;
  const y0t = iy + 4;
  b.p((ctx) => {
    const ty = wrap(ctx, "Aero Buds Pro — Adaptive Noise Cancelling Wireless Earbuds with 40h Battery, Wireless Charging Case, Multipoint Bluetooth 5.4", rx, y0t + 22, rw, 30, font(23, 600), INK, 3);
    text(ctx, "Visit the Aero Store", rx, ty + 8, font(13.5), LINK);
    stars(ctx, rx, ty + 36, 4.7, 15);
    text(ctx, "4.7   12,408 ratings", rx + 88, ty + 36, font(13.5), LINK);
    fillRR(ctx, rx, ty + 50, 150, 22, 5, "#c2410c");
    text(ctx, "#1 Best Seller in Earbuds", rx + 8, ty + 65, font(11.5, 700), "#ffffff", "left", 140);
    ctx.fillStyle = LINE; ctx.fillRect(rx, ty + 86, rw, 1);
    text(ctx, "-30%", rx, ty + 126, font(26, 400), "#cc0c39");
    text(ctx, "$", rx + 66, ty + 112, font(14, 500), INK);
    text(ctx, "69", rx + 76, ty + 126, font(32, 600), INK);
    text(ctx, "99", rx + 114, ty + 110, font(14, 600), INK);
    text(ctx, "List Price: $99.99", rx, ty + 150, font(13), MUTED);
    text(ctx, `Colour: `, rx, ty + 186, font(14), INK);
    text(ctx, colour, rx + 54, ty + 186, font(14, 700), INK);
  });
  const ys = iy + 4 + 30 * 3 + 22 + 8 + 186;
  // swatches (clickable)
  COLOURS.forEach((c, i) => {
    const e = b.el({ id: `sw-${c.name}`, kind: "button", x: rx + i * 70, y: ys + 14, w: 60, h: 60, label: c.name, role: "radio", act: `colour:${c.name}` });
    b.p((ctx) => {
      const on = colour === c.name;
      fillRR(ctx, e.x, e.y, e.w, e.h, 10, "#ffffff", on ? "#c2410c" : b.hot(e.id) ? "#94a3b8" : LINE, on ? 2 : 1);
      earbuds(ctx, e.x + 4, e.y + 4, e.w - 8, e.h - 8, c.hex, AERO.bg, 7);
    });
  });
  const y = ys + 96;
  // buy box
  const bx = rx, bw = Math.min(rw, 360);
  b.p((ctx) => {
    fillRR(ctx, bx, y, bw, 262, 12, "#ffffff", LINE);
    text(ctx, "$69.99", bx + 18, y + 36, font(22, 600), INK);
    text(ctx, "FREE delivery ", bx + 18, y + 62, font(13.5), INK);
    text(ctx, "Tomorrow, 9 AM – 1 PM", bx + 110, y + 62, font(13.5, 700), INK);
    text(ctx, "In Stock", bx + 18, y + 90, font(16, 500), "#15803d");
  });
  const qty = b.el({ id: "qty", kind: "select", x: bx + 18, y: y + 104, w: 120, h: 32, label: `Quantity: ${b.st.qty}`, role: "combobox", field: "Quantity", options: ["1", "2", "3", "4", "5"] });
  b.p((ctx) => {
    fillRR(ctx, qty.x, qty.y, qty.w, qty.h, 8, b.hot("qty") ? "#e2e8f0" : "#f1f5f9", "#cbd5e1");
    text(ctx, `Qty: ${b.st.qty}`, qty.x + 12, qty.y + 21, font(13.5), INK);
    text(ctx, "▾", qty.x + qty.w - 20, qty.y + 21, font(12), INK);
  });
  const add = b.el({ id: "add", kind: "button", x: bx + 18, y: y + 150, w: bw - 36, h: 40, label: "Add to cart", role: "button", act: "add" });
  button(b, add, "amber");
  const buy = b.el({ id: "buy", kind: "button", x: bx + 18, y: y + 200, w: bw - 36, h: 40, label: "Buy now", role: "button", href: "signin" });
  button(b, buy, "orange");
  const about = Math.max(y + 290, iy + imgW + 40);
  b.p((ctx) => {
    ctx.fillStyle = LINE; ctx.fillRect(cx, about, cw, 1);
    text(ctx, "About this item", cx, about + 40, font(20, 700), INK);
    const bullets = [
      "ADAPTIVE NOISE CANCELLING: tunes itself 50,000 times a second to the world around you, so music stays clear on trains, planes and busy streets.",
      "40 HOURS OF PLAY: 8 hours per charge plus 32 more in the case; ten minutes of charging gives two hours of listening.",
      "COMFORT ALL DAY: four sizes of memory-foam tips and a 4.8 g bud that stays put through workouts.",
      "MULTIPOINT: connect your phone and laptop at the same time and switch without re-pairing.",
      "WHAT'S IN THE BOX: Aero Buds Pro, wireless charging case, USB-C cable, four pairs of ear tips.",
    ];
    let by = about + 72;
    for (const t of bullets) {
      ctx.fillStyle = INK; ctx.beginPath(); ctx.arc(cx + 6, by - 5, 2.5, 0, Math.PI * 2); ctx.fill();
      by = wrap(ctx, t, cx + 18, by, Math.min(cw - 18, 900), 21, font(14.5), INK, 3) + 8;
    }
    text(ctx, "Customer reviews", cx, by + 40, font(20, 700), INK);
    stars(ctx, cx, by + 70, 4.7, 18);
    text(ctx, "4.7 out of 5", cx + 110, by + 70, font(15, 500), INK);
    const bars = [78, 13, 5, 2, 2];
    bars.forEach((pc, i) => {
      const yy = by + 100 + i * 28;
      text(ctx, `${5 - i} star`, cx, yy + 13, font(13.5), LINK);
      fillRR(ctx, cx + 56, yy, 260, 18, 4, "#f1f5f9", "#cbd5e1");
      fillRR(ctx, cx + 56, yy, (260 * pc) / 100, 18, 4, "#f59e0b");
      text(ctx, `${pc}%`, cx + 328, yy + 13, font(13.5), LINK);
    });
  });
  return done(b, about + 720, "#ffffff");
}

function cart(b: B): Built {
  const { vw } = b.st;
  const { cx, cw } = container(vw);
  const top = header(b);
  if (b.st.loaded < 0.6) { skeleton(b, top); return done(b, top + 600, "#eaeded"); }
  const y0 = top + 24;
  if (b.st.banner) {
    b.p((ctx) => {
      fillRR(ctx, cx, y0, cw, 64, 12, "#ffffff", "#bbf7d0");
      check(ctx, cx + 34, y0 + 32, 13, "#16a34a");
      text(ctx, "Added to cart", cx + 58, y0 + 38, font(18, 700), "#15803d");
    });
  }
  const y = b.st.banner ? y0 + 84 : y0;
  const lw = cw - 320 - 20;
  const n = b.st.cart;
  b.p((ctx) => {
    fillRR(ctx, cx, y, lw, n ? 250 : 120, 12, "#ffffff");
    text(ctx, n ? "Shopping Cart" : "Your Northwind Cart is empty", cx + 24, y + 44, font(26, 500), INK);
    if (n) {
      text(ctx, "Price", cx + lw - 24, y + 70, font(13), MUTED, "right");
      ctx.fillStyle = LINE; ctx.fillRect(cx + 24, y + 80, lw - 48, 1);
      const hex = COLOURS.find((c) => c.name === b.st.colour)?.hex ?? AERO.color;
      earbuds(ctx, cx + 24, y + 96, 132, 132, hex, AERO.bg, 10);
      wrap(ctx, "Aero Buds Pro — Adaptive Noise Cancelling Wireless Earbuds, 40h Battery", cx + 176, y + 116, lw - 330, 22, font(17, 500), INK, 2);
      text(ctx, "In Stock", cx + 176, y + 170, font(13), "#15803d");
      text(ctx, `Colour: ${b.st.colour}`, cx + 176, y + 192, font(13), INK);
      text(ctx, "$69.99", cx + lw - 24, y + 118, font(18, 700), INK, "right");
    }
  });
  if (n) {
    const del = b.el({ id: "del", kind: "link", x: cx + 176 + 110, y: y + 202, w: 50, h: 24, label: "Delete", role: "link", act: "delete" });
    b.p((ctx) => {
      fillRR(ctx, cx + 176, y + 204, 92, 26, 13, "#ffffff", "#ffb020", 2);
      text(ctx, `–  ${b.st.qty}  +`, cx + 222, y + 222, font(13, 600), INK, "center");
      text(ctx, "Delete", del.x + 4, del.y + 17, font(13), b.hot("del") ? "#c2410c" : LINK);
    });
  }
  const sx = cx + lw + 20;
  b.p((ctx) => {
    fillRR(ctx, sx, y, 320, n ? 170 : 90, 12, "#ffffff");
    if (n) {
      check(ctx, sx + 30, y + 30, 9, "#16a34a");
      text(ctx, "Your order qualifies for FREE delivery.", sx + 46, y + 35, font(12.5), "#15803d", "left", 260);
      text(ctx, `Subtotal (${n} item${n > 1 ? "s" : ""}): `, sx + 20, y + 72, font(17), INK);
      text(ctx, "$69.99", sx + 300, y + 72, font(17, 700), INK, "right");
    } else text(ctx, "Subtotal (0 items): $0.00", sx + 20, y + 52, font(16), INK);
  });
  if (n) {
    const go = b.el({ id: "checkout", kind: "button", x: sx + 20, y: y + 96, w: 280, h: 42, label: "Proceed to checkout", role: "button", href: "signin" });
    button(b, go, "amber");
  }
  return done(b, Math.max(y + 480, top + 700), "#eaeded");
}

function signin(b: B): Built {
  const { vw } = b.st;
  const w = 380, x = Math.round((vw - w) / 2);
  const y = 40;
  b.p((ctx) => {
    ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 3.5; ctx.lineCap = "round"; ctx.lineJoin = "round";
    const lx = vw / 2 - 62;
    ctx.beginPath(); ctx.moveTo(lx, y + 30); ctx.lineTo(lx, y + 10); ctx.lineTo(lx + 16, y + 30); ctx.lineTo(lx + 16, y + 8); ctx.stroke();
    text(ctx, "northwind", lx + 26, y + 28, font(22, 700), INK);
  });
  if (b.st.loaded < 0.6) {
    b.p((ctx) => fillRR(ctx, x, y + 64, w, 400, 12, "#f8fafc", LINE));
    return done(b, 800, "#ffffff");
  }
  const cy = y + 64;
  b.p((ctx) => {
    fillRR(ctx, x, cy, w, b.st.signinError ? 468 : 432, 12, "#ffffff", "#d5d9e0");
    text(ctx, "Sign in", x + 26, cy + 50, font(27, 500), INK);
  });
  const ey = cy + 76;
  if (b.st.signinError) {
    b.p((ctx) => {
      fillRR(ctx, x + 26, ey, w - 52, 30, 6, "#fef2f2", "#fecaca");
      text(ctx, "Enter your email and password", x + 38, ey + 20, font(13, 500), "#b91c1c");
    });
  }
  const fy = b.st.signinError ? ey + 40 : ey;
  b.p((ctx) => text(ctx, "Email or mobile phone number", x + 26, fy + 12, font(13.5, 700), INK));
  const email = b.el({ id: "email", kind: "input", x: x + 26, y: fy + 20, w: w - 52, h: 36, label: "Email or mobile phone number", role: "textbox", field: "Email" });
  inputBox(b, email, { radius: 6 });
  b.p((ctx) => text(ctx, "Password", x + 26, fy + 86, font(13.5, 700), INK));
  const forgot = b.el({ id: "forgot", kind: "link", x: x + w - 26 - 130, y: fy + 72, w: 130, h: 20, label: "Forgot your password?", role: "link" });
  b.p((ctx) => text(ctx, "Forgot your password?", forgot.x + forgot.w, forgot.y + 14, font(13), b.hot("forgot") ? "#c2410c" : LINK, "right"));
  const pw = b.el({ id: "password", kind: "input", x: x + 26, y: fy + 94, w: w - 52, h: 36, label: "Password", role: "textbox", field: "Password", secret: true });
  inputBox(b, pw, { radius: 6 });
  const go = b.el({ id: "signin-go", kind: "button", x: x + 26, y: fy + 148, w: w - 52, h: 38, label: "Sign in", role: "button", act: "signin" });
  button(b, go, "amber", 8);
  const keep = b.el({ id: "keep", kind: "check", x: x + 26, y: fy + 200, w: 170, h: 20, label: "Keep me signed in", role: "checkbox", act: "keep" });
  b.p((ctx) => {
    fillRR(ctx, keep.x, keep.y + 2, 16, 16, 3, b.st.keepSigned ? LINK : "#ffffff", b.st.keepSigned ? LINK : "#94a3b8");
    if (b.st.keepSigned) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(keep.x + 4, keep.y + 10); ctx.lineTo(keep.x + 7, keep.y + 13); ctx.lineTo(keep.x + 12, keep.y + 6); ctx.stroke(); }
    text(ctx, "Keep me signed in", keep.x + 24, keep.y + 15, font(13.5), INK);
    wrap(ctx, "By continuing, you agree to Northwind's Conditions of Use and Privacy Notice.", x + 26, fy + 252, w - 52, 18, font(12), MUTED, 2);
    ctx.fillStyle = LINE; ctx.fillRect(x + 26, fy + 296, w - 52, 1);
    text(ctx, "New to Northwind?", vw / 2, fy + 322, font(12.5), MUTED, "center");
  });
  const create = b.el({ id: "create", kind: "button", x: x, y: cy + (b.st.signinError ? 488 : 452), w, h: 34, label: "Create your Northwind account", role: "button" });
  button(b, create, "ghost", 8);
  b.p((ctx) => {
    ctx.fillStyle = "#f8fafc"; ctx.fillRect(0, 700, vw, 200);
    text(ctx, "Conditions of Use     Privacy Notice     Help", vw / 2, 744, font(12), LINK, "center");
    text(ctx, "© 2026, Northwind (a fictional store in NEO's mock browser)", vw / 2, 768, font(11.5), MUTED, "center");
  });
  return done(b, 900, "#ffffff");
}

function checkout(b: B): Built {
  const { vw } = b.st;
  const { cx, cw } = container(vw);
  b.p((ctx) => {
    const g = ctx.createLinearGradient(0, 0, 0, 64);
    g.addColorStop(0, "#f8fafc"); g.addColorStop(1, "#eef2f7");
    ctx.fillStyle = g; ctx.fillRect(0, 0, vw, 64);
    ctx.fillStyle = LINE; ctx.fillRect(0, 63, vw, 1);
    ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(cx + 2, 42); ctx.lineTo(cx + 2, 24); ctx.lineTo(cx + 18, 42); ctx.lineTo(cx + 18, 22); ctx.stroke();
    text(ctx, "northwind", cx + 28, 39, font(20, 700), INK);
    text(ctx, "Checkout (1 item)", vw / 2, 40, font(22, 500), INK, "center");
    lockIcon(ctx, cx + cw - 16, 22, "#475569");
  });
  if (b.st.loaded < 0.6) { skeleton(b, 64); return done(b, 760, "#ffffff"); }
  const lw = cw - 340 - 28, y = 96;
  const sec = (n: number, title: string, lines: string[], yy: number, h: number) => {
    b.p((ctx) => {
      fillRR(ctx, cx, yy, lw, h, 12, "#ffffff", LINE);
      text(ctx, `${n}`, cx + 24, yy + 36, font(18, 700), INK);
      text(ctx, title, cx + 54, yy + 36, font(18, 700), INK);
      lines.forEach((l, i) => text(ctx, l, cx + 280, yy + 34 + i * 21, font(14), INK, "left", lw - 380));
    });
  };
  sec(1, "Delivery address", ["Alex Rivera", "221 Market Street, Apt 4", "San Francisco, CA 94105"], y, 110);
  const ch1 = b.el({ id: "chg-addr", kind: "link", x: cx + lw - 90, y: y + 20, w: 66, h: 22, label: "Change", role: "link" });
  sec(2, "Payment method", ["Visa ending in 4242", "Billing address: same as delivery"], y + 126, 90);
  const ch2 = b.el({ id: "chg-pay", kind: "link", x: cx + lw - 90, y: y + 146, w: 66, h: 22, label: "Change", role: "link" });
  b.p((ctx) => {
    for (const c of [ch1, ch2]) text(ctx, "Change", c.x + c.w, c.y + 16, font(13.5), b.hot(c.id) ? "#c2410c" : LINK, "right");
    const yy = y + 232;
    fillRR(ctx, cx, yy, lw, 190, 12, "#ffffff", LINE);
    text(ctx, "3", cx + 24, yy + 36, font(18, 700), INK);
    text(ctx, "Review items and delivery", cx + 54, yy + 36, font(18, 700), INK);
    text(ctx, "Arriving tomorrow", cx + 24, yy + 72, font(15, 700), "#15803d");
    const hex = COLOURS.find((c) => c.name === b.st.colour)?.hex ?? AERO.color;
    earbuds(ctx, cx + 24, yy + 86, 88, 88, hex, AERO.bg, 8);
    wrap(ctx, "Aero Buds Pro — Adaptive Noise Cancelling Wireless Earbuds", cx + 128, yy + 104, lw - 170, 20, font(14.5, 600), INK, 2);
    text(ctx, `$69.99 · Colour: ${b.st.colour} · Qty: ${b.st.qty}`, cx + 128, yy + 150, font(13.5), MUTED);
  });
  const sx = cx + lw + 28;
  const place = b.el({ id: "place", kind: "button", x: sx + 20, y: y + 20, w: 300, h: 44, label: "Place your order", role: "button", act: "place" });
  b.p((ctx) => fillRR(ctx, sx, y, 340, 330, 12, "#ffffff", LINE));
  button(b, place, "orange", 10);
  b.p((ctx) => {
    wrap(ctx, "By placing your order, you agree to Northwind's privacy notice and conditions of use.", sx + 20, y + 90, 300, 17, font(11.5), MUTED, 2);
    ctx.fillStyle = LINE; ctx.fillRect(sx + 20, y + 124, 300, 1);
    text(ctx, "Order Summary", sx + 20, y + 154, font(17, 700), INK);
    const rows: [string, string][] = [["Items:", "$69.99"], ["Delivery:", "$0.00"], ["Total before tax:", "$69.99"], ["Estimated tax:", "$6.21"]];
    rows.forEach(([k, v], i) => { text(ctx, k, sx + 20, y + 184 + i * 22, font(13.5), INK); text(ctx, v, sx + 320, y + 184 + i * 22, font(13.5), INK, "right"); });
    ctx.fillStyle = LINE; ctx.fillRect(sx + 20, y + 270, 300, 1);
    text(ctx, "Order total:", sx + 20, y + 304, font(19, 700), "#b12704");
    text(ctx, "$76.20", sx + 320, y + 304, font(19, 700), "#b12704", "right");
  });
  return done(b, 760, "#ffffff");
}

function confirmed(b: B): Built {
  const { vw } = b.st;
  const { cw } = container(vw);
  const top = header(b);
  if (b.st.loaded < 0.6) { skeleton(b, top); return done(b, top + 600, "#f1f5f9"); }
  const w = Math.min(cw, 760), x = Math.round((vw - w) / 2), y = top + 36;
  b.p((ctx) => {
    fillRR(ctx, x, y, w, 300, 16, "#ffffff", "#bbf7d0");
    check(ctx, x + 56, y + 62, 24, "#16a34a");
    text(ctx, "Order placed, thank you!", x + 96, y + 70, font(26, 700), "#15803d");
    text(ctx, "A confirmation will be sent to your email.", x + 96, y + 100, font(15), INK);
    ctx.fillStyle = LINE; ctx.fillRect(x + 32, y + 130, w - 64, 1);
    text(ctx, "Arriving tomorrow by 9 PM", x + 32, y + 170, font(18, 700), INK);
    text(ctx, "Order # NW-208-4417", x + 32, y + 198, font(14), MUTED);
    const hex = COLOURS.find((c) => c.name === b.st.colour)?.hex ?? AERO.color;
    earbuds(ctx, x + w - 190, y + 150, 150, 120, hex, AERO.bg, 10);
    text(ctx, "Aero Buds Pro · " + b.st.colour, x + 32, y + 240, font(15, 600), INK);
    text(ctx, "Delivering to Alex Rivera, San Francisco, CA", x + 32, y + 264, font(13.5), MUTED);
  });
  const more = b.el({ id: "keep-shopping", kind: "button", x: x, y: y + 330, w: 200, h: 42, label: "Continue shopping", role: "button", href: "home" });
  button(b, more, "ghost");
  return done(b, y + 460, "#f1f5f9");
}

function generic(b: B): Built {
  const vw = b.st.vw;
  let host = b.st.genericUrl;
  try { host = new URL(b.st.genericUrl).hostname; } catch { /* keep */ }
  b.p((ctx) => {
    text(ctx, host, vw / 2, 190, font(40, 700), INK, "center", vw - 80);
    text(ctx, "A stand-in page from NEO's mock browser (dev only).", vw / 2, 232, font(16), MUTED, "center");
    text(ctx, b.st.genericUrl, vw / 2, 262, font(13), "#94a3b8", "center", vw - 80);
  });
  const back = b.el({ id: "gen-home", kind: "button", x: vw / 2 - 110, y: 300, w: 220, h: 42, label: "Go to northwind.store", role: "button", href: "home" });
  button(b, back, "dark");
  return done(b, 700, "#ffffff");
}

function done(b: B, docH: number, bg: string): Built {
  return { els: b.els, docH, bg, paint: (ctx) => { for (const op of b.ops) op(ctx); } };
}

export function build(st: PState): Built {
  const b = new B(st);
  switch (st.key) {
    case "home": return home(b);
    case "search": return search(b);
    case "product": return product(b);
    case "cart": return cart(b);
    case "signin": return signin(b);
    case "checkout": return checkout(b);
    case "confirmed": return confirmed(b);
    case "generic": return generic(b);
  }
}

/** Paint the visible viewport into `ctx` (already sized vw×dpr by vh×dpr). */
export function render(ctx: Ctx, st: PState, dpr: number, built = build(st)) {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = built.bg;
  ctx.fillRect(0, 0, st.vw, st.vh);
  ctx.save();
  ctx.translate(0, -st.scrollY);
  built.paint(ctx);
  ctx.restore();
  // Chromium-style overlay scrollbar while the page is taller than the view.
  if (built.docH > st.vh) {
    const h = Math.max(40, (st.vh / built.docH) * st.vh), y = (st.scrollY / (built.docH - st.vh)) * (st.vh - h);
    fillRR(ctx, st.vw - 9, y + 2, 6, h - 4, 3, "rgba(15,23,42,0.28)");
  }
  void FONT;
}

export function hit(built: Built, x: number, docY: number): El | null {
  for (let i = built.els.length - 1; i >= 0; i--) {
    const e = built.els[i];
    if (x >= e.x && x <= e.x + e.w && docY >= e.y && docY <= e.y + e.h) return e;
  }
  return null;
}
