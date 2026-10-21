// Search Screen's card in the notch: an area's results in NotchHub's panel, instead of the sheet
// beside the selection — the same results and controls (./Sheet.tsx's SheetBody), on the island's
// black. The page shows it in place of the chat while it is live (App.tsx), and asks NotchHub for
// room for it: it measures how tall its content would be (`onNatural`, at every change — a forced
// layout, so it works before NotchHub's panel is open), and App sizes the panel from that.
//
// Keys (NotchHub gives the panel the keyboard without taking it from the user's app): the question
// box takes them when the panel opens; Return asks, ⌘Return continues in NEO; ⌘C copies the area's
// text unless something is selected or the box has the keys (as the key, or — when a host sends the
// key straight to WebKit as its Copy command — as the copy that follows); a printable key goes into
// the box; Esc closes the card and folds the panel (NotchHub folds on Esc too).
//
// Closed, it stays as it was while NotchHub's panel folds it away (./card.ts `folding`): no key, click
// or copy acts on it, and it never measures itself again — the panel keeps its size to the end.

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { SheetBody } from "./Sheet";
import { naturalHeight, type CardSnap, type LensCardHub } from "./card";
import { hideWindow, listen } from "../shell";
import "./lens.css";

/** How tall the card's content would be with room for all of it (./card.ts naturalHeight): the
 *  fixed rows (question box, footer) plus the scrolling column's children — measured themselves, as
 *  the column flexes to whatever room the panel gives it now. */
function natural(root: HTMLElement, scroll: HTMLElement): number {
  let bottom: number | null = null;
  for (const el of Array.from(scroll.children)) {
    const b = el.getBoundingClientRect().bottom;
    if (bottom === null || b > bottom) bottom = b;
  }
  const cs = getComputedStyle(scroll);
  return naturalHeight({
    rootH: root.offsetHeight, scrollH: scroll.offsetHeight, scrollTop: scroll.getBoundingClientRect().top, scrolled: scroll.scrollTop,
    padTop: parseFloat(cs.paddingTop) || 0, padBottom: parseFloat(cs.paddingBottom) || 0, contentBottom: bottom,
  });
}

/** ⌘C's own copy: the area's text, when nothing is selected and the question box hasn't the keys. */
function copiesArea(composer: HTMLInputElement | null): boolean {
  return !(composer && document.activeElement === composer) && !window.getSelection()?.toString();
}

export function LensCard({ hub, snap, onNatural }: { hub: LensCardHub; snap: CardSnap; onNatural: (h: number) => void }) {
  const [q, setQ] = useState("");
  const root = useRef<HTMLDivElement>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLInputElement>(null);

  const measure = useCallback(() => {
    if (hub.folding) return; // (folding away: the panel keeps its size to the end)
    if (root.current && scroll.current) onNatural(natural(root.current, scroll.current));
  }, [hub, onNatural]);
  // Every change of the results, measured before it paints (the size request reaches NotchHub
  // before its panel opens); later changes of size (a section unfolded, a font in) by observing the
  // column's children — and a child coming or going (the inline language list) re-observes them.
  useLayoutEffect(measure, [measure, snap.v, q]);
  useEffect(() => {
    const sc = scroll.current;
    if (!sc) return;
    const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => measure());
    const watch = () => {
      if (!ro) return;
      ro.disconnect();
      ro.observe(sc);
      for (const el of Array.from(sc.children)) ro.observe(el);
    };
    watch();
    const mo = typeof MutationObserver === "undefined" ? null : new MutationObserver(() => { watch(); measure(); });
    mo?.observe(sc, { childList: true });
    return () => { ro?.disconnect(); mo?.disconnect(); };
  }, [measure, snap.v]);

  // A new card starts with an empty question box.
  useEffect(() => { setQ(""); }, [snap.cid]);

  // NotchHub opened its panel for it (and gave it the keys): the question box takes them.
  useEffect(() => {
    const un = listen("neo://show", () => composer.current?.focus());
    return () => { un.then((f) => f()).catch(() => {}); };
  }, []);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (hub.folding) return; // (closed: the panel is folding it away)
      const inBox = document.activeElement === composer.current && !!composer.current;
      if (e.isComposing || e.keyCode === 229) return; // (the input method's own keys)
      if (e.key === "Escape") {
        e.preventDefault();
        if (hub.close("esc")) void hideWindow(); // (NotchHub folds its panel on Esc as well)
        return;
      }
      if (e.metaKey && !e.shiftKey && !e.altKey && !e.ctrlKey && e.key.toLowerCase() === "c" && copiesArea(composer.current)) {
        const copy = hub.snap()?.r.local?.actions.find((a) => a.kind === "copy_text");
        if (copy) { e.preventDefault(); hub.op({ t: "chip", kind: "copy_text", id: copy.id }); }
        return;
      }
      const onControl = !!(document.activeElement && document.activeElement !== document.body && document.activeElement.closest?.("button, [role=listbox]"));
      if (!inBox && !onControl && e.key.length === 1 && e.key !== " " && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        composer.current?.focus();
        setQ((v) => (v + e.key).slice(0, 500));
      }
    };
    // A host that sends ⌘C to WebKit as its Copy command (no keydown reaches the page): with nothing
    // selected the command is enabled only when the page says it copies (beforecopy), and the copy
    // itself is the area's text.
    const beforeCopy = (e: Event) => { if (!hub.folding && copiesArea(composer.current)) e.preventDefault(); };
    const copy = (e: Event) => {
      if (hub.folding || !copiesArea(composer.current)) return;
      const act = hub.snap()?.r.local?.actions.find((a) => a.kind === "copy_text");
      if (!act) return;
      e.preventDefault();
      hub.op({ t: "chip", kind: "copy_text", id: act.id });
    };
    window.addEventListener("keydown", down);
    document.addEventListener("beforecopy", beforeCopy);
    document.addEventListener("copy", copy);
    return () => {
      window.removeEventListener("keydown", down);
      document.removeEventListener("beforecopy", beforeCopy);
      document.removeEventListener("copy", copy);
    };
  }, [hub]);

  return (
    <div className="lxc" ref={root} role="dialog" aria-label="Search Screen results" key={snap.cid}>
      <SheetBody
        variant="card"
        r={snap.r}
        hello={snap.hello}
        thumb={snap.thumb}
        translateTo={snap.translateTo}
        listening={snap.listening}
        act={(a) => hub.op(a)}
        onClose={() => { if (hub.close("x")) void hideWindow(); }}
        closeLabel="Close"
        composer={composer}
        q={q}
        setQ={setQ}
        scrollRef={scroll}
      />
    </div>
  );
}
