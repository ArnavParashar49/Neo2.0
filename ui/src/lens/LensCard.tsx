// Search Screen's card in the notch: an area's results in NotchHub's panel, instead of the sheet
// beside the selection — the same results and controls (./Sheet.tsx's SheetBody), on the island's
// black. The page shows it in place of the chat while it is live (App.tsx), and asks NotchHub for
// room for it: it measures how tall its content would be (`onNatural`, at every change — a forced
// layout, so it works before NotchHub's panel is open), and App sizes the panel from that.
//
// Keys (NotchHub gives the panel the keyboard without taking it from the user's app): the question
// box takes them when the panel opens; Return asks, ⌘Return continues in NEO; ⌘C copies the area's
// text unless something is selected or the box has the keys; a printable key goes into the box;
// Esc closes the card and folds the panel (NotchHub folds on Esc too).

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { SheetBody } from "./Sheet";
import type { CardSnap, LensCardHub } from "./card";
import { hideWindow, listen } from "../shell";
import "./lens.css";

/** How tall the card's content would be with room for all of it: the fixed rows (question box,
 *  footer) plus the scrolling column's whole height. */
function natural(root: HTMLElement, scroll: HTMLElement): number {
  return root.offsetHeight - scroll.offsetHeight + scroll.scrollHeight;
}

export function LensCard({ hub, snap, onNatural }: { hub: LensCardHub; snap: CardSnap; onNatural: (h: number) => void }) {
  const [q, setQ] = useState("");
  const root = useRef<HTMLDivElement>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLInputElement>(null);

  const measure = useCallback(() => {
    if (root.current && scroll.current) onNatural(natural(root.current, scroll.current));
  }, [onNatural]);
  // Every change of the results, measured before it paints (the size request reaches NotchHub
  // before its panel opens); later changes of size (a section unfolded, a font in) by observing.
  useLayoutEffect(measure, [measure, snap.v, q]);
  useEffect(() => {
    if (typeof ResizeObserver === "undefined" || !scroll.current) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(scroll.current);
    for (const el of Array.from(scroll.current.children)) ro.observe(el);
    return () => ro.disconnect();
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
      const inBox = document.activeElement === composer.current && !!composer.current;
      if (e.isComposing || e.keyCode === 229) return; // (the input method's own keys)
      if (e.key === "Escape") {
        e.preventDefault();
        hub.close("esc");
        void hideWindow(); // (NotchHub folds its panel on Esc as well)
        return;
      }
      if (e.metaKey && !e.shiftKey && !e.altKey && !e.ctrlKey && e.key.toLowerCase() === "c" && !inBox && !window.getSelection()?.toString()) {
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
    window.addEventListener("keydown", down);
    return () => window.removeEventListener("keydown", down);
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
        onClose={() => { hub.close("x"); void hideWindow(); }}
        closeLabel="Close"
        composer={composer}
        q={q}
        setQ={setQ}
        scrollRef={scroll}
      />
    </div>
  );
}
