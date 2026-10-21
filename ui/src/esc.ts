// Esc folds NEO's chat, as the header's "−" does — wherever the keyboard is.
//
// The page has the keys (NEO's window, or the notch's panel once it took them): `escFolds`, from a
// window-level listener that runs after everything on the page with an Esc of its own had its turn —
// the confirm card (Esc = No) and the erase dialog by name; the browser card's dialogs, menus,
// option lists, address bar and step drawer stop the key or prevent its default; an input method
// composing keeps it; Search Screen's card in the notch has its own. In the notch NotchHub folds its
// panel on Esc as well while the panel has the keys (NotchWindow.sendEvent): the page sees the key
// first, then the panel folds.
//
// Neither the page nor NotchHub has the keys: the chat is on screen while the user's app has the
// keyboard (it opened by itself — a summon in the notch, a login page, the camera — or the user
// clicked back into their app). NEO.app's ⌥⌘ tap sees a plain Esc go by (never taken: the app gets
// its Esc as always) and passes it on — in NEO's window to this page, in the notch through the core,
// and only while this page said it is on screen without the keys (`wantsAwayEsc`: `esc_watch`).
// `awayEscFolds` decides; it is more careful than a key the page got itself, since that Esc was the
// user's app's too: never over a question, a login page, the camera or a song, never a pinned chat,
// never one that came up a moment ago.

/** An Esc that arrived as the chat opened was meant for what was there before. */
export const ESC_SETTLE_MS = 350;

export type EscKey = Pick<KeyboardEvent, "key" | "isComposing" | "keyCode" | "repeat" | "metaKey" | "ctrlKey" | "altKey" | "defaultPrevented">;

/** What the chat shows when an Esc comes. */
export type ChatEsc = {
  /** The chat is on screen (NEO's window: expanded, not the orb; the notch: its panel is open). */
  open: boolean;
  /** A question waits for an OK: its own Esc says No. */
  confirming: boolean;
  /** "Erase everything NEO remembers?": its own Esc says Cancel. */
  erasing: boolean;
  /** Search Screen's card fills the notch's panel: its own Esc closes it. */
  lensCard: boolean;
};

/** Esc, not part of a shortcut, an input method's own, or a key held down. (⇧ doesn't change it.) */
export function plainEsc(e: EscKey): boolean {
  return e.key === "Escape" && !e.isComposing && e.keyCode !== 229 && !e.repeat && !e.metaKey && !e.ctrlKey && !e.altKey;
}

/** A key that reached the page: whether it folds the chat. */
export function escFolds(e: EscKey, s: ChatEsc): boolean {
  return plainEsc(e) && !e.defaultPrevented && s.open && !s.confirming && !s.erasing && !s.lensCard;
}

/** An Esc the user's app got, passed on by NEO.app's shell. */
export type AwayEsc = ChatEsc & {
  /** The page has the keys after all: the key reached it, and `escFolds` had it. */
  focused: boolean;
  pinned: boolean;
  /** Something on the chat must stay in sight: a login page or a turn for the user, the camera's
   *  preview, a song being listened for, the browser card on stage. */
  holds: boolean;
  /** How long the chat has been on screen. */
  openForMs: number;
};

/** An Esc pressed in another app while the chat is on screen: whether it folds the chat. */
export function awayEscFolds(s: AwayEsc): boolean {
  return s.open && !s.focused && !s.confirming && !s.erasing && !s.lensCard && !s.pinned && !s.holds && s.openForMs >= ESC_SETTLE_MS;
}

/** NEO's page in the notch asks to hear an Esc pressed elsewhere only while its panel is on screen
 *  and the keys are not its own (then the key reaches it, or NotchHub, directly). */
export function wantsAwayEsc(s: { visible: boolean; focused: boolean }): boolean {
  return s.visible && !s.focused;
}
