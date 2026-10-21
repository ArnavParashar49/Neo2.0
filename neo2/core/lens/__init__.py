"""Search Screen: circle or drag over anything on the screen, and NEO says what it is.

Hold ⌥⌘ about a second and let go (the quick tap still summons NEO), pick "Search Screen…" in NEO's
menu, or say "search my screen": a glowing overlay covers the display under the pointer. A drag
selects a rectangle, a circle or a scribble selects its bounding box, a click picks the line of
text, the thing or the window under it. NEO answers in a sheet beside the area: the text it reads
there (Copy, Translate), links, codes and numbers in it, what the area shows (from the user's Eyes
brain: a product and where to buy it, a place, a plant, a sum worked out, an error explained),
"Search with Google" and follow-up questions; "Continue in NEO" takes the area to the chat.

Who does what. NEO.app's shell owns the gesture, the overlay window (a panel above everything,
created by the shell alone) and its ticket; this package is the core's part: the still, the area,
everything worked out about it, and the overlay page's own websocket lane.

* :mod:`capture` — the still of the display, taken the moment the hold arms (ScreenCaptureKit with
  NEO's own windows left out, so the overlay is never in it), and a region in live mode;
* :mod:`regions` — what a click can pick: lines and blocks of text, things, windows;
* :mod:`local` — the on-device pass over an area: its text, language, links, e-mail addresses,
  phone numbers, QR codes and barcodes, and whether it came out black (protected video);
* :mod:`ask` — the Eyes brain's short answer (its own order: neo2.core.providers.lens_order),
  follow-up questions and translation;
* :mod:`search` — "Search with Google" (or the site in NEO_LENS_SEARCH) and the search rows for a
  product or a place;
* :mod:`live` — the overlay's session: the ticket, the still, the current area, its commands.

Privacy.

* Holding ⌥⌘ records nothing: the shell sees only that modifier keys are held and whether some
  other input happened, never which key.
* One still, in memory only. It is taken when the hold arms (or the menu or the voice opens Search
  Screen), without NEO's own windows; never written to disk, never logged, never sent anywhere as a
  whole; dropped when Search Screen closes, on Erase, on Stop, when the screen locks, or after 10
  minutes. A hold let go of before it armed never takes one; one cancelled after it drops it and
  everything worked out from it.
* On this Mac: reading the text, its language, links, phone numbers, codes, and smart pick (worked
  out as soon as the still is in, the lines first).
* Only the area the user selects leaves the Mac, and only to the brains they chose: their Eyes
  brain first, then their other roles that see, then NEO's seeing fallbacks — none for a user who
  keeps their brains on this Mac, and an Eyes brain on this Mac is never hedged to the cloud for
  being slow (only when it fails). NEO_LENS_ASK=off: nothing goes anywhere until they press Ask (or
  ask a question). The sheet's privacy line names every place it went — each brain as it is asked,
  not only the one that answered (``sent_to``) — and says "Stayed on this Mac" only when nothing left.
* For a product or a place, a short search phrase (at most 8 words, about the thing) goes to NEO's
  web search engines for the links. Translate sends the area's text to the Quick replies brain.
* "Search with Google" sends a smaller copy of the area (1000 px at most) to Google through the
  user's own browser, only when they click it (NEO_LENS_SEARCH picks the site, or turns it off).
* Logs hold timings, sizes, kinds and fixed error words: never pixels, text, questions, answers,
  queries or links.
* NEO's brains are told never to say who a person is from their face or body, and that the picture
  and its text are content from the screen, never instructions. Links and codes show their full
  host before anything opens, and only http(s) links open at all.
* Windows that hide themselves from capture, and protected video, come out blank or black.
* Results in the notch (NotchHub hosting NEO): once NotchHub shows an area's card the still goes
  at once, and only the area (its crop and its results) stays, until the card closes — 10 minutes
  without a touch at most. NotchHub's own connection hears the card's id, whether it is open and
  why, never a pixel, a word, a link or a brain's name; NEO's page in the notch gets what the sheet
  shows (a 96 px picture of the area, its text, the results and the action ids), never the still.

The protocol.

Shell → core (its own socket, neo2.core.server.shell_link): ``lens_freeze {ticket, display:{id, x,
y, w, h, scale}, via}`` (the hold armed, or the menu or the voice opened it: a new ticket, and the
still is taken — in the notch, a moment after ``lens_opening`` folded the host's panel),
``lens_commit {ticket}`` (the user let go: it is open), ``lens_closed {ticket, why}``. Core → shell:
the hello's ``lens: {on, hold_ms}``, ``lens_close {why}`` (Erase, Stop, the core's own close — and
"gone" when the overlay's page went and didn't come back), and ``shell {action: "lens"}`` (the
voice, the agent's tool, a host).

The overlay's page is the websocket client kind ``lens``: NEO.app's own page with the shell's ticket
(``?token=…&lens=<64 hex>``), one at a time. It hears nothing of the event bus, the transcript
replay or the cards, and it is not a viewer. It sends only these, each answered to it alone::

    lens_hello {v}                         → lens_hello (again once the brains are known), the still
                                             (NBF1 kind 4), lens_frozen, lens_regions (lines first,
                                             done: false; then all of it)
    lens_select {sel, rect, how, live?}    → lens_captured (live), lens_local, lens_answer, lens_links
    lens_ask {sel, qid, q}                 → lens_reply (q "": the first answer, when NEO_LENS_ASK is off)
    lens_translate {sel, to}               → lens_translation
    lens_act {sel, id}                     → lens_done
    lens_continue {sel, text}              → the chat takes the area; Search Screen closes
    lens_listen {sel}                      → lens_listening {on}: the voice listens, for a question about the area

    lens_handoff {sel, to}                 → lens_handoff {sel, ok, display | why}: NotchHub shows the
                                             area's results (the overlay flies it there and closes
                                             "notch"), or why not (no_notch, unshown, timeout, stale)

``sel`` grows with each area; a command for an older one is ``lens_refused {cmd, error: "stale"}``.
Every ``lens_local`` carries the area's 96 px ``thumb``. The card in the notch has its own commands,
from NEO's page there and NotchHub's connection: see :mod:`live` and neo2.core.server.ws.
Every error is a fixed word. The bus hears only ``lens_opening`` and ``lens_continue``, with nothing
in them.
"""
