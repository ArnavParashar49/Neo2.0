"""Build tests/data/ground_eval.jsonl: the resolver's benchmark (neo.tools.computer.resolve).

    python tests/data/build_ground_eval.py --os <os_en_hi.json>

Deterministic (sha256 seeds). Sources:

* **generated** — a hand-written bank of 79 generic macOS controls (the design-laya research track's
  ui_concepts.py): how people ask for each in English, Hinglish and Hindi, and the frames they put
  around a control's name. Each request is scored on windows of 12, 50 and 200 elements with
  same-group lookalikes and static text (the research's `screen()` rule).
* **os** — Hindi-framed requests around a control's OS Hindi form ("नया गेम पर क्लिक करो"), and
  requests on a Hindi-localized window (its labels are the OS Hindi forms), in both scripts.
* **hand** — the design-grounding track's 45 hand-built cases (dialogs, modal duplicates, icons).
* **selftest** — phrases for the controls self-test window (neo/selftest/layout.py).
* **nested** — labels that hold a frame word next to the shorter label it contains ("Copy" and
  "Copy Link", "New" and "New Tab", "Try" and "Try Again", "नया" and "नया टैब"): the frame stripper
  must never turn the longer one into the shorter one.

The OS forms (macOS's own English→Hindi UI strings, from system .loctable files) of the bank's
labels are stored in the first line, so the file is self-contained. Evaluation data only: no
decision code reads it.

Line 1: {"kind": "vocab", "elements": [[role, label], ...], "os": {label: [hindi, ...]},
         "concepts": {id: {"labels": [...], "commit": bool}}}
Then:   {"kind": "row", "id", "text", "lang": en|hing|hi, "script", "type", "source", "commit",
         "gold": [labels], "screens": {"12": [vocab ids], "50": [...], "200": [...]}}   (generated/os)
   or   {"kind": "row", ..., "screen": [[role, label, modal], ...], "gold": [labels]}         (hand/selftest)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "ground_eval.jsonl"
SIZES = (12, 50, 200)

# id, group, labels, en, hinglish, hindi, commit  (design-laya/ui_concepts.py)
C = [
    ("new_game", "game", ["New Game"], ["start a new game", "begin another game", "restart the game from scratch"],
     ["naya game shuru karo", "new game wala button dabao"], ["नया गेम शुरू करो", "नया खेल"], False),
    ("start", "game", ["Start"], ["begin", "let's go, start it", "kick it off"],
     ["start karo", "shuru karo"], ["शुरू करो", "चालू करो"], False),
    ("cancel", "dialog", ["Cancel"], ["never mind, cancel", "don't do it", "back out of this"],
     ["cancel kar do", "rehne do, cancel"], ["रद्द करो", "कैंसल करो"], False),
    ("done", "dialog", ["Done"], ["i'm finished here", "that's done, close it"],
     ["ho gaya, done dabao", "done karo"], ["हो गया", "काम पूरा हुआ"], False),
    ("undo", "edit", ["Undo", "Undo Move"], ["take that back", "undo my last move", "revert the last change"],
     ["undo karo", "pichla move wapas lo"], ["पिछली चाल वापस लो", "पूर्ववत करो"], False),
    ("redo", "edit", ["Redo"], ["do it again", "bring the change back"], ["redo karo", "wapas se karo"], ["फिर से करो"], False),
    ("hint", "game", ["Hint", "Show Hint"], ["give me a hint", "help me with a clue"],
     ["hint dikhao", "koi hint do"], ["संकेत दिखाओ", "हिंट दो"], False),
    ("resign", "game", ["Resign"], ["give up", "i forfeit", "concede the game"],
     ["haar maan lo", "game chhod do"], ["हार मान लो", "खेल छोड़ दो"], False),
    ("settings", "app", ["Settings", "Preferences…"], ["open preferences", "go to settings", "the options screen"],
     ["settings kholo", "setting mein jao"], ["सेटिंग्स खोलो", "सेटिंग में जाओ"], False),
    ("search", "app", ["Search"], ["find something", "look it up here"], ["search karo", "dhoondo"], ["खोजो", "सर्च करो"], False),
    ("back", "nav", ["Back", "Go Back"], ["go back", "previous page", "return to the last page"],
     ["peeche jao", "wapas jao"], ["पीछे जाओ", "वापस जाओ"], False),
    ("forward_nav", "nav", ["Go Forward"], ["go forward a page", "the next page i was on"],
     ["aage jao"], ["आगे जाओ"], False),
    ("reload", "nav", ["Reload this page", "Refresh"], ["refresh", "reload the page"],
     ["page refresh karo", "reload karo"], ["पेज रीलोड करो", "रिफ्रेश करो"], False),
    ("new_tab", "nav", ["New Tab"], ["open another tab", "a fresh tab please"], ["naya tab kholo", "ek aur tab"], ["नया टैब खोलो"], False),
    ("close_window", "window", ["Close", "close button"], ["close the window", "shut this window"],
     ["window band karo"], ["विंडो बंद करो"], False),
    ("minimize", "window", ["Minimize", "minimize button"], ["hide the window in the dock", "shrink it down"],
     ["minimize karo", "window chhoti karo"], ["विंडो छोटी करो"], False),
    ("fullscreen", "window", ["Full Screen", "Enter Full Screen"], ["make it fill the screen", "go fullscreen"],
     ["full screen karo", "poori screen pe dikhao"], ["पूरी स्क्रीन पर दिखाओ"], False),
    ("compose", "mail", ["Compose", "New Message"], ["write a new email", "start a new message"],
     ["naya mail likho", "nayi email banao"], ["नया ईमेल लिखो", "नया संदेश"], False),
    ("reply", "mail", ["Reply"], ["answer this email", "respond to him"], ["reply karo", "jawab do"], ["जवाब दो", "उत्तर दो"], False),
    ("reply_all", "mail", ["Reply All"], ["reply to everyone", "answer all of them"], ["sabko reply karo"], ["सबको जवाब दो"], False),
    ("forward_mail", "mail", ["Forward"], ["forward this email", "pass this mail on to someone"],
     ["ye mail forward karo"], ["यह मेल आगे भेजो"], True),
    ("archive", "mail", ["Archive"], ["file this away", "move it to the archive"], ["archive kar do"], ["आर्काइव करो", "संग्रह में डालो"], False),
    ("junk", "mail", ["Junk", "Move to Junk"], ["mark as spam", "this is spam"], ["spam mein daalo"], ["स्पैम में डालो"], False),
    ("flag", "mail", ["Flag"], ["flag this", "mark it important"], ["flag laga do"], ["फ्लैग करो", "झंडा लगाओ"], False),
    ("inbox", "mail", ["Inbox"], ["show my inbox", "go to incoming mail"], ["inbox kholo"], ["इनबॉक्स खोलो"], False),
    ("drafts", "mail", ["Drafts"], ["my unfinished emails", "open drafts"], ["drafts dikhao"], ["ड्राफ्ट खोलो"], False),
    ("sent", "mail", ["Sent"], ["mail i sent", "sent messages"], ["bheje hue mail dikhao"], ["भेजे गए मेल दिखाओ"], False),
    ("downloads", "nav", ["Downloads"], ["show my downloads", "the files i downloaded"], ["downloads kholo"], ["डाउनलोड दिखाओ"], False),
    ("sidebar", "view", ["Show Sidebar", "Toggle Sidebar"], ["open the side panel", "show the sidebar"],
     ["sidebar dikhao"], ["साइडबार दिखाओ"], False),
    ("bold", "format", ["Bold"], ["make it bold", "thicker text"], ["bold karo"], ["बोल्ड करो", "मोटा करो"], False),
    ("italic", "format", ["Italic"], ["slanted text", "italicize this"], ["italic karo", "tircha karo"], ["तिरछा करो"], False),
    ("zoom_in", "view", ["Zoom In"], ["make it bigger", "magnify"], ["bada karo", "zoom in karo"], ["बड़ा करो", "ज़ूम इन करो"], False),
    ("zoom_out", "view", ["Zoom Out"], ["make it smaller", "shrink the view"], ["chhota dikhao", "zoom out karo"], ["ज़ूम आउट करो"], False),
    ("play", "media", ["Play"], ["start playing", "resume the video"], ["chalao", "play karo"], ["चलाओ", "प्ले करो"], False),
    ("pause", "media", ["Pause"], ["pause it", "hold the video"], ["rok do", "pause karo"], ["रोको", "रोक दो"], False),
    ("next_track", "media", ["Next Track", "Next"], ["skip this song", "next one"], ["agla gaana", "next karo"], ["अगला गाना", "अगला"], False),
    ("prev_track", "media", ["Previous Track", "Previous"], ["go to the last song", "previous one"], ["pichla gaana"], ["पिछला गाना"], False),
    ("mute", "media", ["Mute"], ["silence it", "turn the sound off"], ["awaaz band karo", "mute karo"], ["आवाज़ बंद करो", "म्यूट करो"], False),
    ("shuffle", "media", ["Shuffle"], ["random order", "mix up the songs"], ["shuffle karo"], ["गाने मिलाओ", "शफल करो"], False),
    ("repeat", "media", ["Repeat"], ["loop it", "play on repeat"], ["repeat pe lagao"], ["दोहराओ", "रिपीट करो"], False),
    ("sign_in", "account", ["Sign In", "Log In"], ["log me in", "sign into my account"], ["login karo"], ["लॉग इन करो", "साइन इन करो"], False),
    ("sign_up", "account", ["Create Account", "Sign Up"], ["register a new account", "make an account"],
     ["naya account banao"], ["नया खाता बनाओ"], False),
    ("help", "app", ["Help"], ["i need help", "show the help docs"], ["madad chahiye", "help kholo"], ["मदद करो", "सहायता"], False),
    ("share", "file", ["Share"], ["share this", "send this to someone"], ["share karo"], ["शेयर करो"], True),
    ("print", "file", ["Print…"], ["print this", "send it to the printer"], ["print karo", "print nikalo"], ["प्रिंट करो", "छापो"], False),
    ("save", "file", ["Save"], ["save my work", "keep the changes"], ["save karo", "save kar do"], ["सेव करो", "सहेजो"], False),
    ("open_file", "file", ["Open…"], ["open a file", "browse for a document"], ["file kholo"], ["फ़ाइल खोलो"], False),
    ("new_folder", "file", ["New Folder"], ["make a folder", "create a directory"], ["naya folder banao"], ["नया फ़ोल्डर बनाओ"], False),
    ("rename", "file", ["Rename"], ["change the name", "give it a new name"], ["naam badlo"], ["नाम बदलो"], False),
    ("copy", "edit", ["Copy"], ["copy it", "duplicate the text"], ["copy karo"], ["कॉपी करो", "नकल करो"], False),
    ("paste", "edit", ["Paste"], ["put it here", "paste the clipboard"], ["paste karo", "yahan chipkao"], ["चिपकाओ", "पेस्ट करो"], False),
    ("accept_cookies", "web", ["Accept All"], ["allow all cookies", "agree to the cookies"], ["cookies accept karo"],
     ["सभी कुकीज़ स्वीकार करो"], True),
    ("reject_cookies", "web", ["Reject All", "Decline"], ["decline the cookies", "refuse tracking"],
     ["cookies reject karo", "mana kar do"], ["कुकीज़ अस्वीकार करो"], False),
    ("next_page", "web", ["Next Page"], ["go to the next page of results", "more results"], ["agla page"], ["अगला पेज"], False),
    ("wifi", "settings", ["Wi-Fi"], ["network settings", "wireless"], ["wifi kholo"], ["वाई-फ़ाई खोलो"], False),
    ("bluetooth", "settings", ["Bluetooth"], ["bluetooth settings"], ["bluetooth kholo"], ["ब्लूटूथ खोलो"], False),
    ("battery", "settings", ["Battery"], ["power settings", "check the battery"], ["battery dikhao"], ["बैटरी दिखाओ"], False),
    ("displays", "settings", ["Displays"], ["screen settings", "monitor options"], ["display settings kholo"], ["डिस्प्ले सेटिंग खोलो"], False),
    ("notifications", "settings", ["Notifications"], ["alert settings"], ["notification settings kholo"], ["सूचनाएँ खोलो"], False),
    ("today", "calendar", ["Today"], ["jump to today", "go to the current date"], ["aaj pe jao"], ["आज पर जाओ"], False),
    ("add_event", "calendar", ["Add Event", "New Event"], ["create an appointment", "put a meeting in"],
     ["naya event banao"], ["नया इवेंट जोड़ो"], False),
    ("delete", "file", ["Delete", "Move to Trash"], ["bin this", "throw it away", "get rid of it"],
     ["delete kar do", "hata do"], ["हटा दो", "मिटा दो"], True),
    ("send", "mail", ["Send"], ["fire it off", "send it"], ["bhej do", "send karo"], ["भेज दो", "भेजो"], True),
    ("buy", "shop", ["Buy Now", "Place Order"], ["purchase it", "check out now"], ["khareed lo", "order kar do"], ["खरीद लो"], True),
    ("filter", "web", ["Filter"], ["narrow the results", "show only some of them"], ["filter lagao"], ["फ़िल्टर लगाओ"], False),
    ("sort", "web", ["Sort By"], ["change the order", "arrange the list"], ["sort karo"], ["क्रम बदलो"], False),
    ("more", "app", ["More", "More Options"], ["the three dots", "other options", "the overflow menu"],
     ["aur options dikhao"], ["और विकल्प दिखाओ"], False),
    ("edit", "edit", ["Edit"], ["change this", "modify it"], ["edit karo"], ["एडिट करो", "बदलाव करो"], False),
    ("attach", "mail", ["Attach", "Add Attachment"], ["add a file to the email", "the paperclip"], ["file attach karo"], ["फ़ाइल जोड़ो"], False),
    ("emoji", "edit", ["Emoji"], ["the smiley face", "insert an emoji"], ["emoji daalo"], ["इमोजी डालो"], False),
    ("home", "nav", ["Home"], ["go to the start page", "the main page"], ["home pe jao"], ["होम पर जाओ"], False),
    ("profile", "account", ["Profile", "Account"], ["my account page", "my profile"], ["profile kholo"], ["प्रोफ़ाइल खोलो"], False),
    ("difficulty", "game", ["Difficulty"], ["make it harder", "change the level"], ["level badlo"], ["कठिनाई बदलो"], False),
    ("vs_computer", "game", ["Human vs Computer"], ["play against the computer", "me against the machine"],
     ["computer ke khilaaf khelo"], ["कंप्यूटर के ख़िलाफ़ खेलो"], False),
    ("vs_human", "game", ["Human vs Human"], ["two players on this mac", "play against a friend"],
     ["dost ke saath khelo"], ["दोस्त के साथ खेलो"], False),
    ("rotate_board", "game", ["Rotate Board"], ["flip the board", "turn the board around"], ["board ghumao"], ["बोर्ड घुमाओ"], False),
    ("subtitles", "media", ["Subtitles", "Captions"], ["turn on captions", "show the subtitles"], ["subtitle chalu karo"], ["सबटाइटल दिखाओ"], False),
    ("quit", "window", ["Quit"], ["exit the app", "close the program completely"], ["app band karo"], ["ऐप बंद करो"], False),
    ("address_bar", "nav", ["Address and Search Bar"], ["the url bar", "where i type the website"], ["url bar mein likho"], ["पता बार"], False),
]  # fmt: skip
CONCEPTS = [{"id": c[0], "group": c[1], "labels": c[2], "en": c[3], "hing": c[4], "hi": c[5], "commit": c[6]} for c in C]
FRAMES = {
    "en": ["click {x}", "tap {x}", "press the {x} button", "click on {x}", "hit {x}", "select {x}"],
    "hing": ["{x} pe click karo", "{x} dabao", "{x} wala button dabao", "{x} pe tap karo", "{x} select karo"],
    "hi": ["{x} पर क्लिक करो", "{x} दबाओ", "{x} वाला बटन दबाओ", "{x} चुनो"],
}
FILLER = [
    "Last updated 3 minutes ago", "Untitled", "report-final-v2.pdf", "Chapter 3", "Tuesday, 14 October",
    "10:42 AM", "Priya Sharma", "Re: quarterly numbers", "Loading…", "No results", "12 items", "Page 2 of 7",
    "Terms of Service", "Privacy Policy", "© 2026", "Welcome back", "Documents", "Desktop", "Recents",
    "White to move", "Black to move", "e4", "Nf3", "Move 12", "4.2 GB free", "Connected", "Not connected",
    "Ready", "Draft saved", "Sent 2h ago", "Invoice #4471", "Order summary", "Qty 1", "₹1,299", "In stock",
    "Show more", "Read more", "Sponsored", "Trending now", "Up next", "3:41", "Autoplay", "Library",
    "Playlists", "Recently Added", "Artists", "Albums", "Songs", "Queue", "Lyrics", "Volume", "Brightness",
    "General", "Appearance", "Accessibility", "Privacy & Security", "Desktop & Dock", "Keyboard", "Trackpad",
    "Mouse", "Users & Groups", "Passwords", "Software Update", "Storage", "About", "Name", "Kind",
    "Date Modified", "Size", "Tags", "Favorites", "iCloud", "Locations", "Shared", "Network",
]  # fmt: skip
_ROW_CONCEPTS = {"inbox", "drafts", "sent", "wifi", "bluetooth", "battery", "displays", "notifications"}
_FIELD_CONCEPTS = {"address_bar"}


def role_of(cid: str) -> str:
    return "AXRow" if cid in _ROW_CONCEPTS else "AXTextField" if cid in _FIELD_CONCEPTS else "AXButton"


# design-grounding/ground_eval.py: (role, label, where); "sheet …" = in an open sheet
HAND_SCREENS = {
    "game": [
        ("Button", "Start", "sheet"), ("Button", "Cancel", "sheet"), ("Button", "Start", "sidebar"),
        ("Button", "New Game", "sidebar"), ("Button", "Undo Move", "sidebar"), ("Button", "Hint", "sidebar"),
        ("Button", "Resign", "sidebar"), ("Button", "Settings", "sidebar"), ("MenuBarItem", "Game", "menu bar"),
        ("MenuBarItem", "View", "menu bar"), ("Button", "close", "title bar"), ("Button", "minimize", "title bar"),
    ],
    "mail": [
        ("Button", "Compose new message", "toolbar"), ("Button", "Reply", "toolbar"), ("Button", "Reply All", "toolbar"),
        ("Button", "Forward", "toolbar"), ("Button", "Delete", "toolbar"), ("Button", "Archive", "toolbar"),
        ("Button", "Junk", "toolbar"), ("SearchField", "Search", "toolbar"), ("Row", "Inbox", "sidebar"),
        ("Row", "Sent", "sidebar"), ("Row", "Drafts", "sidebar"), ("Button", "Flag", "toolbar"),
    ],
    "settings": [
        ("Row", "Wi-Fi", "sidebar"), ("Row", "Bluetooth", "sidebar"), ("Row", "Displays", "sidebar"),
        ("Row", "Sound", "sidebar"), ("Row", "Battery", "sidebar"), ("CheckBox", "Show battery percentage", "pane"),
        ("Slider", "Output volume", "pane"), ("CheckBox", "Mute", "pane"), ("PopUpButton", "Alert sound", "pane"),
        ("Button", "Done", "pane"), ("SearchField", "Search", "sidebar"), ("Button", "Back", "toolbar"),
    ],
    "browser": [
        ("Button", "Back", "toolbar"), ("Button", "Forward", "toolbar"), ("Button", "Reload this page", "toolbar"),
        ("TextField", "Address and search bar", "toolbar"), ("Button", "New Tab", "tab bar"),
        ("Tab", "Inbox (3) - Mail", "tab bar"), ("Tab", "YouTube", "tab bar"), ("Button", "Bookmark this tab", "toolbar"),
        ("Link", "Sign in", "page"), ("Button", "Accept all", "page"), ("Button", "Reject all", "page"),
        ("Button", "Downloads", "toolbar"),
    ],
    "save_dialog": [
        ("TextField", "Save As:", "sheet"), ("PopUpButton", "Where", "sheet"), ("Button", "Save", "sheet"),
        ("Button", "Cancel", "sheet"), ("Button", "New Folder", "sheet"), ("Button", "Bold", "toolbar"),
        ("Button", "Italic", "toolbar"), ("Button", "Share", "toolbar"), ("TextArea", "document body", "window"),
    ],
    # the controls self-test window (neo/selftest/layout.py): two buttons, a field, the Test menu, a list
    "selftest": [
        ("StaticText", "NEO controls self-test - please don't touch the mouse or keyboard", "window"),
        ("Button", "Alpha", "window"), ("Button", "Bravo", "window"), ("TextField", "Type here", "window"),
        ("MenuBarItem", "Test", "menu bar"), ("MenuItem", "Test Action", "menu"),
        *[("Row", f"Row {i}", "list") for i in range(1, 25)],
    ],
    # a label that holds a frame word, beside the shorter label it contains
    "nested": [
        ("MenuItem", "Copy", "menu"), ("MenuItem", "Copy Link", "menu"), ("MenuItem", "New", "menu"),
        ("MenuItem", "New Tab", "menu"), ("MenuItem", "Close", "menu"), ("MenuItem", "Close Tab", "menu"),
        ("Button", "Try", "page"), ("Button", "Try Again", "page"), ("Button", "Buy", "page"),
        ("Button", "Buy Now", "page"), ("Button", "Select", "toolbar"), ("Button", "Select All", "toolbar"),
        ("Button", "Main", "page"), ("Button", "Main Menu", "page"), ("Button", "Remove", "page"),
        ("Button", "Remove Item", "page"), ("Link", "Open", "page"), ("Link", "Open Link", "page"),
    ],
    "nested_long": [
        ("MenuItem", "Copy", "menu"), ("MenuItem", "Copy Link Address", "menu"), ("MenuItem", "Paste", "menu"),
        ("Button", "Start", "page"), ("Link", "Start Button Tutorial", "page"), ("Button", "Cancel", "page"),
    ],
    "nested_hi": [
        ("MenuItem", "नया", "menu"), ("MenuItem", "नया टैब", "menu"), ("MenuItem", "बंद करें", "menu"),
        ("MenuItem", "टैब बंद करें", "menu"), ("Button", "फिर से कोशिश करें", "page"), ("Button", "कोशिश करें", "page"),
    ],
}  # fmt: skip
# (screen, request, index of the right element, the gold label commits)
HAND_CASES = [
    ("game", "click Start", 0), ("game", "press start in the dialog", 0), ("game", "start the new game", 0),
    ("game", "cancel that", 1), ("game", "click the hint button", 5), ("game", "take back my last move", 4),
    ("game", "give up the game", 6), ("game", "open the game menu", 8), ("game", "close the window", 10),
    ("game", "start wala button dabao", 0), ("game", "hint dikhao", 5), ("game", "गेम छोड़ दो", 6),
    ("mail", "write a new email", 0), ("mail", "reply to everyone", 2), ("mail", "click reply", 1),
    ("mail", "move this to the bin", 4), ("mail", "mark it as spam", 6), ("mail", "go to my sent mail", 9),
    ("mail", "open drafts", 10), ("mail", "flag this message", 11), ("mail", "search my mail", 7),
    ("mail", "is mail ko forward karo", 3), ("mail", "ड्राफ्ट खोलो", 10),
    ("settings", "turn off bluetooth settings page", 1), ("settings", "show the battery percentage", 5),
    ("settings", "change the alert sound", 8), ("settings", "mute the sound", 7), ("settings", "go to displays", 2),
    ("settings", "wifi kholo", 0), ("settings", "click done", 9),
    ("browser", "go back", 0), ("browser", "refresh the page", 2), ("browser", "open a new tab", 4),
    ("browser", "switch to the youtube tab", 6), ("browser", "log in", 8), ("browser", "decline the cookies", 10),
    ("browser", "bookmark this", 7), ("browser", "type in the url bar", 3), ("browser", "naya tab kholo", 4),
    ("save_dialog", "save it", 2), ("save_dialog", "name the file", 0), ("save_dialog", "make a new folder", 4),
    ("save_dialog", "choose where to save", 1), ("save_dialog", "make the text bold", 5), ("save_dialog", "don't save, cancel", 3),
    ("selftest", "click Alpha", 1), ("selftest", "press the bravo button", 2), ("selftest", "alpha pe click karo", 1),
    ("selftest", "bravo dabao", 2), ("selftest", "tap alpha", 1), ("selftest", "hit bravo", 2),
    ("selftest", "click the test action menu item", 5), ("selftest", "click row 12", 17), ("selftest", "row 3 pe click karo", 8),
    ("selftest", "run the test action", 5), ("selftest", "अल्फा दबाओ", 1), ("selftest", "click in the type here field", 3),
    ("nested", "click copy link", 1), ("nested", "click copy", 0), ("nested", "click new tab", 3),
    ("nested", "new tab pe click karo", 3), ("nested", "click the new button", 2), ("nested", "close tab dabao", 5),
    ("nested", "click the close tab button", 5), ("nested", "close pe click karo", 4), ("nested", "click try again", 7),
    ("nested", "tap try", 6), ("nested", "buy now pe click karo", 9), ("nested", "hit the buy button", 8),
    ("nested", "click select all", 11), ("nested", "main menu pe click karo", 13), ("nested", "remove item dabao", 15),
    ("nested", "click remove", 14), ("nested", "click open link", 17), ("nested", "click buy now please", 9),
    ("nested_long", "click copy link", 1), ("nested_long", "click the start button", 3), ("nested_long", "press start", 3),
    ("nested_hi", "नया टैब दबाओ", 1), ("nested_hi", "नया दबाओ", 0), ("nested_hi", "टैब बंद करें पर क्लिक करो", 3),
    ("nested_hi", "बंद करें दबाओ", 2), ("nested_hi", "फिर से कोशिश करें दबाओ", 4), ("nested_hi", "कोशिश करें पर क्लिक करो", 5),
]  # fmt: skip
_HAND_COMMIT = {"Forward", "Delete", "Accept all", "Share", "Send"}


def stable(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def lang_of(text: str) -> str:
    if re.search("[ऀ-ॿ]", text):
        return "hi"
    if re.search(r"\b(?:karo|kar do|kholo|dabao|dikhao|wala|pe|jao|lo|do|banao|lagao|daalo|chalao|badlo|ghumao|khelo|mein)\b", text):
        return "hing"
    return "en"


def queries() -> list[dict]:
    """The research's 703 requests: every paraphrase, and two frames per concept in EN and Hinglish."""
    rnd = random.Random(1)
    out = []
    for c in CONCEPTS:
        for lang in ("en", "hing", "hi"):
            for t in c[lang]:
                out.append({"text": t, "lang": lang, "type": "paraphrase", "cid": c["id"]})
        for lang in ("en", "hing"):
            for fr in rnd.sample(FRAMES[lang], 2):
                lab = rnd.choice(c["labels"]).rstrip("…").lower()
                out.append({"text": fr.format(x=lab), "lang": lang, "type": "framed", "cid": c["id"]})
    return out


def os_forms(os_map: dict[str, list[str]], label: str) -> list[str]:
    base = label.rstrip("…").strip()
    return list(os_map.get(label) or os_map.get(base) or os_map.get(base.title()) or [])[:3]


def screen(cid: str, n: int, rnd: random.Random) -> list[tuple[str, str]]:
    """(label, concept id or '') — the gold control, same-group lookalikes first, then others, then
    static text up to n (design-laya bench_ground.screen)."""
    gold = next(c for c in CONCEPTS if c["id"] == cid)
    same = [c for c in CONCEPTS if c["group"] == gold["group"] and c["id"] != gold["id"]]
    other = [c for c in CONCEPTS if c["group"] != gold["group"]]
    rnd.shuffle(same)
    rnd.shuffle(other)
    ctrl = [gold] + same[:5] + other
    ctrl = ctrl[: min(len(ctrl), n if n <= 12 else max(12, int(n * 0.6)))]
    els = [(rnd.choice(c["labels"]), c["id"]) for c in ctrl]
    fill = FILLER[:]
    rnd.shuffle(fill)
    while len(els) < n and fill:
        els.append((fill.pop(), ""))
    k = 0
    while len(els) < n:
        els.append((f"Row {k + 1}: item {k * 7 % 31}", ""))
        k += 1
    rnd.shuffle(els)
    return els


class Vocab:
    def __init__(self) -> None:
        self.items: list[list[str]] = []
        self.ids: dict[tuple[str, str], int] = {}

    def id(self, role: str, label: str) -> int:
        k = (role, label)
        if k not in self.ids:
            self.ids[k] = len(self.items)
            self.items.append([role, label])
        return self.ids[k]


def build(os_map: dict[str, list[str]]) -> list[dict]:
    vocab = Vocab()
    concept_of = {c["id"]: c for c in CONCEPTS}
    rows: list[dict] = []

    def windows(cid: str, key: str, hindi_ui: bool = False) -> dict[str, list[int]]:
        out = {}
        for n in SIZES:
            els = screen(cid, n, random.Random(stable(key, n)))
            ids = []
            for lab, c in els:
                role = role_of(c) if c else "AXStaticText"
                if hindi_ui and c and (forms := os_forms(os_map, lab)):
                    lab = forms[0]
                ids.append(vocab.id(role, lab))
            out[str(n)] = ids
        return out

    def gold_labels(cid: str, hindi_ui: bool = False) -> list[str]:
        labs = concept_of[cid]["labels"]
        if hindi_ui:
            return [os_forms(os_map, lab)[0] if os_forms(os_map, lab) else lab for lab in labs]
        return list(labs)

    for i, q in enumerate(queries()):
        c = concept_of[q["cid"]]
        rows.append({
            "kind": "row", "id": f"g{i:04d}", "text": q["text"], "lang": q["lang"],
            "script": "devanagari" if q["lang"] == "hi" else "latin", "type": q["type"], "source": "generated",
            "concept": c["id"], "commit": c["commit"], "gold": gold_labels(c["id"]),
            "screens": windows(c["id"], f"g|{q['text']}"),
        })  # fmt: skip
    # The OS's Hindi forms: a Hindi frame around one (an English window), and a Hindi-localized window
    # asked in English and in Hindi.
    rnd = random.Random(2)
    k = 0
    for c in CONCEPTS:
        forms = [(lab, os_forms(os_map, lab)) for lab in c["labels"]]
        forms = [(lab, f) for lab, f in forms if f]
        if not forms:
            continue
        lab, f = forms[0]
        hi = f[0].rstrip("…")
        for kind, text, hindi_ui in (
            ("os_hi_framed", rnd.choice(FRAMES["hi"]).format(x=hi), False),
            ("hindi_ui_en", rnd.choice(FRAMES["en"]).format(x=lab.rstrip("…").lower()), True),
            ("hindi_ui_hi", rnd.choice(FRAMES["hi"]).format(x=hi), True),
        ):
            rows.append({
                "kind": "row", "id": f"o{k:04d}", "text": text, "lang": lang_of(text),
                "script": "devanagari" if lang_of(text) == "hi" else "latin", "type": kind, "source": "os",
                "concept": c["id"], "commit": c["commit"], "gold": gold_labels(c["id"], hindi_ui),
                "screens": windows(c["id"], f"o|{kind}|{text}", hindi_ui),
            })  # fmt: skip
            k += 1
    roles = {"StaticText": "AXStaticText", "MenuBarItem": "AXMenuBarItem", "MenuItem": "AXMenuItem"}
    for i, (scr, req, gi) in enumerate(HAND_CASES):
        els = HAND_SCREENS[scr]
        gold = els[gi][1]
        rows.append({
            "kind": "row", "id": f"h{i:02d}", "text": req, "lang": lang_of(req),
            "script": "devanagari" if lang_of(req) == "hi" else "latin", "type": scr,
            "source": "selftest" if scr == "selftest" else "nested" if scr.startswith("nested") else "hand",
            "commit": gold in _HAND_COMMIT, "gold": [gold],
            "screen": [[roles.get(r, "AX" + r), lab, where in ("sheet",)] for r, lab, where in els],
        })  # fmt: skip
    head = {
        "kind": "vocab",
        "elements": vocab.items,
        "os": {lab: os_forms(os_map, lab) for c in CONCEPTS for lab in c["labels"] if os_forms(os_map, lab)},
        "concepts": {c["id"]: {"labels": c["labels"], "commit": c["commit"], "group": c["group"]} for c in CONCEPTS},
    }
    return [head, *rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--os", required=True, help="the OS English→Hindi strings (design-laya os_en_hi.json)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    os_map = json.loads(Path(a.os).read_text(encoding="utf-8"))
    lines = build(os_map)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in lines:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{a.out}: {len(lines) - 1} rows, {len(lines[0]['elements'])} distinct elements")


if __name__ == "__main__":
    main()
