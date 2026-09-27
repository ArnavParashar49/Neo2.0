"""neo2.core.decide.axkit: the AX facade over media's primitives, and the new finders and writers, on a
fake element tree patched into media (the same targets media's own tests patch). No real UI is
read, no key is pressed and the pasteboard is never touched."""

from __future__ import annotations

import pytest

from neo2.core.decide import axkit, media


class El:
    def __init__(self, role, children=(), **attrs):
        self.attrs = {"AXRole": role, **attrs}
        self.children = list(children)
        self.pressed = 0

    def __repr__(self):
        return f"El({self.attrs.get('AXRole')}, {self.attrs.get('AXTitle') or self.attrs.get('AXValue')!r})"


class FakeAX:
    def __init__(self, monkeypatch, root: El, *, value_sticks: bool = True):
        self.root, self.sets, self.value_sticks = root, [], value_sticks
        monkeypatch.setattr(media, "_attr", self.attr)
        monkeypatch.setattr(media, "_set", self.set)
        monkeypatch.setattr(media, "_perform", self.perform)
        monkeypatch.setattr(media, "_actions", lambda el: ["AXPress"])
        monkeypatch.setattr(media, "_ax_app", lambda pid: self.root)
        monkeypatch.setattr(media, "_trusted", lambda: True)

    @staticmethod
    def attr(el, name, default=None):
        if el is None:
            return default
        if name == "AXChildren":
            return list(el.children)
        return el.attrs.get(name, default)

    def set(self, el, name, value):
        self.sets.append((name, value))
        if name == "AXValue" and not self.value_sticks:
            return True  # a web input that ignores a value set
        el.attrs[name] = value
        return True

    @staticmethod
    def perform(el, action):
        el.pressed += 1
        return True


@pytest.fixture(autouse=True)
def _words(monkeypatch):
    axkit.lexicon.cache_clear()
    monkeypatch.setattr(media, "lexicon", lambda: {"play": frozenset({"Play"}), "search": frozenset({"Search"}),
                                                   "find": frozenset({"Find"}), "pause": frozenset({"Pause"})})
    monkeypatch.setattr(axkit, "_MORE_SOURCES", ())
    yield
    axkit.lexicon.cache_clear()


def test_the_facade_follows_medias_primitives(monkeypatch):
    el = El("AXButton", AXTitle="Go")
    fake = FakeAX(monkeypatch, El("AXApplication", [el]))
    assert axkit.attr(el, "AXTitle") == "Go" and axkit.attr(el, "Nope", 3) == 3
    assert axkit.set_attr(el, "AXFocused", True) and fake.sets == [("AXFocused", True)]
    assert axkit.perform(el, "AXPress") and el.pressed == 1
    assert axkit.actions(el) == ["AXPress"] and axkit.app_el(1) is fake.root and axkit.trusted()
    returns = []
    monkeypatch.setattr(media, "_post_return", returns.append)
    axkit.post_return(42)
    assert returns == [42]


def test_lexicon_adds_send_and_new_to_medias_words():
    words = axkit.lexicon()
    assert "Send" in words["send"] and "New" in words["new"] and "Play" in words["play"]


def menu_bar():
    new = El("AXMenuItem", AXTitle="New Note", AXMenuItemCmdChar="N", AXMenuItemCmdModifiers=0)
    new_folder = El("AXMenuItem", AXTitle="New Folder", AXMenuItemCmdChar="N", AXMenuItemCmdModifiers=1)
    off = El("AXMenuItem", AXTitle="Send Later", AXMenuItemCmdChar="D", AXMenuItemCmdModifiers=1, AXEnabled=False)
    return El("AXMenuBar", [El("AXMenuBarItem", [El("AXMenu", [new_folder, new, off])])]), new, new_folder


def test_menu_items_are_found_by_their_shortcut_not_their_title(monkeypatch):
    bar, new, new_folder = menu_bar()
    FakeAX(monkeypatch, El("AXApplication", AXMenuBar=bar))
    assert axkit.menu_item(1, "n") == (new, "New Note")
    assert axkit.menu_item(1, "n", 1) == (new_folder, "New Folder")
    assert axkit.menu_item(1, "d", 1) is None  # disabled
    assert axkit.menu_item(1, "t") is None


def test_focused_editable_is_never_a_password_or_search_field(monkeypatch):
    app = El("AXApplication")
    FakeAX(monkeypatch, app)
    for el, ok in (
        (El("AXTextArea"), True),
        (El("AXTextField", AXPlaceholderValue="Title"), True),
        (El("AXGroup", AXEditableAncestor=object()), True),  # contenteditable in a web view
        (El("AXTextField", AXSubrole="AXSecureTextField"), False),
        (El("AXTextField", AXSubrole="AXSearchField"), False),
        (El("AXTextField", AXPlaceholderValue="Search messages"), False),
        (El("AXButton"), False),
    ):
        app.attrs["AXFocusedUIElement"] = el
        assert (axkit.focused_editable(1) is el) is ok, el


def chat_window():
    search = El("AXTextField", AXSubrole="AXSearchField")
    rows = [El("AXRow", [El("AXStaticText", AXValue="Priya Sharma")]), El("AXRow", [El("AXStaticText", AXValue="Priyanka")]),
            El("AXRow", [El("AXStaticText", AXValue="Priya Sharma")]), El("AXCell", AXTitle="Book club: Priya, Sam")]
    transcript = El("AXGroup", [El("AXStaticText", AXValue="running  late"), El("AXStaticText", AXValue="ok")])
    composer = El("AXTextArea", AXValue="")
    send = El("AXButton", AXDescription="Send")
    heading = El("AXHeading", AXValue="Priya Sharma")
    win = El("AXWindow", [search, El("AXTable", rows), heading, transcript, composer, send], AXTitle="")
    return win, search, rows, composer, send


def test_composer_rows_header_transcript_and_send_button(monkeypatch):
    win, search, rows, composer, send = chat_window()
    FakeAX(monkeypatch, El("AXApplication", [win]))
    assert axkit.composer(win) is composer and axkit.composer(win, exclude=(composer,)) is None
    got = axkit.rows_matching(win, "priya")
    assert [label for _el, label in got] == ["Priya Sharma", "Book club: Priya, Sam"]  # whole words, one per label
    assert got[0][0] is rows[0]
    assert axkit.rows_matching(win, "priya sharma")[0][1] == "Priya Sharma" and axkit.rows_matching(win, " ") == []
    assert axkit.header_title(win) == "Priya Sharma"
    win.attrs["AXTitle"] = "Chat with Sam"
    assert axkit.header_title(win) == "Chat with Sam"
    assert axkit.transcript_count(win, "Running late") == 1 and axkit.transcript_count(win, "") == 0
    assert axkit.commit_button(win) is send
    win.children.append(El("AXButton", AXTitle="send"))
    assert axkit.commit_button(win) is None  # two: not sure which


def test_set_text_by_value_and_by_typing_into_that_process_only(monkeypatch):
    box = El("AXTextArea", AXValue="")
    fake = FakeAX(monkeypatch, El("AXApplication", [box]))
    assert axkit.set_text(box, "hi\nthere") and box.attrs["AXValue"] == "hi\nthere"
    assert ("AXFocused", True) in fake.sets
    # a web input ignores the value set: type into its process, lines joined by the chord asked for
    web = El("AXTextArea", AXValue="")
    fake = FakeAX(monkeypatch, El("AXApplication", [web]), value_sticks=False)
    typed = []
    monkeypatch.setattr(axkit, "_pid_of", lambda el: 321)

    def type_(pid, text, newline):
        typed.append((pid, text, newline))
        web.attrs["AXValue"] = text

    monkeypatch.setattr(axkit, "_type", type_)
    assert axkit.set_text(web, "नमस्ते\ndost", newline="shift+return")
    assert typed == [(321, "नमस्ते\ndost", "shift+return")]
    # never into a password field; never on top of a half-set value
    assert not axkit.set_text(El("AXTextField", AXSubrole="AXSecureTextField"), "x")
    web.attrs["AXValue"] = "partial"
    typed.clear()
    assert not axkit.set_text(web, "hello") and typed == []


def test_clear_text_only_takes_back_neos_own_text(monkeypatch):
    box = El("AXTextArea", AXValue="running late")
    FakeAX(monkeypatch, El("AXApplication", [box]))
    assert not axkit.clear_text(box, "something else") and box.attrs["AXValue"] == "running late"
    assert axkit.clear_text(box, "running  late") and box.attrs["AXValue"] == ""


class Url:
    def absoluteString(self):
        return "https://example.test/search?q=x"


def test_web_url_of_the_front_window(monkeypatch):
    area = El("AXWebArea", [El("AXStaticText", AXValue="Hello")], AXURL=Url())
    win = El("AXWindow", [area])
    FakeAX(monkeypatch, El("AXApplication", [win], AXFocusedWindow=win))
    assert axkit.web_url(1) == "https://example.test/search?q=x"
    FakeAX(monkeypatch, El("AXApplication", [El("AXWindow")], AXFocusedWindow=El("AXWindow")))
    assert axkit.web_url(1) == "" and axkit.web_page(1) == ("", "", "")


def T(text):
    return El("AXStaticText", AXValue=text)


def P(*kids, **attrs):
    return El("AXGroup", list(kids), **attrs)


STORY = "The river rose two metres overnight and the old bridge closed at dawn. " * 4


def article_page():
    banner = P(T("Daily Planet"), El("AXLink", [T("Sign in")]), AXSubrole="AXLandmarkBanner")
    nav = P(El("AXLink", [T("World")]), El("AXLink", [T("Sport")]), AXSubrole="AXLandmarkNavigation")
    cookie = P(T("We use cookies"), El("AXButton", [T("Accept")]), AXDOMClassList=["cookie-banner"])
    main = P(
        El("AXHeading", [T("Floods close the bridge")], AXValue=1),
        P(T("By Lois Lane,"), T("reporting from the river")),
        P(T("Read the"), El("AXLink", [T("full timeline")]), T(". Updated hourly.")),
        P(T(STORY.strip())),
        El("AXButton", [T("Share")]),
        El("AXImage", AXDescription="The bridge at dawn"),
        El("AXHeading", [T("Prices")], AXValue=2),
        El("AXTable", [El("AXRow", [El("AXCell", [T("Sandbags")]), El("AXCell", [T("40 rupees")])])]),
        P(T("Menu of the day: dal and rice"), AXDOMClassList=["menu", "nav"]),  # inside main: content
        El("AXTextField", AXValue="search words"),
        AXSubrole="AXLandmarkMain",
    )
    aside = P(T("Most read: a cat on a train"), AXSubrole="AXLandmarkComplementary")
    after = P(El("AXHeading", [T("Comments (2)")], AXValue=2), P(T("Stay safe everyone")), P(T("The bridge was due for repair")))
    footer = P(T("Copyright Daily Planet"), AXSubrole="AXLandmarkContentInfo")
    return El("AXWebArea", [banner, nav, cookie, main, aside, after, footer], AXURL=Url(), AXTitle="Floods | Daily Planet")


def page_app(monkeypatch, area):
    win = El("AXWindow", [El("AXToolbar", [El("AXTextField", AXValue="example.test")]), area], AXTitle="Window title")
    FakeAX(monkeypatch, El("AXApplication", [win], AXFocusedWindow=win))


def test_a_page_is_read_main_content_first_without_the_sites_furniture(monkeypatch):
    page_app(monkeypatch, article_page())
    url, title, text = axkit.web_page(1)
    assert url == "https://example.test/search?q=x" and title == "Floods | Daily Planet"
    assert text == "\n".join([
        "# Floods close the bridge",
        "By Lois Lane, reporting from the river",  # a lower-case start carries the sentence on
        "Read the full timeline. Updated hourly.",  # a link's words are part of their sentence
        STORY.strip(),
        "## Prices",
        "Sandbags | 40 rupees",  # a table row's cells on one line
        "Menu of the day: dal and rice",
        "",
        "[elsewhere on the page]",
        "## Comments (2)",
        "Stay safe everyone",
        "The bridge was due for repair",
        "Most read: a cat on a train",  # the sidebar last
    ])
    for gone in ("Daily Planet\n", "Sign in", "World", "cookies", "Accept", "Share", "search words", "Copyright", "bridge at dawn"):
        assert gone not in text, gone


def test_a_page_is_cut_at_a_line_and_a_page_without_landmarks_reads_in_order(monkeypatch):
    page_app(monkeypatch, article_page())
    text = axkit.web_page(1, limit=200)[2]
    assert text.endswith("\n…") and len(text) <= 202 and text.startswith("# Floods close the bridge\nBy Lois Lane")
    plain = El("AXWebArea", [
        P(El("AXLink", [T("Home")]), El("AXLink", [T("About")]), AXDOMIdentifier="site-nav"),
        El("AXHeading", [T("Recipes")]),  # no level: a second-level heading
        P(T(STORY.strip())),
        P(T("Sidebar words"), AXDOMClassList=["left-sidebar"]),
    ])
    page_app(monkeypatch, plain)
    assert axkit.web_page(1)[1] == "Window title"  # the page has no title of its own
    assert axkit.web_page(1)[2] == "## Recipes\n" + STORY.strip()


def test_a_page_the_filters_would_empty_is_read_whole(monkeypatch):
    """Everything sits in a container named like page furniture: the filters would leave nothing."""
    area = El("AXWebArea", [P(P(T("Welcome to the tiny site of a village school, open since 1952.")), AXDOMClassList=["nav-wrap"])])
    page_app(monkeypatch, area)
    assert axkit.web_page(1)[2] == "Welcome to the tiny site of a village school, open since 1952."


def test_a_web_editors_document_is_read_but_a_forms_field_is_not(monkeypatch):
    doc = "Minutes of the meeting. We agreed to repaint the hall in June and to hire a new cook."
    area = El("AXWebArea", [El("AXTextArea", AXValue=doc), El("AXTextArea", AXValue="my comment draft"),
                            El("AXTextField", AXSubrole="AXSecureTextField", AXValue="hunter2")])
    page_app(monkeypatch, area)
    assert axkit.web_page(1)[2] == doc


def test_reading_a_page_switches_a_chromium_apps_web_tree_on_and_back(monkeypatch):
    app = El("AXApplication")
    fake = FakeAX(monkeypatch, app)
    assert axkit.enable_web_reading(app, "native") == [] and fake.sets == []
    changed = axkit.enable_web_reading(app, "chromium")
    assert changed == [("AXManualAccessibility", False), ("AXEnhancedUserInterface", False)]
    axkit.restore_web_tree(app, changed)
    assert app.attrs["AXManualAccessibility"] is False and app.attrs["AXEnhancedUserInterface"] is False
    fake.sets.clear()
    assert axkit.enable_web_reading(app, "electron") == [("AXManualAccessibility", False)]
    app.attrs.update(AXManualAccessibility=False, AXEnhancedUserInterface=True)  # the user's VoiceOver
    fake.sets.clear()
    assert axkit.enable_web_reading(app, "chromium") == [] and fake.sets == []



@pytest.fixture(autouse=True)
def _no_held_switches():
    """What one test's reads left switched on never counts as another's reader."""
    yield
    held = getattr(axkit, "_WEB_HELD", None)
    if held is not None:
        held.clear()


def test_a_switch_already_on_is_never_joined_by_the_other(monkeypatch):
    """AXManualAccessibility already on (a tool the user runs set it): the tree is there, and
    AXEnhancedUserInterface (slow window animations while on) is not turned on as well."""
    for rt in ("electron", "chromium"):
        app = El("AXApplication", AXManualAccessibility=True)
        fake = FakeAX(monkeypatch, app)
        assert axkit.enable_web_reading(app, rt) == [] and fake.sets == []
        assert "AXEnhancedUserInterface" not in app.attrs


def test_two_reads_of_one_app_restore_its_switches_when_the_last_one_ends(monkeypatch):
    """A second read (or a site search's watch) starts while the first holds the switches on: the
    first one ending must not switch the tree off under the second."""
    app = El("AXApplication")
    FakeAX(monkeypatch, app)
    first = axkit.enable_web_reading(app, "chromium")
    second = axkit.enable_web_tree(app, "chromium")  # sees NEO's own switch on: changes nothing
    assert first and second == []
    axkit.restore_web_tree(app, first)
    assert app.attrs["AXManualAccessibility"] is True and app.attrs["AXEnhancedUserInterface"] is True
    axkit.restore_web_tree(app, second)
    assert app.attrs["AXManualAccessibility"] is False and app.attrs["AXEnhancedUserInterface"] is False
    # and a later read starts afresh
    again = axkit.enable_web_reading(app, "chromium")
    assert again == [("AXManualAccessibility", False), ("AXEnhancedUserInterface", False)]
    axkit.restore_web_tree(app, again)
    assert app.attrs["AXEnhancedUserInterface"] is False


MORE = "Engineers will inspect the pillars on Monday; until then buses take the ring road and add twenty minutes."


def _wrapped_page(wrapper_classes):
    banner = P(T("Daily Planet"), AXSubrole="AXLandmarkBanner")
    promo = P(T("Subscribe today and get the first three months of the Daily Planet for the price of one. " * 3))
    main = P(El("AXHeading", [T("Floods close the bridge")], AXValue=1), P(T(STORY.strip())), P(T(MORE)), AXSubrole="AXLandmarkMain")
    aside = P(T("Most read: a cat on a train"), AXSubrole="AXLandmarkComplementary")
    return El("AXWebArea", [banner, promo, P(main, aside, AXDOMClassList=wrapper_classes)], AXURL=Url())


@pytest.mark.parametrize("classes", [["content-sidebar-wrap"], ["no-ads", "has-nav"], ["nav-wrap"], ["sidebar"]])
def test_a_container_named_like_furniture_never_hides_the_main_content(monkeypatch, classes):
    """A wrapper whose class says sidebar/ads/nav (a theme's "content-sidebar-wrap", a state class
    like "no-ads") holds the article: the article is read, first."""
    page_app(monkeypatch, _wrapped_page(classes))
    text = axkit.web_page(1)[2]
    assert text.startswith(f"# Floods close the bridge\n{STORY.strip()}\n{MORE}\n\n[elsewhere on the page]\n")
    assert "Subscribe today" in text
    # a wrapper named only by state or by other words is no furniture: its sidebar is read, last
    assert text.endswith("Most read: a cat on a train") == (classes[0] in ("content-sidebar-wrap", "no-ads"))


def test_class_hints_take_whole_names_and_keep_a_title(monkeypatch):
    """Without landmarks: "cookie-recipe" or "has-sidebar" is no page furniture, "site-footer"
    is; a first-level heading inside furniture is still the page's title."""
    recipe = "Chocolate chip cookies: cream the butter and sugar, fold in the chips, bake for twelve minutes. " * 2
    plain = El("AXWebArea", [
        P(El("AXHeading", [T("Cookie of the week")], AXValue=1), El("AXLink", [T("Home")]), AXDOMClassList=["top-nav"]),
        P(P(T(recipe.strip()), AXDOMClassList=["cookie-recipe"]), AXDOMClassList=["has-sidebar"]),
        P(T("Copyright the bakery, all rights reserved"), AXDOMClassList=["site-footer"]),
    ])
    page_app(monkeypatch, plain)
    text = axkit.web_page(1)[2]
    assert text == "# Cookie of the week\n" + recipe.strip()


def test_a_read_stops_when_told(monkeypatch):
    import threading

    page_app(monkeypatch, article_page())
    stop = threading.Event()
    stop.set()
    assert axkit.web_page(1, stop=stop)[2] == ""
    assert axkit.web_page(1)[2].startswith("# Floods")


def test_the_part_of_the_window_a_page_shows(monkeypatch):
    """The web area's frame, cut to the scroll area it sits in (Safari's web area is the whole
    document), for a screenshot of the page without the browser's tabs."""
    FakeAX(monkeypatch, El("AXApplication"))
    area = El("AXWebArea", AXPosition=(100.0, -1900.0), AXSize=(1200.0, 9000.0))
    area.attrs["AXParent"] = El("AXScrollArea", [area], AXPosition=(100.0, 120.0), AXSize=(1200.0, 700.0))
    assert axkit.web_view(area) == (100.0, 120.0, 1200.0, 700.0)
    chrome = El("AXWebArea", AXPosition=(0.0, 88.0), AXSize=(1440.0, 780.0), AXParent=El("AXGroup"))
    assert axkit.web_view(chrome) == (0.0, 88.0, 1440.0, 780.0)
    assert axkit.web_view(El("AXWebArea")) is None  # no frame to tell
    assert axkit.web_view(El("AXWebArea", AXPosition=(0.0, 0.0), AXSize=(0.0, 0.0))) is None

def test_dock_badge(monkeypatch):
    items = El("AXList", [El("AXDockItem", AXSubrole="AXApplicationDockItem", AXTitle="Postbox", AXStatusLabel="3"),
                          El("AXDockItem", AXSubrole="AXApplicationDockItem", AXTitle="Notebook")])
    FakeAX(monkeypatch, El("AXApplication", [items]))
    monkeypatch.setattr(axkit, "_dock_pid", lambda: 10)
    assert axkit.dock_badge("postbox") == "3" and axkit.dock_badge("Notebook") is None and axkit.dock_badge("Nope") is None
    monkeypatch.setattr(axkit, "_dock_pid", lambda: None)
    assert axkit.dock_badge("Postbox") is None


def test_set_search_takes_search_and_address_fields_but_never_a_password(monkeypatch):
    search = El("AXTextField", AXSubrole="AXSearchField", AXPlaceholderValue="Search", AXValue="")
    address = El("AXComboBox", AXValue="https://example.test/")
    FakeAX(monkeypatch, El("AXApplication", [search, address]))
    assert not axkit.set_text(search, "cats")  # a message never lands in a search box
    assert axkit.set_search(search, "cats") and search.attrs["AXValue"] == "cats"
    assert axkit.set_search(address, "cats site:example.test") and address.attrs["AXValue"] == "cats site:example.test"
    assert not axkit.set_search(El("AXTextField", AXSubrole="AXSecureTextField"), "hunter2")
    assert not axkit.set_search(El("AXButton"), "x") and not axkit.set_search(None, "x")


def test_web_area_pid_and_sign_in_come_from_medias_primitives(monkeypatch):
    area = El("AXWebArea")
    win = El("AXWindow", [El("AXGroup", [area]), El("AXTextField", AXSubrole="AXSecureTextField")])
    FakeAX(monkeypatch, El("AXApplication", [win], AXFocusedWindow=win))
    assert axkit.web_area(1) is area
    assert axkit.needs_sign_in(win) and not axkit.needs_sign_in(area)
    monkeypatch.setattr(media, "_pid_of", lambda bid: 55 if bid == "test.webby" else None)
    assert axkit.pid_of("test.webby") == 55 and axkit.pid_of("test.nope") is None


def test_in_web_tells_web_content_from_the_apps_own_controls(monkeypatch):
    page_field = El("AXTextField")
    area = El("AXWebArea", [El("AXGroup", [page_field])])
    bar = El("AXTextField")
    win = El("AXWindow", [El("AXToolbar", [bar]), area])
    FakeAX(monkeypatch, El("AXApplication", [win]))
    for parent in (win, *win.children, *area.children):  # the parent links the fake tree implies
        for child in parent.children:
            child.attrs["AXParent"] = parent
    assert axkit.in_web(page_field) and axkit.in_web(area)
    assert not axkit.in_web(bar) and not axkit.in_web(None)
    deep = El("AXTextField")
    deep.attrs["AXParent"] = deep  # a parent loop: can't tell, so treated as web content
    assert axkit.in_web(deep)
