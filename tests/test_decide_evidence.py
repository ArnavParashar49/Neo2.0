"""neo.decide.evidence + the inventory's new fields and helpers: what bundles declare beyond their
scripting dictionary (SiriKit intents, App Intents, share extensions, services, usage keys, the
document class), read from fixture bundles in a temp dir. Nothing real is scanned or launched."""

from __future__ import annotations

import json
import os
import plistlib
from pathlib import Path

import pytest

from neo.decide import evidence
from neo.decide import inventory as inv

FIX = Path(__file__).parent / "fixtures" / "decide"


def write_plist(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps(data))


def make_bundle(root: Path, name: str, info: dict, *, exts: dict[str, dict] | None = None,
                actions: dict | None = None) -> Path:
    app = root / f"{name}.app"
    write_plist(app / "Contents" / "Info.plist", {"CFBundleIdentifier": f"test.{name.lower()}", **info})
    for rel, plist in (exts or {}).items():
        write_plist(app / "Contents" / rel / "Contents" / "Info.plist", plist)
    if actions is not None:
        meta = app / "Contents" / "Resources" / "Metadata.appintents"
        meta.mkdir(parents=True)
        (meta / "extract.actionsdata").write_text(json.dumps(actions))
    return app


def appex(point: str, **attrs) -> dict:
    return {"NSExtension": {"NSExtensionPointIdentifier": point, "NSExtensionAttributes": attrs}}


SHARE = "vendor.share-services"  # any reverse-DNS point id ending in .share-services
PROTO = "vendor.link.systemProtocol"
TEXT_RULE = (
    'SUBQUERY (extensionItems, $extensionItem, SUBQUERY ($extensionItem.attachments, $attachment, '
    'ANY $attachment.registeredTypeIdentifiers UTI-CONFORMS-TO "public.plain-text").@count == 1).@count == 1'
)
URL_RULE = 'SUBQUERY (extensionItems, $e, ANY $e.registeredTypeIdentifiers UTI-CONFORMS-TO "public.url").@count == 1'


def read(app: Path) -> evidence.Evidence:
    contents = str(app / "Contents")
    return evidence.read(contents, plistlib.loads((app / "Contents" / "Info.plist").read_bytes()))


def test_intents_from_plugins_and_extensions(tmp_path):
    app = make_bundle(tmp_path, "Chatter", {}, exts={
        "PlugIns/Intents.appex": appex("vendor.intents-service", IntentsSupported=["INSendMessageIntent", "INSearchForMessagesIntent"]),
        "Extensions/More.appex": {"EXAppExtensionAttributes": {"EXExtensionPointIdentifier": "vendor.x", "IntentsSupported": ["INAddTasksIntent"]}},
        "PlugIns/Widget.appex": appex("vendor.widgetkit-extension"),
    })
    ev = read(app)
    assert ev.intents == ("INSendMessageIntent", "INSearchForMessagesIntent", "INAddTasksIntent")
    assert not ev.share_text and ev.ai_schemas == ()


def test_app_intents_schemas_and_protocols_in_the_app_and_its_extensions(tmp_path):
    actions = {"actions": {
        "Compose": {"assistantDefinedSchemas": [{"domain": "mail", "name": "CreateDraftIntent", "version": "1"}],
                    "systemProtocols": [f"{PROTO}.AssistantIntent"]},
        "Send": {"assistantDefinedSchemas": [], "systemProtocols": [f"{PROTO}.SendMail"]},
        "Junk": "not a dict",
        "Bad": {"assistantDefinedSchemas": [{"domain": 3}], "systemProtocols": "SendMail"},
    }}
    app = make_bundle(tmp_path, "Mailer", {}, actions=actions)
    ext_meta = app / "Contents" / "PlugIns" / "W.appex" / "Contents" / "Resources" / "Metadata.appintents"
    write_plist(app / "Contents" / "PlugIns" / "W.appex" / "Contents" / "Info.plist", appex("vendor.widgetkit-extension"))
    ext_meta.mkdir(parents=True)
    (ext_meta / "extract.actionsdata").write_text(json.dumps({"actions": {"Open": {"systemProtocols": [f"{PROTO}.OpenEntity"]}}}))
    ev = read(app)
    assert ev.ai_schemas == ("schema:mail.CreateDraftIntent", "protocol:AssistantIntent", "protocol:SendMail", "protocol:OpenEntity")


def test_share_text_in_both_rule_forms(tmp_path):
    as_dict = make_bundle(tmp_path, "A", {}, exts={"PlugIns/S.appex": appex(SHARE, NSExtensionActivationRule={
        "NSExtensionActivationSupportsText": True, "NSExtensionActivationSupportsWebURLWithMaxCount": 1})})
    as_pred = make_bundle(tmp_path, "B", {}, exts={"PlugIns/S.appex": appex(SHARE, NSExtensionActivationRule=TEXT_RULE)})
    anything = make_bundle(tmp_path, "C", {}, exts={"PlugIns/S.appex": appex(SHARE, NSExtensionActivationRule="TRUEPREDICATE")})
    urls = make_bundle(tmp_path, "D", {}, exts={"PlugIns/S.appex": appex(SHARE, NSExtensionActivationRule=URL_RULE)})
    no_text = make_bundle(tmp_path, "E", {}, exts={"PlugIns/S.appex": appex(SHARE, NSExtensionActivationRule={
        "NSExtensionActivationSupportsImageWithMaxCount": 10})})
    not_share = make_bundle(tmp_path, "F", {}, exts={"PlugIns/S.appex": appex("vendor.action", NSExtensionActivationRule="TRUEPREDICATE")})
    assert [read(a).share_text for a in (as_dict, as_pred, anything, urls, no_text, not_share)] == [
        True, True, True, False, False, False]


def test_services_usage_keys_and_document_class(tmp_path):
    app = make_bundle(tmp_path, "Writer", {
        "NSServices": [
            {"NSMenuItem": {"default": "Writer/New Note With Selection"}, "NSSendTypes": ["NSStringPboardType", 3]},
            {"NSMenuItem": {}, "NSSendTypes": ["public.utf8-plain-text"]},  # no title: skipped
            "junk",
        ],
        "NSCalendarsUsageDescription": "adds events",
        "NSContactsUsageDescription": "",  # empty: not declared
        "NSMicrophoneUsageDescription": "dictation",
        "CFBundleDocumentTypes": [{"CFBundleTypeRole": "Editor", "NSDocumentClass": "WriterDocument"}],
    })
    ev = read(app)
    assert ev.services == (("Writer/New Note With Selection", ("NSStringPboardType",)),)
    assert ev.usage == {"NSCalendarsUsageDescription", "NSMicrophoneUsageDescription"}
    assert ev.doc_based
    plain = make_bundle(tmp_path, "Plain", {"CFBundleDocumentTypes": [{"CFBundleTypeRole": "Editor"}], "NSServices": {"x": 1}})
    assert not read(plain).doc_based and read(plain).services == ()


def test_oversized_malformed_and_odd_files_are_skipped(tmp_path, monkeypatch):
    app = make_bundle(tmp_path, "Odd", {}, exts={
        "PlugIns/Big.appex": appex("vendor.intents-service", IntentsSupported=["INSendMessageIntent"]),
        "PlugIns/Ok.appex": appex("vendor.intents-service", IntentsSupported=["INCreateNoteIntent"]),
    })
    big = app / "Contents" / "PlugIns" / "Big.appex" / "Contents" / "Info.plist"
    big.write_bytes(big.read_bytes() + b" " * (evidence.MAX_PLIST + 1))  # over the cap
    bad = app / "Contents" / "PlugIns" / "Bad.appex" / "Contents"
    bad.mkdir(parents=True)
    (bad / "Info.plist").write_bytes(b"<plist><dict><key>NSExtension")  # malformed
    fifo = app / "Contents" / "PlugIns" / "Pipe.appex" / "Contents"
    fifo.mkdir(parents=True)
    os.mkfifo(fifo / "Info.plist")  # never block on a pipe
    meta = app / "Contents" / "Resources" / "Metadata.appintents"
    meta.mkdir(parents=True)
    (meta / "extract.actionsdata").write_text("{not json")
    ev = read(app)
    assert ev.intents == ("INCreateNoteIntent",) and ev.ai_schemas == ()
    (meta / "extract.actionsdata").write_text(json.dumps({"actions": {"A": {"systemProtocols": [f"{PROTO}.SendMail"]}}}))
    monkeypatch.setattr(evidence, "MAX_ACTIONS", 10)
    assert read(app).ai_schemas == ()  # over its cap
    monkeypatch.setattr(evidence, "MAX_EXTENSIONS", 0)
    assert read(app).intents == ()


def test_the_responsible_apps_usage_keys(tmp_path, monkeypatch):
    from neo.decide import media

    app = make_bundle(tmp_path, "Host", {"NSCalendarsFullAccessUsageDescription": "x", "NSContactsUsageDescription": "y"})
    exe = app / "Contents" / "MacOS" / "Host"
    monkeypatch.setattr(media, "responsible_pid", lambda pid: 4242)
    monkeypatch.setattr(media, "_proc_path", lambda pid: str(exe) if pid == 4242 else "")
    evidence.responsible_usage.cache_clear()
    try:
        assert evidence.responsible_usage() == {"NSCalendarsFullAccessUsageDescription", "NSContactsUsageDescription"}
        evidence.responsible_usage.cache_clear()
        monkeypatch.setattr(media, "_proc_path", lambda pid: "/usr/bin/python3")  # not inside an app
        assert evidence.responsible_usage() == frozenset()
    finally:
        evidence.responsible_usage.cache_clear()


# ---- through the inventory ----------------------------------------------------------------------


@pytest.fixture
def mac(tmp_path, monkeypatch):
    apps = tmp_path / "Applications"
    apps.mkdir()
    m = monkeypatch
    m.setattr(inv, "APP_DIRS", (str(apps),))
    m.setattr(inv, "SERVICES_DIR", str(tmp_path / "CoreServices"))
    m.setattr(inv, "_spotlight_app_paths", lambda: [])
    m.setattr(inv, "_launchservices_app_paths", lambda: [])
    m.setattr(inv, "_spotlight_usage", lambda paths: {p: (None, 0, 0, "", "") for p in paths})
    m.setattr(inv, "_running_reals", lambda: set())
    m.setattr(inv, "_dock_ids", lambda: set())
    m.setattr(inv, "_scheme_defaults", lambda: {})
    m.setattr(inv, "_WORDS_FILE", str(FIX / "words.txt"))
    m.setattr(inv, "_CONNECTIVES_FILE", str(FIX / "connectives.txt"))
    m.setattr(inv, "_words_known", {})
    m.setattr(inv, "_connectives", None)
    inv._reset()
    yield apps
    inv._reset()


def test_inventory_apps_carry_the_evidence_and_it_is_cached_by_the_plist(mac):
    app = make_bundle(mac, "Chatter", {"NSContactsUsageDescription": "find people"}, exts={
        "PlugIns/Intents.appex": appex("vendor.intents-service", IntentsSupported=["INSendMessageIntent"]),
        "PlugIns/Share.appex": appex(SHARE, NSExtensionActivationRule={"NSExtensionActivationSupportsText": 1}),
    })
    a = next(x for x in inv.inventory() if x.bundle_id == "test.chatter")
    assert a.intents == ("INSendMessageIntent",) and a.share_text and a.usage == {"NSContactsUsageDescription"}
    assert a.kind == "app" and a.note == "" and not a.doc_based
    # an extension changes, the app's own Info.plist doesn't: the cached reading stands
    write_plist(app / "Contents" / "PlugIns" / "Intents.appex" / "Contents" / "Info.plist",
                appex("vendor.intents-service", IntentsSupported=["INCreateNoteIntent"]))
    assert next(x for x in inv.inventory() if x.bundle_id == "test.chatter").intents == ("INSendMessageIntent",)
    # an app update rewrites its Info.plist: read again
    info = app / "Contents" / "Info.plist"
    write_plist(info, {**plistlib.loads(info.read_bytes()), "CFBundleVersion": "2"})
    assert next(x for x in inv.inventory() if x.bundle_id == "test.chatter").intents == ("INCreateNoteIntent",)


def test_snapshot_and_lookups_never_start_a_scan(mac, monkeypatch):
    from neo.decide import media

    make_bundle(mac, "Chatter", {"CFBundleName": "Chatter"})
    assert inv.snapshot() is None and inv.app_for_bundle("test.chatter") is None
    inv.inventory()
    snap = inv.snapshot()
    assert [a.bundle_id for a in snap] == ["test.chatter"]
    snap.clear()  # a copy: the cache is untouched
    assert inv.app_for_bundle("TEST.CHATTER").name == "Chatter" and inv.app_for_bundle("") is None
    monkeypatch.setattr(media, "responsible_pid", lambda pid: 77 if pid == 78 else pid)
    monkeypatch.setattr(media, "_bundle_id_of_pid", lambda pid: {77: "test.chatter"}.get(pid, ""))
    assert inv.app_for_pid(78).bundle_id == "test.chatter"  # a helper process: its app
    assert inv.app_for_pid(99) is None


def mk(name: str, bid: str, **kw) -> inv.App:
    return inv.App(bundle_id=bid, name=name, path=f"/Applications/{name}.app", **kw)


@pytest.fixture
def words(monkeypatch):
    monkeypatch.setattr(inv, "_WORDS_FILE", str(FIX / "words.txt"))
    monkeypatch.setattr(inv, "_CONNECTIVES_FILE", str(FIX / "connectives.txt"))
    monkeypatch.setattr(inv, "_words_known", {})
    monkeypatch.setattr(inv, "_connectives", None)


@pytest.mark.parametrize(
    "text, want",
    [
        ("message john on Chatterbox", "Chatterbox"),
        ("Neo,   open   Chatterbox now", "Chatterbox"),  # positions are the text's own (runs of spaces)
        ("hey neo play 505 on quill mail.", "quill mail"),
        ("open\tChatterbox", "Chatterbox"),
        ("play something", None),
    ],
)
def test_named_span_points_into_the_text_as_given(words, text, want):
    apps = [mk("Chatterbox", "test.chatterbox"), mk("Quill Mail", "test.quill")]
    hit = inv.named_span(text, apps)
    if want is None:
        assert hit is None and inv.named_in(text, apps) is None
        return
    app, a, b = hit
    assert text[a:b] == want and app is inv.named_in(text, apps)
