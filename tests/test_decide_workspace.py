"""neo.decide.workspace: LaunchServices' default handlers, read only, on faked lookups."""

from __future__ import annotations

from neo.decide import inventory, workspace

PATHS = {"/Applications/Postbox.app": "test.postbox", "/Applications/Browsa.app": "test.browsa",
         "/Applications/Datebook.app": "test.datebook"}


def fake(monkeypatch, defaults: dict[str, str], handlers: dict[str, list[str]]):
    seen = []

    def default_path(s):
        seen.append(s)
        return defaults.get(s)

    monkeypatch.setattr(inventory, "_ls_default_path", default_path)
    monkeypatch.setattr(workspace, "_handler_paths", lambda scheme: handlers.get(scheme, []))
    monkeypatch.setattr(workspace, "_bundle_id", lambda path: PATHS.get(path))
    return seen


def test_default_handlers_by_scheme_and_extension(monkeypatch):
    seen = fake(monkeypatch, {"mailto:": "/Applications/Postbox.app", "ics": "/Applications/Datebook.app"},
                {"https": ["/Applications/Browsa.app", "/Applications/Other.app", "/Applications/Browsa.app"]})
    assert workspace.default_handler("mailto") == workspace.default_handler("MAILTO:") == "test.postbox"
    assert workspace.default_handler("sms") is None and workspace.default_handler("") is None
    assert workspace.default_for_extension(".ics") == workspace.default_for_extension("ICS") == "test.datebook"
    assert workspace.default_for_extension("com.vendor.type") is None  # an extension, not a type id
    assert workspace.default_for_extension("../x") is None
    assert workspace.handlers("https") == ["test.browsa"]  # unreadable bundles skipped, each once
    assert workspace.handlers("") == []
    assert "" not in seen and "com.vendor.type" not in seen
