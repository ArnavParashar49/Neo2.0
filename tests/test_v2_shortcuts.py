"""Browser and window commands run as the Mac's standard shortcuts, instantly (English + Hinglish)."""

import pytest


@pytest.mark.parametrize(
    "text,keys",
    [
        ("new tab", "cmd+t"),
        ("naya tab kholo", "cmd+t"),
        ("close this tab", "cmd+w"),
        ("ye tab band karo", "cmd+w"),
        ("reopen the last closed tab", "cmd+shift+t"),
        ("bring back the tab I just closed", "cmd+shift+t"),
        ("band kiya hua tab wapas kholo", "cmd+shift+t"),
        ("next tab", "ctrl+tab"),
        ("agla tab", "ctrl+tab"),
        ("pichla tab", "ctrl+shift+tab"),
        ("go to the third tab", "cmd+3"),
        ("switch to tab 2", "cmd+2"),
        ("open an incognito window", "cmd+shift+n"),
        ("close all tabs", "cmd+shift+w"),
        ("go to the address bar", "cmd+l"),
        ("bookmark this page", "cmd+d"),
        ("hard refresh", "cmd+shift+r"),
        ("page refresh karo", "cmd+r"),
    ],
)
def test_shortcut_commands(text, keys):
    from neo.agent import fastpath
    from neo.tools import load_all

    load_all()
    (tool, args), = fastpath.plan(text)
    assert tool == "hotkey" and args["keys"] == keys


def test_the_hotkey_tool_teaches_the_standard_shortcuts():
    from neo.agent.registry import registry
    from neo.tools import load_all

    load_all()
    d = registry().get("hotkey").description
    for s in ("reopen closed tab cmd+shift+t", "new tab cmd+t", "close tab cmd+w", "address bar cmd+l"):
        assert s in d
