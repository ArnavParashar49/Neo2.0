"""Step 0's merge of the controls fixes, their self-test and main: the rules each side brought hold
together — protected targets are refused before anything is brought forward, and the self-test's
own window (NEO's child: macOS holds NEO responsible for it) is the one exception, by pid, only
while the test runs."""

from __future__ import annotations

import asyncio

import pytest

from neo.tools import computer
from neo.tools.computer import ax, pointer


@pytest.fixture
def neo_family(monkeypatch):
    from neo.decide import media

    monkeypatch.setattr(media, "_self_family", lambda: frozenset({10}))
    monkeypatch.setattr(media, "responsible_pid", lambda pid: 10 if pid in (10, 4242) else pid)


def test_the_test_window_is_neos_own_only_outside_a_self_test(neo_family):
    assert ax.is_neo(4242, "NEO Self-Test AB12")  # NEO started it: NEO's family
    with computer.allow_test_window(4242):
        assert not ax.is_neo(4242, "NEO Self-Test AB12") and not ax.is_neo(4242, "NEO")
        assert ax.is_neo(10) and ax.is_neo(999, "neo-ui")  # NEO itself stays NEO's
        assert not computer._protected("NEO", 4242)
        assert computer._protected("NEO", 999) and computer._protected("neo-ui")
    assert ax.is_neo(4242) and computer._test_window_pid == 0


def test_a_probe_refuses_neo_and_protected_apps_but_not_the_test_window(neo_family, monkeypatch):
    from neo import config

    monkeypatch.setenv("NEO_PROTECTED_APPS", '["HostApprover"]')
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(ax, "app_names", lambda pid: set())
    hits = {
        (1, 1): ax.Hit(pid=4242, app="NEO Self-Test AB12", regular=True),
        (2, 2): ax.Hit(pid=10, app="NEO", regular=True),
        (3, 3): ax.Hit(pid=77, app="HostApprover", regular=True),
        (4, 4): ax.Hit(pid=78, app="Finder", regular=True),
    }
    monkeypatch.setattr(ax, "hit", lambda x, y: hits[(x, y)])

    def probe(pt):
        try:
            asyncio.run(pointer.probe([pt]))
        except pointer.Refused as r:
            return "protected" if r.protected else "refused"
        return "ok"

    assert [probe(p) for p in hits] == ["protected", "protected", "protected", "ok"]
    with computer.allow_test_window(4242):
        assert [probe(p) for p in hits] == ["ok", "protected", "protected", "ok"]


def test_a_protected_app_is_never_brought_forward_by_the_pointer(monkeypatch):
    """bring() checks the app it would activate — an element's own app too — before activating it."""
    from neo import config
    from neo.tools.computer import apps, focus

    monkeypatch.setenv("NEO_PROTECTED_APPS", '["HostApprover"]')
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(ax, "app_names", lambda pid: {"HostApprover"} if pid == 77 else set())
    monkeypatch.setattr(ax, "regular", lambda pid: True)
    monkeypatch.setattr(focus, "front_app", lambda: ("Finder", 78))
    brought: list[int] = []

    async def bring(pid):
        brought.append(pid)
        return True, "x"

    monkeypatch.setattr(apps, "bring_pid_front", bring)
    t = pointer.Target([ax.Hit(pid=78, app="Finder", regular=True)])
    with pytest.raises(pointer.Refused) as r:
        asyncio.run(pointer.bring(t, [(5.0, 5.0)], want_pid=77))  # an element of the approver's
    assert r.value.protected and brought == []
