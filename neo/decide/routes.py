"""Destinations that are no app: the Mac's shared calendar and reminder stores, as candidates.

When access to a store is granted, the engine weighs it like an app, with the user's use of the
store (days with items in the last 30) standing in for an app's use. Before NEO may read the
store its use is unknown (:func:`pending`), and the engine counts it as much as the most-used app
of the kind. A store the user refused NEO, or one NEO could never ask for, is no candidate.
What Laya reads about a store is counts only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neo.decide import eventkit

if TYPE_CHECKING:
    from neo.decide.inventory import App

__all__ = ["ROUTES", "route_app", "pending"]

ROUTES: dict[str, eventkit.Entity] = {"eventkit.events": "event", "eventkit.reminders": "reminder"}
_NAMES = {"event": "your Mac's calendar store", "reminder": "your Mac's reminders store"}
_ITEMS = {"event": ("calendar", "calendars", "events"), "reminder": ("list", "lists", "reminders added")}


def route_app(kind: str, route: str) -> App | None:
    """``route`` as a candidate for ``kind``, or None when its store can't be used."""
    from neo.decide.inventory import App

    entity = ROUTES.get(route)
    if entity is None or eventkit.available(entity):
        return None
    st = eventkit.status(entity)
    if st in ("denied", "restricted"):
        return None
    name, (one, many, items) = _NAMES[entity], _ITEMS[entity]
    shown = f"{name} (every {one} app here shows it)"
    if st != "full":  # can't read it: nothing to count
        access = "access not asked yet" if st == "not_determined" else "NEO access: add only"
        return App(bundle_id=f"route:{route}", name=name, path="", kind="route", note=f"{shown}; {access}")
    c = eventkit.counts(entity)
    return App(
        bundle_id=f"route:{route}",
        name=name,
        path="",
        kind="route",
        note=f"{shown}: {c.writable} writable {one if c.writable == 1 else many}, "
        f"{items} on {c.days_30} of the last 30 days; NEO access: full",
        last_used=c.last,
        days_used_30=c.days_30,
    )


def pending(app: App) -> bool:
    """NEO can't read the route's store yet (not asked, or add-only), so its use is unknown."""
    entity = ROUTES.get(app.bundle_id.removeprefix("route:"))
    return entity is not None and eventkit.status(entity) in ("not_determined", "write_only")
