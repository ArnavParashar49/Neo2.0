"""Weather in ~0.5 s from Open-Meteo (free, no key) — instead of a web search and a page read.

"Here" is the place the user told NEO (memory profile `city`), else `NEO_HOME_LOCATION`, else a
rough location from the network (ipinfo.io, cached for the session).
"""

from __future__ import annotations

import time

import httpx

from neo.agent.registry import ToolContext, tool

_FORECAST = "https://api.open-meteo.com/v1/forecast"
_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}
_WHEN = {"now": 0, "today": 0, "tonight": 0, "tomorrow": 1}
_here: tuple[float, float, str, float] | None = None  # lat, lon, name, when looked up


def _geocode(place: str, client: httpx.Client) -> tuple[float, float, str] | None:
    r = client.get(_GEOCODE, params={"name": place, "count": 1}, timeout=6)
    hits = (r.json() or {}).get("results") or []
    if not hits:
        return None
    h = hits[0]
    name = ", ".join(x for x in (h.get("name"), h.get("country")) if x)
    return float(h["latitude"]), float(h["longitude"]), name


def _home(client: httpx.Client) -> tuple[float, float, str] | None:
    global _here
    from neo.config import settings

    for place in (_remembered_city(), settings().home_location):
        if place and (g := _geocode(place, client)):
            return g
    if _here and time.time() - _here[3] < 6 * 3600:
        return _here[0], _here[1], _here[2]
    try:
        j = client.get("https://ipinfo.io/json", timeout=4).json()
        lat, lon = (float(x) for x in j["loc"].split(","))
        _here = (lat, lon, ", ".join(x for x in (j.get("city"), j.get("country")) if x), time.time())
        return _here[:3]
    except Exception:  # noqa: BLE001
        return None


def _remembered_city() -> str:
    try:
        from neo.memory.store import store

        db = store()._db
        row = db.execute(
            "SELECT text FROM memory WHERE kind='profile' AND key IN ('city','location','home') ORDER BY updated DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else ""
    except Exception:  # noqa: BLE001
        return ""


def forecast(place: str = "", when: str = "today") -> str:
    with httpx.Client() as client:
        loc = _geocode(place, client) if place else _home(client)
        if loc is None:
            return f"Error: I couldn't find {place!r}." if place else "Error: I don't know where you are — tell me your city."
        lat, lon, name = loc
        r = client.get(
            _FORECAST,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 7,
            },
            timeout=6,
        ).json()
    cur, day = r["current"], r["daily"]

    def daily(i: int) -> str:
        return (
            f"{_WMO.get(day['weather_code'][i], 'mixed')}, high {round(day['temperature_2m_max'][i])}°, "
            f"low {round(day['temperature_2m_min'][i])}°, {day['precipitation_probability_max'][i] or 0}% chance of rain"
        )

    w = " ".join((when or "today").lower().split()).removeprefix("on ")
    w = {"tomorrow morning": "tomorrow", "tomorrow afternoon": "tomorrow", "tomorrow evening": "tomorrow",
         "tomorrow night": "tomorrow", "this morning": "today", "this afternoon": "today", "this evening": "today",
         "later": "today", "later today": "today", "next week": "week", "next weekend": "week"}.get(w, w)
    days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    if w in days:  # "on friday": that day's forecast
        import datetime as _dt

        ahead = (days.index(w) - _dt.date.fromisoformat(day["time"][0]).weekday()) % 7
        if ahead < len(day["time"]):
            return f"{name} on {w.capitalize()}: {daily(ahead)}."
    if w in ("week", "this week", "weekend", "this weekend", "next few days", "the week"):
        days = [f"{day['time'][i]}: {daily(i)}" for i in range(len(day["time"]))]
        return f"{name} — next 7 days:\n" + "\n".join(days)
    idx = _WHEN.get(w, 0)
    now = (
        f"{name}: {round(cur['temperature_2m'])}°C now (feels like {round(cur['apparent_temperature'])}°), "
        f"{_WMO.get(cur['weather_code'], 'mixed')}, humidity {cur['relative_humidity_2m']}%, wind {round(cur['wind_speed_10m'])} km/h."
    )
    if idx == 0:
        return f"{now} Today: {daily(0)}. Tomorrow: {daily(1)}."
    return f"{name} tomorrow: {daily(1)}. (Now: {round(cur['temperature_2m'])}°C, {_WMO.get(cur['weather_code'], 'mixed')}.)"


# Time words, so a place never swallows them ("paris next week", "london tomorrow morning").
_TIME = (
    r"(?:today|tonight|tomorrow(?:\s+(?:morning|afternoon|evening|night))?|this\s+(?:morning|afternoon|evening)|"
    r"later(?:\s+today)?|(?:this|next)\s+week(?:end)?|week(?:end)?|now|right\s+now|"
    r"(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))"
)
# A place: 1–4 words starting with a letter, never a time word, never "the/my/our …" (that's a
# thing — "the forecast for our sales" — not a city).
_PLACE = (
    r"(?:\s+(?:in|at|around)\s+(?!(?:the|my|our|your|this|that|here)\b)(?!" + _TIME + r"\b)"
    r"(?P<place>[a-z][\w.'-]*(?:\s+(?!" + _TIME + r"\b)[a-z][\w.'-]*){0,3}?))?"
)
_WHEN_RE = r"(?:\s+(?:for\s+)?(?:the\s+)?(?P<when>" + _TIME + r"))?"
_WHEN2 = _WHEN_RE.replace("<when>", "<when2>")  # "in new york tomorrow": the time may come after the place
_ARGS = {"place": "<place>", "when": "<when>", "when2": "<when2>"}
_Q = r"\s*[?.!]*\s*$"


@tool(
    "weather",
    "Current weather and forecast. place: a city (omit for where the user is). when: now | today | "
    "tomorrow | week.",
    {
        "type": "object",
        "properties": {"place": {"type": "string"}, "when": {"type": "string"}},
    },
    parallel_safe=True,
    category="web",
    timeout=15,
    fast_path=[
        (rf"^\s*(?:what(?:'s| is)\s+the\s+)?weather(?:\s+like)?(?:\s+(?:going\s+to\s+be|gonna\s+be))?{_WHEN_RE}{_PLACE}{_WHEN2}{_Q}", _ARGS),
        (rf"^\s*(?:how(?:'s| is)\s+the\s+weather|what(?:'s| is)\s+it\s+like\s+outside){_WHEN_RE}{_PLACE}{_WHEN2}{_Q}", _ARGS),
        (rf"^\s*(?:is\s+it|will\s+it|is\s+it\s+going\s+to|it\s+will)\s+(?:rain|snow|be\s+(?:hot|cold|sunny|cloudy|windy))(?:ing)?{_WHEN_RE}{_PLACE}{_WHEN2}{_Q}", _ARGS),
        (rf"^\s*how\s+(?:hot|cold|warm)\s+is\s+it(?:\s+outside)?{_WHEN_RE}{_PLACE}{_WHEN2}{_Q}", _ARGS),
        # "temperature"/"forecast" only count with "outside", "weather", a place or a time —
        # "what's the temperature of the sun" and "the forecast for Q3 sales" are not weather.
        (rf"^\s*what(?:'s| is)\s+the\s+(?:temperature|temp)\s+outside{_WHEN_RE}{_PLACE}{_WHEN2}{_Q}", _ARGS),
        (rf"^\s*what(?:'s| is)\s+the\s+(?:temperature|temp|forecast)(?=\s+(?:in|at|around|for\s+(?:today|tonight|tomorrow|the\s+week)|today|tonight|tomorrow|this|next))(?:\s+outside)?{_WHEN_RE}{_PLACE}{_WHEN2}{_Q}", _ARGS),
        (rf"^\s*(?:the\s+)?weather\s+forecast{_WHEN_RE}{_PLACE}{_WHEN2}{_Q}", _ARGS),
        (rf"^\s*(?:the\s+)?forecast(?=\s+(?:for\s+)?(?:the\s+)?(?:today|tonight|tomorrow|week|weekend|this|next|in\s))" + rf"{_WHEN_RE}{_PLACE}{_WHEN2}{_Q}", _ARGS),
        (rf"^\s*do\s+i\s+need\s+an\s+umbrella{_WHEN_RE}{_PLACE}{_WHEN2}{_Q}", _ARGS),
    ],
)
async def weather(a: dict, c: ToolContext) -> str:
    import asyncio

    place = (a.get("place") or "").strip()
    if place.lower() in ("here", "outside", "my area", "my city"):
        place = ""
    return await asyncio.to_thread(forecast, place, (a.get("when") or a.get("when2") or "today"))
