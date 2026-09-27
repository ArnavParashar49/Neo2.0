"""The distribution: a small core, extras for everything else, NEO.app's `app` extra exactly the
dependency list it always had, the licence and links as the owner set them, and the MCP console
script and the typing marker shipped."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.portable  # Linux or macOS, any install (what needs an extra skips): CI's sdk-core job

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
PROJECT = PYPROJECT["project"]
EXTRAS: dict[str, list[str]] = PROJECT["optional-dependencies"]
SELF = re.compile(r"^neo\[([a-z0-9_,-]+)\]$")

# NEO.app's dependencies before they were split into core + extras (pyproject.toml at 4764b73).
APP_BEFORE = {
    "google-genai>=2.20",
    "openai>=2",
    "anthropic>=1",
    "laya>=0.3",
    "mlx-audio; sys_platform == 'darwin'",
    "mlx-lm; sys_platform == 'darwin'",
    "misaki[en]; sys_platform == 'darwin'",
    "vosk",
    "sounddevice",
    "soundfile",
    "silero-vad",
    "torch",
    "sentence-transformers",
    "playwright",
    "pyobjc-framework-Cocoa; sys_platform == 'darwin'",
    "pyobjc-framework-Quartz; sys_platform == 'darwin'",
    "pyobjc-framework-ApplicationServices; sys_platform == 'darwin'",
    "pyobjc-framework-EventKit>=12.2,<13; sys_platform == 'darwin'",
    "pyobjc-framework-Contacts>=12.2,<13; sys_platform == 'darwin'",
    "pillow",
    "httpx",
    "ddgs",
    "tld",
    "lxml",
    "trafilatura",
    "pypdf",
    "python-dateutil",
    "numpy",
    "pydantic>=2",
    "pydantic-settings>=2.7",
    "python-dotenv",
    "websockets>=13",
}
HEAVY = ("torch", "laya", "mlx", "pyobjc", "playwright", "sentence-transformers", "google-genai", "openai", "anthropic")


def _flat(extra: str, seen: tuple[str, ...] = ()) -> set[str]:
    """An extra's requirements, with NEO's own extras (neo[a,b]) expanded."""
    assert extra in EXTRAS, f"no extra named {extra!r}"
    assert extra not in seen, f"extras include each other in a loop: {seen + (extra,)}"
    out: set[str] = set()
    for req in EXTRAS[extra]:
        m = SELF.match(req)
        if m:
            for e in m.group(1).split(","):
                out |= _flat(e, seen + (extra,))
        else:
            out.add(req)
    return out


def _name(req: str) -> str:
    return re.split(r"[\s;\[<>=!~]", req, maxsplit=1)[0].lower()


def test_neo_app_gets_exactly_the_dependencies_it_always_had():
    assert set(PROJECT["dependencies"]) | _flat("app") == APP_BEFORE


def test_the_core_is_small_and_nothing_heavy():
    core = {_name(r) for r in PROJECT["dependencies"]}
    assert core == {"numpy", "httpx", "pydantic", "pydantic-settings", "python-dotenv"}
    assert not [n for n in core if n.startswith(HEAVY)]


def test_the_extras():
    assert _flat("brains") == set(EXTRAS["gemini"]) | set(EXTRAS["openai"]) | set(EXTRAS["claude"])
    assert _flat("all") == _flat("app") | set(EXTRAS["mcp"]) | set(EXTRAS["langchain"])
    assert EXTRAS["mcp"] == ["mcp>=2.2,<3"]  # neo2.mcp uses mcp 2.2's low-level Server and mcp_types
    assert EXTRAS["langchain"] == ["langchain-core>=1,<2"]
    assert EXTRAS["web"] == ["ddgs", "trafilatura", "pypdf"]
    assert EXTRAS["browser"] == ["playwright"]
    # No extra the SDK's own users pick pulls in a model, a macOS framework or a browser.
    for extra in ("web", "brains", "mcp", "langchain"):
        heavy = [r for r in _flat(extra) if _name(r).startswith(("torch", "laya", "mlx", "pyobjc", "playwright"))]
        assert heavy == [], (extra, heavy)
    for extra in EXTRAS:  # every neo[...] names real extras
        _flat(extra)
    assert set(EXTRAS["dev"]) == {"pytest", "pytest-asyncio", "ruff", "mypy"}
    assert EXTRAS["train"] == ["datasets"]


def test_licence_author_and_links_are_the_owners():
    assert PROJECT["license"] == "MIT" and PROJECT["license-files"] == ["LICENSE"]
    assert PROJECT["authors"] == [{"name": "Arnav Parashar"}]
    assert PROJECT["urls"] == {
        "Homepage": "https://github.com/ArnavParashar49/Neo2.0",
        "Repository": "https://github.com/ArnavParashar49/Neo2.0",
        "GitHub": "https://github.com/ArnavParashar49",
        "LinkedIn": "https://www.linkedin.com/in/arnavparashar49/",
    }
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License\n\nCopyright (c) 2026 Arnav Parashar\n")
    readme = (ROOT / "README.md").read_text()
    assert "## Author" in readme and "[LinkedIn](https://www.linkedin.com/in/arnavparashar49/)" in readme
    assert "[MIT](LICENSE)" in readme


def test_python_version():
    # neo/decide uses Python 3.12 syntax (PEP 695); 3.13 waits for the owner's decision.
    assert PROJECT["requires-python"] == ">=3.12,<3.13"


def test_console_scripts_and_typing_marker():
    import neo2
    import neo2.mcp

    assert PROJECT["scripts"] == {"neo": "neo.main:main", "neo2-mcp": "neo2.mcp:main"}
    assert callable(neo2.mcp.main)
    assert (ROOT / "neo2" / "py.typed").is_file()
    assert PYPROJECT["tool"]["setuptools"]["package-data"]["neo2"] == ["py.typed"]
    assert neo2.__version__ == PROJECT["version"]


def test_the_portable_marker_is_registered():
    markers = PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("portable:") for m in markers)
