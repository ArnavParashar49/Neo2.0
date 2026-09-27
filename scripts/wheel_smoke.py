"""Check an installed NEO wheel as another app would use it (CI's sdk-core job).

    <venv>/bin/python scripts/wheel_smoke.py [--extra web]

Run it with the python of a clean venv the wheel was installed into. It checks that neo2 comes
from that venv (not a checkout); that `import neo2` loads only the standard library and
`from neo2 import AsyncAgent` nothing heavy; that none of torch, pyobjc, playwright,
sentence-transformers, laya, mlx, a vendor SDK, mcp or langchain-core is installed (with
`--extra web`: that ddgs, trafilatura and pypdf are); and that an agent with a scripted brain
answers, with and without NEO's web tools. No key, no network, and nothing written to ~/.neo.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

NEVER = ("torch", "objc", "AppKit", "Quartz", "playwright", "sentence_transformers", "laya", "mlx",
         "openai", "anthropic", "google.genai", "mcp", "langchain_core")  # fmt: skip
WEB = ("ddgs", "trafilatura", "pypdf")


def installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # its parent package isn't there
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", choices=("web",), help="the wheel was installed with this extra")
    extra = ap.parse_args().extra
    problems: list[str] = []
    neo_home = Path.home() / ".neo"
    had_neo_home = neo_home.exists()

    wrongly = [m for m in NEVER if installed(m)]
    if wrongly:
        problems.append(f"installed but must not be: {wrongly}")
    web = {m: installed(m) for m in WEB}
    if extra == "web" and not all(web.values()):
        problems.append(f"the web extra is missing packages: {web}")
    if extra is None and any(web.values()):
        problems.append(f"no extras asked for, but the web extra's packages are here: {web}")

    before = set(sys.modules)
    import neo2

    outside = sorted({m.split(".")[0] for m in set(sys.modules) - before} - set(sys.stdlib_module_names) - {"neo2"})
    if outside:
        problems.append(f"import neo2 loaded more than the standard library: {outside}")
    if not Path(neo2.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        problems.append(f"neo2 came from {neo2.__file__}, not this venv ({sys.prefix})")

    t = time.perf_counter()
    from neo2 import AsyncAgent

    seconds = round(time.perf_counter() - t, 3)
    from neo2.testing import ScriptedBrain, ToolCall, Turn

    async def ask() -> list[str]:
        out = []
        async with AsyncAgent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain(["Hello."])) as agent:
            out.append((await agent.ask("hi")).status)
        brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "clock", {})]), "It's noon."])
        async with AsyncAgent(neo2.AgentConfig(tools="web", fast_paths=False), provider=brain) as agent:
            r = await agent.ask("what time is it?")
            out.append(f"{r.status}:{','.join(s.tool for s in r.steps)}")
        return out

    asks = asyncio.run(ask())
    if asks != ["done", "done:clock"]:
        problems.append(f"the scripted agent didn't answer: {asks}")
    loaded = [m for m in NEVER if m in sys.modules]
    if loaded:
        problems.append(f"loaded while answering: {loaded}")
    if neo_home.exists() and not had_neo_home:
        problems.append("~/.neo was created")

    print(json.dumps({"neo2": neo2.__file__, "from neo2 import AsyncAgent (s)": seconds, "asks": asks,
                      "web extra": web, "problems": problems}, indent=1))  # fmt: skip
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
