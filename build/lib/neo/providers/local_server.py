"""Launch / reuse a local ``mlx_lm.server`` (OpenAI-compatible) on demand.

The model is downloaded from Hugging Face on first use into the HF cache. Gemma 4 12B
4-bit is ~6.7 GB — the right size for an 18 GB Mac running everything else too.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import httpx

from neo.config import settings

_proc: subprocess.Popen | None = None


async def is_up(port: int = 8080) -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.5) as c:
            r = await c.get(f"http://127.0.0.1:{port}/v1/models")
            return r.status_code == 200
    except Exception:
        return False


async def ensure_running(port: int = 8080, model: str | None = None, wait_s: float = 120) -> bool:
    """Start the server if it isn't already serving. Returns True when ready."""
    global _proc
    if await is_up(port):
        return True
    model = model or settings().local_model
    log = settings().data_dir / "local_server.log"
    _proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server", "--model", model, "--port", str(port)],
        stdout=open(log, "ab"),  # noqa: SIM115 — must outlive this function
        stderr=subprocess.STDOUT,
        cwd=str(Path.home()),
    )
    deadline = asyncio.get_event_loop().time() + wait_s
    while asyncio.get_event_loop().time() < deadline:
        if await is_up(port):
            return True
        if _proc.poll() is not None:
            return False
        await asyncio.sleep(1.0)
    return False


def stop() -> None:
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
    _proc = None
