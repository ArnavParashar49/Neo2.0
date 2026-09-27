"""Agent: AsyncAgent for code that isn't async.

It runs a private event loop on one daemon thread (NEO's session locks, a provider's HTTP client
and the browser all belong to one loop) and calls into it. Inside a running event loop use
:class:`AsyncAgent` instead: blocking that loop on the agent would stall both.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import weakref
from collections.abc import Callable, Coroutine, Iterator, Mapping
from typing import Any, Literal

from neo2.events import Event
from neo2.sdk.agent import AsyncAgent
from neo2.sdk.approval import Approval, ApprovalRequest, guard_reentry
from neo2.sdk.config import AgentConfig
from neo2.sdk.hosttools import HostTool, host_tool
from neo2.sdk.results import Result, ToolInfo, ToolOutcome
from neo2.sdk.tools_api import ToolTurn

__all__ = ["Agent"]

_END = object()


class Agent:
    """The sync twin of :class:`AsyncAgent`: the same keywords, the same results.

        with neo2.Agent(neo2.AgentConfig(tools="web"), provider=brain) as neo:
            r = neo.ask("what's the weather in Lisbon?")

    Callable from any thread except inside a running event loop (RuntimeError: use AsyncAgent).
    close() (or leaving the `with`) stops its loop and joins its thread."""

    def __init__(self, config: AgentConfig | None = None, /, **kwargs: Any) -> None:
        _not_in_a_loop()
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="neo2-agent", daemon=True)
        self._thread.start()
        try:
            self._agent = self._run(_make(config, kwargs))
        except BaseException:
            self._stop()
            raise
        self._thread.name = f"neo2-agent-{self._agent.name}"
        # an Agent dropped without close(): its loop stops with it rather than idling forever
        self._finalizer = weakref.finalize(self, _stop_loop, self._loop)

    # ---- plumbing ------------------------------------------------------------------------------
    def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError("use AsyncAgent: this is the agent's own event loop")
        try:
            _not_in_a_loop()
        except RuntimeError:
            coro.close()
            raise
        if self._closed:
            coro.close()
            raise RuntimeError("this agent is closed")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _stop(self) -> None:
        async def drain() -> None:
            me = asyncio.current_task()
            rest = [t for t in asyncio.all_tasks() if t is not me]
            for t in rest:
                t.cancel()
            await asyncio.gather(*rest, return_exceptions=True)

        if self._loop.is_running():
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(drain(), self._loop).result(timeout=10)
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        if not self._thread.is_alive():
            self._loop.close()

    # ---- the API -------------------------------------------------------------------------------
    @property
    def async_agent(self) -> AsyncAgent:
        """The AsyncAgent underneath (use it only from inside this agent's own loop)."""
        return self._agent

    @property
    def name(self) -> str:
        return self._agent.name

    @property
    def closed(self) -> bool:
        return self._closed

    def ask(self, text: str, **kw: Any) -> Result:
        guard_reentry()  # checked here: the approver's context doesn't cross into the agent's loop
        return self._run(self._agent.ask(text, **kw))

    def stream(self, text: str, **kw: Any) -> Iterator[Event]:
        """ask(), as its events while it runs (the last one is Completed)."""
        guard_reentry()
        _not_in_a_loop()
        if self._closed:
            raise RuntimeError("this agent is closed")
        return self._stream(text, kw)

    def _stream(self, text: str, kw: dict[str, Any]) -> Iterator[Event]:
        q: queue.Queue[Any] = queue.Queue()

        async def pump() -> None:
            try:
                async for e in self._agent.stream(text, **kw):
                    q.put(e)
            except BaseException as e:  # noqa: BLE001 — handed to the caller's thread
                q.put(_Failed(e))
            finally:
                q.put(_END)

        fut = asyncio.run_coroutine_threadsafe(pump(), self._loop)
        try:
            while True:
                item = q.get()
                if item is _END:
                    return
                if isinstance(item, _Failed):
                    raise item.error
                yield item
        finally:
            if not fut.done():
                fut.cancel()

    def approve(self, action_id: str, *, via: Approval) -> Result:
        return self._run(self._agent.approve(action_id, via=via))

    def deny(self, action_id: str, *, via: Approval | None = None) -> Result:
        return self._run(self._agent.deny(action_id, via=via))

    def pending(self, conversation: str = "default") -> ApprovalRequest | None:
        return self._run(_call(self._agent.pending, conversation))

    def cancel(self, conversation: str | None = None) -> int:
        return self._run(self._agent.cancel(conversation))

    def tool(self, fn: Callable[..., Any] | None = None, /, **host_tool_kwargs: Any) -> Any:
        """Decorator: the function becomes one of this agent's tools (see neo2.host_tool)."""

        def deco(f: Callable[..., Any]) -> HostTool:
            t = host_tool(f, **host_tool_kwargs)
            self.add_tool(t)
            return t

        return deco(fn) if fn is not None else deco

    def add_tool(self, t: HostTool | Callable[..., Any]) -> None:
        self._run(_call(self._agent.add_tool, t))

    def remove_tool(self, name: str) -> None:
        self._run(_call(self._agent.remove_tool, name))

    @property
    def tools(self) -> list[ToolInfo]:
        return self._agent.tools

    def export_tools(self, fmt: str, *, prefix: str = "") -> list[dict[str, Any]]:
        return self._agent.export_tools(fmt, prefix=prefix)

    @contextlib.contextmanager
    def turn(
        self,
        *,
        tool_context: Mapping[str, Any] | None = None,
        conversation: str = "default",
        source: Literal["user", "model"] = "model",
    ) -> Iterator[ToolTurn]:
        """One request of the host's own model loop (see AsyncAgent.turn)."""
        if self._closed:
            raise RuntimeError("this agent is closed")
        t = self._agent._new_turn(tool_context, conversation, source)
        try:
            yield t
        finally:
            t.closed = True

    def call_tool(self, name: str, args: Mapping[str, Any] | None = None, *, turn: ToolTurn | None = None) -> ToolOutcome:
        return self._run(self._agent.call_tool(name, args, turn=turn))

    def history(self, conversation: str = "default") -> list[dict[str, Any]]:
        return self._run(_call(self._agent.history, conversation))

    def reset(self, conversation: str = "default") -> None:
        self._run(_call(self._agent.reset, conversation))

    def forget_all(self) -> dict[str, int]:
        return self._run(self._agent.forget_all())

    def close(self) -> None:
        """Close the agent, stop its loop and join its thread. Idempotent."""
        if self._closed:
            return
        if threading.current_thread() is self._thread:
            raise RuntimeError("close() from inside the agent's own loop would never return")
        try:
            if self._loop.is_running():
                asyncio.run_coroutine_threadsafe(self._agent.aclose(), self._loop).result(timeout=30)
        finally:
            self._closed = True
            self._finalizer.detach()
            self._stop()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Agent {self._agent.name!r}{' closed' if self._closed else ''}>"


class _Failed:
    def __init__(self, error: BaseException) -> None:
        self.error = error


async def _make(config: AgentConfig | None, kwargs: dict[str, Any]) -> AsyncAgent:
    return AsyncAgent(config, **kwargs)


async def _call(fn: Callable[..., Any], *a: Any) -> Any:
    return fn(*a)


def _stop_loop(loop: asyncio.AbstractEventLoop) -> None:
    if not loop.is_closed():
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(loop.stop)


def _not_in_a_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("use AsyncAgent: neo2.Agent blocks, and this thread is running an event loop")
