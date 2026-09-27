"""A host's own functions as an agent's tools (:func:`host_tool`, ``AsyncAgent.tool``).

    @agent.tool(risk="destructive", confirm=lambda a: f"refund {a['amount']} on order {a['order_id']}")
    async def refund(order_id: str, amount: float, ctx: neo2.ToolContext) -> str:
        return await billing.refund(order_id, amount, tenant=ctx.extras["tenant"])

- The schema comes from the typed signature (or `parameters`: a JSON schema or a pydantic model);
  a parameter annotated ToolContext is handed NEO's context (`ctx.extras` = the request's
  `tool_context`). A NEO-style handler ``(args: dict, ctx: ToolContext)`` — both annotated — works
  as is, with its schema given as `parameters`.
- A gated tool (risk="destructive", or `confirm`) asks before its function runs: NEO stages the
  exact validated arguments, and the function runs only after the host approves that card, with
  those arguments. `detail` (default: the arguments as JSON) is what the approval UI shows.
- Output is third-party content by default (`untrusted=True`): it's fenced for the model and the
  request counts as having read something untrusted (memory writes and data-carrying links ask
  first afterwards) — whatever it says, and when the function fails too (an exception's message,
  or a ToolOutput(ok=False) the host built, can carry the text it read). Only a raise or ok=False
  is a failure: returned text that starts with "Error" is still the tool's output. Pass
  untrusted=False for output the host vouches for.
- A model can't pass `confirm`, `cancel` or `_`-prefixed arguments (NEO drops them), so a tool
  can't declare them. Nor can a host tool take a built-in tool's name.
"""

from __future__ import annotations

import asyncio
import collections.abc
import copy
import dataclasses
import inspect
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_origin, get_type_hints

import pydantic

from neo.agent import confirm as _confirm
from neo.agent.confirm import Halt
from neo.agent.registry import RegisteredTool, ToolContext, ToolOutput, fence

__all__ = ["HostTool", "host_tool", "RESERVED_NAMES"]

Risk = Literal["safe", "reversible", "destructive"]
_RISKS = ("safe", "reversible", "destructive")
_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")  # what every model API accepts as a tool name
# Never a tool's name: they read as an approval channel, or are NEO's own meta-tool — or are the
# labels a request's taint gives the outer model's own words ("caller": a model wrote the task;
# "context": the context passed with it), which sit beside the names of the untrusted tools it read.
RESERVED_NAMES = frozenset({"approve", "confirm", "cancel", "deny", "more_tools", "caller", "context"})
_DROPPED_ARGS = ("confirm", "cancel")  # neo.agent.registry.Registry.invoke drops these (and "_…")
# What a failing untrusted tool said, at most: its fenced error stays whole where NEO cuts a result
# short (800 characters after an approval, neo.agent.session).
_ERROR_CHARS = 600


def builtin_names() -> frozenset[str]:
    """Every name NEO's own tools use — also those not loaded in this process or given to no agent."""
    from neo.agent.registry import catalog
    from neo.tools import PRESETS, REFUSED

    return frozenset({*PRESETS["all"], *REFUSED, *(t.name for t in catalog().all())})


@dataclass(frozen=True)
class _Binding:
    """How validated arguments reach the host's function."""

    kind: Literal["native", "model", "typed", "loose"]
    model: type[pydantic.BaseModel] | None = None
    ctx: str | None = None  # the parameter that gets the ToolContext
    model_param: str | None = None  # "model": the parameter that gets the model instance
    fields: tuple[tuple[str, str], ...] = ()  # "typed": (parameter, model attribute)
    accepts: frozenset[str] | None = None  # "loose": the parameters it takes (None: **kwargs)

    def prepare(self, args: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """(what the function gets, the normalised arguments that are staged and shown)."""
        if self.model is not None:
            inst = self.model.model_validate(dict(args))
            params = inst.model_dump(mode="json", by_alias=True)
            if self.kind == "model":
                return inst, params
            if self.kind == "native":
                return params, params
            return {p: getattr(inst, attr) for p, attr in self.fields}, params
        params = dict(args)
        if self.kind == "loose" and self.accepts is not None:
            return {k: v for k, v in params.items() if k in self.accepts}, params
        return params, params


@dataclass(frozen=True, eq=False)
class HostTool:
    """A host function registered as a tool of one agent. Calling it calls the function."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    risk: Risk = "reversible"
    confirm: bool | Callable[[dict[str, Any]], str] | None = None
    detail: Callable[[dict[str, Any]], str] | None = None
    untrusted: bool = True
    read_only: bool = False
    timeout: float = 120.0
    binding: _Binding = field(default=_Binding("loose"), repr=False)

    @property
    def gated(self) -> bool:
        """Asks before every run: risk="destructive" (unless confirm=False), or `confirm` given."""
        if self.confirm is None:
            return self.risk == "destructive"
        return self.confirm is not False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def registered(self) -> RegisteredTool:
        """This tool as an entry of an agent's registry."""
        return RegisteredTool(
            name=self.name,
            description=self.description,
            parameters=copy.deepcopy(self.parameters),
            handler=self.handle,
            risk=self.risk,
            parallel_safe=self.read_only,
            category="host",
            timeout=self.timeout,
            untrusted=self.untrusted,
            origin="host",
            read_only=self.read_only,
        )

    async def handle(self, args: dict[str, Any], ctx: ToolContext) -> str | ToolOutput:
        """NEO's handler: validate, ask first when gated, then run the host's function."""
        try:
            call, params = self.binding.prepare(args)
        except pydantic.ValidationError as e:
            return ToolOutput(f"Error: bad arguments for {self.name}: {_brief(e)}", ok=False)
        if self.gated:
            try:
                summary = self.confirm(params) if callable(self.confirm) else f"run {self.name}"
                detail = (
                    self.detail(params)
                    if self.detail is not None
                    else json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
                )
            except Exception as e:  # noqa: BLE001 — can't show what would run: it doesn't run
                return ToolOutput(f"Error: couldn't describe {self.name} for approval: {e}", ok=False)
            released = _confirm.gated(self.name, params, str(summary), str(detail))
            if isinstance(released, str):
                return released  # NEO's own question (a Halt): the run stops here
            call, _ = self.binding.prepare(released)  # exactly what the human approved
        try:
            out = _output(await self._run(call, ctx))
        except Exception as e:  # noqa: BLE001 — a failure is data for the model, like NEO's own
            out = ToolOutput(f"Error: {self.name} failed: {e}", ok=False)
            return self._failed(out, str(e), ctx)
        if isinstance(out, ToolOutput) and not out.ok and not out.halt:
            return self._failed(out, out.text, ctx)
        return out

    def _failed(self, out: ToolOutput, said: str, ctx: ToolContext) -> ToolOutput:
        """The host's function ran and failed. From an untrusted tool, what it said can be the
        third-party text it read: fenced, and the request has read it (NEO's registry and loop
        fence and taint a result only when it's ok)."""
        if not self.untrusted:
            return out
        body = said if len(said) <= _ERROR_CHARS else said[:_ERROR_CHARS] + " …"
        if self.name not in ctx.taint:
            ctx.taint.append(self.name)
        return dataclasses.replace(out, text=f"Error: {self.name} failed:\n{fence(self.name, body)}")

    async def _run(self, call: Any, ctx: ToolContext) -> Any:
        b = self.binding
        pos: list[Any] = []
        kw: dict[str, Any] = {}
        if b.kind == "native":
            pos = [call, ctx]
        elif b.kind == "model":
            kw[b.model_param or ""] = call
        else:
            kw.update(call)
        if b.ctx and b.kind != "native":
            kw[b.ctx] = ctx
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(*pos, **kw)
        # A plain function runs off the event loop (it carries the agent with it, like any thread
        # asyncio starts); one that returns an awaitable has it awaited here.
        r = await asyncio.to_thread(self.fn, *pos, **kw)
        if inspect.isawaitable(r):
            r = await r
        return r


def host_tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | type[pydantic.BaseModel] | None = None,
    risk: Risk = "reversible",
    confirm: bool | Callable[[dict[str, Any]], str] | None = None,
    detail: Callable[[dict[str, Any]], str] | None = None,
    untrusted: bool = True,
    read_only: bool = False,
    timeout: float = 120.0,
) -> Any:
    """Make `fn` a tool an agent can be given (``AsyncAgent(host_tools=[...])`` or ``add_tool``).
    Used bare (``@host_tool``) or with options (``@host_tool(risk="destructive")``)."""
    opts = dict(
        name=name,
        description=description,
        parameters=parameters,
        risk=risk,
        confirm=confirm,
        detail=detail,
        untrusted=untrusted,
        read_only=read_only,
        timeout=timeout,
    )
    if fn is None:
        return lambda f: _build(f, **opts)
    return _build(fn, **opts)


def _build(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    parameters: dict[str, Any] | type[pydantic.BaseModel] | None,
    risk: Risk,
    confirm: bool | Callable[[dict[str, Any]], str] | None,
    detail: Callable[[dict[str, Any]], str] | None,
    untrusted: bool,
    read_only: bool,
    timeout: float,
) -> HostTool:
    if isinstance(fn, HostTool):
        raise TypeError(f"{fn.name} is already a host tool")
    if not callable(fn):
        raise TypeError("a host tool is a function")
    tname = name or getattr(fn, "__name__", "")
    check_name(tname)
    if risk not in _RISKS:
        raise ValueError(f"risk must be one of {', '.join(_RISKS)}")
    if confirm is not None and not isinstance(confirm, bool) and not callable(confirm):
        raise TypeError("confirm is True/False or a function of the arguments returning the approval summary")
    if detail is not None and not callable(detail):
        raise TypeError("detail is a function of the arguments returning what the approval UI shows")
    if not timeout or timeout <= 0:
        raise ValueError("timeout must be positive")
    desc = (description if description is not None else inspect.getdoc(fn)) or tname
    schema, binding = _schema_and_binding(fn, tname, parameters)
    check_parameters(tname, schema)
    return HostTool(
        name=tname,
        description=desc.strip(),
        parameters=schema,
        fn=fn,
        risk=risk,
        confirm=confirm,
        detail=detail,
        untrusted=bool(untrusted),
        read_only=bool(read_only),
        timeout=float(timeout),
        binding=binding,
    )


def check_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"tool name {name!r} must be 1-64 letters, digits, '_' or '-'")
    if name in RESERVED_NAMES:
        raise ValueError(f"{name!r} can't be a tool's name (it reads as an approval, or is NEO's own)")
    if name in builtin_names():
        raise ValueError(f"{name!r} is one of NEO's own tools: give the host tool another name")


def check_parameters(tool: str, schema: Mapping[str, Any]) -> None:
    if schema.get("type", "object") != "object":
        raise ValueError(f"{tool}: parameters must be a JSON schema of type 'object'")
    for p in schema.get("properties") or {}:
        if p in _DROPPED_ARGS or str(p).startswith("_"):
            raise ValueError(
                f"{tool}: a parameter can't be named {p!r} — NEO drops 'confirm', 'cancel' and '_…' "
                "arguments from every model call, so it would never arrive"
            )


# ---- schemas ------------------------------------------------------------------------------------


def _schema_and_binding(
    fn: Callable[..., Any], name: str, parameters: dict[str, Any] | type[pydantic.BaseModel] | None
) -> tuple[dict[str, Any], _Binding]:
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001 — unresolvable annotations: read them as written
        hints = {}
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        raise ValueError(f"{name}: *args can't be a tool's arguments")
    ann = {p.name: hints.get(p.name, p.annotation) for p in params}
    ctx = next((p.name for p in params if _is_ctx(ann[p.name])), None)
    is_model = isinstance(parameters, type) and issubclass(parameters, pydantic.BaseModel)

    # NEO's own handler style: (args: dict, ctx: ToolContext), both annotated. Its arguments come
    # as one dict, so its schema can only come from `parameters` (never guessed as "no arguments").
    if len(params) == 2 and _is_args(params[0], ann) and ctx == params[1].name:
        if is_model:
            return _model_schema(parameters), _Binding("native", model=parameters)  # type: ignore[arg-type]
        if not isinstance(parameters, Mapping):
            raise ValueError(
                f"{name} takes (args: dict, ctx: ToolContext), NEO's own handler style: pass "
                "parameters= (a JSON schema or a pydantic model) to say what the arguments are"
            )
        return dict(parameters), _Binding("native")

    rest = [p for p in params if p.name != ctx and p.kind is not inspect.Parameter.VAR_KEYWORD]
    if is_model:
        model: type[pydantic.BaseModel] = parameters  # type: ignore[assignment]
        a0 = ann[rest[0].name] if len(rest) == 1 else None
        if len(rest) == 1 and (
            a0 is model or a0 is inspect.Parameter.empty or (isinstance(a0, str) and a0.rsplit(".", 1)[-1] == model.__name__)
        ):
            return _model_schema(model), _Binding("model", model=model, ctx=ctx, model_param=rest[0].name)
        fields = tuple((n, n) for n in model.model_fields)
        return _model_schema(model), _Binding("typed", model=model, ctx=ctx, fields=fields)
    if isinstance(parameters, Mapping):
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
        accepts = None if takes_any else frozenset(p.name for p in rest)
        return dict(parameters), _Binding("loose", ctx=ctx, accepts=accepts)
    if parameters is not None:
        raise TypeError("parameters is a JSON schema (dict) or a pydantic model class")

    # From the typed signature. Each field gets a neutral attribute name with the parameter's name
    # as its alias: a parameter called "json" or "model_id" never clashes with pydantic's own.
    defs: dict[str, Any] = {}
    fields_map: list[tuple[str, str]] = []
    for i, p in enumerate(rest):
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(f"{name}: positional-only parameters can't be a tool's arguments")
        if p.name in _DROPPED_ARGS or p.name.startswith("_"):
            check_parameters(name, {"properties": {p.name: {}}})
        a = ann[p.name]
        a = Any if a is inspect.Parameter.empty else a
        default = ... if p.default is inspect.Parameter.empty else p.default
        attr = f"f{i}"
        defs[attr] = (a, pydantic.Field(default, alias=p.name))
        fields_map.append((p.name, attr))
    model = pydantic.create_model(  # type: ignore[call-overload]
        f"{name}_arguments",
        __config__=pydantic.ConfigDict(extra="ignore", protected_namespaces=(), populate_by_name=False),
        **defs,
    )
    return _model_schema(model), _Binding("typed", model=model, ctx=ctx, fields=tuple(fields_map))


def _model_schema(model: type[pydantic.BaseModel]) -> dict[str, Any]:
    return _untitled(model.model_json_schema(by_alias=True))


def _untitled(node: Any) -> Any:
    """pydantic's generated "title" strings are noise to a model (a property *named* title stays)."""
    if isinstance(node, dict):
        return {k: _untitled(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))}
    if isinstance(node, list):
        return [_untitled(v) for v in node]
    return node


def _is_ctx(a: Any) -> bool:
    if a is ToolContext:
        return True
    return isinstance(a, str) and a.rsplit(".", 1)[-1] == "ToolContext"


def _is_args(p: inspect.Parameter, ann: Mapping[str, Any]) -> bool:
    """Annotated as a dict or a Mapping (a plain parameter merely named "args" is an argument)."""
    a = ann[p.name]
    if a is inspect.Parameter.empty:
        return False
    if a in (dict, Mapping, collections.abc.Mapping):
        return True
    if isinstance(a, str):
        return a.split("[", 1)[0].rsplit(".", 1)[-1] in ("dict", "Mapping")
    return get_origin(a) in (dict, collections.abc.Mapping)


# ---- results ------------------------------------------------------------------------------------


def _output(r: Any) -> str | ToolOutput:
    """A host function's return value as NEO's tool output. Only NEO's own code stops a run to ask
    something (a Halt its gate produced): a ToolOutput a host builds never halts, whatever it says.
    Anything returned is a success (a ToolOutput: never NEO's guess from how its text starts, so
    third-party text beginning with "Error" is still fenced and still taints)."""
    if isinstance(r, Halt):
        return r
    if isinstance(r, ToolOutput):
        if isinstance(r.text, Halt):
            return r
        return dataclasses.replace(r, text=str(r.text), halt=False, staged="")
    if r is None:
        text = "Done."
    elif isinstance(r, str):
        text = (str(r) or "Done.").strip()  # a plain str, never a subclass that could pass for NEO's own
    elif isinstance(r, pydantic.BaseModel):
        text = r.model_dump_json()
    elif isinstance(r, (bytes, bytearray)):
        text = bytes(r).decode("utf-8", "replace")
    elif isinstance(r, (dict, list, tuple)):
        text = json.dumps(r, ensure_ascii=False, default=str)
    else:
        text = str(r)
    return ToolOutput(text, ok=True)


def _brief(e: pydantic.ValidationError) -> str:
    parts = []
    for d in e.errors(include_url=False, include_input=False):
        loc = ".".join(str(x) for x in d.get("loc", ())) or "arguments"
        parts.append(f"{loc}: {d.get('msg', 'invalid')}")
    return "; ".join(parts)[:400]
