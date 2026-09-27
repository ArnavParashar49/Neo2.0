"""Check a dict against the installed vendor SDKs' own request types, offline.

openai and anthropic describe request params as TypedDicts; `conforms(value, TypedDict)` walks one
(required keys present, no key it doesn't declare unless the SDK allows extras, every value of its
declared type, Literals exact) and returns what's wrong. google-genai's types are pydantic models
with extra="forbid", so they validate themselves.
"""

from __future__ import annotations

import collections.abc
import types
import typing
from collections.abc import Mapping
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import typing_extensions
from typing_extensions import is_typeddict

_SEQUENCES = {
    list,
    tuple,
    collections.abc.Iterable,
    collections.abc.Sequence,
    collections.abc.Collection,
}
_MAPPINGS = {dict, collections.abc.Mapping, collections.abc.MutableMapping}


def conforms(value: Any, tp: Any, path: str = "$") -> list[str]:
    """What's wrong with `value` as a `tp` (empty when it fits)."""
    if tp is Any or tp is object:
        return []
    if isinstance(tp, typing.TypeAliasType):  # type X = ...
        return conforms(value, tp.__value__, path)
    if is_typeddict(tp):
        return _typeddict(value, tp, path)
    origin = get_origin(tp)
    if origin is typing.Annotated:
        return conforms(value, get_args(tp)[0], path)
    if origin in (Union, types.UnionType):
        branches = [conforms(value, a, path) for a in get_args(tp)]
        if any(not b for b in branches):
            return []
        return [f"{path}: {value!r:.80} fits none of {tp}"]
    if origin is Literal:
        return [] if value in get_args(tp) else [f"{path}: {value!r} not one of {get_args(tp)}"]
    if origin in _SEQUENCES or _is_sequence_not_str(tp):
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
            return [f"{path}: expected a list, got {type(value).__name__}"]
        args = get_args(tp)
        item = args[0] if args else Any
        return [e for i, v in enumerate(value) for e in conforms(v, item, f"{path}[{i}]")]
    if origin in _MAPPINGS or tp in (dict, Mapping):
        if not isinstance(value, Mapping):
            return [f"{path}: expected an object, got {type(value).__name__}"]
        args = get_args(tp)
        kt, vt = args if args else (Any, Any)
        errs = [e for k in value for e in conforms(k, kt, f"{path}.<key>")]
        return errs + [e for k, v in value.items() for e in conforms(v, vt, f"{path}.{k}")]
    if tp is type(None) or tp is None:
        return [] if value is None else [f"{path}: expected null"]
    if tp is bool:
        return [] if isinstance(value, bool) else [f"{path}: expected a bool"]
    if tp is int:
        return [] if isinstance(value, int) and not isinstance(value, bool) else [f"{path}: expected an int"]
    if tp is float:
        return (
            []
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else [f"{path}: expected a number"]
        )
    if tp is str:
        return [] if isinstance(value, str) else [f"{path}: expected a string, got {type(value).__name__}"]
    if isinstance(tp, type):
        return (
            [] if isinstance(value, tp) else [f"{path}: expected {tp.__name__}, got {type(value).__name__}"]
        )
    # anything else (a Protocol like Base64FileInput): the SDK's pydantic validation decides
    from pydantic import TypeAdapter, ValidationError

    try:
        TypeAdapter(tp, config={"arbitrary_types_allowed": True}).validate_python(value)
    except (ValidationError, TypeError) as e:  # TypeError: a type pydantic can't build
        return [f"{path}: {e}"]
    return []


def _is_sequence_not_str(tp: Any) -> bool:
    return (
        getattr(tp, "__name__", "") == "SequenceNotStr"
        or getattr(get_origin(tp), "__name__", "") == "SequenceNotStr"
    )


def required_keys(tp: Any) -> frozenset[str]:
    """A TypedDict's required keys. Not ``__required_keys__``: the SDKs declare their TypedDicts
    under ``from __future__ import annotations``, and then the class never sees its Required[...]
    markers (that set comes out empty) — resolve the annotations instead."""
    out = set()
    for k, h in get_type_hints(tp, include_extras=True).items():
        o = get_origin(h)
        if o in (typing.Required, typing_extensions.Required) or (
            o not in (typing.NotRequired, typing_extensions.NotRequired) and _declared_total(tp, k)
        ):
            out.add(k)
    return frozenset(out)


def _declared_total(tp: Any, key: str) -> bool:
    for b in getattr(tp, "__orig_bases__", ()):
        if is_typeddict(b) and key in getattr(b, "__annotations__", {}):
            return _declared_total(b, key)
    return bool(getattr(tp, "__total__", True))


def _typeddict(value: Any, tp: Any, path: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path}: expected an object for {tp.__name__}, got {type(value).__name__}"]
    hints = get_type_hints(tp)
    errs = [f"{path}: missing {k!r} ({tp.__name__})" for k in sorted(required_keys(tp) - value.keys())]
    config = getattr(tp, "__pydantic_config__", None) or {}
    if config.get("extra") != "allow":
        errs += [f"{path}: {tp.__name__} has no key {k!r}" for k in sorted(value.keys() - hints.keys())]
    for k, v in value.items():
        if k in hints:
            errs += conforms(v, hints[k], f"{path}.{k}")
    return errs


def assert_conforms(value: Any, tp: Any) -> None:
    errs = conforms(value, tp)
    assert errs == [], "\n".join(errs)


def assert_rejected(value: Any, tp: Any) -> None:
    """The checker itself: a wrong shape must fail."""
    assert conforms(value, tp), f"{value!r} unexpectedly fits {tp}"
