"""Tool base class and parameter validation.

A tool's schema is part of the *model contract*: the model decides what to call
by reading the name, description, and parameter schema. Changing any of them
changes model behaviour, so they are treated as an interface rather than an
implementation detail.
"""

from __future__ import annotations

import inspect
import json
import types
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin, get_type_hints

from vela.core.contracts import ToolResult

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ValidationError(ValueError):
    """Raised when arguments do not satisfy a tool's schema."""


class ContextVarProvider:
    """Per-request context carried without threading arguments through every call.

    Tool execution nests several layers deep (runner -> dispatcher -> tool ->
    helper), and those layers have different signatures. A context variable lets
    deep code reach request-scoped data such as the workspace root without every
    intermediate function having to forward it, while still isolating concurrent
    requests -- which plain globals could not do.
    """


_request_scope: ContextVar[dict[str, Any] | None] = ContextVar(
    "vela_request_scope", default=None
)


def bind_request_scope(**values: Any) -> Token:
    """Bind request-scoped values; pair with :func:`reset_request_scope`."""
    return _request_scope.set(values)


def reset_request_scope(token: Token) -> None:
    """Restore the previous scope. Must run in ``finally`` or values will leak."""
    _request_scope.reset(token)


def current_scope() -> dict[str, Any]:
    return _request_scope.get() or {}


@dataclass
class ToolContext:
    """Values tools may read from the active request."""

    workspace: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Validate arguments against a JSON Schema fragment.

    Returns human-readable errors rather than raising immediately, so the
    runner can hand *all* problems back to the model in one round trip --
    fixing them one at a time wastes an iteration each time.

    Implemented directly instead of pulling in a JSON Schema library: the
    validator must treat ``bool`` as distinct from ``int`` (``True`` is not a
    valid integer despite Python's type hierarchy), and its error strings become
    part of the model-facing contract.
    """
    errors: list[str] = []
    properties: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = schema.get("required") or []

    for name in required:
        if name not in arguments:
            errors.append(f"missing required parameter '{name}'")

    for name, value in arguments.items():
        definition = properties.get(name)
        if definition is None:
            allowed = ", ".join(sorted(properties)) or "(none)"
            errors.append(f"unknown parameter '{name}'; allowed: {allowed}")
            continue
        errors.extend(_validate_value(name, definition, value))
    return errors


def _validate_value(path: str, schema: dict[str, Any], value: Any) -> list[str]:
    errors: list[str] = []

    expected = schema.get("type")
    if expected and not _matches_type(value, expected):
        return [f"'{path}' should be {expected}, got {type(value).__name__}"]

    enum = schema.get("enum")
    if enum and value not in enum:
        options = ", ".join(repr(item) for item in enum)
        errors.append(f"'{path}' must be one of [{options}], got {value!r}")

    if (
        isinstance(value, str)
        and (min_length := schema.get("minLength"))
        and len(value) < min_length
    ):
        errors.append(f"'{path}' must be at least {min_length} characters")

    if isinstance(value, dict):
        nested = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"'{path}.{name}' is required")
        for name, item in value.items():
            if name in nested:  # unknown keys are reported at the top level
                errors.extend(_validate_value(f"{path}.{name}", nested[name], item))

    if isinstance(value, list) and (items := schema.get("items")):
        for index, item in enumerate(value):
            errors.extend(_validate_value(f"{path}[{index}]", items, item))

    return errors


def _matches_type(value: Any, expected: str | list[str]) -> bool:
    """Type check that keeps booleans out of numeric types.

    ``isinstance(True, int)`` is ``True`` in Python, so a naive check would let
    ``{"count": true}`` pass an integer schema and fail deep inside tool code
    instead of at the boundary where the model could be told.
    """
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        if name == "null":
            if value is None:
                return True
            continue
        python_type = _JSON_TYPES.get(name)
        if python_type is None:
            return True  # Unknown type keyword: do not block the call.
        if name in {"integer", "number"} and isinstance(value, bool):
            continue
        if isinstance(value, python_type):
            return True
    return False


def schema_from_signature(func: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema from a function signature.

    Lets a tool author write a plain typed function and get a correct schema for
    free, removing the usual failure mode where the two drift apart.
    """
    try:
        hints = get_type_hints(func)
    except Exception:  # pragma: no cover - exotic annotations
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    signature = inspect.signature(func)

    for name, parameter in signature.parameters.items():
        # 使用 signature(func).parameters 可以得到一个有序字典，其中按照参数的顺序给出。
        # param.default：参数的默认值
        # param.annotation：参数要求的数据类型
        # param.kind：参数本身的类型
        if name in {"self", "ctx"} or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        json_type = _json_type_of(hints.get(name, str))
        properties[name] = {"type": json_type}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


def _json_type_of(annotation: Any) -> str:
    """Map a Python annotation onto a JSON Schema type name.

    Optional parameters are the common case (``count: int | None``,
    ``tags: list[str] | None``), so the ``None`` half of a union is discarded
    before inspection. Without that, every optional parameter would fall through
    to ``"string"`` and the model would be told to send a number as text.
    """
    origin = get_origin(annotation)

    # Union / X | None: use the first non-None member.
    if origin is Union or origin is types.UnionType:
        members = [member for member in get_args(annotation) if member is not type(None)]
        if not members:
            return "null"
        if len(members) == 1:
            return _json_type_of(members[0])
        # A heterogeneous union has no single JSON type; "string" is the
        # conservative choice because it accepts anything at validation time.
        return "string"

    if origin is not None:
        if origin in {list, tuple, set}:
            return "array"
        if origin is dict:
            return "object"
        annotation = origin

    if annotation is bool:
        # Checked before int: bool is a subclass of int, and a boolean
        # parameter must not be advertised as a number.
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is str:
        return "string"
    if annotation in {list, tuple, set, dict}:
        return "array" if annotation in {list, tuple, set} else "object"
    return "string"


# --------------------------------------------------------------------------
# Tool base class
# --------------------------------------------------------------------------


class Tool(ABC):
    """Base class for everything the model can invoke."""

    #: Stable, model-facing identifier. Renaming this breaks model behaviour.
    name: str = ""
    #: Model-facing description; this is prompt surface, not documentation.
    description: str = ""
    #: JSON Schema for the accepted arguments.
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    async def run(self, arguments: dict[str, Any]) -> str:
        """Execute the tool and return text for the model to read."""

    async def execute(self, arguments: dict[str, Any], *, call_id: str = "") -> ToolResult:
        """Validate then run, converting any failure into an error result.

        ``call_id`` is passed separately rather than injected into
        ``arguments``: the arguments come from the model and are validated
        against the schema, so smuggling bookkeeping fields in there would make
        every tool reject its own call.

        Exceptions become error results instead of propagating: a crashing tool
        must not abort the turn. The model can read the message and correct its
        next attempt, which keeps the conversation alive.
        """
        errors = validate_arguments(self.parameters, arguments)
        if errors:
            joined = "; ".join(errors)
            return ToolResult(
                call_id=call_id,
                name=self.name,
                content=f"Invalid parameters for '{self.name}': {joined}",
                status="error",
            )
        try:
            content = await self.run(arguments)
        except Exception as exc:  # noqa: BLE001 - deliberate boundary
            return ToolResult(
                call_id=call_id,
                name=self.name,
                content=f"{type(exc).__name__}: {exc}",
                status="error",
            )
        return ToolResult(
            call_id=call_id,
            name=self.name,
            content=content,
        )

    def spec(self):
        """Advertise this tool to a provider."""
        from vela.providers.base import ToolSpec

        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


class FunctionTool(Tool):
    """Adapter turning a plain function into a :class:`Tool`."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").strip()
        self.parameters = parameters or schema_from_signature(func)
        self._is_async = inspect.iscoroutinefunction(func)

    async def run(self, arguments: dict[str, Any]) -> str:
        payload = {k: v for k, v in arguments.items() if not k.startswith("__")}
        if self._is_async:
            value = await self._func(**payload)
        else:
            value = await _maybe_await(self._func(**payload))
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator building a :class:`FunctionTool` from a function."""

    def decorate(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            func, name=name, description=description, parameters=parameters
        )

    return decorate
