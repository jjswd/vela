"""Built-in tools.

Small, dependency-free tools that demonstrate the extension surface: schema
validation, workspace scoping, and error-as-value behaviour.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from vela.tools.base import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Echo the given text back to the user. Useful for testing the tool loop."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1}},
        "required": ["text"],
    }

    async def run(self, arguments: dict) -> str:
        return str(arguments["text"])


class NowTool(Tool):
    name = "current_time"
    description = "Get the current local date and time as an ISO-8601 string."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class MathTool(Tool):
    name = "calculate"
    description = "Evaluate an arithmetic expression, for example '(12 + 8) * 3'."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "minLength": 1}},
        "required": ["expression"],
    }

    # Only digits and operators reach the evaluator: eval() on model-generated
    # text would otherwise be a straightforward code-execution hole.
    _ALLOWED = set("0123456789.+-*/%() \t")

    async def run(self, arguments: dict) -> str:
        expression = str(arguments["expression"])
        illegal = sorted(set(expression) - self._ALLOWED)
        if illegal:
            # Raising keeps the failure distinguishable from a result: returning
            # this text as a normal answer would let the model read a rejected
            # expression as a computed value.
            raise ValueError(f"unsupported characters: {''.join(illegal)}")
        if not expression.strip():
            raise ValueError("empty expression")
        try:
            value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - sanitized
        except Exception as exc:  # noqa: BLE001
            # A failed evaluation yields no number, so reporting it as a normal
            # result would hand the model text where it expects a value.
            raise ValueError(f"cannot evaluate {expression!r}: {exc}") from exc
        return str(round(value, 10) if isinstance(value, float) else value)


class WorkspaceFileTool(Tool):
    """Read a file from the workspace."""

    name = "read_file"
    description = "Read a UTF-8 text file from the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "max_bytes": {"type": "integer"},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: str) -> None:
        self.workspace = Path(workspace)

    async def run(self, arguments: dict) -> str:
        resolved = self._resolve(str(arguments["path"]))
        if not resolved.exists():
            return f"No such file: {arguments['path']}"
        if resolved.is_dir():
            return f"{arguments['path']} is a directory."
        limit = int(arguments.get("max_bytes") or 20_000)
        data = resolved.read_bytes()[:limit]
        text = data.decode("utf-8", errors="replace")
        if len(data) == limit:
            text += f"\n\n[truncated at {limit} bytes]"
        return text

    def _resolve(self, candidate: str) -> Path:
        """Resolve inside the workspace, refusing traversal escapes.

        Absolute paths and ``..`` segments are both common in model output. The
        resolved path must remain under the workspace root, otherwise a crafted
        argument could read arbitrary host files.
        """
        path = Path(candidate)
        target = (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()
        root = self.workspace.resolve()
        if root != target and root not in target.parents:
            raise PermissionError(f"path escapes workspace: {candidate}")
        return target


class WorkspaceListTool(Tool):
    name = "list_files"
    description = "List files in the workspace, optionally one directory deep."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }

    def __init__(self, workspace: str) -> None:
        self.workspace = Path(workspace)

    async def run(self, arguments: dict) -> str:
        base = self.workspace / str(arguments.get("path") or ".")
        entries = []
        for item in sorted(base.glob("*")):
            if item.name.startswith("."):
                continue
            kind = "dir " if item.is_dir() else "file"
            size = "" if item.is_dir() else f" {item.stat().st_size}B"
            entries.append(f"{kind} {item.name}{size}")
        return "\n".join(entries) if entries else "(empty)"


def builtin_tools(workspace: str) -> list[Tool]:
    os.makedirs(workspace, exist_ok=True)
    return [
        EchoTool(),
        NowTool(),
        MathTool(),
        WorkspaceFileTool(workspace),
        WorkspaceListTool(workspace),
    ]
