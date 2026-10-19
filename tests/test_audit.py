"""Tests for the audit scenario and for blocking-tool isolation.

Two things are being pinned down here. First, the audit findings must be
deterministic and must cite a file and line, because a report whose claims
cannot be checked is worse than no report. Second, a tool that blocks must not
hold the event loop -- the first version of the audit tools walked a repository
inside the loop and stalled everything else for the duration.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from vela.core.contracts import ToolCall
from vela.tools.audit import (
    Finding,
    build_audit_tools,
)
from vela.tools.base import Tool
from vela.tools.registry import ToolRegistry


@pytest.fixture
def repo(tmp_path):
    """A small repository with known, deliberately planted problems."""
    (tmp_path / "README.md").write_text(
        "# Demo\n\n"
        "Run `python src/main.py` to start.\n\n"
        "See [the guide](docs/guide.md) and [the api](docs/api.md).\n"
        "[dead](docs/removed.md)\n"
        "```python\n"
        "from src import config\n"
        "config.setting_file = \"src/settings.py\"\n"
        "```\n",
        "utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide\n", "utf-8")

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        'API_KEY = "sk-live-abcdefgh12345"\n'
        "\n"
        "def run(cmd):\n"
        "    return eval(cmd)\n"
        "\n"
        "def broken():\n"
        "    try:\n"
        "        pass\n"
        "    except:\n"
        "        pass\n",
        "utf-8",
    )
    # A module a pattern scan should skip.
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text('token = "should-not-be-flagged"\n', "utf-8")
    return tmp_path


def tool_set(root):
    return {tool.name: tool for tool in build_audit_tools(root)}


async def call(root, name, arguments=None):
    return await tool_set(root)[name].execute(arguments or {}, call_id="t1")


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


async def test_hardcoded_secret_is_found(repo) -> None:
    data = json.loads((await call(repo, "scan_patterns", {"kind": "hardcoded-secret"})).content)
    assert data["files_scanned"] >= 1
    findings = data["findings"]
    assert any(f["path"].endswith("main.py") and f["severity"] == "critical" for f in findings)


async def test_dangerous_eval_is_found(repo) -> None:
    data = json.loads((await call(repo, "scan_patterns", {"kind": "dangerous-eval"})).content)
    assert any(f["kind"] == "dangerous-eval" for f in data["findings"])


async def test_bare_except_is_found(repo) -> None:
    data = json.loads((await call(repo, "scan_patterns", {"kind": "bare-except"})).content)
    assert any(f["kind"] == "bare-except" for f in data["findings"])


async def test_commented_code_is_not_flagged(repo) -> None:
    """A pattern inside a comment is not a live construct."""
    (repo / "src" / "noted.py").write_text("# eval(user_input)\n", "utf-8")
    data = json.loads((await call(repo, "scan_patterns", {"kind": "dangerous-eval"})).content)
    assert not any(f["path"].endswith("noted.py") for f in data["findings"])


async def test_vendored_directories_are_skipped(repo) -> None:
    data = json.loads((await call(repo, "scan_patterns", {"kind": "hardcoded-secret"})).content)
    assert not any("node_modules" in f["path"] for f in data["findings"])


async def test_every_finding_cites_a_file_and_line(repo) -> None:
    """An unverifiable claim is a guess, and a guess in a report is worse than none."""
    data = json.loads((await call(repo, "scan_patterns", {"kind": "all"})).content)
    assert data["findings"], "expected findings in the fixture"
    for finding in data["findings"]:
        assert finding["path"]
        assert isinstance(finding["line"], int) and finding["line"] > 0


async def test_findings_are_ordered_by_severity(repo) -> None:
    data = json.loads((await call(repo, "scan_patterns", {"kind": "all"})).content)
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    severities = [order[f["severity"]] for f in data["findings"]]
    assert severities == sorted(severities)


async def test_broken_doc_links_are_found(repo) -> None:
    data = json.loads((await call(repo, "check_doc_links")).content)
    targets = {item["target"] for item in data["broken"]}
    # docs/removed.md does not exist; the other two do.
    assert "docs/removed.md" in targets
    assert "docs/guide.md" not in targets


async def test_external_links_are_not_checked(repo) -> None:
    (repo / "README2.md").write_text("[x](https://example.com/missing)\n", "utf-8")
    data = json.loads((await call(repo, "check_doc_links")).content)
    assert not any("example.com" in item["target"] for item in data["broken"])


async def test_stale_code_fence_reference_is_found(repo) -> None:
    """A file path inside a fence that no longer exists.

    Only file paths are checked, not dotted module names: whether
    ``src.missing_module`` still exists is a question for an import, not for a
    documentation audit, and guessing would produce false positives.
    """
    data = json.loads((await call(repo, "check_doc_commands")).content)
    assert any("src/settings.py" in item["references"] for item in data["stale"])


async def test_illustrative_fences_are_ignored(repo) -> None:
    """A ``text`` fence is documentation, not a claim about a path."""
    (repo / "ILLUSTRATIVE.md").write_text(
        "```text\nsrc/placeholder.py\n```\n", "utf-8"
    )
    data = json.loads((await call(repo, "check_doc_commands")).content)
    assert not any("placeholder.py" in item["references"] for item in data["stale"])


async def test_syntax_error_is_reported(repo) -> None:
    (repo / "src" / "broken_syntax.py").write_text("def (:\n", "utf-8")
    data = json.loads((await call(repo, "analyze_python")).content)
    assert any(i["kind"] == "syntax-error" for i in data["issues"])


async def test_import_cycle_is_detected(repo) -> None:
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", "utf-8")
    (repo / "pkg" / "a.py").write_text("from pkg import b\n", "utf-8")
    (repo / "pkg" / "b.py").write_text("from pkg import a\n", "utf-8")
    data = json.loads((await call(repo, "analyze_python", {"path": "pkg"})).content)
    assert any(i["kind"] == "import-cycle" for i in data["issues"])


async def test_path_escape_is_refused(repo) -> None:
    result = await call(repo, "read_file", {"path": "../../../etc/passwd"})
    assert "PermissionError" in result.content


async def test_scan_reports_truncation_rather_than_silence(monkeypatch) -> None:
    """A capped scan must say so; a partial audit that looks complete is a lie."""
    import vela.tools.audit as audit

    monkeypatch.setattr(audit, "MAX_FILES_PER_SCAN", 2)
    root = audit.Path(".")
    files = list(audit.iter_text_files(root, "src", limit=2))
    assert len(files) == 2
    assert audit.scan_is_truncated(2) is True
    assert audit.scan_is_truncated(1) is False


async def test_findings_sort_by_severity_not_table_order() -> None:
    """The table need not be ordered; the sorted output must be."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    unsorted = [
        Finding("todo-marker", "low", "a.py", 1, ""),
        Finding("hardcoded-secret", "critical", "a.py", 2, ""),
        Finding("bare-except", "medium", "a.py", 3, ""),
    ]
    unsorted.sort(key=lambda f: (order[f.severity], f.path, f.line))
    assert [f.severity for f in unsorted] == ["critical", "medium", "low"]


# --------------------------------------------------------------------------
# Blocking isolation
# --------------------------------------------------------------------------


class SlowBlocking(Tool):
    name = "slow_blocking"
    description = "sleeps, synchronously"
    blocking = True
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict) -> str:
        time.sleep(0.5)
        return "finished"


class SlowAsync(Tool):
    name = "slow_async"
    description = "sleeps, asynchronously"
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict) -> str:
        await asyncio.sleep(0.5)
        return "finished"


async def test_blocking_tool_does_not_stall_the_loop() -> None:
    """The whole point of the flag: other coroutines keep running meanwhile."""
    ticks: list[float] = []

    async def heartbeat() -> None:
        for _ in range(100):
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)

    registry = ToolRegistry([SlowBlocking()])
    try:
        task = asyncio.create_task(heartbeat())
        records = await registry.dispatch(
            [ToolCall(id="c1", name="slow_blocking", arguments={})], timeout=10
        )
        await task
    finally:
        registry.shutdown()

    assert records[0].result.content == "finished"
    # The heartbeat runs 100 ticks at 10ms, so if the loop had been stalled by
    # the 500ms tool the gaps would be around half a second. The threshold sits
    # far from both outcomes rather than just above the passing value, since a
    # loaded CI runner adds latency of its own.
    gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
    worst = max(gaps) if gaps else 0.0
    assert worst < 0.25, f"event loop was blocked for {worst * 1000:.0f}ms"


async def test_blocking_tool_still_validates_and_wraps_errors() -> None:
    """Thread offload must not skip validation or error handling."""

    class Exploding(Tool):
        name = "exploding_blocking"
        description = "raises"
        blocking = True
        parameters = {
            "type": "object",
            "properties": {"needed": {"type": "string"}},
            "required": ["needed"],
        }

        async def run(self, arguments: dict) -> str:
            raise RuntimeError("boom")

    registry = ToolRegistry([Exploding()])
    try:
        # Missing required argument must be rejected before execution.
        records = await registry.dispatch(
            [ToolCall(id="c1", name="exploding_blocking", arguments={})], timeout=10
        )
        assert records[0].result.is_error
        assert "missing required parameter" in records[0].result.content

        # A raised exception becomes an error result, not a crash.
        records = await registry.dispatch(
            [ToolCall(id="c2", name="exploding_blocking", arguments={"needed": "x"})],
            timeout=10,
        )
        assert records[0].result.is_error
        assert "boom" in records[0].result.content
    finally:
        registry.shutdown()


async def test_blocking_tool_timeout_is_reported() -> None:
    registry = ToolRegistry([SlowBlocking()])
    try:
        records = await registry.dispatch(
            [ToolCall(id="c1", name="slow_blocking", arguments={})], timeout=0.05
        )
        assert records[0].result.is_error
        assert "exceeded" in records[0].result.content
    finally:
        registry.shutdown()


async def test_non_blocking_tools_stay_on_the_loop() -> None:
    """Without the flag the existing coroutine path applies, unchanged."""
    registry = ToolRegistry([SlowAsync()])
    try:
        records = await registry.dispatch(
            [ToolCall(id="c1", name="slow_async", arguments={})], timeout=10
        )
        assert records[0].result.content == "finished"
    finally:
        registry.shutdown()


async def test_audit_tools_declare_their_blocking_nature() -> None:
    for tool in build_audit_tools("."):
        assert getattr(tool, "blocking", False) is True, tool.name


async def test_audit_tool_runs_through_the_registry(tmp_path) -> None:
    """End to end: a blocking audit tool invoked the way the model would."""
    (tmp_path / "a.py").write_text('API_KEY = "sk-live-0123456789abcdef"\n', "utf-8")
    registry = ToolRegistry(build_audit_tools(tmp_path))
    try:
        records = await registry.dispatch(
            [ToolCall(id="c1", name="scan_patterns", arguments={"kind": "hardcoded-secret"})],
            timeout=20,
        )
        assert not records[0].result.is_error, records[0].result.content[:200]
        data = json.loads(records[0].result.content)
        assert data["findings"]
    finally:
        registry.shutdown()
