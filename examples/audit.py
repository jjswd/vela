"""Scenario: repository audit, end to end.

    python examples/audit.py                     # scan this repository
    python examples/audit.py /path/to/repo       # scan another

Answers a question every repository owner eventually has: *is the
documentation still true?* Broken relative links, file paths referenced in code
fences that no longer exist, credentials left in source, and untested modules.

The scanning is deterministic and lives in ``vela.tools.audit``. The runtime's
job is to decide what to check, interpret what came back, and present it in a
form someone will actually read. That division is the point: a model is good at
judgement and bad at reliably matching patterns across fifty files, so it is
given the former role and denied the latter.

Two modes:

- **Deterministic** (default, no API key): runs the tools directly and renders
  the report. This is what CI can gate on, because the output never varies.
- **Model-driven** (needs VELA_API_KEY): routes the same tools through the
  agent loop, so the model chooses what to check and writes the summary.

The same findings either way, which is the property worth having: the model
changes the presentation, never the facts.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vela.observability import Instrumentor  # noqa: E402
from vela.tools.audit import AuditResult, Finding, build_audit_tools  # noqa: E402

SEVERITY_ORDER = ["critical", "high", "medium", "low"]
SEVERITY_MARK = {"critical": "!!", "high": "! ", "medium": "~ ", "low": ". "}


def collect(root: Path) -> AuditResult:
    """Run every deterministic check and gather the results."""
    tools = {tool.name: tool for tool in build_audit_tools(root)}
    result = AuditResult()

    async def run() -> None:
        # Pattern scan.
        response = await tools["scan_patterns"].execute({"kind": "all"}, call_id="scan")
        payload = json.loads(response.content)
        result.files_scanned = payload.get("files_scanned", 0)
        for item in payload.get("findings", []):
            result.findings.append(
                Finding(
                    kind=item["kind"],
                    severity=item["severity"],
                    path=item["path"],
                    line=item["line"],
                    detail=item["detail"],
                    snippet=item.get("snippet", ""),
                )
            )

        # Broken documentation links.
        response = await tools["check_doc_links"].execute({}, call_id="links")
        payload = json.loads(response.content)
        for item in payload.get("broken", []):
            result.findings.append(
                Finding(
                    kind="broken-doc-link",
                    severity="medium",
                    path=item["path"],
                    line=item["line"],
                    detail=f"link target does not exist: {item['target']}",
                )
            )

        # Stale paths in documentation code blocks.
        response = await tools["check_doc_commands"].execute({}, call_id="cmds")
        payload = json.loads(response.content)
        for item in payload.get("stale", []):
            result.findings.append(
                Finding(
                    kind="stale-doc-reference",
                    severity="medium",
                    path=item["path"],
                    line=item["line"],
                    detail=f"references a file that does not exist: {item['references']}",
                )
            )

        # Static analysis.
        response = await tools["analyze_python"].execute({}, call_id="py")
        payload = json.loads(response.content)
        result.files_scanned = max(result.files_scanned, payload.get("files_analysed", 0))
        severity_by_kind = {
            "syntax-error": "critical",
            "import-cycle": "high",
            "high-complexity": "medium",
            "untested-module": "low",
        }
        for item in payload.get("issues", []):
            result.findings.append(
                Finding(
                    kind=item["kind"],
                    severity=severity_by_kind.get(item["kind"], "low"),
                    path=item["path"],
                    line=item.get("line", 1),
                    detail=item["detail"],
                )
            )

    asyncio.run(run())
    return result


def render(root: Path, result: AuditResult) -> str:
    """Present findings grouped by severity, each with a citation."""
    lines: list[str] = []
    name = root.resolve().name
    counts = result.summary()

    lines.append(f"repository audit -- {name}")
    lines.append("=" * 60)
    lines.append(
        f"scanned {result.files_scanned} files  |  "
        + "  ".join(f"{level}: {counts.get(level, 0)}" for level in SEVERITY_ORDER)
    )
    lines.append("")

    if not result.findings:
        lines.append("no findings.")
        return "\n".join(lines)

    for level in SEVERITY_ORDER:
        group = result.by_severity(level)
        if not group:
            continue
        lines.append(f"{level.upper()} ({len(group)})")
        lines.append("-" * 60)
        for finding in group:
            location = f"{finding.path}:{finding.line}"
            lines.append(f"  {SEVERITY_MARK[level]}{location}")
            lines.append(f"     {finding.kind}: {finding.detail}")
            if finding.snippet:
                snippet = finding.snippet[:100]
                suffix = "..." if len(finding.snippet) > 100 else ""
                lines.append(f"     | {snippet}{suffix}")
            lines.append("")
        lines.append("")

    actionable = counts.get("critical", 0) + counts.get("high", 0)
    if actionable:
        lines.append(
            f"{actionable} finding(s) at critical or high severity. "
            "Each cites a file and line, so each is independently checkable."
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You audit a software repository for problems a reader can verify.

Use the provided tools. They are deterministic: the same repository always
produces the same findings, so rely on their output rather than searching
for problems yourself.

Work in this order:
1. scan_patterns for risky constructs and unfinished markers
2. check_doc_links and check_doc_commands for documentation that has drifted
3. analyze_python for structural problems

Then write a report. Rules:
- Every claim cites a file and line. A claim you cannot cite is a guess.
- Sort by severity. A report that flags everything as critical teaches the
  reader to ignore it.
- Say plainly when something looks wrong but you could not confirm it.

Do not repeat the raw tool output; interpret it.
"""


async def run_with_model(root: Path, backend: str, model: str, api_key: str) -> str:
    """Route the audit through the agent loop."""
    from vela.core.loop import AgentLoop, SessionStore
    from vela.providers.factory import create_provider
    from vela.tools.registry import ToolRegistry

    provider = create_provider(backend, model=model, api_key=api_key)
    tools = ToolRegistry(build_audit_tools(root))
    instrumentor = Instrumentor()

    loop = AgentLoop(
        provider,
        tools,
        SessionStore(root / ".vela-audit-sessions"),
        workspace=str(root),
        system=SYSTEM_PROMPT,
        max_iterations=24,
        instrumentor=instrumentor,
        trace=True,
    )
    try:
        result = await loop.run(
            "audit",
            f"Audit this repository at {root}. Produce the report described in your instructions.",
        )
        print()
        print("turn trace")
        print("-" * 60)
        print(loop.last_trace.to_text())
        print()
        print("metrics")
        print("-" * 60)
        snapshot = instrumentor.metrics.snapshot()
        print("  tools used   :", ", ".join(dict.fromkeys(result.tools_used)) or "none")
        print("  iterations   :", result.iterations)
        print("  tokens       :", result.usage.input_tokens, "in /", result.usage.output_tokens, "out")
        for name, value in snapshot["counters"].items():
            print(f"  {name:<44} {value}")
        return result.content
    finally:
        await loop.aclose()


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if not target.is_dir():
        print(f"error: {target} is not a directory", file=sys.stderr)
        return 2

    api_key = os.environ.get("VELA_API_KEY")
    backend = os.environ.get("VELA_BACKEND", "openai")
    model = os.environ.get("VELA_MODEL", "gpt-4o-mini")

    if api_key:
        print(f"model-driven audit via {backend}/{model}\n")
        asyncio.run(run_with_model(target, backend, model, api_key))
        return 0

    # Deterministic mode. Same tools, no model in the loop.
    result = collect(target)
    print()
    print(render(target, result))

    counts = result.summary()
    blocking = counts.get("critical", 0)
    print()
    print(
        "This report was produced without a model. Set VELA_API_KEY to route the "
        "same tools through the agent instead -- the findings are identical either way."
    )
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
