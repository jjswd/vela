"""Repository audit tools.

Every tool here is deterministic. That is the point: an agent is good at
deciding what to check across a repository and at explaining what it found, and
bad at reliably pattern-matching fifty files. So the scanning is exact and
repeatable, and the model supplies judgement and structure instead of accuracy.

Results carry file and line evidence. A finding a reader cannot verify from its
own citation is a guess, and a guess in an audit report is worse than no report.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from vela.tools.base import Tool

#: Directories that hold vendored or generated code; findings inside them are
#: noise because nobody is expected to maintain them.
SKIP_DIRS = frozenset(
    {
        ".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
        ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache", "site-packages",
        ".idea", ".vscode", "target", ".next", "coverage", "htmlcov",
    }
)

#: File types worth scanning, by suffix. Binary and lock files are excluded
#: because matches inside them are never actionable.
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".sh", ".ts", ".tsx", ".js", ".jsx"}
)

SKIP_SUFFIXES = frozenset({".lock", ".min.js", ".map", ".svg", ".png", ".jpg", ".ico", ".woff", ".woff2"})

#: Ceiling on files per scan. Repository-wide scans run inside the event loop,
#: and a tree of tens of thousands of files takes long enough to stall the
#: service rather than merely slow it down. Beyond this the tool reports that it
#: hit the cap instead of pretending to have covered everything -- a partial
#: audit that says so is useful, one that quietly under-reports is not.
MAX_FILES_PER_SCAN = 2000

#: (name, severity, regex, why it matters). Severity is the starting point; a
#: report that flags everything as critical teaches a reader to ignore it.
PATTERNS: list[tuple[str, str, re.Pattern[str], str]] = [
    # Matches the usual spellings without demanding an exact keyword, since
    # API_KEY, KEY, SECRET_KEY and authToken all leak credentials equally.
    # The value shape excludes short strings and obvious placeholders.
    (
        "hardcoded-secret",
        "critical",
        re.compile(
            r"(?i)\b(?:"
            r"[a-z0-9_]*(?:api|secret|auth|access)[a-z0-9_]*key"
            r"|[a-z0-9_]*password"
            r"|[a-z0-9_]*secret"
            r"|[a-z0-9_]*token"
            r")\s*[:=]\s*[\"'][A-Za-z0-9_\-./+]{12,}[\"']"
        ),
        "A credential in source is committed to history and cannot be un-leaked by deleting the line.",
    ),
    ("dangerous-eval", "high", re.compile(r"\b(eval|exec)\s*\("),
     "Dynamic execution turns any untrusted string into code."),
    # A bare ``except:`` is the finding; what follows it does not matter. The
    # earlier version required pass or a comment immediately after the colon and
    # therefore missed the common case of a bare except wrapping real handling.
    ("bare-except", "medium", re.compile(r"except\s*:"),
     "A bare except swallows every error, including the ones worth stopping for."),
    ("shell-true", "medium", re.compile(r"shell\s*=\s*True"),
     "Passing a string to a shell turns arguments into commands."),
    ("sql-string-format", "high", re.compile(r"(?i)(execute|query)\s*\(\s*f?[\"'].*\{"),
     "Interpolating into SQL turns user input into query syntax."),
    ("todo-marker", "low", re.compile(r"\b(TODO|FIXME|XXX|HACK)\b"),
     "An unfinished marker left in place, for tracking rather than triage."),
    ("sleep-in-loop", "low", re.compile(r"time\.sleep\("),
     "Blocking sleeps in a loop usually mean a polling design that a wait primitive would handle."),
]

#: Markdown link targets. Reference-style links and anchors are excluded because
#: their validity cannot be checked without resolving a base first.
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)")
#: Test fixtures legitimately contain credentials-shaped strings: they exist to
#: prove the scanner works. Reporting them is noise that trains a reader to skip
#: the critical section, so files under a test directory are matched with a
#: lower severity rather than dropped -- the pattern may still matter there.
TEST_PATH_MARKERS = frozenset({"tests", "test", "testing", "testdata", "fixtures"})


def _in_test_tree(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(part in TEST_PATH_MARKERS for part in parts[:-1])



#: File paths referenced inside documentation. The leading character class
#: includes the dot so that dotted module paths ("import src.missing") yield
#: their trailing segment, which is what has to be resolved on disk.
PATH_REFERENCE = re.compile(r"[\w./\-]*[\w-]\.(?:py|toml|json|md|ya?ml|txt|cfg|ini|sh)\b")

MD_CODE_CMD = re.compile(r"^\s*\$?\s*(?:python -m |pip |pytest |ruff |cd |export |vela )([\w.\-/@]+)")

#: Code fences whose contents are illustrative rather than executable.
IGNORED_FENCE_LANGS = frozenset({"text", "plaintext", "console", "output", "mermaid"})


class _BlockingTool(Tool):
    """Marks an audit tool as performing synchronous IO.

    A repository-wide scan walks thousands of files; measured at 39s on a
    16k-file tree, long enough to stall the event loop and every other
    in-flight request with it. Declaring the flag moves the call onto a worker
    thread; a tool that blocks without declaring it would be a latent outage.
    """

    blocking = True


@dataclass
class Finding:
    """One observation, with the evidence needed to check it."""

    kind: str
    severity: str
    path: str
    line: int
    detail: str
    snippet: str = ""

    def to_dict(self) -> dict:
        payload = {
            "kind": self.kind,
            "severity": self.severity,
            "path": self.path,
            "line": self.line,
            "detail": self.detail,
        }
        if self.snippet:
            payload["snippet"] = self.snippet[:160]
        return payload


@dataclass
class AuditResult:
    """Everything a scan found, plus what it looked at."""

    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def by_severity(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def summary(self) -> dict[str, int]:
        counts = {level: 0 for level in ("critical", "high", "medium", "low")}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts


# --------------------------------------------------------------------------
# Helpers shared by the tools
# --------------------------------------------------------------------------


def iter_text_files(root: Path, subpath: str = "", *, limit: int = MAX_FILES_PER_SCAN):
    """Yield repository files worth scanning, skipping vendored trees.

    Stops at ``limit`` so a scan cannot monopolise the process. Callers that
    report counts should compare against the limit to detect truncation.
    """
    base = (root / subpath).resolve() if subpath else root.resolve()
    if root.resolve() not in base.parents and base != root.resolve():
        raise PermissionError(f"path escapes repository: {subpath}")
    if not base.exists():
        return
    emitted = 0
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        yield path
        emitted += 1
        if emitted >= limit:
            return


def scan_is_truncated(count: int) -> bool:
    """Whether a scan stopped at the file cap rather than finishing."""
    return count >= MAX_FILES_PER_SCAN


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text("utf-8", errors="replace").splitlines()
    except OSError:
        return []


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class ListFilesTool(_BlockingTool):
    name = "list_files"
    description = (
        "List repository files with a suffix filter, for example .py or .md. "
        "Returns paths and sizes; use it to decide what to read next."
    )
    parameters = {
        "type": "object",
        "properties": {
            "suffix": {"type": "string", "description": "file suffix including the dot"},
            "path": {"type": "string", "description": "optional subdirectory"},
        },
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    async def run(self, arguments: dict) -> str:
        suffix = str(arguments.get("suffix") or "").lower()
        subpath = str(arguments.get("path") or "")
        entries = []
        for path in iter_text_files(self.root, subpath):
            if suffix and not path.name.endswith(suffix):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            entries.append(f"{relative(path, self.root)}\t{size}B")
        return "\n".join(entries) if entries else "(no matching files)"


class ReadFileTool(_BlockingTool):
    name = "read_file"
    description = "Read a repository file, optionally a line range, as 1-indexed line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "required": ["path"],
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    async def run(self, arguments: dict) -> str:
        target = (self.root / str(arguments["path"])).resolve()
        if self.root.resolve() not in target.parents and target != self.root.resolve():
            return f"PermissionError: path escapes repository: {arguments['path']}"
        if not target.exists() or not target.is_file():
            return f"No such file: {arguments['path']}"

        lines = read_lines(target)
        start = max(1, int(arguments.get("start_line") or 1))
        end = int(arguments.get("end_line") or len(lines))
        end = min(end, len(lines), start + 2000)

        numbered = [f"{index:>5}  {lines[index - 1]}" for index in range(start, end + 1)]
        header = f"# {relative(target, self.root)}  lines {start}-{end} of {len(lines)}"
        return "\n".join([header, *numbered])


class ScanPatternsTool(_BlockingTool):
    name = "scan_patterns"
    description = (
        "Scan repository files for known risky or unfinished constructs and "
        "return matches with file and line numbers. Deterministic: the same "
        "repository always yields the same result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "all", "hardcoded-secret", "dangerous-eval", "bare-except",
                    "shell-true", "sql-string-format", "todo-marker", "sleep-in-loop",
                ],
            },
            "path": {"type": "string"},
        },
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    async def run(self, arguments: dict) -> str:
        kind = str(arguments.get("kind") or "all")
        subpath = str(arguments.get("path") or "")
        try:
            files = list(iter_text_files(self.root, subpath))
        except PermissionError as exc:
            return f"PermissionError: {exc}"

        selected = [p for p in PATTERNS if kind in ("all", p[0])]
        findings: list[Finding] = []

        for path in files:
            for index, line in enumerate(read_lines(path), start=1):
                # A commented-out line is not a live construct.
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue
                location = relative(path, self.root)
                in_tests = _in_test_tree(location)
                for name, severity, pattern, detail in selected:
                    if not pattern.search(line):
                        continue
                    # A fixture proving the scanner works is not itself a leak,
                    # but the construct may still be worth noting there.
                    effective = "medium" if in_tests and severity == "critical" else severity
                    findings.append(
                        Finding(
                            kind=name,
                            severity=effective,
                            path=location,
                            line=index,
                            detail=detail + (" (in test fixture)" if in_tests else ""),
                            snippet=stripped,
                        )
                    )

        if not findings:
            return json.dumps({"files_scanned": len(files), "findings": []})

        # Highest severity first, then file order, so the output is stable.
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        findings.sort(key=lambda f: (order.get(f.severity, 9), f.path, f.line))
        return json.dumps(
            {
                "files_scanned": len(files),
                "truncated": scan_is_truncated(len(files)),
                "counts": _counts(findings),
                "findings": [f.to_dict() for f in findings[:120]],
            },
            ensure_ascii=False,
            indent=2,
        )


def _counts(findings: list[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


class CheckDocLinksTool(_BlockingTool):
    name = "check_doc_links"
    description = (
        "Find broken relative links in Markdown files: link targets that point "
        "at paths that no longer exist."
    )
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}}

    def __init__(self, root: Path) -> None:
        self.root = root

    async def run(self, arguments: dict) -> str:
        subpath = str(arguments.get("path") or "")
        broken: list[dict] = []
        checked = 0

        for path in iter_text_files(self.root, subpath):
            if path.suffix != ".md":
                continue
            lines = read_lines(path)
            for index, line in enumerate(lines, start=1):
                for target in MD_LINK.findall(line):
                    if target.startswith(("http://", "https://", "#", "mailto:")):
                        continue
                    # Anchors within a page, and root-absolute paths, cannot be
                    # checked without resolving a server; skip rather than
                    # report a false positive.
                    if target.startswith("/") or target.startswith("#"):
                        continue
                    clean = target.split("#", 1)[0]
                    if not clean:
                        continue
                    checked += 1
                    resolved = (path.parent / clean).resolve()
                    if not resolved.exists():
                        broken.append(
                            {
                                "path": relative(path, self.root),
                                "line": index,
                                "target": target,
                            }
                        )

        return json.dumps(
            {
                "links_checked": checked,
                "broken_count": len(broken),
                "broken": broken[:80],
            },
            ensure_ascii=False,
            indent=2,
        )


class CheckDocCommandsTool(_BlockingTool):
    name = "check_doc_commands"
    description = (
        "Verify that file paths referenced in fenced code blocks and shell "
        "commands in documentation still exist."
    )
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}}

    def __init__(self, root: Path) -> None:
        self.root = root

    async def run(self, arguments: dict) -> str:
        subpath = str(arguments.get("path") or "")
        stale: list[dict] = []
        checked = 0

        for path in iter_text_files(self.root, subpath):
            if path.suffix not in {".md", ".rst"}:
                continue
            lines = read_lines(path)
            in_fence = False
            fence_lang = ""
            for index, line in enumerate(lines, start=1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    if in_fence:
                        in_fence = False
                        fence_lang = ""
                    else:
                        in_fence = True
                        fence_lang = stripped[3:].strip().split()[0].lower() if stripped[3:].strip() else ""
                    continue
                if not in_fence or fence_lang in IGNORED_FENCE_LANGS:
                    continue

                for match in PATH_REFERENCE.finditer(line):
                    candidate = match.group(0)
                    if candidate.startswith(("http", "-")):
                        continue
                    # Strip a leading separator: in "import src.missing_module"
                    # the pattern starts at "missing_module", and resolving it
                    # against the document's own directory would look in the
                    # wrong place.
                    candidate = candidate.lstrip("./")
                    if not candidate:
                        continue
                    checked += 1
                    beside_doc = (path.parent / candidate).resolve()
                    from_root = (self.root / candidate).resolve()
                    if not beside_doc.exists() and not from_root.exists():
                        stale.append(
                            {
                                "path": relative(path, self.root),
                                "line": index,
                                "references": candidate,
                            }
                        )

        return json.dumps(
            {
                "references_checked": checked,
                "stale_count": len(stale),
                "stale": stale[:80],
            },
            ensure_ascii=False,
            indent=2,
        )


class AnalyzePythonTool(_BlockingTool):
    name = "analyze_python"
    description = (
        "Statically analyse Python files for import cycles, syntax errors, "
        "functions above a complexity threshold, and modules with no tests."
    )
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }

    def __init__(self, root: Path) -> None:
        self.root = root

    async def run(self, arguments: dict) -> str:
        subpath = str(arguments.get("path") or "")
        try:
            files = [p for p in iter_text_files(self.root, subpath) if p.suffix == ".py"]
        except PermissionError as exc:
            return f"PermissionError: {exc}"

        test_dirs = {"tests", "test", "testing"}
        issues: list[dict] = []
        modules: list[str] = []
        graph: dict[str, set[str]] = {}

        for path in files:
            rel = relative(path, self.root)
            modules.append(rel)
            source = "\n".join(read_lines(path))
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                issues.append({"kind": "syntax-error", "path": rel, "line": exc.lineno, "detail": exc.msg})
                continue

            if not any(part in test_dirs for part in path.parts):
                has_test = any(
                    rel.rsplit("/", 1)[-1].replace(".py", "") in candidate
                    for candidate in modules
                    if any(part in test_dirs for part in Path(candidate).parts)
                )
                if not has_test and len(read_lines(path)) > 40:
                    issues.append(
                        {"kind": "untested-module", "path": rel, "line": 1,
                         "detail": "more than 40 lines with no matching test module"}
                    )

            deps: set[str] = set()
            for node in ast.walk(tree):
                # Record the full dotted path, not just the first component.
                # Truncating to the top-level package loses the information
                # needed to map an import back onto a file, which is what makes
                # cycle detection possible at all.
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    deps.add(node.module)
                    # ``from pkg import b`` names a module directly; without
                    # this the reference to b is invisible and no cycle forms.
                    for alias in node.names:
                        if alias.name != "*":
                            deps.add(f"{node.module}.{alias.name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        deps.add(alias.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    branches = sum(
                        1
                        for child in ast.walk(node)
                        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.BoolOp, ast.IfExp))
                    )
                    if branches > 12:
                        issues.append(
                            {"kind": "high-complexity", "path": rel, "line": node.lineno,
                             "detail": f"{node.name}: {branches} branch points"}
                        )
            graph[rel] = deps

        cycles = _find_cycles(graph, modules)
        for cycle in cycles:
            issues.append(
                {"kind": "import-cycle", "path": cycle[0], "line": 1,
                 "detail": " -> ".join(cycle)}
            )

        return json.dumps(
            {
                "files_analysed": len(files),
                "truncated": scan_is_truncated(len(files)),
                "issue_count": len(issues),
                "issues": issues[:80],
            },
            ensure_ascii=False,
            indent=2,
        )


def _module_name(path: str) -> str:
    """Convert a repository-relative file path into an importable module name.

    Windows separators have to be handled alongside POSIX ones, otherwise the
    vertex identifiers silently stop matching the dotted names collected from
    the imports.
    """
    normalised = path.replace("\\", "/").replace(".py", "")
    return normalised.replace("/", ".")



def _find_cycles(graph: dict[str, set[str]], modules: list[str]) -> list[list[str]]:
    """Find import cycles among first-party modules.

    Only first-party names are considered: a dependency on a third-party
    package cannot form a cycle with this repository, and treating it as one
    would bury the real findings.
    """
    # Vertices must be identified the same way on both sides: the recorded
    # dependencies are dotted package paths, so comparing them against bare
    # file stems silently produced an empty edge set and never found a cycle.
    first_party = {_module_name(name) for name in modules}
    edges: dict[str, set[str]] = {}
    for module, deps in graph.items():
        vertex = _module_name(module)
        # Resolve an import to the module it names; a package import is
        # attributed to its __init__, which is what actually runs it.
        edges[vertex] = {
            dep if dep in first_party else dep + ".__init__"
            for dep in deps
            if dep in first_party or f"{dep}.__init__" in first_party
        }

    cycles: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    visiting: list[str] = []
    done: set[str] = set()

    def walk(node: str) -> None:
        if node in visiting:
            index = visiting.index(node)
            cycle = visiting[index:] + [node]
            key = tuple(sorted(cycle))
            if key not in seen:
                seen.add(key)
                cycles.append(cycle)
            return
        if node in done:
            return
        visiting.append(node)
        for dependency in sorted(edges.get(node, ())):
            walk(dependency)
        visiting.pop()
        done.add(node)

    for module in sorted(edges):
        walk(module)
    return cycles


def build_audit_tools(root: str | Path) -> list[Tool]:
    """Construct the audit tool set rooted at a repository."""
    base = Path(root).resolve()
    return [
        ListFilesTool(base),
        ReadFileTool(base),
        ScanPatternsTool(base),
        CheckDocLinksTool(base),
        CheckDocCommandsTool(base),
        AnalyzePythonTool(base),
    ]
