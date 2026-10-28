"""Command line entry point.

    vela serve                # start the HTTP server
    vela chat "hello"         # one-shot from the terminal
    vela verify               # run the offline checks
    vela providers            # list registered backends

Deliberately argument-parsed by hand rather than with argparse or a CLI
framework: the surface is four commands, and a dependency here would be the
heaviest one in the project.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

USAGE = """vela - a small agent runtime

usage:
  vela serve [options]        start the HTTP server
  vela chat <message>         one-shot completion, non-interactive
  vela verify                 run the offline checks
  vela providers              list registered backends
  vela doctor                 check this environment

serve options:
  --backend NAME       provider backend (default: openai)
  --model NAME         model id (default: gpt-4o-mini)
  --base-url URL       override the endpoint
  --host HOST          bind address (default: 127.0.0.1)
  --port PORT          port (default: 8080)
  --token TOKEN        require this bearer token
  --workspace PATH     workspace for tools (default: .)
  --log-level LEVEL    debug/info/warning (default: info)

environment:
  VELA_API_KEY         credential for the selected backend
  VELA_BACKEND, VELA_MODEL, VELA_BASE_URL  defaults for serve

examples:
  vela serve --backend anthropic --model claude-sonnet-4-5
  VELA_API_KEY=sk-... vela chat "summarise this repository"
"""


def _env(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback)


def _require_key(backend: str) -> str:
    """Fetch the credential for the chosen backend.

    Raises rather than starting a server that would only fail on first request:
    a misconfigured deployment should say so at startup.
    """
    mapping = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "moonshot": "MOONSHOT_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    if backend == "ollama":
        return "not-needed"
    variable = mapping.get(backend, "OPENAI_API_KEY")
    key = _env(variable) or _env("VELA_API_KEY")
    if not key:
        raise SystemExit(
            f"error: no credential found. Set {variable} (or VELA_API_KEY)."
        )
    return key


def cmd_serve(args: list[str]) -> int:
    try:
        from vela.server import ServerConfig, VelaServer
    except ImportError:
        # FastAPI is optional, so its absence is a normal configuration state
        # rather than a crash.
        print(
            "error: the server needs FastAPI.\n"
            "       install it with: pip install 'vela[server]'\n"
            "       or: pip install fastapi 'uvicorn[standard]'",
            file=sys.stderr,
        )
        return 1

    options = _parse_flags(args)
    backend = options.get("backend") or _env("VELA_BACKEND", "openai")
    model = options.get("model") or _env("VELA_MODEL", "gpt-4o-mini")
    base_url = options.get("base-url") or _env("VELA_BASE_URL") or None
    host = options.get("host", "127.0.0.1")
    port = int(options.get("port", "8080"))
    token = options.get("token", "")
    workspace = options.get("workspace", ".")

    api_key = _require_key(backend)

    from vela.core.loop import build_default_loop
    from vela.observability import Instrumentor

    instrumentor = Instrumentor()

    def factory() -> Any:
        loop = build_default_loop(
            backend=backend,
            api_key=api_key,
            model=model,
            base_url=base_url,
            workspace=workspace,
        )
        # Share the server's collector with the runtime, so one scrape covers
        # the whole request rather than only part of it.
        loop.instrumentor = instrumentor
        loop._runner.instrumentor = instrumentor  # noqa: SLF001
        return loop

    server = VelaServer(
        factory,
        ServerConfig(host=host, port=int(port), api_key=token),
        workspace=workspace,
        instrumentor=instrumentor,
    )

    print(f"vela listening on http://{host}:{port}")
    print(f"  backend   {backend} ({model})")
    print(f"  workspace {workspace}")
    print(f"  auth      {'bearer token required' if token else 'open (loopback only)'}")
    print("\n  POST /v1/chat/completions   OpenAI-compatible")
    print("  GET  /health                liveness")
    print("  GET  /metrics               Prometheus exposition")
    print("  GET  /docs                  interactive API documentation\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_chat(args: list[str]) -> int:
    if not args:
        raise SystemExit("error: chat requires a message")
    text = " ".join(args)

    backend = _env("VELA_BACKEND", "openai")
    model = _env("VELA_MODEL", "gpt-4o-mini")
    api_key = _require_key(backend)

    from vela.core.loop import build_default_loop
    from vela.core.runner import StreamingEmitter

    loop = build_default_loop(
        backend=backend,
        api_key=api_key,
        model=model,
        base_url=_env("VELA_BASE_URL") or None,
        root=_env("VELA_ROOT", "./.vela"),
        workspace=_env("VELA_WORKSPACE", "."),
    )

    async def run() -> str:
        emitter = StreamingEmitter(on_text=lambda delta: print(delta, end="", flush=True))
        result = await loop.run(
            _env("VELA_SESSION", "cli"),
            text,
            emitter=emitter,
            ephemeral=True,
        )
        print()
        usage = result.usage
        print(
            f"[{result.stop_reason} · {result.iterations} iterations · "
            f"{usage.input_tokens} in / {usage.output_tokens} out]"
        )
        return result.content

    try:
        asyncio.run(run())
    finally:
        pass
    return 0


def cmd_providers(_args: list[str]) -> int:
    from vela import available_providers, create_provider

    print("registered backends:\n")
    for name in available_providers():
        try:
            provider = create_provider(name, model="<any>", api_key="probe")
            implementation = type(provider).__name__
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            implementation = f"unavailable ({type(exc).__name__})"
        print(f"  {name:<12} {implementation}")
    print("\nSet VELA_API_KEY and use --backend to choose one.")
    return 0


def cmd_doctor(_args: list[str]) -> int:
    """Report whether this environment is set up correctly.

    The most common way to lose time with a CLI is not knowing which of three
    things is wrong: the interpreter, the installed package, or the PATH. This
    checks all three and says which, because "command not found" without that
    context is a guess.
    """
    import importlib.util
    import shutil

    problems: list[str] = []
    notes: list[str] = []

    print("vela doctor")
    print("=" * 60)

    # 1. Which interpreter is running.
    print(f"python      {sys.executable}")
    print(f"version     {sys.version.split()[0]}")
    if sys.version_info < (3, 11):  # noqa: UP036
        # The package metadata already requires 3.11, so pip would refuse the
        # install. This catches the case the metadata cannot: someone running
        # from a source checkout on an older interpreter, which is exactly the
        # situation a doctor command is meant to explain.
        problems.append(
            f"Python 3.11 or newer is required; this is {sys.version_info.major}."
            f"{sys.version_info.minor}."
        )

    # 2. Is the package importable, and from where.
    spec = importlib.util.find_spec("vela")
    if spec is None:
        problems.append(
            "the vela package is not importable in this environment. "
            "Install it with: pip install -e '.[dev,server]'  "
            "(or 'uv pip install -e \".[dev,server]\"' in a uv environment)"
        )
        location = "(not found)"
    else:
        location = spec.origin or "(namespace package)"
        print(f"vela        {location}")
        # An editable install points into the source tree, which is what
        # someone working on the project wants; a site-packages copy is what a
        # user of the published package wants. Neither is wrong, but the
        # difference explains "my edits are not taking effect".
        if "site-packages" in (location or ""):
            notes.append(
                "vela is installed as a copy, not an editable install. "
                "Use 'pip install -e .' if you are working on the source."
            )
        elif "src" in (location or ""):
            notes.append("vela is an editable install, so source edits apply immediately.")

    # 3. Is the command itself reachable.
    executable = shutil.which("vela")
    if executable:
        print(f"command     {executable}")
    else:
        problems.append(
            "the 'vela' command is not on PATH. Activate the environment "
            "(.venv\\Scripts\\Activate.ps1) or run the module directly: "
            "python -m vela.cli"
        )

    # 4. Optional pieces.
    for module, label, extra in (
        ("fastapi", "fastapi", "vela[server]"),
        ("uvicorn", "uvicorn", "vela[server]"),
    ):
        if importlib.util.find_spec(module) is None:
            notes.append(f"{label} is not installed; 'vela serve' needs it. Install {extra}.")
        else:
            print(f"{label:<11} installed")

    # 5. Credentials, reported without printing them.
    backend = _env("VELA_BACKEND", "openai")
    if _env("ANTHROPIC_API_KEY") or _env("OPENAI_API_KEY") or _env("VELA_API_KEY"):
        print(f"credential  present (backend default: {backend})")
    else:
        notes.append(
            "no API key found. 'vela chat' and 'vela serve' need one; "
            "'vela verify' and 'vela providers' do not."
        )

    # An explicit blank line before the sections, so the output does not run
    # into the last check. A bare newline inside the string is avoided on
    # purpose: some Windows shells concatenate output that arrives with a
    # line feed only.
    print()
    if notes:
        print("notes")
        print("-" * 60)
        for note in notes:
            print(f"  - {note}")

    if problems:
        print()
        print("problems")
        print("-" * 60)
        for problem in problems:
            print(f"  ! {problem}")
        return 1

    print()
    print("no problems found.")
    return 0


def cmd_verify(_args: list[str]) -> int:
    """Delegate to the offline checks so `vela verify` matches the script."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(os.path.dirname(here), "examples", "verify.py")
    if not os.path.exists(script):
        raise SystemExit("error: examples/verify.py not found in this installation")
    import subprocess

    return subprocess.call([sys.executable, script])


def _parse_flags(args: list[str]) -> dict[str, str]:
    """Parse ``--key value`` and ``--flag`` pairs."""
    options: dict[str, str] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:]
        if "=" in key:
            name, _, value = key.partition("=")
            options[name] = value
            index += 1
            continue
        following = args[index + 1] if index + 1 < len(args) else ""
        if following.startswith("--") or not following:
            options[key] = "true"
            index += 1
        else:
            options[key] = following
            index += 2
    return options


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    command, rest = args[0], args[1:]
    handlers = {
        "serve": cmd_serve,
        "chat": cmd_chat,
        "providers": cmd_providers,
        "verify": cmd_verify,
        "doctor": cmd_doctor,
    }
    handler = handlers.get(command)
    if handler is None:
        print(f"error: unknown command {command!r}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
