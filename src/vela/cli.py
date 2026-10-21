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
    }
    handler = handlers.get(command)
    if handler is None:
        print(f"error: unknown command {command!r}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
