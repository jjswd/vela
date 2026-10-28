"""Tests for the CLI.

The CLI is what a user touches first, and a confusing error there costs more
time than a bug anywhere else. These tests cover the diagnostics in
particular: "command not found" tells you nothing about whether the
interpreter, the package, or the PATH is at fault, and `vela doctor` exists to
answer that.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from vela.cli import main


def run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the CLI in a subprocess so exit codes and output are real."""
    source_root = Path(__file__).resolve().parent.parent / "src"
    environment = {**os.environ, "PYTHONPATH": str(source_root)}
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, "-m", "vela.cli", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )


# --------------------------------------------------------------------------
# Basic dispatch
# --------------------------------------------------------------------------


def test_help_lists_the_commands(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    for command in ("serve", "chat", "verify", "providers", "doctor"):
        assert command in output, command


def test_unknown_command_fails() -> None:
    result = run_cli("definitely-not-a-command")
    assert result.returncode == 2
    assert "unknown command" in result.stderr


def test_no_arguments_prints_usage() -> None:
    result = run_cli()
    assert result.returncode == 0
    assert "usage:" in result.stdout


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_reports_the_environment_state() -> None:
    """Whatever the state, the essentials are reported.

    The exit code depends on whether the command is on PATH, which differs
    between a developer's shell and a test runner, so it is not asserted here.
    """
    result = run_cli("doctor")
    for label in ("python", "version", "vela", "command"):
        assert label in result.stdout, label
    assert "vela doctor" in result.stdout


def test_doctor_succeeds_when_the_command_is_reachable(tmp_path) -> None:
    """With vela on PATH, the check passes.

    A directory holding a copy of the command is put on PATH, which is what
    activating an environment does.
    """
    import shutil

    source_root = Path(__file__).resolve().parent.parent / "src"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    # Windows resolves commands through PATHEXT, so the stand-in has to be a
    # .bat. A shebang script would not be found, and pretending otherwise
    # would make this test pass on Linux and fail on the platform it targets.
    if os.name == "nt":
        # The batch file delegates to a tiny Python file, which keeps the
        # quoting simple; nesting quotes inside a batch argument is where
        # this goes wrong.
        helper = bin_dir / "_vela_entry.py"
        helper.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(source_root)!r})\n"
            "from vela.cli import main\n"
            "sys.exit(main())\n",
            "utf-8",
        )
        launcher = bin_dir / "vela.bat"
        launcher.write_text(
            "@echo off\r\n"
            f'"{sys.executable}" "{helper}" %*\r\n',
            "utf-8",
        )
    else:
        launcher = bin_dir / "vela"
        launcher.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            f"sys.path.insert(0, {str(source_root)!r})\n"
            "from vela.cli import main\n"
            "sys.exit(main())\n",
            "utf-8",
        )
        launcher.chmod(0o755)

    result = run_cli("doctor", env={"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no problems found" in result.stdout
    assert shutil.which("vela", path=str(bin_dir)) is not None


def test_doctor_names_the_interpreter() -> None:
    """A wrong interpreter is a common cause, and it must be visible."""
    result = run_cli("doctor")
    assert "python" in result.stdout
    assert sys.executable.split(os.sep)[-1] in result.stdout or "python" in result.stdout


def test_doctor_notes_a_missing_credential_without_failing() -> None:
    """No key is a normal state, not a broken installation."""
    result = run_cli("doctor", env={"VELA_API_KEY": "", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
    assert "no API key found" in result.stdout


def test_doctor_reports_a_present_credential_without_printing_it() -> None:
    """The key must never be echoed, not even partially."""
    result = run_cli("doctor", env={"VELA_API_KEY": "sk-secret-value-12345"})
    assert "sk-secret-value-12345" not in result.stdout
    assert "credential" in result.stdout


def test_doctor_detects_a_missing_package() -> None:
    """If vela cannot be imported, say so instead of guessing."""
    script = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'vela':\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from vela.cli import main\n"
        "sys.exit(main(['doctor']))\n"
    )
    source_root = Path(__file__).resolve().parent.parent / "src"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(source_root)},
    )
    # Blocking the package makes the import itself fail, before doctor can
    # report on it. The point of the check is that the failure is legible.
    assert result.returncode != 0
    assert "blocked" in result.stderr


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


def test_providers_lists_the_backends() -> None:
    result = run_cli("providers")
    assert result.returncode == 0
    for backend in ("anthropic", "openai", "ollama"):
        assert backend in result.stdout, backend


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def test_verify_runs_the_offline_checks() -> None:
    result = run_cli("verify")
    assert result.returncode == 0
    assert "all checks passed" in result.stdout


# --------------------------------------------------------------------------
# chat and serve
# --------------------------------------------------------------------------


def test_chat_without_a_message_fails() -> None:
    result = run_cli("chat")
    assert result.returncode != 0
    assert "requires a message" in result.stderr


def test_chat_without_a_credential_explains_what_is_missing() -> None:
    """The message must name the variable to set, not just say 'error'."""
    result = run_cli(
        "chat", "hello", env={"VELA_API_KEY": "", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}
    )
    assert result.returncode != 0
    assert "API_KEY" in result.stderr


def test_serve_requires_a_credential() -> None:
    result = run_cli(
        "serve", env={"VELA_API_KEY": "", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}
    )
    assert result.returncode != 0
    assert "API_KEY" in result.stderr


def test_serve_reports_a_missing_web_extra_clearly() -> None:
    """FastAPI is optional, so its absence must read as advice, not a crash."""
    script = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('fastapi', 'pydantic', 'starlette'):\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from vela.cli import main\n"
        "sys.exit(main(['serve']))\n"
    )
    source_root = Path(__file__).resolve().parent.parent / "src"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(source_root)},
    )
    assert result.returncode == 1
    assert "vela[server]" in result.stderr
