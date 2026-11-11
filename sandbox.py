"""Run model-written solvers with a timeout and a small denylist."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

FORBIDDEN = re.compile(
    r"("
    r"\bsubprocess\b|\bsocket\b|\bshutil\b|\bctypes\b|\bmultiprocessing\b|"
    r"\burllib\b|\brequests\b|\bhttp\.client\b|\bwebbrowser\b|"
    r"\bos\.system\b|\bos\.popen\b|\bos\.remove\b|\bos\.unlink\b|"
    r"\b__import__\b|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(|"
    r"\bopen\s*\("
    r")",
    re.IGNORECASE,
)


class SandboxError(Exception):
    pass


def check_source(code: str) -> None:
    if not code or not code.strip():
        raise SandboxError("empty program")
    if FORBIDDEN.search(code):
        raise SandboxError("program uses a blocked operation")


def run_python(code: str, timeout: float = 20.0) -> str:
    """Execute code and return the single answer line printed to stdout."""
    check_source(code)
    with tempfile.TemporaryDirectory(prefix="agent_solver_") as folder:
        path = Path(folder) / "solver.py"
        path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=folder,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"program timed out after {timeout:.0f}s") from exc
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise SandboxError(detail[-2000:])
    if not stdout:
        raise SandboxError("program printed nothing")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SandboxError(
            "program must print exactly one line; got:\n" + "\n".join(lines[:8])
        )
    return lines[0]
