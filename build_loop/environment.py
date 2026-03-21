"""EnvironmentSnapshot: captures host reality before policy decisions.

No LLM calls. Bounded side effects: runs tool --version subprocesses,
performs a DNS lookup to check network availability, and writes/deletes
a temporary file to test output directory writability. All side effects
are read-like probes that leave no lasting state.

The policy engine consumes this snapshot to decide what's feasible on
this machine right now, not in the abstract.

Schema version is explicit for journal/cache compatibility.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


class ToolAvailability(BaseModel):
    """Whether a specific tool is available and its version."""
    name: str
    available: bool
    version: str = ""
    path: str = ""


class EnvironmentSnapshot(BaseModel):
    """A point-in-time snapshot of the host environment.

    Captured once at pipeline start. Policy engine consumes this to decide
    what's feasible on this machine right now, not in the abstract.
    """
    schema_version: str = SCHEMA_VERSION

    # OS
    os_name: str = Field(description="e.g. Darwin, Linux, Windows")
    os_version: str = ""
    arch: str = Field(description="e.g. arm64, x86_64")

    # Python
    python_version: str = ""
    python_path: str = ""

    # Toolchain
    tools: list[ToolAvailability] = Field(default_factory=list)

    # Docker
    docker_available: bool = False
    docker_version: str = ""

    # Network
    network_available: bool = False

    # Secrets (which required env vars are set — never log values)
    secrets_present: list[str] = Field(
        default_factory=list,
        description="Names of environment variables that are set (not their values)"
    )
    secrets_missing: list[str] = Field(
        default_factory=list,
        description="Names of required env vars that are NOT set"
    )

    # Disk
    output_dir_writable: bool = True


_BUILTIN_TOOLS = ["git", "curl", "gh", "docker", "node", "npm", "cargo", "go"]


def capture_snapshot(
    output_dir: str = ".",
    required_secrets: list[str] | None = None,
    required_tools: list[str] | None = None,
) -> EnvironmentSnapshot:
    """Capture the current environment.

    Bounded side effects: runs --version subprocesses, DNS lookup,
    and a write/delete probe on the output directory.

    Args:
        required_tools: Additional tool names to probe (from contract
            SYSTEM_TOOL capability requirements). These are checked
            alongside the built-in tool list so policy can match by name.
    """
    required_secrets = required_secrets or []
    required_tools = required_tools or []

    snap = EnvironmentSnapshot(
        os_name=platform.system(),
        os_version=platform.release(),
        arch=platform.machine(),
        python_version=platform.python_version(),
        python_path=sys.executable,
    )

    # Check built-in tools + any demanded by the contract
    all_tools = list(dict.fromkeys(_BUILTIN_TOOLS + [t.lower() for t in required_tools]))
    for tool_name in all_tools:
        tool = _check_tool(tool_name)
        snap.tools.append(tool)
        if tool_name == "docker":
            snap.docker_available = tool.available
            snap.docker_version = tool.version

    # Network check (DNS resolution, not a full HTTP round-trip)
    snap.network_available = _check_network()

    # Secrets
    for name in required_secrets:
        if os.environ.get(name):
            snap.secrets_present.append(name)
        else:
            snap.secrets_missing.append(name)

    # Output dir writable
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        test_file = out / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        snap.output_dir_writable = True
    except Exception:
        snap.output_dir_writable = False

    return snap


def _check_tool(name: str) -> ToolAvailability:
    """Check if a CLI tool is installed and get its version."""
    path = shutil.which(name)
    if not path:
        return ToolAvailability(name=name, available=False)

    version = ""
    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        # Take first line of output, strip common prefixes
        raw = (proc.stdout or proc.stderr).strip().split("\n")[0]
        version = raw[:100]
    except Exception:
        pass

    return ToolAvailability(name=name, available=True, version=version, path=path)


def _check_network() -> bool:
    """Quick DNS check — can we resolve a hostname?"""
    import socket
    try:
        socket.getaddrinfo("dns.google", 53, socket.AF_INET, socket.SOCK_STREAM)
        return True
    except (socket.gaierror, OSError):
        return False
