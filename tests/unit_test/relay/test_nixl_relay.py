# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]
FAKE_API = """
class nixl_agent:
    source = {source!r}


class nixl_agent_config:
    pass
"""


def write_fake_package(root: Path, package_name: str, api_source: str | None) -> None:
    package = root / package_name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    if api_source is not None:
        (package / "_api.py").write_text(
            FAKE_API.format(source=api_source), encoding="utf-8"
        )
    else:
        pass


def run_isolated_import(root: Path, source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{root}{os.pathsep}{existing_python_path}"
        if existing_python_path
        else str(root)
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_nixl_imports_cuda13_wheel_without_meta_package(tmp_path: Path) -> None:
    write_fake_package(tmp_path, "nixl_cu13", "cuda13")
    write_fake_package(tmp_path, "nixl", None)
    result = run_isolated_import(
        tmp_path,
        """
import sglang_omni.relay.nixl as relay

assert relay.NIXL_AVAILABLE is True
assert relay.NixlAgent.source == "cuda13"
""",
    )
    assert result.returncode == 0, result.stderr


def test_nixl_falls_back_to_generic_package(tmp_path: Path) -> None:
    write_fake_package(tmp_path, "nixl_cu13", None)
    write_fake_package(tmp_path, "nixl", "generic")
    result = run_isolated_import(
        tmp_path,
        """
import sglang_omni.relay.nixl as relay

assert relay.NIXL_AVAILABLE is True
assert relay.NixlAgent.source == "generic"
""",
    )
    assert result.returncode == 0, result.stderr


def test_nixl_connection_reports_import_failure(tmp_path: Path) -> None:
    write_fake_package(tmp_path, "nixl_cu13", None)
    write_fake_package(tmp_path, "nixl", None)
    result = run_isolated_import(
        tmp_path,
        """
import sglang_omni.relay.nixl as relay

assert relay.NIXL_AVAILABLE is False
try:
    relay.Connection("test")
except RuntimeError as error:
    assert "NIXL is unavailable" in str(error)
else:
    raise AssertionError("Connection accepted an unavailable NIXL backend")
""",
    )
    assert result.returncode == 0, result.stderr
