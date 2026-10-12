# SPDX-License-Identifier: Apache-2.0
"""GPU admission gates for the MiniCPM-o handoff validation entry point."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import test_minicpm_handoff as handoff


@pytest.mark.parametrize("gpu", ["0", "GPU-test-device"])
@pytest.mark.parametrize("memory_used_mib", [0, 256])
def test_idle_gpu_is_accepted_and_queries_are_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu: str, memory_used_mib: int
) -> None:
    commands: list[list[str]] = []

    def query(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert timeout == 30
        assert capture_output and text and not check
        output = (
            "" if "--query-compute-apps=pid" in command else f"{memory_used_mib}, 0\n"
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(handoff.subprocess, "run", query)
    log_path = tmp_path / "admission.log"
    handoff.check_gpu_idle(gpu, log_path, max_idle_memory_mib=256)
    assert len(commands) == 2
    assert all(command[1:3] == ["-i", gpu] for command in commands)
    assert "--query-compute-apps=pid" in log_path.read_text()
    assert "--query-gpu=memory.used,utilization.gpu" in log_path.read_text()


@pytest.mark.parametrize(
    ("processes", "activity"),
    [
        ("123\n", "0, 0\n"),
        ("N/A\n", "0, 0\n"),
        ("", "257, 0\n"),
        ("", "0, 1\n"),
        ("", ""),
        ("", "N/A, 0\n"),
        ("", "0, N/A\n"),
        ("", "0, 0\n0, 0\n"),
        ("", "-1, 0\n"),
    ],
)
def test_busy_or_unknown_gpu_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, processes: str, activity: str
) -> None:
    responses = [
        subprocess.CompletedProcess([], 0, processes, ""),
        subprocess.CompletedProcess([], 0, activity, ""),
    ]
    monkeypatch.setattr(handoff.subprocess, "run", Mock(side_effect=responses))
    with pytest.raises(RuntimeError):
        handoff.check_gpu_idle("0", tmp_path / "admission.log", max_idle_memory_mib=256)


@pytest.mark.parametrize("failure", ["exit", "warning", "timeout"])
def test_query_failures_stop_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure == "timeout":
        query = Mock(side_effect=subprocess.TimeoutExpired("nvidia-smi", 30))
        expected_error = subprocess.TimeoutExpired
    else:
        query = Mock(
            return_value=subprocess.CompletedProcess(
                [], 1 if failure == "exit" else 0, "", "driver query failed"
            )
        )
        expected_error = RuntimeError
    monkeypatch.setattr(handoff.subprocess, "run", query)
    with pytest.raises(expected_error):
        handoff.check_gpu_idle("0", tmp_path / "admission.log", max_idle_memory_mib=256)


def test_busy_gpu_prevents_environment_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handoff.sys, "platform", "linux")
    monkeypatch.setattr(handoff.shutil, "which", Mock(return_value="nvidia-smi"))
    admission = Mock(side_effect=RuntimeError("GPU busy"))
    execute = Mock()
    monkeypatch.setattr(handoff, "check_gpu_idle", admission)
    monkeypatch.setattr(handoff, "execute", execute)
    settings = argparse.Namespace(gpu="0", max_idle_memory_mib=256)
    with pytest.raises(RuntimeError, match="GPU busy"):
        handoff.prepare_environment(settings, tmp_path, dict(os.environ), tmp_path)
    execute.assert_not_called()
    admission.assert_called_once_with(
        "0", tmp_path / "gpu-admission.log", max_idle_memory_mib=256
    )


def test_busy_gpu_prevents_trial_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admission = Mock(side_effect=RuntimeError("GPU busy"))
    launch = Mock()
    monkeypatch.setattr(handoff, "check_gpu_idle", admission)
    monkeypatch.setattr(handoff.subprocess, "Popen", launch)
    trial = handoff.Trial(
        phase="smoke", arm="p1", concurrency=1, repetition=1, samples=8
    )
    settings = argparse.Namespace(gpu="0", max_idle_memory_mib=256)
    with pytest.raises(RuntimeError, match="GPU busy"):
        handoff.run_trial(
            trial, tmp_path, tmp_path, dict(os.environ), tmp_path, settings
        )
    launch.assert_not_called()
    admission.assert_called_once_with(
        "0", tmp_path / trial.name / "gpu-admission.log", max_idle_memory_mib=256
    )


def test_idle_memory_limit_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        handoff.check_gpu_idle("0", tmp_path / "admission.log", max_idle_memory_mib=-1)


@pytest.mark.parametrize("memory_limit", ["-1", "invalid"])
def test_cli_rejects_invalid_idle_memory_limit(memory_limit: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(handoff.__file__)),
            "--dry-run",
            "--max-idle-memory-mib",
            memory_limit,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "--max-idle-memory-mib" in result.stderr
