# SPDX-License-Identifier: Apache-2.0
"""Run the PR 2330 H100 correctness, component, and serving experiment."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


BASELINE_REVISION = "12f7b6670cc236b63cca4f53a9a495b9aea93cf5"
SUITE_PREFIX = "tests/experiments/pr2330_h100/"
CHECKPOINT_REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
DATASET_REVISION = "8f5e1aa2a35d42f42e940074c1983358b9491f89"


def run_stage(
    name: str,
    command: list[str],
    output: Path,
    source: Path,
    environment: dict[str, str],
) -> None:
    log_path = output / f"{name}.log"
    print(f"Running {name}: {' '.join(command)}", flush=True)
    with log_path.open("x") as log:
        result = subprocess.run(
            command,
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed ({result.returncode}); inspect {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Pinned checkpoint snapshot"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Pinned SeedTTS dataset snapshot"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New results directory"
    )
    parser.add_argument(
        "--asr-model", type=Path, help="Local ASR checkpoint for quality scoring"
    )
    parser.add_argument("--gpu", default="0", help="Physical H100 index or UUID")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()

    source = Path(__file__).resolve().parents[1]
    model = arguments.model.resolve()
    dataset = arguments.dataset.resolve()
    output = arguments.output.resolve()
    if model.name != CHECKPOINT_REVISION or not (model / "config.json").is_file():
        parser.error(f"--model must be a complete snapshot at {CHECKPOINT_REVISION}")
    if dataset.name != DATASET_REVISION:
        parser.error(f"--dataset must be the snapshot at {DATASET_REVISION}")
    meta = dataset / "en/meta.lst"
    warmup_meta = dataset / "zh/meta.lst"
    if not meta.is_file() or not warmup_meta.is_file():
        parser.error("The snapshot needs en/meta.lst and zh/meta.lst")
    if arguments.repeats < 1 or not 0 < arguments.port < 65536:
        parser.error("--repeats and --port must be positive")
    if not arguments.smoke_only and arguments.asr_model is None:
        parser.error("--asr-model is required for the full experiment")
    if arguments.asr_model is not None and not arguments.asr_model.is_dir():
        parser.error("--asr-model must be a local checkpoint directory")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if subprocess.run(
        ["git", "status", "--porcelain", "--", "sglang_omni/models/minicpm_o"],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    ).stdout:
        parser.error(
            "MiniCPM-o source has uncommitted changes; commit or stash it first"
        )

    archive = source / "tests/pr2330-h100-experiment.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(f"Experiment suite is missing: {archive}")
    config = json.loads((model / "config.json").read_text())
    vocabulary_size = config["tts_config"]["num_audio_tokens"]
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=arguments.gpu, OMP_NUM_THREADS="4")
    baseline = output / "baseline-checkout"
    run_stage(
        "baseline-checkout",
        ["git", "worktree", "add", "--detach", str(baseline), BASELINE_REVISION],
        output,
        source,
        environment,
    )

    with tempfile.TemporaryDirectory(prefix="pr2330-suite-") as temporary:
        suite = Path(temporary) / "pr2330_h100"
        suite.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                filename = Path(member.name).name
                if member.isdir() or filename.startswith("._"):
                    continue
                if not member.isfile() or not member.name.startswith(SUITE_PREFIX):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                if member.name != SUITE_PREFIX + filename:
                    raise ValueError(f"Unexpected archive path: {member.name}")
                contents = bundle.extractfile(member)
                assert contents is not None
                (suite / filename).write_bytes(contents.read())

        common = [
            "--baseline",
            str(baseline),
            "--candidate",
            str(source),
            "--model",
            str(model),
            "--meta",
            str(meta),
            "--warmup-meta",
            str(warmup_meta),
            "--gpu",
            arguments.gpu,
            "--port",
            str(arguments.port),
        ]
        run_stage(
            "talker-tests",
            [sys.executable, "-m", "pytest", "-q", "-rs", "tests/unit_test/minicpm_o"],
            output,
            source,
            environment,
        )
        run_stage(
            "harness-tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-rs",
                str(suite / "test_harness.py"),
            ],
            output,
            source,
            environment,
        )
        run_stage(
            "smoke",
            [
                sys.executable,
                str(suite / "run_ab.py"),
                *common,
                "--output",
                str(output / "smoke"),
                "--repeats",
                "1",
                "--samples",
                "8",
                "--concurrencies",
                "1",
            ],
            output,
            source,
            environment,
        )
        if arguments.smoke_only:
            print(f"Smoke passed; results: {output}")
            return

        for repetition in range(1, arguments.repeats + 1):
            for dtype in ("float32", "float16", "bfloat16"):
                for steps in (1, 32):
                    name = f"penalty-r{repetition}-{dtype}-steps{steps}"
                    run_stage(
                        name,
                        [
                            sys.executable,
                            "-m",
                            "benchmarks.benchmark_minicpm_talker_penalty",
                            "--device",
                            "cuda",
                            "--vocab-size",
                            str(vocabulary_size),
                            "--batch-sizes",
                            "1",
                            "2",
                            "4",
                            "8",
                            "16",
                            "32",
                            "--dtype",
                            dtype,
                            "--samples",
                            "200",
                            "--warmup",
                            "50",
                            "--steps-per-sample",
                            str(steps),
                            "--output",
                            str(output / f"{name}.json"),
                        ],
                        output,
                        source,
                        environment,
                    )
            order = (
                ("baseline", "candidate")
                if repetition % 2
                else ("candidate", "baseline")
            )
            for label in order:
                checkout = baseline if label == "baseline" else source
                checkout_environment = environment | {"PYTHONPATH": str(checkout)}
                run_stage(
                    f"prefill-r{repetition}-{label}",
                    [
                        sys.executable,
                        str(suite / "prefill.py"),
                        "--config",
                        str(model / "config.json"),
                        "--dtype",
                        "bfloat16",
                        "--output",
                        str(output / f"prefill-r{repetition}-{label}.json"),
                    ],
                    output,
                    checkout,
                    checkout_environment,
                )
        run_stage(
            "full",
            [
                sys.executable,
                str(suite / "run_ab.py"),
                *common,
                "--output",
                str(output / "full"),
                "--repeats",
                str(arguments.repeats),
            ],
            output,
            source,
            environment,
        )
        quality_points = {
            label: output / "full" / f"r1-c16-{label}" / "client"
            for label in ("baseline", "candidate")
        }
        for point in quality_points.values():
            records = [
                json.loads(line)
                for line in (point / "measured.jsonl").read_text().splitlines()
            ]
            generated = [
                {
                    "sample_id": record["sample_id"],
                    "target_text": record["target_text"],
                    "wav_path": str(point / "measured" / f"{index:04d}.wav"),
                    "is_success": not record["error"],
                    "latency_s": record["ready_s"],
                    "audio_duration_s": record["audio_s"],
                    "error": record["error"],
                }
                for index, record in enumerate(records)
            ]
            (point / "generated.json").write_text(json.dumps(generated, indent=2))

        asr_model = arguments.asr_model.resolve()
        asr_command = [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            str(asr_model),
            "--model-name",
            str(asr_model),
            "--host",
            "127.0.0.1",
            "--port",
            str(arguments.port),
        ]
        (output / "asr-command.json").write_text(json.dumps(asr_command, indent=2))
        with (output / "asr-server.log").open("x") as log:
            asr_server = subprocess.Popen(
                asr_command,
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 1200
                while True:
                    if asr_server.poll() is not None:
                        raise RuntimeError("ASR server exited; inspect asr-server.log")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("ASR server startup timed out")
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{arguments.port}/v1/models", timeout=2
                        ) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(1)
                for label, point in quality_points.items():
                    run_stage(
                        f"quality-{label}-transcribe",
                        [
                            sys.executable,
                            "-m",
                            "benchmarks.eval.benchmark_omni_seedtts",
                            "--transcribe-only",
                            "--model",
                            "MiniCPM-o-4_5",
                            "--meta",
                            str(meta),
                            "--lang",
                            "en",
                            "--output-dir",
                            str(point),
                            "--port",
                            str(arguments.port),
                            "--asr-model-path",
                            str(asr_model),
                        ],
                        output,
                        source,
                        environment,
                    )
            finally:
                try:
                    os.killpg(asr_server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    asr_server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(asr_server.pid, signal.SIGKILL)
                    asr_server.wait(timeout=20)
        for label, point in quality_points.items():
            for mode in ("similarity", "utmos"):
                run_stage(
                    f"quality-{label}-{mode}",
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.eval.benchmark_omni_seedtts",
                        f"--{mode}-only",
                        "--model",
                        "MiniCPM-o-4_5",
                        "--meta",
                        str(meta),
                        "--lang",
                        "en",
                        "--output-dir",
                        str(point),
                        "--device",
                        "cuda:0",
                    ],
                    output,
                    source,
                    environment,
                )
    print(f"Performance experiment complete: {output / 'full/comparison.md'}")
    print("Quality scores are saved beside the selected c16 audio cohorts.")
    print("Listen to matched audio before accepting a speedup.")


if __name__ == "__main__":
    main()
