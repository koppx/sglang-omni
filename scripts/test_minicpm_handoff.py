# SPDX-License-Identifier: Apache-2.0
"""Run gated MiniCPM-o handoff validation on an idle Linux CUDA GPU."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FrameType

if sys.version_info < (3, 10):
    raise RuntimeError("Start this script with Python 3.10 or newer")
else:
    pass

BASELINE_REVISION = "7f78bf3dc5c3053d6eb270764eb19232b617dec4"
MODEL_REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
UNIT_FILES = [
    f"tests/unit_test/minicpm_o/{name}.py"
    for name in (
        "test_request_builders",
        "test_thinker_model_runner",
        "test_sglang_talker",
        "test_native_config",
        "test_preprocessor_prompt",
    )
] + [
    "tests/unit_test/pipeline/test_stage.py",
    "tests/unit_test/pipeline/test_payload_device_restore.py",
    "tests/unit_test/benchmarks/test_minicpm_handoff.py",
]
GPU_CASES = [
    f"{UNIT_FILES[1]}::test_gpu_hidden_survives_graph_replay_on_another_stream",
    f"{UNIT_FILES[2]}::test_gpu_conditioning_matches_cpu_handoff_across_streams",
    f"{UNIT_FILES[5]}::test_stage_materializes_fenced_same_process_payload",
    f"{UNIT_FILES[6]}::test_cuda_payload_survives_delayed_cross_process_consumption",
]
METRICS = [
    "latency_median_s",
    "latency_p95_s",
    "throughput_qps",
    "audio_throughput_s_per_s",
    "audio_duration_mean_s",
]
DOWNLOAD = """
import hashlib
import json
import shutil
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import load_seedtts_samples
settings_path, destination = sys.argv[1:]
settings = json.loads(Path(settings_path).read_text())
manifest = {}
if settings["meta"]:
    manifest["meta"] = settings["meta"]
    manifest["dataset_source"] = "local"
else:
    directory = Path(settings["cache_dir"]) / "seedtts" / SEEDTTS_DATASET_REVISION / settings["lang"]
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / "meta.lst"
    if not metadata.exists():
        samples = load_seedtts_samples(
            SEEDTTS_DATASET_ID, split=settings["lang"], revision=SEEDTTS_DATASET_REVISION
        )
        rows = []
        for sample in samples:
            source = Path(sample.ref_audio)
            audio_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            audio_path = directory / (audio_hash + source.suffix)
            shutil.copyfile(source, audio_path)
            fields = [sample.sample_id, sample.ref_text, str(audio_path), sample.target_text]
            if any("|" in field or "\\n" in field or "\\r" in field for field in fields):
                raise ValueError(f"Unsupported metadata field: {sample.sample_id}")
            else:
                rows.append("|".join(fields))
        temporary = directory / "meta.lst.tmp"
        temporary.write_text("\\n".join(rows) + "\\n")
        temporary.replace(metadata)
    else:
        pass
    manifest.update(
        meta=str(metadata), dataset_source=SEEDTTS_DATASET_ID,
        dataset_revision=SEEDTTS_DATASET_REVISION, split=settings["lang"]
    )
if settings["checkpoint"]:
    manifest.update(checkpoint=settings["checkpoint"], model_source="local")
else:
    checkpoint = snapshot_download(
        "openbmb/MiniCPM-o-4_5", revision=settings["model_revision"]
    )
    manifest.update(
        checkpoint=checkpoint, model_source="openbmb/MiniCPM-o-4_5",
        model_revision=settings["model_revision"]
    )
Path(destination).write_text(json.dumps(manifest, indent=2))
"""
PROBE = """
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import torch
import sglang_omni
from benchmarks.dataset.seedtts import load_seedtts_samples
root, meta, warmup_meta, lang, warmup_count, smoke_count, destination = sys.argv[1:]
assert sys.platform == "linux", "Linux CUDA is required"
assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version
assert torch.__version__.split("+")[0] == "2.13.0", torch.__version__
assert importlib.metadata.version("sglang") == "0.5.21"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert Path(sglang_omni.__file__).resolve().is_relative_to(Path(root))
samples = load_seedtts_samples(meta, split=lang)
warmup = load_seedtts_samples(warmup_meta, int(warmup_count), split=lang)
assert len(samples) > int(smoke_count) and len(warmup) == int(warmup_count)
assert len({sample.sample_id for sample in samples}) == len(samples)
assert not {sample.sample_id for sample in samples} & {sample.sample_id for sample in warmup}
assert not {sample.target_text for sample in samples} & {sample.target_text for sample in warmup}
assert not {str(Path(sample.ref_audio).resolve()) for sample in samples} & {str(Path(sample.ref_audio).resolve()) for sample in warmup}
assert not {hashlib.sha256(Path(sample.ref_audio).read_bytes()).hexdigest() for sample in samples} & {hashlib.sha256(Path(sample.ref_audio).read_bytes()).hexdigest() for sample in warmup}
digest = hashlib.sha256()
for sample in samples + warmup:
    digest.update(json.dumps(vars(sample), sort_keys=True).encode())
    digest.update(Path(sample.ref_audio).read_bytes())
Path(destination).write_text(json.dumps({
    "samples": len(samples), "input_sha256": digest.hexdigest(),
    "torch": torch.__version__, "sglang": importlib.metadata.version("sglang"),
    "python": sys.version, "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0), "source": sglang_omni.__file__
}, indent=2))
"""


@dataclass(kw_only=True)
class MetadataRow:
    sample_id: str
    ref_text: str
    ref_audio: str
    target_text: str
    audio_sha256: str

    @property
    def line(self) -> str:
        return "|".join(
            (self.sample_id, self.ref_text, self.ref_audio, self.target_text)
        )


def read_metadata(path: Path) -> list[MetadataRow]:
    rows: list[MetadataRow] = []
    sample_ids: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        else:
            fields = line.strip().split("|")
        if len(fields) != 4 or any(not value.strip() for value in fields):
            raise ValueError(f"Expected four nonempty fields: {path}:{number}")
        elif fields[0] in sample_ids:
            raise ValueError(f"Duplicate sample ID: {path}:{number}")
        else:
            sample_ids.add(fields[0])
        audio_path = (path.parent / fields[2]).resolve(strict=True)
        audio_bytes = audio_path.read_bytes()
        if not audio_bytes:
            raise ValueError(f"Empty reference audio: {audio_path}")
        else:
            rows.append(
                MetadataRow(
                    sample_id=fields[0],
                    ref_text=fields[1],
                    ref_audio=str(audio_path),
                    target_text=fields[3],
                    audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
                )
            )
    return rows


def split_metadata(settings: argparse.Namespace, output: Path) -> None:
    samples = read_metadata(Path(settings.meta))
    if settings.warmup_meta:
        warmup = read_metadata(Path(settings.warmup_meta))[: settings.warmup_samples]
        measurement = samples
        policy = "external-warmup"
    elif len(samples) <= settings.warmup_samples + settings.smoke_samples:
        raise ValueError("Dataset is too small for disjoint warmup and full trials")
    else:
        selected = set(
            random.Random(settings.seed).sample(
                range(len(samples)), settings.warmup_samples
            )
        )
        warmup = [sample for i, sample in enumerate(samples) if i in selected]
        measurement = [sample for i, sample in enumerate(samples) if i not in selected]
        policy = "deterministic-holdout"
    warmup_ids = {sample.sample_id for sample in warmup}
    warmup_texts = {sample.target_text for sample in warmup}
    warmup_audio = {sample.audio_sha256 for sample in warmup}
    disjoint = [
        sample
        for sample in measurement
        if sample.sample_id not in warmup_ids
        and sample.target_text not in warmup_texts
        and sample.audio_sha256 not in warmup_audio
    ]
    excluded = len(measurement) - len(disjoint)
    if settings.warmup_meta and excluded:
        raise ValueError(
            "External warmup overlaps measurement IDs, text or audio bytes"
        )
    elif (
        len(warmup) != settings.warmup_samples
        or len(disjoint) <= settings.smoke_samples
    ):
        raise ValueError("Insufficient samples after removing warmup overlap")
    else:
        pass
    for name, rows in (("measurement", disjoint), ("warmup", warmup)):
        (output / f"{name}.lst").write_text(
            "\n".join(sample.line for sample in rows) + "\n"
        )
    fingerprint = hashlib.sha256()
    for sample in disjoint + warmup:
        fingerprint.update(
            json.dumps(
                [
                    sample.sample_id,
                    sample.ref_text,
                    sample.target_text,
                    sample.audio_sha256,
                ],
                ensure_ascii=False,
            ).encode()
        )
    settings.resources.update(
        split_policy=policy,
        split_seed=settings.seed,
        source_samples=len(samples),
        warmup_samples=len(warmup),
        measurement_samples=len(disjoint),
        extra_overlap_removed=excluded,
        content_sha256=fingerprint.hexdigest(),
        warmup_ids=[sample.sample_id for sample in warmup],
        measurement_ids=[sample.sample_id for sample in disjoint],
    )
    settings.meta = str(output / "measurement.lst")
    settings.warmup_meta = str(output / "warmup.lst")
    (output / "resources.json").write_text(json.dumps(settings.resources, indent=2))


def check_gpu_idle(gpu: str, log_path: Path, *, max_idle_memory_mib: int) -> None:
    """Refuse active or unobservable GPUs before launching CUDA work."""
    if max_idle_memory_mib < 0:
        raise ValueError("max_idle_memory_mib must be nonnegative")
    else:
        pass
    observations: dict[str, str] = {}
    with log_path.open("w") as log:
        for name, query in (
            ("processes", "--query-compute-apps=pid"),
            ("activity", "--query-gpu=memory.used,utilization.gpu"),
        ):
            command = ["nvidia-smi", "-i", gpu, query, "--format=csv,noheader,nounits"]
            log.write(f"command: {shlex.join(command)}\n")
            log.flush()
            observation = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
            log.write(observation.stdout)
            log.write(observation.stderr)
            if observation.returncode != 0 or observation.stderr.strip():
                raise RuntimeError(f"Cannot verify GPU {gpu} is idle: {log_path}")
            else:
                observations[name] = observation.stdout.strip()
        if observations["processes"]:
            raise RuntimeError(
                f"GPU {gpu} has compute processes or unknown state: {log_path}"
            )
        else:
            pass
        fields = observations["activity"].split(",")
        if len(fields) != 2 or not all(value.strip().isdigit() for value in fields):
            raise RuntimeError(f"Invalid GPU activity report for {gpu}: {log_path}")
        else:
            memory_used_mib, utilization_percent = (
                int(value.strip()) for value in fields
            )
        if memory_used_mib > max_idle_memory_mib or utilization_percent != 0:
            raise RuntimeError(
                f"GPU {gpu} is not idle: memory={memory_used_mib} MiB, "
                f"utilization={utilization_percent}%; {log_path}"
            )
        else:
            pass


def prepare_environment(
    settings: argparse.Namespace,
    repository: Path,
    environment: dict[str, str],
    output: Path,
) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Linux with an NVIDIA CUDA driver is required")
    elif shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is missing; provision an NVIDIA CUDA host first")
    else:
        check_gpu_idle(
            settings.gpu,
            output / "gpu-admission.log",
            max_idle_memory_mib=settings.max_idle_memory_mib,
        )
    if settings.use_current_env:
        settings.python = sys.executable
    else:
        cache = Path(settings.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        uv_path = shutil.which("uv")
        if uv_path is None:
            bootstrap = cache / "bootstrap"
            execute(
                [sys.executable, "-m", "venv", str(bootstrap)],
                repository,
                environment,
                output / "bootstrap-venv.log",
                120,
            )
            execute(
                [
                    str(bootstrap / "bin/python"),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "uv",
                ],
                repository,
                environment,
                output / "bootstrap-uv.log",
                settings.setup_timeout_seconds,
            )
            uv_path = str(bootstrap / "bin/uv")
        else:
            pass
        virtualenv = cache / "environments" / settings.candidate
        if not (virtualenv / "bin/python").is_file():
            execute(
                [uv_path, "venv", "--python", "3.12", "--seed", str(virtualenv)],
                repository,
                environment,
                output / "environment-create.log",
                settings.setup_timeout_seconds,
            )
        else:
            pass
        settings.python = str(virtualenv / "bin/python")
        execute(
            [
                uv_path,
                "pip",
                "install",
                "--python",
                settings.python,
                "--prerelease=allow",
                "-e",
                f"{repository}[minicpm-o]",
            ],
            repository,
            environment,
            output / "environment-install.log",
            settings.setup_timeout_seconds,
        )
    environment["PATH"] = (
        str(Path(settings.python).parent) + os.pathsep + environment["PATH"]
    )
    environment.pop("PYTHONHOME", None)
    environment["VIRTUAL_ENV"] = str(Path(settings.python).parent.parent)
    check_gpu_idle(
        settings.gpu,
        output / "gpu-admission-environment-check.log",
        max_idle_memory_mib=settings.max_idle_memory_mib,
    )
    execute(
        [
            settings.python,
            "-c",
            "import importlib.metadata; import torch; "
            "assert torch.__version__.split('+')[0] == '2.13.0', torch.__version__; "
            "assert importlib.metadata.version('sglang') == '0.5.21'; "
            "assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "
            "'The selected GPU must be visible to the installed CUDA runtime'",
        ],
        repository,
        environment,
        output / "environment-check.log",
        120,
    )


@dataclass(kw_only=True)
class Trial:
    phase: str
    arm: str
    concurrency: int
    repetition: int
    samples: int
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.phase}-{self.arm}-c{self.concurrency}-r{self.repetition}"


def trial_plan(sample_count: int, settings: argparse.Namespace) -> list[Trial]:
    trials = [
        Trial(
            phase="smoke",
            arm=arm,
            concurrency=1,
            repetition=1,
            samples=settings.smoke_samples,
        )
        for arm in ("baseline", "p0", "p1")
    ]
    for repetition in range(1, settings.repeats + 1):
        arms = ("baseline", "p0", "p1")
        offset = (repetition - 1) % len(arms)
        for concurrency in settings.concurrencies:
            for arm in arms[offset:] + arms[:offset]:
                trials.append(
                    Trial(
                        phase="full",
                        arm=arm,
                        concurrency=concurrency,
                        repetition=repetition,
                        samples=sample_count,
                    )
                )
    return trials


def stop_process(
    process: subprocess.Popen[bytes], gpu: str | None = None
) -> None:
    """Terminate only this launch's process group, including remaining workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    except ProcessLookupError:
        process.wait(timeout=10)
    if gpu is None:
        return
    # Engine workers can escape the process group (spawned subprocess setsid)
    # and keep holding GPU memory, which trips the next admission check. On a
    # Slurm-exclusive allocation every compute process on the selected GPU
    # belongs to this job, so reap leftovers and wait for the memory to drain.
    for _ in range(12):
        query = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [
            int(token)
            for token in query.stdout.split()
            if token.strip().isdigit()
        ]
        if not pids:
            return
        else:
            pass
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(5)


def execute(
    command: list[str],
    checkout: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> None:
    print(f"Running {log_path.name}", flush=True)
    with log_path.open("w") as log:
        log.write(f"cwd: {checkout}\ncommand: {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=checkout,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
            if return_code:
                raise RuntimeError(f"Exit {return_code}: {log_path}")
            else:
                pass
        finally:
            stop_process(process)


def validate_junit(path: Path, minimum_cases: int, reject_skips: bool) -> None:
    cases = list(ET.parse(path).iter("testcase"))
    bad = [
        case.get("name")
        for case in cases
        if case.find("failure") is not None
        or case.find("error") is not None
        or (reject_skips and case.find("skipped") is not None)
    ]
    executed = [case for case in cases if case.find("skipped") is None]
    if bad or len(executed) < minimum_cases:
        raise RuntimeError(
            f"JUnit gate failed: {path}; executed={len(executed)}, bad={bad}"
        )
    else:
        pass


def validate_speed(path: Path, expected_samples: int) -> dict[str, float]:
    data = json.loads(path.read_text())
    summary = data["summary"]
    requests = data["per_request"]
    if (
        summary["total_requests"] != expected_samples
        or summary["completed_requests"] != expected_samples
        or summary["failed_requests"] != 0
        or len(requests) != expected_samples
    ):
        raise RuntimeError(f"Incomplete or failed requests: {path}")
    else:
        pass
    for request in requests:
        if (
            request["is_success"] is not True
            or not math.isfinite(request["audio_duration_s"])
            or request["audio_duration_s"] <= 0
            or not Path(request["wav_path"]).is_file()
            or Path(request["wav_path"]).stat().st_size == 0
        ):
            raise RuntimeError(f"Invalid audio result: {path}; id={request['id']}")
        else:
            pass
    metrics = {name: float(summary[name]) for name in METRICS}
    if not all(math.isfinite(value) and value > 0 for value in metrics.values()):
        raise RuntimeError(f"Invalid performance metrics: {path}")
    else:
        return metrics


def write_report(
    output: Path, status: str, trials: list[Trial], settings: argparse.Namespace
) -> None:
    record = {
        "status": status,
        "baseline": BASELINE_REVISION,
        "settings": vars(settings),
        "trials": [asdict(trial) for trial in trials],
    }
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(output / "report.json")
    lines = [
        "# MiniCPM-o handoff validation",
        "",
        f"Status: **{status}**",
        "",
        f"Resources: {output / 'resources.json'} (source, revisions, split and content fingerprint).",
        f"Measurement samples: {vars(settings).get('resources', {}).get('measurement_samples', 'pending')}; "
        f"warmup samples: {settings.warmup_samples}.",
        "",
        "Generation checks only: audio quality, full-model tensor parity and handoff device latency remain unverified.",
        "Latency is non-streaming request completion, not incremental audio first-chunk latency.",
        "Smoke timings include COMM_TRACE; use only full trials for performance comparisons.",
        "",
        "| Trial | Samples | E2E p50 s | E2E p95 s | req/s | audio s/s | mean audio s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for trial in trials:
        if trial.metrics:
            lines.append(
                f"| {trial.name} | {trial.samples} | "
                + " | ".join(f"{trial.metrics[name]:.4f}" for name in METRICS)
                + " |"
            )
        else:
            pass
    lines += [
        "",
        "## Full-run median comparisons",
        "",
        "| Concurrency | Comparison | Throughput change | p95 change |",
        "|---|---|---:|---:|",
    ]
    for concurrency in settings.concurrencies:
        for before, after in (("baseline", "p0"), ("p0", "p1"), ("baseline", "p1")):
            before_runs = [
                trial
                for trial in trials
                if trial.phase == "full"
                and trial.concurrency == concurrency
                and trial.arm == before
                and trial.metrics
            ]
            after_runs = [
                trial
                for trial in trials
                if trial.phase == "full"
                and trial.concurrency == concurrency
                and trial.arm == after
                and trial.metrics
            ]
            if len(before_runs) == len(after_runs) == settings.repeats:
                changes = [
                    100
                    * (
                        statistics.median(trial.metrics[name] for trial in after_runs)
                        / statistics.median(
                            trial.metrics[name] for trial in before_runs
                        )
                        - 1
                    )
                    for name in ("throughput_qps", "latency_p95_s")
                ]
                lines.append(
                    f"| {concurrency} | {after} vs {before} | {changes[0]:+.2f}% | {changes[1]:+.2f}% |"
                )
            else:
                pass
    lines += [
        "",
        "Positive throughput is better; negative p95 is better. Differences are observations, not significance or quality acceptance.",
        "Raw logs, JUnit XML, speed_results.json, per-request CSV and WAVs are retained. No test source is deleted.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def run_trial(
    trial: Trial,
    checkout: Path,
    client_checkout: Path,
    environment: dict[str, str],
    output: Path,
    settings: argparse.Namespace,
) -> None:
    directory = output / trial.name
    directory.mkdir()
    check_gpu_idle(
        settings.gpu,
        directory / "gpu-admission.log",
        max_idle_memory_mib=settings.max_idle_memory_mib,
    )
    server_environment = dict(
        environment,
        PYTHONPATH=str(checkout),
        SGLANG_OMNI_COMM_TRACE="1" if trial.phase == "smoke" else "0",
    )
    if trial.phase == "smoke":
        # The size gate routes small payloads to the pooled relay, so force
        # the direct-IPC branch during smoke to keep its assertion meaningful.
        server_environment["SGLANG_OMNI_DIRECT_IPC_MIN_BYTES"] = "1"
    else:
        pass
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", settings.port))
    command = [
        settings.python,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        settings.checkpoint,
        "--host",
        "127.0.0.1",
        "--port",
        str(settings.port),
        "--thinker.engine.mem_fraction_static",
        "0.55",
        "--talker.engine.mem_fraction_static",
        "0.15",
    ]
    if trial.arm != "baseline":
        command += ["--enable_gpu_handoff", str(trial.arm == "p1").lower()]
    else:
        pass
    with (directory / "server.log").open("w") as log:
        log.write(f"cwd: {checkout}\ncommand: {shlex.join(command)}\n")
        log.flush()
        server = subprocess.Popen(
            command,
            cwd=checkout,
            env=server_environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + settings.startup_timeout_seconds
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited: {directory / 'server.log'}")
                elif time.monotonic() >= deadline:
                    raise TimeoutError(f"Server startup timed out: {directory}")
                else:
                    pass
                try:
                    with opener.open(
                        f"http://127.0.0.1:{settings.port}/v1/models", timeout=2
                    ) as response:
                        models = json.load(response)
                    model_name = models["data"][0]["id"]
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            benchmark = [
                settings.python,
                "-m",
                "benchmarks.eval.benchmark_omni_seedtts",
                "--host",
                "127.0.0.1",
                "--port",
                str(settings.port),
                "--model",
                model_name,
                "--meta",
                settings.meta,
                "--warmup-meta",
                settings.warmup_meta,
                "--lang",
                settings.lang,
                "--generate-only",
                "--voice-clone",
                "--reference-audio-field",
                "audio.ref_audio",
                "--warmup",
                str(settings.warmup_samples),
                "--max-samples",
                str(trial.samples),
                "--max-concurrency",
                str(trial.concurrency),
                "--temperature",
                "0.7",
                "--max-new-tokens",
                "256",
                "--seed",
                str(settings.seed),
                "--fingerprint",
                "--output-dir",
                str(directory),
            ]
            with (directory / "gpu-memory.csv").open("w") as memory_log:
                monitor = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "-i",
                        settings.gpu,
                        "--query-gpu=timestamp,memory.used,utilization.gpu",
                        "--format=csv,nounits",
                        "-l",
                        "1",
                    ],
                    stdout=memory_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    execute(
                        benchmark,
                        client_checkout,
                        environment,
                        directory / "benchmark.log",
                        settings.trial_timeout_seconds,
                    )
                    if monitor.poll() is not None:
                        raise RuntimeError(
                            f"GPU sampling stopped: {directory / 'gpu-memory.csv'}"
                        )
                    else:
                        pass
                finally:
                    stop_process(monitor)
            trial.metrics = validate_speed(
                directory / "speed_results.json", trial.samples
            )
        finally:
            stop_process(server, gpu=settings.gpu)
    if trial.phase == "smoke" and trial.arm == "p1":
        trace = (directory / "server.log").read_text()
        if (
            '"event": "comm_payload_device_ready"' not in trace
            or '"transport": "torch_cuda_ipc"' not in trace
        ):
            raise RuntimeError(
                f"P1 direct IPC was not observed: {directory / 'server.log'}"
            )
        else:
            pass
    else:
        pass


def interrupt_run(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"Received signal {signum}")


def run(settings: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = Path(settings.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Results: {output}", flush=True)
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=settings.gpu,
        OMP_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    environment.pop("CUDA_LAUNCH_BLOCKING", None)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONOPTIMIZE", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTEST_ADDOPTS", None)
    environment["SGLANG_OMNI_COMM_TRACE"] = "0"
    trials: list[Trial] = []
    write_report(output, "RUNNING: preflight", trials, settings)
    checkouts: list[Path] = []
    cache_lock: int | None = None
    try:
        settings.candidate = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            text=True,
        ).strip():
            raise RuntimeError("Commit tracked changes before validation")
        else:
            pass
        for name in ("checkpoint", "meta", "warmup_meta"):
            value = vars(settings)[name]
            if value:
                vars(settings)[name] = str(
                    Path(value).expanduser().resolve(strict=True)
                )
            else:
                pass
        settings.cache_dir = str(Path(settings.cache_dir).expanduser().resolve())
        Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)
        cache_lock = os.open(
            Path(settings.cache_dir) / "run.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            fcntl.flock(cache_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another run is using this cache; select a separate --cache-dir"
            ) from error
        environment.setdefault("HF_HOME", str(Path(settings.cache_dir) / "huggingface"))
        environment.setdefault("UV_CACHE_DIR", str(Path(settings.cache_dir) / "uv"))
        environment.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        environment.setdefault("UV_HTTP_TIMEOUT", "120")
        write_report(output, "RUNNING: preparing Python environment", trials, settings)
        prepare_environment(settings, repository, environment, output)
        write_report(output, "RUNNING: preparing model and dataset", trials, settings)
        (output / "resource-settings.json").write_text(
            json.dumps(vars(settings), indent=2)
        )
        execute(
            [
                settings.python,
                "-c",
                DOWNLOAD,
                str(output / "resource-settings.json"),
                str(output / "downloads.json"),
            ],
            repository,
            environment,
            output / "resources.log",
            settings.setup_timeout_seconds,
        )
        settings.resources = json.loads((output / "downloads.json").read_text())
        settings.checkpoint = settings.resources["checkpoint"]
        settings.meta = settings.resources["meta"]
        split_metadata(settings, output)
        config = Path(settings.checkpoint) / "config.json"
        settings.checkpoint_config_sha256 = hashlib.sha256(
            config.read_bytes()
        ).hexdigest()
        if subprocess.run(
            ["git", "cat-file", "-e", f"{BASELINE_REVISION}^{{commit}}"],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode:
            execute(
                ["git", "fetch", "--no-tags", "origin", BASELINE_REVISION],
                repository,
                environment,
                output / "baseline-fetch.log",
                settings.setup_timeout_seconds,
            )
        else:
            pass
        with tempfile.TemporaryDirectory(prefix="minicpm-handoff-") as temporary:
            try:
                for arm, revision in (
                    ("baseline", BASELINE_REVISION),
                    ("candidate", settings.candidate),
                ):
                    checkout = Path(temporary) / arm
                    execute(
                        ["git", "worktree", "add", "--detach", str(checkout), revision],
                        repository,
                        environment,
                        output / f"checkout-{arm}.log",
                        120,
                    )
                    checkouts.append(checkout)
                baseline, candidate = checkouts
                environment["PYTHONPATH"] = str(candidate)
                check_gpu_idle(
                    settings.gpu,
                    output / "gpu-admission-preflight.log",
                    max_idle_memory_mib=settings.max_idle_memory_mib,
                )
                execute(
                    [
                        settings.python,
                        "-c",
                        PROBE,
                        str(candidate),
                        settings.meta,
                        settings.warmup_meta,
                        settings.lang,
                        str(settings.warmup_samples),
                        str(settings.smoke_samples),
                        str(output / "environment.json"),
                    ],
                    candidate,
                    environment,
                    output / "preflight.log",
                    settings.trial_timeout_seconds,
                )
                sample_count = json.loads((output / "environment.json").read_text())[
                    "samples"
                ]
                trials = trial_plan(sample_count, settings)
                execute(
                    [settings.python, "-m", "pip", "freeze"],
                    candidate,
                    environment,
                    output / "packages.log",
                    120,
                )
                execute(["nvidia-smi"], candidate, environment, output / "gpu.log", 30)
                for label, targets, minimum, reject_skips in (
                    (
                        "unit-small",
                        [UNIT_FILES[0], "-k", "handoff or projection"],
                        21,
                        True,
                    ),
                    *[
                        (f"gpu-core-{repetition}", GPU_CASES, 5, True)
                        for repetition in range(1, 4)
                    ],
                ):
                    write_report(output, f"RUNNING: {label}", trials, settings)
                    check_gpu_idle(
                        settings.gpu,
                        output / f"gpu-admission-{label}.log",
                        max_idle_memory_mib=settings.max_idle_memory_mib,
                    )
                    execute(
                        [
                            settings.python,
                            "-m",
                            "pytest",
                            *targets,
                            "-q",
                            "-rs",
                            "--tb=short",
                            "-p",
                            "no:cacheprovider",
                            f"--junitxml={output / (label + '.xml')}",
                        ],
                        candidate,
                        environment,
                        output / f"{label}.log",
                        settings.trial_timeout_seconds,
                    )
                    validate_junit(output / f"{label}.xml", minimum, reject_skips)
                for trial in trials:
                    if trial.phase == "full" and not settings.full_regression_done:
                        write_report(
                            output, "RUNNING: full unit regression", trials, settings
                        )
                        check_gpu_idle(
                            settings.gpu,
                            output / "gpu-admission-unit-full.log",
                            max_idle_memory_mib=settings.max_idle_memory_mib,
                        )
                        execute(
                            [
                                settings.python,
                                "-m",
                                "pytest",
                                *UNIT_FILES,
                                "-q",
                                "-rs",
                                "--tb=short",
                                "-p",
                                "no:cacheprovider",
                                f"--junitxml={output / 'unit-full.xml'}",
                            ],
                            candidate,
                            environment,
                            output / "unit-full.log",
                            settings.trial_timeout_seconds,
                        )
                        validate_junit(output / "unit-full.xml", 26, False)
                        settings.full_regression_done = True
                    else:
                        pass
                    write_report(output, f"RUNNING: {trial.name}", trials, settings)
                    print(
                        f"Starting {trial.name} ({trial.samples} samples)", flush=True
                    )
                    run_trial(
                        trial,
                        baseline if trial.arm == "baseline" else candidate,
                        candidate,
                        environment,
                        output,
                        settings,
                    )
                    write_report(output, f"COMPLETED: {trial.name}", trials, settings)
            finally:
                for checkout in reversed(checkouts):
                    execute(
                        ["git", "worktree", "remove", str(checkout)],
                        repository,
                        environment,
                        output / f"cleanup-{checkout.name}.log",
                        120,
                    )
        write_report(
            output,
            "COMPLETED: generation validation; quality review pending",
            trials,
            settings,
        )
    except BaseException as error:
        write_report(
            output, f"FAILED: {type(error).__name__}: {error}", trials, settings
        )
        raise
    finally:
        if cache_lock is not None:
            os.close(cache_lock)
        else:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        help="Local immutable checkpoint; default: download the pinned MiniCPM-o model",
    )
    parser.add_argument(
        "--meta", help="Local SeedTTS meta.lst; default: download pinned Arrow data"
    )
    parser.add_argument(
        "--warmup-meta",
        help="Independent local warmup metadata; default: hold out 16 rows from the dataset",
    )
    parser.add_argument(
        "--output",
        default=f"handoff-results/{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}",
        help="New result directory; existing paths are never overwritten",
    )
    parser.add_argument(
        "--cache-dir", default=str(Path.home() / ".cache/sglang-omni/handoff")
    )
    parser.add_argument(
        "--use-current-env",
        action="store_true",
        help="Use the current Python environment without installing dependencies",
    )
    parser.add_argument(
        "--model-revision",
        default=MODEL_REVISION,
        help="Immutable model commit to download",
    )
    parser.add_argument("--gpu", default="0", help="One physical GPU index or UUID")
    parser.add_argument(
        "--max-idle-memory-mib",
        type=int,
        default=256,
        help="Admission ceiling for driver/display memory; compute processes are always rejected",
    )
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--smoke-samples", type=int, default=8)
    parser.add_argument("--warmup-samples", type=int, default=16)
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1200)
    parser.add_argument("--trial-timeout-seconds", type=int, default=7200)
    parser.add_argument("--setup-timeout-seconds", type=int, default=14400)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without loading dependencies or starting processes",
    )
    settings = parser.parse_args()
    settings.full_regression_done = False
    positive = [
        settings.smoke_samples,
        settings.warmup_samples,
        settings.repeats,
        settings.startup_timeout_seconds,
        settings.trial_timeout_seconds,
        settings.setup_timeout_seconds,
        *settings.concurrencies,
    ]
    if min(positive) <= 0 or not 0 < settings.port < 65536 or "," in settings.gpu:
        parser.error(
            "Counts/timeouts must be positive; select one GPU and a valid port"
        )
    elif settings.max_idle_memory_mib < 0:
        parser.error("--max-idle-memory-mib must be nonnegative")
    elif len(settings.model_revision) != 40 or any(
        character not in "0123456789abcdef" for character in settings.model_revision
    ):
        parser.error("--model-revision must be a full lowercase commit SHA")
    elif settings.dry_run:
        print(
            json.dumps(
                {
                    "baseline": BASELINE_REVISION,
                    "environment": (
                        "current"
                        if settings.use_current_env
                        else "automatic Python 3.12 virtualenv"
                    ),
                    "checkpoint": settings.checkpoint
                    or f"openbmb/MiniCPM-o-4_5@{settings.model_revision}",
                    "dataset": settings.meta or "pinned SeedTTS Arrow snapshot",
                    "warmup": settings.warmup_meta
                    or f"hold out {settings.warmup_samples} rows with seed {settings.seed}",
                    "full_samples": "remaining disjoint rows; exact count recorded after preparation",
                    "output": settings.output,
                    "trials": [asdict(trial) for trial in trial_plan(0, settings)],
                },
                indent=2,
            )
        )
    else:
        signal.signal(signal.SIGTERM, interrupt_run)
        run(settings)


if __name__ == "__main__":
    main()
else:
    pass
