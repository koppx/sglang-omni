#!/usr/bin/env python3
"""Bounded, non-interactive A100 acceptance runner. Bootstrap uses stdlib only."""
from __future__ import annotations

import argparse
import fcntl
import html
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def stop(process):
    # Children inherit this process group; do not use pkill or a GPU PID list.
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # Parent may exit while a worker remains alive in its group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def report(out, state):
    rows = state["stages"]
    bad = any(r["status"] == "FAIL" for r in rows)
    incomplete = any(r["status"] in ("PENDING", "RUNNING", "INCOMPLETE", "BLOCKED") for r in rows)
    state["status"] = "FAIL" if bad else "INCOMPLETE" if incomplete else "PASS"
    state["updated_at"] = time.time()
    save(out / "summary.json", state)
    lines = ["# MiniCPM-o A100 acceptance report", "", f"Status: **{state['status']}**", "",
             f"Commit: `{state.get('commit', 'unknown')}`", f"Run: `{out.name}`", "",
             "Quality thresholds are provisional smoke gates, not published model scores.",
             "Subsample evaluations are regression checks, not leaderboard results.", "",
             "| Stage | Status | Seconds | Evidence / error |", "|---|---|---:|---|"]
    suite = ET.Element("testsuite", name="minicpm-a100", tests=str(len(rows)))
    for r in rows:
        note = r.get("error", "").replace("|", "/").replace("\n", " ")
        log = r.get("log")
        evidence = f"[{log}]({log})" if log else ""
        lines.append(f"| {r['name']} | {r['status']} | {r.get('seconds', 0):.1f} | {evidence} {note} |")
        case = ET.SubElement(suite, "testcase", name=r["name"], time=str(r.get("seconds", 0)))
        if r["status"] == "FAIL":
            ET.SubElement(case, "failure", message=note).text = note
        elif r["status"] != "PASS":
            ET.SubElement(case, "skipped", message=note or r["status"])
    for file in sorted(out.glob("results/*.json")):
        if ".partial." in file.name or ".last_batch." in file.name:
            continue
        raw = file.read_text()
        if len(raw) > 18000:
            data = json.loads(raw)
            compact = {k: (f"{len(v)} entries; see raw JSON" if isinstance(v, list) else v) for k, v in data.items()}
            raw = json.dumps(compact, ensure_ascii=False, indent=2)
        lines += ["", f"## {file.stem}", "", f"Raw result: [{file.name}]({file.relative_to(out)})", "",
                  "```json", raw, "```"]
    lines += ["", "## Interpretation", "",
              "GPU utilization is not SM/Tensor occupancy. CPU samples and traces are diagnostic runs, excluded from baseline timing.",
              "A/B differences are hypotheses unless repeated, above measured A/A noise, and correctness checks pass.",
              "A successful diagnostic configuration does not erase a default-startup failure.",
              "No remote deployment occurred merely by generating this report."]
    md = "\n".join(lines) + "\n"
    (out / "report.md").write_text(md)
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>MiniCPM A100 report</title>'
        '<style>body{max-width:1200px;margin:40px auto;font:15px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>MiniCPM A100 report</h1><p><a href="report.md">Markdown</a> · <a href="summary.json">JSON</a></p><pre>'
        + html.escape(md) + '</pre>')
    suite.set("failures", str(sum(r["status"] == "FAIL" for r in rows)))
    suite.set("skipped", str(sum(r["status"] not in ("PASS", "FAIL") for r in rows)))
    ET.ElementTree(suite).write(out / "junit.xml", encoding="utf-8", xml_declaration=True)


class Blocked(RuntimeError):
    """An independent prerequisite is unavailable; not a quality failure."""


class Runner:
    def __init__(self, args):
        self.args = args
        self.config = json.loads(Path(args.config).read_text())
        if args.gpu is not None:
            self.config["gpu"] = args.gpu
        self.config["mode"] = getattr(args, "mode", "bug-regression")
        if getattr(args, "model_path", None):
            self.config["model_path"] = str(Path(args.model_path).expanduser().resolve())
        if getattr(args, "dependency_policy", None):
            self.config["dependency_policy"] = args.dependency_policy
        self.out = Path(args.output or ROOT / ".minicpm-runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / ".lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.out / "summary.json").exists():
            self.lock.close()
            raise ValueError("Output already contains a run; choose a new directory (use --recover-report for interrupted runs).")
        self.deadline = time.monotonic() + self.config["budget_seconds"]
        self.env = os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES=str(self.config["gpu"]), PYTHONUNBUFFERED="1",
                        HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
        self.cache = Path(args.cache_dir).expanduser().resolve()
        self.env["HF_HOME"] = str(self.cache / "huggingface")
        self.env["PYTHONPATH"] = str(ROOT)
        self.python = sys.executable
        self.reference_python = None
        self.server = None
        self.sampler = None
        self.process_sampler = None
        self.current = None
        self.handles = []
        self.launch_args = []
        self.gpu_lock = None
        names = ["preflight", "environment", "resources", "unit", "review_probes", "default_startup",
                 "diagnostic_startup", "functional", "boundary", "asr", "mmmu", "tts", "reference_collect", "performance",
                 "soak", "profile", "chunked_prefill", "ab_text", "ab_no_graph", "aa_restart", "reference", "tts_score", "analysis", "cleanup"]
        names += ["reference_prepare", "reference_environment", "reference_preflight", "reference_export", "empty_output", "cancel_recovery",
                  "content_blocks", "alignment_baseline", "alignment_chunked", "alignment_compare"]
        names += [f"oom_c{c}" for c in self.config["concurrencies"]]
        self.state = {"status": "RUNNING", "started_at": time.time(), "config": self.config,
                      "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      "dirty": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                      "stages": [{"name": n, "status": "PENDING"} for n in names]}
        (self.out / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--", "sglang_omni", "pyproject.toml", "scripts/minicpm_a100"], cwd=ROOT))
        shutil.copytree(HERE, self.out / "runner-source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        save(self.out / "config.json", self.config)
        report(self.out, self.state)

    def spawn(self, command, log):
        handle = (self.out / log).open("ab")
        self.handles.append(handle)
        handle.write(("$ " + shlex.join(map(str, command)) + "\n").encode())
        handle.flush()
        return subprocess.Popen(list(map(str, command)), cwd=ROOT, env=self.env,
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)

    def command(self, command, log, timeout, *, allow_incomplete=False):
        remaining = min(timeout, self.deadline - time.monotonic() - 30)
        if remaining <= 0:
            raise TimeoutError("Global eight-hour budget exhausted")
        self.current = self.spawn(command, log)
        try:
            rc = self.current.wait(timeout=remaining)
            if rc == 2 and allow_incomplete:
                raise Blocked(f"Required verification evidence unavailable; see {log} and stage result JSON")
            if rc:
                raise RuntimeError(f"exit={rc}; see {log}")
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command exceeded {remaining:.0f}s; see {log}")
        finally:
            stop(self.current)
            self.current = None

    def stage(self, name, fn):
        row = next(r for r in self.state["stages"] if r["name"] == name)
        row.update(status="RUNNING", log=name + ".log")
        started = time.monotonic()
        report(self.out, self.state)
        print(f"[{name}] starting", flush=True)
        try:
            if self.deadline - time.monotonic() < 30 and name not in ("cleanup", "analysis"):
                raise TimeoutError("Global budget exhausted")
            fn()
            row["status"] = "PASS"
        except Exception as exc:
            row.update(status="BLOCKED" if isinstance(exc, Blocked) else "INCOMPLETE" if isinstance(exc, TimeoutError) else "FAIL", error=str(exc))
            with (self.out / row["log"]).open("a") as log:
                traceback.print_exc(file=log)
        finally:
            row["seconds"] = time.monotonic() - started
            report(self.out, self.state)
        print(f"[{name}] {row['status']}", flush=True)
        return row["status"] == "PASS"

    def skip(self, name, reason, optional=False):
        row = next(r for r in self.state["stages"] if r["name"] == name)
        row.update(status="N/A" if optional else "BLOCKED", error=reason)
        report(self.out, self.state)

    def preflight(self):
        if platform.system() != "Linux":
            raise RuntimeError("GPU run requires Linux; local tests use unittest without deployment")
        if not (3, 10) <= sys.version_info < (3, 13):
            raise RuntimeError("Bootstrap with Python 3.10–3.12")
        self.cache.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.cache).free < self.config["min_disk_gb"] * 1024**3:
            raise RuntimeError("Insufficient free disk for model, datasets and environment")
        self.command(["nvidia-smi", "-i", self.config["gpu"],
                      "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv,noheader,nounits"], "preflight.log", 20)
        line = (self.out / "preflight.log").read_text().splitlines()[-1]
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 6 or "A100" not in fields[1] or float(fields[2]) < 79000:
            raise RuntimeError("Selected device is not an A100 80GB")
        if float(fields[3]) > self.config["max_idle_memory_mib"] or float(fields[4]) > self.config["max_idle_util_percent"]:
            raise RuntimeError("GPU is occupied; no foreign process will be stopped")
        self.state["gpu_fingerprint"] = fields
        self.gpu_lock = (Path("/tmp") / ("sglang-omni-minicpm-" + fields[0] + ".lock")).open("a")
        fcntl.flock(self.gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.config["port"]))

    def environment(self):
        venv = self.out / "venv"
        self.command([sys.executable, "-m", "venv", str(venv)], "environment.log", 120)
        self.python = str(venv / "bin/python")
        policy = self.config.setdefault("dependency_policy", "strict")
        requirements = ["-e", str(ROOT) + ("[minicpm-o]" if policy == "strict" else ""), "py-spy", "aiohttp"]
        if policy == "minicpmo-override":
            requirements += self.config["minicpmo_runtime_dependencies"]
        self.command([self.python, "-m", "pip", "install", "--disable-pip-version-check", *requirements],
                     "environment.log", self.config["environment_timeout_seconds"])
        if policy == "minicpmo-override":
            self.command([self.python, "-m", "pip", "install", "--no-deps", "minicpmo-utils==1.0.6"], "environment.log", 300)
        self.command([self.python, "-m", "pip", "freeze"], "dependencies.txt", 120)
        self.command([self.python, HERE / "dependency_check.py", "--run-dir", self.out], "environment.log", 120)
        self.command([self.python, "-c", "import torch; print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(); x=torch.ones(16,16,device='cuda',dtype=torch.bfloat16); print((x@x).sum().item())"], "environment.log", 120)

    def worker(self, action, log=None, timeout=None):
        self.command([self.python, HERE / "workload.py", "--run-dir", self.out, "--action", action],
                     log or action + ".log", timeout or self.config["stage_timeout_seconds"], allow_incomplete=True)

    def resources(self):
        self.command([self.python, HERE / "prepare.py", "--run-dir", self.out], "resources.log", self.config["download_timeout_seconds"])
        manifest = json.loads((self.out / "resources.json").read_text())
        self.env["MINICPMO_CHECKPOINT"] = manifest["model_path"]
        self.env["HF_HUB_OFFLINE"] = "1"
        self.env["HF_DATASETS_OFFLINE"] = "1"

    def unit(self):
        path = self.out / "unit.xml"
        self.command([self.python, "-m", "pytest", "-q", "tests/unit_test/minicpm_o",
                      "--junitxml=" + str(path)], "unit.log", 900)
        root = ET.parse(path).getroot()
        skipped = root.findall(".//testcase/skipped")
        if skipped:
            raise RuntimeError(f"{len(skipped)} MiniCPM tests were skipped; required GPU/checkpoint coverage is missing; see unit.xml")

    def start_server(self, extra, name):
        self.stop_server()
        self.launch_args = list(extra)
        model = json.loads((self.out / "resources.json").read_text())["model_path"]
        cmd = [self.python, "-m", "sglang_omni.cli", "serve", "--model-path", model,
               "--host", "127.0.0.1", "--port", str(self.config["port"]), "--model-name", "minicpm-a100", *extra]
        self.server = self.spawn(cmd, name + ".log")
        save(self.out / "server.json", {"pid": self.server.pid, "command": list(map(str, cmd)), "configuration": name})
        until = min(self.deadline - 30, time.monotonic() + self.config["startup_timeout_seconds"])
        while time.monotonic() < until:
            if self.server.poll() is not None:
                raise RuntimeError(f"Server exited {self.server.returncode}; see {name}.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.config['port']}/health", timeout=2) as response:
                    if response.status == 200:
                        self.worker("warmup", name + ".log", 600)
                        return
            except (OSError, TimeoutError):
                pass
            time.sleep(1)
        raise TimeoutError(f"Server never ready; see {name}.log")

    def stop_server(self):
        stop(self.server)
        self.server = None

    def cleanup(self):
        stop(self.current)
        self.stop_server()
        stop(self.sampler)
        stop(self.process_sampler)
        self.current = self.sampler = None
        for handle in self.handles:
            handle.close()
        if self.gpu_lock is not None:
            self.gpu_lock.close()
            self.gpu_lock = None
        (self.out / "cleanup.log").write_text("Stopped owned process groups. Model/data cache retained.\n")

    def reference_environment(self):
        path = self.out / "reference-venv"
        self.command([sys.executable, "-m", "venv", path], "reference_environment.log", 120)
        self.reference_python = str(path / "bin/python")
        self.command([self.reference_python, "-m", "pip", "install", *self.config["reference_dependencies"]],
                     "reference_environment.log", self.config["environment_timeout_seconds"])
        self.command([self.reference_python, "-m", "pip", "check"], "reference_environment.log", 120)
        self.command([self.reference_python, "-m", "pip", "freeze"], "reference-dependencies.txt", 120)

    def reference_export(self, preflight=False):
        if not self.reference_python:
            raise Blocked("Reference environment unavailable")
        cmd = [self.reference_python, HERE / "reference_hf.py", "--run-dir", self.out]
        if preflight:
            cmd += ["--preflight"]
        self.command(cmd, "reference_preflight.log" if preflight else "reference_export.log", 1800)

    def health(self, label):
        if self.server is None or self.server.poll() is not None:
            raise RuntimeError("Server process unavailable")
        with urllib.request.urlopen(f"http://127.0.0.1:{self.config['port']}/health", timeout=10) as response:
            if response.status != 200:
                raise RuntimeError("Server health failed")
        self.worker("health", label + ".log", 90)

    def isolated(self, action, extra, *, alignment=False):
        if alignment:
            self.env["MINICPM_ALIGNMENT_DIR"] = str(self.out / action)
            self.env["PYTHONPATH"] = str(HERE / "alignment_hook") + os.pathsep + str(ROOT)
        try:
            try:
                self.start_server(extra, action)
                self.health(action)
            except Exception as exc:
                raise Blocked(f"Isolated server unavailable: {exc}") from exc
            try:
                self.worker(action, timeout=self.config["soak_seconds"] + 300 if action == "soak" else None)
            finally:
                # A request failure must not hide a dead stage or broken next request.
                self.health(action)
        finally:
            self.stop_server()
            self.env.pop("MINICPM_ALIGNMENT_DIR", None)
            self.env["PYTHONPATH"] = str(ROOT)

    def execute(self):
        try:
            for name, fn in [("preflight", self.preflight), ("environment", self.environment), ("resources", self.resources)]:
                if not self.stage(name, fn):
                    return
            ref_env = self.stage("reference_environment", self.reference_environment)
            ref_ready = ref_env and self.stage("reference_preflight", lambda: self.reference_export(True))
            if not ref_env:
                self.skip("reference_preflight", "Reference environment failed")
            self.stage("unit", self.unit)
            self.stage("review_probes", lambda: self.command([self.python, HERE / "review_probes.py", "--output", self.out / "results/review_probes.json"], "review_probes.log", 120))
            ready = self.stage("default_startup", lambda: self.start_server([], "default_startup"))
            if ready:
                self.skip("diagnostic_startup", "Default configuration started", optional=True)
            else:
                ready = self.stage("diagnostic_startup", lambda: self.start_server(self.config["diagnostic_engine_args"], "diagnostic_startup"))
            if not ready:
                return
            baseline = self.launch_args[:]
            self.stop_server()
            self.sampler = self.spawn(["nvidia-smi", "-i", self.config["gpu"], "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw", "--format=csv", "-l", "1"], "gpu.csv")
            self.process_sampler = self.spawn(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv", "-l", "1"], "gpu-processes.csv")
            for action in ("empty_output", "cancel_recovery", "content_blocks", "boundary"):
                self.stage(action, lambda a=action: self.isolated(a, baseline))
            for action, chunk in (("alignment_baseline", 0), ("alignment_chunked", 256)):
                self.stage(action, lambda a=action, c=chunk: self.isolated(a, baseline + ["--thinker.engine.chunked_prefill_size", str(c)], alignment=True))
            self.stage("alignment_compare", lambda: self.command([self.python, HERE / "alignment_compare.py", "--run-dir", self.out], "alignment_compare.log", 120))
            for concurrency in self.config["concurrencies"]:
                action = f"oom_c{concurrency}"
                self.stage(action, lambda a=action: self.isolated(a, baseline))
            # Quality/reference work is independent of failures above; each gets a fresh service.
            for action in ("functional", "asr", "mmmu", "tts", "reference_collect"):
                self.stage(action, lambda a=action: self.isolated(a, baseline))
            prepared = self.stage("reference_prepare", lambda: self.worker("reference_prepare"))
            if ref_ready and prepared:
                exported = self.stage("reference_export", self.reference_export)
                if exported:
                    self.stage("reference", lambda: self.worker("reference"))
                else:
                    self.skip("reference", "HF export failed; no valid baseline")
            else:
                self.skip("reference_export", "HF preflight or native preparation unavailable")
                self.skip("reference", "HF preflight or native preparation unavailable")
            self.stage("tts_score", lambda: self.worker("tts_score"))
            gates = {"unit", "review_probes", "empty_output", "cancel_recovery", "content_blocks", "boundary", "alignment_compare", "reference", "tts_score", "asr", "mmmu", "functional"}
            gates.update(f"oom_c{c}" for c in self.config["concurrencies"])
            regression_ok = all(r["status"] == "PASS" for r in self.state["stages"] if r["name"] in gates)
            if self.config["mode"] == "full" and regression_ok:
                for action in ("performance", "soak", "profile", "chunked_prefill", "ab_text", "ab_no_graph", "aa_restart"):
                    extra = baseline[:]
                    if action == "chunked_prefill":
                        extra += ["--thinker.engine.chunked_prefill_size", "256"]
                    elif action == "ab_no_graph":
                        extra += ["--thinker.engine.disable_cuda_graph", "true", "--talker.engine.disable_cuda_graph", "true"]
                    elif action == "ab_text":
                        extra = [x for pair in zip(extra[::2], extra[1::2]) if not pair[0].startswith("--talker") for x in pair] + ["--text-only"]
                    self.stage(action, lambda a=action, e=extra: self.isolated(a, e))
            else:
                for action in ("performance", "soak", "profile", "chunked_prefill", "ab_text", "ab_no_graph", "aa_restart"):
                    self.skip(action, "Not selected in bug-regression mode" if self.config["mode"] != "full" else "Required regression gates failed; full acceptance blocked", optional=self.config["mode"] != "full")
        finally:
            self.stage("cleanup", self.cleanup)
            def summarize():
                from analyze import analyze
                result = analyze(self.out)
                save(self.out / "results/analysis.json", result)
                if result["failed"]:
                    raise RuntimeError("Some diagnostic artifacts could not be parsed")
            self.stage("analysis", summarize)
            for row in self.state["stages"]:
                if row["status"] in ("RUNNING", "PENDING"):
                    row.update(status="BLOCKED", error="Prerequisite unavailable or budget exhausted")
            report(self.out, self.state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(HERE / "config.json"))
    p.add_argument("--output")
    p.add_argument("--cache-dir", default=str(ROOT / ".minicpm-cache"))
    p.add_argument("--gpu")
    p.add_argument("--mode", choices=["bug-regression", "full"], default="bug-regression")
    p.add_argument("--model-path", help="Existing complete MiniCPM model directory; skips model download")
    p.add_argument("--dependency-policy", choices=["strict", "minicpmo-override"])
    p.add_argument("--print-plan", action="store_true")
    p.add_argument("--recover-report", metavar="RUN_DIR")
    args = p.parse_args()
    if args.print_plan:
        print(Path(args.config).read_text())
        return 0
    if args.recover_report:
        out = Path(args.recover_report).resolve()
        with (out / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = json.loads((out / "summary.json").read_text())
            for row in state["stages"]:
                if row["status"] in ("RUNNING", "PENDING"):
                    row.update(status="INCOMPLETE", error="Interrupted; no prior PID is killed on recovery")
            report(out, state)
        return 2
    runner = Runner(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        runner.execute()
    except KeyboardInterrupt:
        pass
    print(f"Report: {runner.out / 'report.html'}")
    return 0 if runner.state["status"] == "PASS" else 1 if runner.state["status"] == "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())
