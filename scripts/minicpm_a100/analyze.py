"""Evidence-only summaries: trace interval unions, CPU leaf counts and A/A noise."""
from __future__ import annotations
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import statistics


def union_duration(intervals):
    end = None
    total = 0.0
    for left, right in sorted(intervals):
        if right < left:
            continue
        if end is None or left >= end:
            total += right - left
        elif right > end:
            total += right - end
        end = max(end if end is not None else right, right)
    return total


def trace_summary(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as file:
        data = json.load(file)
    intervals, names, categories = [], Counter(), Counter()
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    for event in events:
        cat = str(event.get("cat", "")).lower()
        # CUDA API launch calls are CPU intervals and must not count as GPU work.
        if event.get("ph") != "X" or cat not in {"kernel", "gpu_memcpy", "gpu_memset", "graph_trace"}:
            continue
        start, duration = event.get("ts"), event.get("dur")
        if start is None or duration is None:
            continue
        intervals.append((float(start), float(start) + float(duration)))
        names[event.get("name", "unknown")] += float(duration)
        categories[cat] += 1
    if not intervals:
        return {"file": str(path), "available": False, "reason": "No recognized GPU activities; do not interpret as 0% busy"}
    window = max(b for _, b in intervals) - min(a for a, _ in intervals)
    return {"file": str(path), "available": True, "activity_window_us": window,
            "busy_union_us": union_duration(intervals), "busy_ratio": union_duration(intervals) / window if window else None,
            "activity_categories": dict(categories), "top_ops_summed_us": names.most_common(15),
            "scope": "This process trace only; summed operator durations may overlap. Missing CUDA graph activity can undercount hardware busy."}


def analyze(out):
    result = {"failed": 0, "cpu": [], "traces": [], "ab": [], "limitations": []}
    for path in sorted((out / "profiles").glob("cpu-*.raw")):
        leaves = Counter()
        for line in path.read_text().splitlines():
            try:
                stack, count = line.rsplit(" ", 1)
                leaves[stack.split(";")[-1]] += int(count)
            except (ValueError, IndexError):
                continue
        count = sum(leaves.values())
        result["cpu"].append({"file": str(path), "samples": count,
                              "top_leaf_samples": leaves.most_common(10),
                              "evidence": "sampling attribution, not proof of causality"})
    for path in sorted((out / "profiles").rglob("*.trace.json*")):
        try:
            result["traces"].append(trace_summary(path))
        except Exception as exc:
            result["traces"].append({"file": str(path), "available": False, "error": str(exc)})
            result["failed"] += 1
    gpu = out / "gpu.csv"
    if gpu.exists():
        rows = []
        for row in csv.reader(gpu.read_text().splitlines()):
            try:
                if len(row) == 5:
                    rows.append({"timestamp": row[0], "util": float(row[2].strip().split()[0]), "memory_mib": float(row[3].strip().split()[0])})
            except ValueError:
                continue
        if rows:
            result["gpu"] = {"samples": len(rows), "peak_memory_mib": max(r["memory_mib"] for r in rows),
                             "mean_util_percent": statistics.mean(r["util"] for r in rows),
                             "scope": "whole monitored window including idle/restarts; per-request logs provide workload boundaries"}
    aa_file = out / "results/aa_restart.json"
    aa = json.loads(aa_file.read_text()) if aa_file.exists() else {}
    baseline_file = out / "results/performance.json"
    baseline = json.loads(baseline_file.read_text()) if baseline_file.exists() else {}
    summary_file = out / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    cfg = summary.get("config", {})
    statuses = {stage["name"]: stage["status"] for stage in summary.get("stages", [])}

    def comparison_ready(name, data):
        if statuses.get(name) != "PASS" or data.get("failed") != 0 or data.get("incomplete"):
            return False
        repeats, requests = cfg.get("repeats"), cfg.get("performance_requests")
        if not isinstance(repeats, int) or repeats <= 0 or not isinstance(requests, int) or requests <= 0:
            return False
        # Match every intended measurement, including repeat identities. A partial
        # baseline must not become valid just because it contains a few text points.
        modalities = ("text", "audio_input", "speech_output") if name == "performance" else ("text",)
        concurrencies = cfg.get("concurrencies", []) if name == "performance" else [1]
        expected = {(shape, modality, c, repeat) for shape in ("short", "long")
                    for modality in modalities for c in concurrencies for repeat in range(repeats)}
        points = data.get("points", [])
        actual = {(p.get("shape"), p.get("modality"), p.get("concurrency"), p.get("repeat")) for p in points}
        return bool(expected) and actual == expected and len(points) == len(expected) and all(
            p.get("attempted") == requests and p.get("success") == requests and p.get("failed") == 0 for p in points)

    noise = {p["shape"]: abs(p["throughput_change_fraction"]) for p in aa.get("baseline_comparison", []) if p["throughput_change_fraction"] is not None}
    for name in ("ab_text", "ab_no_graph"):
        path = out / f"results/{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for compare in data.get("baseline_comparison", []):
            delta = compare["throughput_change_fraction"]
            floor = noise.get(compare["shape"])
            invalid = [label for label, source in (("performance", baseline), ("aa_restart", aa), (name, data))
                       if not comparison_ready(label, source)]
            result["ab"].append({"experiment": name, **compare, "measured_aa_restart_delta_abs": floor,
                "invalid_or_incomplete_runs": invalid,
                "interpretation": "inconclusive" if invalid or floor is None or delta is None or abs(delta) <= floor else "directional; one restart A/A is not a confidence interval",
                "evidence": "medium at best; shared-host drift and cross-process scheduling remain possible"})
    result["limitations"] = ["No automatic kernel-code changes or unbounded tuning.",
        "nvidia-smi utilization is not SM Active or Tensor Active; DCGM/nsys hardware counters were not collected.",
        "A100 execution, reference comparisons and traces are only validated when corresponding stage logs pass.",
        "No sound-quality MOS or verified speaker-identity metric is inferred from ASR consistency."]
    return result
