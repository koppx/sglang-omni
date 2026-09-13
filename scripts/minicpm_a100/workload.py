#!/usr/bin/env python3
"""HTTP workloads, quality gates and bounded diagnostic captures."""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
import traceback

from run import save


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = math.floor(pos)
    return values[low] + (values[math.ceil(pos)] - values[low]) * (pos - low)


def metrics(records, elapsed):
    ok = [r for r in records if r["ok"]]
    lat = [r["latency_s"] for r in ok]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    rtfs = [r["e2e_audio_rtf"] for r in ok if r.get("e2e_audio_rtf") is not None]
    return {"attempted": len(records), "success": len(ok), "failed": len(records) - len(ok),
            "elapsed_s": elapsed, "requests_per_second": len(ok) / elapsed if elapsed > 0 else None,
            "completion_tokens_per_second": sum(r.get("completion_tokens", 0) for r in ok) / elapsed if elapsed > 0 else None,
            "latency_p50_s": percentile(lat, .5), "latency_p95_s": percentile(lat, .95), "latency_p99_s": percentile(lat, .99),
            "latency_max_s": max(lat) if lat else None, "text_ttft_p50_s": percentile(ttft, .5),
            "e2e_audio_rtf_p50": percentile(rtfs, .5), "e2e_audio_rtf_p95": percentile(rtfs, .95),
            "text_ttft_p95_s": percentile(ttft, .95), "failures": [{"id": r["id"], "error": r.get("error")} for r in records if not r["ok"]]}


def uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


class Workload:
    def __init__(self, out, action):
        self.out, self.action = out, action
        self.cfg = json.loads((out / "config.json").read_text())
        self.res = json.loads((out / "resources.json").read_text())
        self.base = f"http://127.0.0.1:{self.cfg['port']}"
        self.raw = out / "requests" / (action + ".jsonl")
        self.raw.parent.mkdir(exist_ok=True)
        (out / "results").mkdir(exist_ok=True)
        (out / "audio").mkdir(exist_ok=True)
        self.sequence = 0

    def chat(self, prompt, **extra):
        body = {"model": "minicpm-a100", "messages": [{"role": "user", "content": prompt}],
                "modalities": ["text"], "max_tokens": 256, "temperature": 0, "seed": self.cfg["seed"]}
        body.update(extra)
        return body

    async def request(self, body, *, sample_id=None, audio=None, lang="en", expected_error=False, contains=None, cancel=False):
        import httpx
        self.sequence += 1
        ident = f"{self.action}-{self.sequence:06d}"
        rec = {"id": ident, "sample_id": sample_id, "started_at": time.time(), "request": body,
               "input_audio": audio, "lang": lang, "ok": False}
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.cfg["request_timeout_seconds"], trust_env=False) as client:
                if audio:
                    response = await client.post(self.base + "/v1/audio/transcriptions", data={"model": "minicpm-a100", "language": lang},
                                                 files={"file": ("sample.wav", Path(audio).read_bytes(), "audio/wav")})
                elif body.get("stream"):
                    text, ttft, done, chunks = "", None, False, []
                    async with client.stream("POST", self.base + "/v1/chat/completions", json=body) as stream:
                        stream.raise_for_status()
                        async for line in stream.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                done = True
                                break
                            chunk = json.loads(data)
                            chunks.append(chunk)
                            if chunk.get("error"):
                                raise RuntimeError(str(chunk["error"]))
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {}).get("content") or ""
                                if delta and ttft is None:
                                    ttft = time.perf_counter() - start
                                text += delta
                            if cancel and ttft is not None:
                                break
                    if not cancel and not done:
                        raise RuntimeError("SSE closed without [DONE]")
                    rec.update(response={"chunks": chunks}, text=text, ttft_s=ttft, cancelled=cancel)
                    if not text:
                        raise RuntimeError("No streaming text received")
                    rec["ok"] = True
                    return rec
                else:
                    response = await client.post(self.base + "/v1/chat/completions", json=body)
                rec["http_status"] = response.status_code
                rec["response"] = response.json()
                if expected_error:
                    # An invalid input must produce a client error, not HTTP 500.
                    if not 400 <= response.status_code < 500:
                        raise RuntimeError(f"Expected 4xx validation error, got {response.status_code}")
                    rec["ok"] = True
                    return rec
                response.raise_for_status()
                data = rec["response"]
                message = data.get("choices", [{}])[0].get("message", {})
                text = data.get("text", message.get("content")) or ""
                rec.update(text=text, completion_tokens=data.get("usage", {}).get("completion_tokens", 0))
                if not text.strip():
                    raise RuntimeError("Empty text response")
                if contains and contains.lower() not in text.lower():
                    raise RuntimeError(f"Expected {contains!r} in response")
                if "audio" in body.get("modalities", []):
                    payload = message.get("audio", {}).get("data")
                    if not payload:
                        raise RuntimeError("No output audio")
                    import numpy as np
                    import soundfile as sf
                    raw = base64.b64decode(payload, validate=True)
                    target = self.out / "audio" / (ident + ".wav")
                    target.write_bytes(raw)
                    wave, sr = sf.read(io.BytesIO(raw), dtype="float32")
                    if not wave.size or not np.isfinite(wave).all() or sr != 24000:
                        raise RuntimeError(f"Invalid waveform or sample rate: {sr}")
                    rms = float(np.sqrt(np.mean(wave ** 2)))
                    clip = float(np.mean(np.abs(wave) >= .999))
                    rec.update(audio=str(target), audio_duration_s=len(wave) / sr, audio_rms=rms, clipping_fraction=clip)
                    if rms < 1e-5 or clip > .05:
                        raise RuntimeError("Silent or heavily clipped audio")
                rec["ok"] = True
        except Exception as exc:
            rec.update(error=str(exc), traceback=traceback.format_exc())
        finally:
            rec["latency_s"] = time.perf_counter() - start
            if rec.get("audio_duration_s"):
                rec["e2e_audio_rtf"] = rec["latency_s"] / rec["audio_duration_s"]
            with self.raw.open("a") as file:
                file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    async def batch(self, cases, concurrency):
        sem = asyncio.Semaphore(concurrency)
        async def one(case):
            async with sem:
                return await self.request(**case)
        start = time.perf_counter()
        results = await asyncio.gather(*(one(case) for case in cases))
        result = metrics(results, time.perf_counter() - start)
        result.update(started_at=min((r["started_at"] for r in results), default=None), ended_at=time.time())
        return results, result

    def smoke_cases(self):
        red, blue = self.out / "media/red.png", self.out / "media/blue.png"
        cases = []
        for i in range(10):
            cases.append({"body": self.chat(f"Reply with exactly MARKER{i}.", max_tokens=32), "contains": f"MARKER{i}"})
        for color, path in [("red", red), ("blue", blue)]:
            for _ in range(5):
                cases.append({"body": self.chat("Name the background color in one word.", images=[uri(path, "image/png")]), "contains": color})
        for sample in self.res["samples"]["librispeech_clean"][:5]:
            cases.append({"body": {}, "audio": sample["audio"], "sample_id": sample["id"]})
            cases.append({"body": self.chat("Transcribe this audio.", audios=[uri(sample["audio"], "audio/wav")]), "sample_id": sample["id"]})
        for i in range(5):
            cases.append({"body": self.chat(f"Reply with STREAM{i}.", stream=True, max_tokens=32)})
            cases.append({"body": self.chat("Say hello in a short sentence.", modalities=["text", "audio"], audio={"format": "wav"})})
        return cases

    async def warmup(self):
        # Cold start, then identical warmup before each measured concurrency point.
        cases = [{"body": self.chat("Say hello.", max_tokens=16)} for _ in range(2)]
        server = json.loads((self.out / "server.json").read_text())
        if "--text-only" not in server["command"]:
            cases += [{"body": self.chat("Say hello.", modalities=["text", "audio"], audio={"format": "wav"})}]
        _, result = await self.batch(cases, 1)
        return result

    async def functional(self):
        _, result = await self.batch(self.smoke_cases(), 1)
        # Same question with media in the original turn; tests standard content blocks.
        body = self.chat("What color was the first image?", messages=[
            {"role": "user", "content": [{"type": "text", "text": "Remember this image."},
                 {"type": "image_url", "image_url": {"url": uri(self.out / "media/red.png", "image/png")}}]},
            {"role": "assistant", "content": "I will remember it."},
            {"role": "user", "content": "What color was that image?"}])
        probe = await self.request(body, contains="red")
        result["multiturn_content_blocks_ok"] = probe["ok"]
        result["failed"] += not probe["ok"]
        return result

    async def boundary(self):
        cases = []
        for _ in range(3):
            cases += [
                {"body": self.chat("Hello", max_tokens=-1), "expected_error": True},
                {"body": self.chat("Hello", messages=[]), "expected_error": True},
                {"body": self.chat("Describe", images=["data:image/png;base64,bm90LWFuLWltYWdl"]), "expected_error": True},
                {"body": self.chat("Transcribe", audios=["data:audio/wav;base64,bm90LWF1ZGlv"]), "expected_error": True},
                {"body": self.chat("Speak", modalities=["text", "audio"], audio={"format": "wav", "ref_audio": "data:audio/wav;base64,bm90LWF1ZGlv"}), "expected_error": True},
                {"body": self.chat("Count from one to one thousand.", max_tokens=1024, stream=True), "cancel": True},
                {"body": self.chat("Reply HEALTHY.", max_tokens=16), "contains": "HEALTHY"},
            ]
        for seconds in (0.01, 1, 29, 31, 60):
            cases.append({"body": {}, "audio": str(self.out / f"media/boundary-{seconds}.wav")})
        cases += [{"body": self.chat("Summarize briefly: " + "the blue sky and green grass. " * n, max_tokens=32)} for n in (100, 600, 900)]
        cases += [{"body": self.chat("Say hello.", max_tokens=1, modalities=["text", "audio"], audio={"format": "wav"})}]
        _, result = await self.batch(cases, 1)
        return result

    async def chunked_prefill(self):
        cases = []
        for count in (100, 300, 600):
            prompt = "Read the final sentence only. Context: " + "Rain fills the river. " * count + "\nFinal sentence: Hello, this is a speech test."
            cases += [{"body": self.chat(prompt, max_tokens=128, modalities=["text", "audio"], audio={"format": "wav"})} for _ in range(4)]
        _, result = await self.batch(cases, 4)
        result["note"] = "Thinker chunk size explicitly 256; probes validate hidden alignment, this workload validates long-prompt speech completion"
        return result

    async def asr(self):
        from benchmarks.tasks.asr import normalize_text
        import jiwer
        result = {"failed": 0, "datasets": {}}
        for key in ("librispeech_clean", "librispeech_other", "fleurs_zh"):
            samples = self.res["samples"][key]
            records, speed = await self.batch([{"body": {}, "audio": s["audio"], "sample_id": s["id"], "lang": s["lang"]} for s in samples], 1)
            refs = [normalize_text(s["target"], s["lang"]) for s in samples]
            hyps = [normalize_text(r.get("text", "") if r["ok"] else "", s["lang"]) for r, s in zip(records, samples)]
            error = jiwer.wer(refs, hyps)  # Chinese normalizer emits space-separated characters.
            metric = "cer" if key == "fleurs_zh" else "wer"
            threshold = self.cfg["quality_thresholds"][key + "_" + metric]
            speed.update(**{metric: error}, threshold=threshold, failed_requests_counted_as_deletions=True)
            result["datasets"][key] = speed
            result["failed"] += speed["failed"] + (error > threshold)
        return result

    async def mmmu(self):
        samples = self.res["samples"]["mmmu"]
        records, result = await self.batch([{"body": self.chat(s["prompt"], images=[uri(p, "image/png") for p in s["images"]], max_tokens=1024), "sample_id": s["id"]} for s in samples], 1)
        correct, parsed = 0, 0
        for s, r in zip(samples, records):
            match = re.findall(r"Answer\s*:\s*\*{0,2}([A-Z])\b", r.get("text", ""), re.I)
            parsed += bool(match)
            correct += bool(r["ok"] and match and match[-1].upper() == s["target"].strip().upper())
        accuracy = correct / len(samples)
        result.update(accuracy=accuracy, parsed=parsed, scoring="strict final Answer: LETTER; no random fallback; multiple-choice subset")
        result["failed"] += accuracy < self.cfg["quality_thresholds"]["mmmu_accuracy_min"]
        return result

    async def reference_collect(self):
        count = self.cfg["reference_per_modality"]
        cases = [{"body": self.chat(f"Reply with exactly REFERENCE{i}.", max_tokens=32)} for i in range(count)]
        cases += [{"body": self.chat(s["prompt"], images=[uri(p, "image/png") for p in s["images"]], max_tokens=256), "sample_id": s["id"]} for s in self.res["samples"]["mmmu"][:count]]
        cases += [{"body": self.chat("Transcribe this audio.", audios=[uri(s["audio"], "audio/wav")], max_tokens=256), "sample_id": s["id"]} for s in self.res["samples"]["librispeech_clean"][:count]]
        _, result = await self.batch(cases, 1)
        return result

    async def tts(self):
        cases = []
        en, zh = self.res["samples"]["librispeech_clean"], self.res["samples"]["fleurs_zh"]
        refs = [en[0]["audio"], en[1]["audio"], None]
        for i in range(self.cfg["samples_tts"]):
            sample = (en if i % 2 == 0 else zh)[i // 2]
            audio = {"format": "wav"}
            if refs[i % 3]:
                audio["ref_audio"] = uri(refs[i % 3], "audio/wav")
            prompt = ("Read the following text verbatim: " if sample["lang"] == "en" else "请逐字朗读以下文字：") + sample["target"]
            cases.append({"body": self.chat(prompt, modalities=["text", "audio"], audio=audio), "sample_id": sample["id"], "lang": sample["lang"]})
        _, result = await self.batch(cases, 1)
        result["reference_schedule"] = "A, B, default (repeated); waveform checks here, independent ASR after serving stops"
        return result

    async def performance(self, quick=False):
        result = {"failed": 0, "points": []}
        concs = [1] if quick else self.cfg["concurrencies"]
        for shape in ("short", "long"):
            for c in concs:
                body = self.chat("Summarize in one sentence: " + "Rain fills the river. " * (5 if shape == "short" else 500), max_tokens=64)
                for repeat in range(self.cfg["repeats"]):
                    _, warmup = await self.batch([{"body": body}] * (2 * c), c)
                    result["failed"] += warmup["failed"]
                    records, point = await self.batch([{"body": body}] * self.cfg["performance_requests"], c)
                    point.update(shape=shape, modality="text", concurrency=c, repeat=repeat, output_texts=[r.get("text") for r in records])
                    result["points"].append(point)
                    result["failed"] += point["failed"]
                    save(self.out / f"results/{self.action}.partial.json", result)
        if not quick:
            for modality in ("audio_input", "speech_output"):
                for shape in ("short", "long"):
                    if modality == "audio_input":
                        audio = self.out / ("media/boundary-1.wav" if shape == "short" else "media/boundary-31.wav")
                        body = self.chat("Transcribe this audio.", audios=[uri(audio, "audio/wav")])
                    else:
                        body = self.chat("Read verbatim: " + "Rain fills the river. " * (1 if shape == "short" else 12),
                                         modalities=["text", "audio"], audio={"format": "wav"})
                    for c in concs:
                        for repeat in range(self.cfg["repeats"]):
                            _, warm = await self.batch([{"body": body}] * (2 * c), c)
                            result["failed"] += warm["failed"]
                            _, point = await self.batch([{"body": body}] * self.cfg["performance_requests"], c)
                            point.update(modality=modality, shape=shape, concurrency=c, repeat=repeat)
                            result["points"].append(point)
                            result["failed"] += point["failed"]
                            save(self.out / f"results/{self.action}.partial.json", result)
        if self.action != "performance":
            baseline_file = self.out / "results/performance.json"
            if baseline_file.exists():
                base = json.loads(baseline_file.read_text())
                compare = []
                for shape in ("short", "long"):
                    a = [p for p in base["points"] if p["concurrency"] == 1 and p["shape"] == shape and p.get("modality") == "text"]
                    b = [p for p in result["points"] if p["shape"] == shape]
                    if a and b:
                        av = statistics.median(p["requests_per_second"] for p in a)
                        bv = statistics.median(p["requests_per_second"] for p in b)
                        same = {t for p in a for t in p["output_texts"]} == {t for p in b for t in p["output_texts"]}
                        compare.append({"shape": shape, "throughput_change_fraction": bv / av - 1 if av else None,
                                        "greedy_output_set_equal": same, "evidence": "medium; inspect aa_restart before claiming a small speedup"})
                        result["failed"] += not same
                result["baseline_comparison"] = compare
        return result

    async def soak(self):
        until = time.monotonic() + self.cfg["soak_seconds"]
        result = {"failed": 0, "batches": 0, "requests": 0, "quiescent_memory": []}
        cases = self.smoke_cases()
        rng = random.Random(self.cfg["seed"])
        next_sample = 0
        while time.monotonic() < until:
            rng.shuffle(cases)
            _, batch = await self.batch(cases[:8], 4)
            result["failed"] += batch["failed"]
            result["requests"] += batch["attempted"]
            result["batches"] += 1
            if time.monotonic() >= next_sample:
                raw = subprocess.check_output(["nvidia-smi", "-i", str(self.cfg["gpu"]),
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=10, text=True)
                result["quiescent_memory"].append({"timestamp": time.time(), "memory_mib": float(raw.strip())})
                next_sample = time.monotonic() + 60
            # Persist progress even if a later request hangs or the process is killed.
            save(self.out / "results/soak.partial.json", result)
        values = result["quiescent_memory"]
        if len(values) >= 8:
            width = max(2, len(values) // 4)
            growth = statistics.median(v["memory_mib"] for v in values[-width:]) - statistics.median(v["memory_mib"] for v in values[:width])
            result["memory_growth_mib"] = growth
            result["memory_growth_guard_mib"] = self.cfg.get("soak_memory_growth_guard_mib", 512)
            result["memory_guard_passed"] = growth <= result["memory_growth_guard_mib"]
            result["failed"] += not result["memory_guard_passed"]
        else:
            result["memory_guard_passed"] = None
            result["memory_guard_note"] = "Too few samples for a memory trend"
        result["memory_interpretation"] = "Between completed batches; caches/allocator/external contention can cause growth. A failed guard is a leak investigation trigger, not a proven leak."
        return result

    async def profile(self):
        import httpx
        result = {"failed": 0, "captures": [], "evidence": "diagnostic only; profiling overhead excluded from performance"}
        profile_dir = self.out / "profiles"
        profile_dir.mkdir(exist_ok=True)
        pid = json.loads((self.out / "server.json").read_text())["pid"]
        pyspy = Path(sys.executable).parent / "py-spy"
        for repeat in range(3):
            target = profile_dir / f"cpu-{repeat}.raw"
            with (profile_dir / f"cpu-{repeat}.log").open("w") as log:
                command = [str(pyspy), "record", "--idle", "--subprocesses", "--pid", str(pid), "--duration", "20", "--rate", "5", "--format", "raw", "--output", str(target)]
                log.write(" ".join(command) + "\n")
                log.flush()
                process = subprocess.Popen(command, stdout=log, stderr=log)
                start = time.monotonic()
                try:
                    while time.monotonic() - start < 21:
                        _, m = await self.batch(self.smoke_cases()[:8], 4)
                        result["failed"] += m["failed"]
                    rc = await asyncio.to_thread(process.wait, timeout=10)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        await asyncio.to_thread(process.wait, timeout=10)
                result["captures"].append({"file": str(target), "exit": rc})
                result["failed"] += rc != 0 or not target.exists() or target.stat().st_size == 0
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            for torch_enabled in (False, True):
                run_id = f"{self.out.name}-{'torch' if torch_enabled else 'events'}"
                payload = {"run_id": run_id, "event_dir": str(profile_dir / run_id / "events"),
                           "enable_torch": torch_enabled, "trace_path_template": str(profile_dir / run_id / "{stage}")}
                started = False
                try:
                    response = await client.post(self.base + "/start_profile", json=payload)
                    response.raise_for_status()
                    started = True
                    _, m = await asyncio.wait_for(self.batch(self.smoke_cases()[10:13] + self.smoke_cases()[-2:], 1), timeout=60)
                    result["failed"] += m["failed"]
                finally:
                    if started:
                        response = await client.post(self.base + "/stop_profile", json={"run_id": run_id})
                        response.raise_for_status()
                # Stop is an async broadcast; wait for nonempty artifacts to settle.
                previous, stable = None, 0
                until = time.monotonic() + 90
                while time.monotonic() < until:
                    artifacts = [p for p in (profile_dir / run_id).rglob("*") if p.is_file()]
                    snapshot = [(str(p), p.stat().st_size) for p in sorted(artifacts)]
                    events_ok = any(p.suffix == ".jsonl" and p.stat().st_size > 0 for p in artifacts)
                    trace_ok = not torch_enabled or any(str(p).endswith(".trace.json.gz") and p.stat().st_size > 0 for p in artifacts)
                    stable = stable + 1 if snapshot == previous and events_ok and trace_ok else 0
                    previous = snapshot
                    if stable >= 3:
                        break
                    await asyncio.sleep(1)
                files = [str(p) for p in artifacts]
                result["captures"].append({"run_id": run_id, "files": files})
                if not events_ok or not trace_ok:
                    result["failed"] += 1
                try:
                    from sglang_omni.profiler.views import build_report
                    save(profile_dir / f"{run_id}-stages.json", build_report(str(profile_dir / run_id / "events")))
                except Exception:
                    result["failed"] += 1
                    result["event_report_error"] = traceback.format_exc()
        return result


async def execute(out, action):
    w = Workload(out, action)
    start = time.perf_counter()
    try:
        if action in ("reference", "tts_score"):
            from reference import run_reference, score_tts
            result = await run_reference(w) if action == "reference" else score_tts(w)
        elif action.startswith("ab_") or action == "aa_restart":
            result = await w.performance(quick=True)
            cases = w.smoke_cases()[:30] if action == "ab_text" else w.smoke_cases()
            # Include a structurally longer input after each graph/deployment change.
            cases += [{"body": {}, "audio": str(w.out / "media/boundary-31.wav")}]
            _, regression = await w.batch(cases, 1)
            result["functional_regression"] = regression
            result["failed"] += regression["failed"]
        else:
            result = await getattr(w, action)()
    except Exception:
        result = {"failed": 1, "error": traceback.format_exc()}
    result["seconds"] = time.perf_counter() - start
    result["request_log"] = str(w.raw)
    save(out / "results" / (action + ".json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--action", required=True)
    args = p.parse_args()
    sys.exit(asyncio.run(execute(args.run_dir, args.action)))
