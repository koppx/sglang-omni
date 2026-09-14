"""CPU-only checks of failure handling and metric semantics. No model downloads."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import Runner, report, save, stop
from analyze import trace_summary, union_duration
from workload import Workload, metrics


class TestMetrics(unittest.TestCase):
    def test_empty_is_not_zero_error_or_zero_latency(self):
        result = metrics([{"ok": False, "id": "bad", "latency_s": 2}], 2)
        self.assertIsNone(result["latency_p50_s"])
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["requests_per_second"], 0)

    def test_quantiles_and_success_denominator(self):
        result = metrics([{"ok": True, "latency_s": 1, "completion_tokens": 20},
                          {"ok": True, "latency_s": 3, "completion_tokens": 40},
                          {"ok": False, "id": "failed", "latency_s": 10}], 10)
        self.assertEqual(result["latency_p50_s"], 2)
        self.assertEqual(result["requests_per_second"], .2)
        self.assertEqual(result["completion_tokens_per_second"], 6)

    def test_overlap_union_not_sum(self):
        self.assertEqual(union_duration([(0, 10), (5, 15), (7, 8), (20, 25)]), 20)

    def test_host_cuda_launch_excluded_from_gpu_busy(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "trace.json"
            save(p, {"traceEvents": [
                {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 0, "dur": 100},
                {"ph": "X", "cat": "kernel", "ts": 10, "dur": 10},
                {"ph": "X", "cat": "gpu_memcpy", "ts": 15, "dur": 10},
                {"ph": "X", "cat": "kernel", "ts": 30, "dur": 5}]})
            self.assertAlmostEqual(trace_summary(p)["busy_ratio"], .8)


class TestLifecycle(unittest.TestCase):
    def test_failed_stage_cannot_be_masked_by_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            state = {"stages": [{"name": "request", "status": "FAIL", "error": "bad <input>"},
                                {"name": "cleanup", "status": "PASS"}]}
            report(Path(temp), state)
            self.assertEqual(state["status"], "FAIL")
            self.assertIn("&lt;input&gt;", (Path(temp) / "report.html").read_text())
            self.assertIn('<failure', (Path(temp) / "junit.xml").read_text())

    def test_missing_stage_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            state = {"stages": [{"name": "gpu", "status": "PENDING"}]}
            report(Path(temp), state)
            self.assertEqual(state["status"], "INCOMPLETE")

    def test_timeout_terminates_child(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = Runner.__new__(Runner)
            runner.out, runner.env = Path(temp), {}
            runner.handles, runner.current = [], None
            runner.deadline = time.monotonic() + 60
            with self.assertRaises(TimeoutError):
                runner.command([sys.executable, "-c", "import time; time.sleep(30)"], "hang.log", .05)
            self.assertIsNone(runner.current)
            for handle in runner.handles:
                handle.close()

    def test_failed_prerequisite_still_reports_and_cleans(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(config=str(Path(__file__).with_name("config.json")), gpu=None,
                                   output=temp, cache_dir=temp)
            runner = Runner(args)
            runner.preflight = lambda: (_ for _ in ()).throw(RuntimeError("fixture failure"))
            runner.execute()
            state = json.loads((Path(temp) / "summary.json").read_text())
            self.assertEqual(state["status"], "FAIL")
            self.assertEqual(next(s for s in state["stages"] if s["name"] == "cleanup")["status"], "PASS")
            self.assertEqual(next(s for s in state["stages"] if s["name"] == "resources")["status"], "BLOCKED")
            runner.lock.close()

    def test_fresh_run_does_not_overwrite_prior_results(self):
        with tempfile.TemporaryDirectory() as temp:
            save(Path(temp) / "summary.json", {"status": "FAIL"})
            args = SimpleNamespace(config=str(Path(__file__).with_name("config.json")), gpu=None,
                                   output=temp, cache_dir=temp)
            with self.assertRaises(ValueError):
                Runner(args)
            self.assertEqual(json.loads((Path(temp) / "summary.json").read_text())["status"], "FAIL")


class TestHTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        save(self.out / "config.json", {"port": 12345, "request_timeout_seconds": 1, "seed": 1})
        save(self.out / "resources.json", {})
        self.worker = Workload(self.out, "fixture")

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def send(self, status, payload=None, text=None, **kwargs):
        import httpx
        original = httpx.AsyncClient
        def handler(request):
            return httpx.Response(status, json=payload) if text is None else httpx.Response(status, text=text)
        transport = httpx.MockTransport(handler)
        def client(**options):
            return original(transport=transport, **options)
        body = kwargs.pop("body", self.worker.chat("hello"))
        with patch("httpx.AsyncClient", client):
            return await self.worker.request(body, **kwargs)

    async def test_valid_response_has_evidence(self):
        rec = await self.send(200, {"choices": [{"message": {"content": "hello"}}], "usage": {"completion_tokens": 1}})
        self.assertTrue(rec["ok"])
        self.assertEqual(json.loads(self.worker.raw.read_text())["text"], "hello")

    async def test_empty_output_probe_does_not_require_waveform(self):
        rec = await self.send(200, {"choices": [{"message": {"content": ""}}]}, allow_empty=True)
        self.assertTrue(rec["ok"])

    async def test_circuit_open_does_not_send_request(self):
        self.worker.circuit_open = True
        rec = await self.send(200, {"choices": [{"message": {"content": "hello"}}]})
        self.assertTrue(rec["blocked"])
        self.assertNotIn("sent", rec)

    async def test_http_200_empty_output_is_failure(self):
        rec = await self.send(200, {"choices": [{"message": {"content": ""}}]})
        self.assertFalse(rec["ok"])
        self.assertIn("Empty", rec["error"])

    async def test_server_error_is_not_expected_validation(self):
        rec = await self.send(500, {"error": "crashed"}, expected_error=True)
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["http_status"], 500)

    async def test_client_error_is_expected_validation(self):
        rec = await self.send(400, {"error": "invalid"}, expected_error=True)
        self.assertTrue(rec["ok"])

    async def test_stream_must_finish(self):
        event = 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        rec = await self.send(200, text=event, body=self.worker.chat("hello", stream=True))
        self.assertFalse(rec["ok"])
        rec = await self.send(200, text=event + 'data: [DONE]\n\n', body=self.worker.chat("hello", stream=True))
        self.assertTrue(rec["ok"])
        self.assertIsNotNone(rec["ttft_s"])

    async def test_cancel_does_not_require_done(self):
        event = 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        rec = await self.send(200, text=event, body=self.worker.chat("hello", stream=True), cancel=True)
        self.assertTrue(rec["ok"])
        self.assertTrue(rec["cancelled"])

    async def test_speech_requires_audio_not_just_text(self):
        rec = await self.send(200, {"choices": [{"message": {"content": "hello"}}]}, body=self.worker.chat("hello", modalities=["text", "audio"]))
        self.assertFalse(rec["ok"])
        self.assertIn("No output audio", rec["error"])


if __name__ == "__main__":
    unittest.main()
