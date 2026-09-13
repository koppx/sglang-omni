---
name: sglang-omni-runtime-profile
description: Profile SGLang Omni model serving with reproducible workloads, request timelines, CPU samples, GPU traces, controlled A/B and correctness checks. Use for Omni runtime performance diagnosis or the unattended MiniCPM-o A100 acceptance suite; not for model training.
---

# SGLang Omni runtime profiling

Use the five-layer methodology in [references/METHODOLOGY.md](references/METHODOLOGY.md).
This is a snapshot from sgl-project/sglang-omni commit
`3060470a85c39dfd5afff9a16eabdbc85c179ea9`, path
`.claude/skills/model-profiling/METHODOLOGY.md`. Its tracking source is
[issue #1798](https://github.com/sgl-project/sglang-omni/issues/1798), not a PR.
Read the relevant methodology sections before choosing measurements. Record the
tested repository and checkpoint commits separately from this methodology version.

## Unattended MiniCPM-o A100 run

The executable suite lives in the user's sglang-omni checkout at
`scripts/minicpm_a100/run.py`; inspect its README and config before invoking it.
Resolve the checkout from the current workspace, never assume a host-specific path.
If the suite is missing, explain that dependency instead of claiming the skill alone
contains a model-serving implementation.

For an already authorized single-A100-80GB run, use Python 3.10–3.12:

```bash
python3 scripts/minicpm_a100/run.py --gpu 0
```

The config defines the 28,800-second total budget, sample counts, provisional quality
gates, retry limits and diagnostic memory caps. The runner creates an isolated venv,
downloads missing complete checkpoints and fixed dataset samples, checks startup,
runs tests, captures evidence, and produces Markdown/HTML/JSON/JUnit reports.
Its raw per-request logs and state survive normal errors. `--recover-report RUN_DIR`
marks interrupted stages incomplete after a host/process failure; it does not kill
old PIDs or pretend the interrupted test passed. Cached HF downloads can be reused.

Honor the user's approved batch scope: no second human pause is necessary between
preauthorized stages or A/B arms. For unattended runs, execute only bounded,
predeclared configuration experiments; newly discovered code optimizations become
recommendations, not unreviewed mutations. The reference upstream planning skill's
two-pause/background-Agent workflow is not required by this adapted workflow.
Do not auto-publish GitHub issues, messages, commits or results.

## Measurement invariants

- Fingerprint GPU UUID/model, driver, framework/dependency versions, dirty diff,
  checkpoint/dataset revisions, seed, sample IDs and command lines. Confirm the
  selected card is free in both memory and compute. Clean up owned processes only.
- Use identical warmup and workload shapes across arms. Separate profiler-enabled
  discovery from benchmark timing. Include short and long sequences, realistic
  concurrency and functional regression after changing graph/backend settings.
- For non-streaming MiniCPM audio, report complete-audio latency and end-to-end
  audio RTF. Text TTFT, server-stage milestones and audio TTFA are different metrics.
- Start with utilization and request-stage evidence. Capture `py-spy --idle
  --subprocesses` at bounded low rates for host-side gaps, and short native torch
  traces for device activity. Verify files actually contain the expected events;
  an HTTP 200 from a profile start endpoint does not prove every worker captured.
- GPU busy uses the union of device activity intervals, not summed kernel durations.
  Exclude host CUDA API launch intervals. Report missing graph/device activities,
  per-process trace scope and capture-window coverage; utilization is not SM/Tensor
  saturation. Consult the methodology for optional DCGM/nsys investigation.
- Use repeated A/A measurements to bound noise before claiming small A/B effects.
  A leaf-frame hotspot or a single restart speedup is evidence for a hypothesis,
  not proof. Grade findings strong/medium/weak using the methodology rubric.
- Reference parity can reproduce a bug present in both implementations. Pair it
  with contract/invariant tests, and distinguish inherited defects from port bugs.

## Verify and report

Read `summary.json`, failed stage logs, raw request records and captured artifacts.
Require all mandatory stages to finish for PASS. Distinguish FAIL, INCOMPLETE and
explicit N/A; missing tools or exhausted budget are never silently successful.
Default-startup failure remains failure even if a smaller diagnostic config works.
Provide the self-contained report location, known limitations, and actionable
findings with their evidence paths. Never fabricate A100 measurements on a CPU host.
