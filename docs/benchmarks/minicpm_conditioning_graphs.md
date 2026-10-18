# Exact-shape MiniCPM-o conditioning projection graphs

## Scope and conclusion

This opt-in optimization reduces Talker semantic-projector host launch overhead while preserving each request's GEMM shape. It is not a claim of faster raw hidden transfer, a general P1 throughput improvement, or a staged conditioning implementation. Default capture shapes are empty, so serving remains eager unless explicitly configured.

The user-expanded optimization scope includes Thinker finalization, transport/control-plane queueing, conditioning and Talker admission. Projection is one measured subcomponent. Formal service guardrails show c16 essentially unchanged and c32 about 2% lower throughput, so this feature is not recommended as a universal default.

## Design

Configure `talker.factory.conditioning_graph_rows`, for example `[8, 16, 17, 32, 64]`. Shapes are exact: no padding or cross-request batching changes numerical GEMM selection. Capture occurs once after model weights load. Unknown shapes, changed device/dtype, and gradient-enabled projector calls use eager execution. The inference-only request builder disables autograd, enabling graph replay without changing the projector's training fallback.

Graph static buffers share a CUDA graph pool. A host lock serializes enqueue and a reused completion event orders replay when the CUDA stream changes. Same-stream calls use natural stream ordering. Each result is cloned before the next replay can overwrite shared output. The clone and completion record occur within the protected enqueue sequence. Checkpoint keys remain unchanged; graph captures are not model parameters. Do not mutate or reload weights after capture without rebuilding the cache.

Memory is bounded by the explicitly configured shape list, not an automatic growing cache. Choose a small representative list before requesting all row counts. Experimental six-shape shared-pool allocation delta was about 104 MB including retained validation tensors; that is not a precise production reservation estimate. Memory/capture cost must be checked for the deployment. Two-GPU deployment and live weight reload are not accepted by these experiments.

## Evidence and corrections

Raw results and runners: `handoff-results/projection-graph-20261016/` and `handoff-results/projection-graph-20261017/`.

1. Frozen actual checkpoint projector: exact-shape graph plus owned-output clone reduced isolated host/device synchronized time from about 105 us to 35–43 us, matching eager bitwise. Shared-pool alternating-stream ownership test retained 30 outputs across later replays, all bitwise equal.
2. Cross-request batching was investigated, with 2–16 requests. It accelerated projection but changed bf16 numerical outputs by up to 0.0009765625, so it was not integrated.
3. Initial integrated mode=graph runs actually used eager because of typed configuration and grad-mode eligibility. Their timings are excluded from graph-benefit claims. Execution-proof route logs were added in isolated snapshots; only subsequent actual replay windows support the numbers below.
4. An instrumented acceptance attempt failed warmup due to missing `time` import. It remains a failed diagnostic artifact, excluded from statistics. A memory failure overlapped another session's service and cannot establish graph memory regression. GPU experiments were serialized with `/tmp/minicpm-handoff-gpu0.lock` after detecting that contention.

## Repeated actual-replay measurements

Same resident service, independent fixed warmup, 128 fixed measurement requests per window, eager/graph order E/G/G/E/E/G after an excluded warmup window. Every eligible graph window recorded 144 graph invocations, not merely a startup capture message. CUDA graphs for the AR engine remain enabled in both modes. Host timing is not isolated GPU compute time.

Projection host medians in microseconds:

| Concurrency | Eager windows | Graph windows |
|---:|---|---|
| 1 | 231.97 / 230.61 / 234.83 | 147.38 / 140.40 / 141.26 |
| 16 | 192.88 / 176.85 / 197.60 | 112.58 / 114.31 / 128.27 |
| 32 | 188.55 / 183.93 / 193.06 | 125.67 / 127.43 / 111.89 |

A separate full-builder diagnostic, including span parsing and request construction, found:

| Concurrency | Eager builder us | Graph builder us |
|---:|---|---|
| 16 | 4246.62 / 4228.32 / 3705.30 | 3660.69 / 3279.80 / 3358.45 |
| 32 | 3372.79 / 3130.99 / 3359.63 | 3011.96 / 2835.02 / 2775.78 |

High-concurrency builder pairs improve in all three directions, roughly 9–22% at c16 and 9–17% at c32. At c1 the full builder did not improve consistently. Diagnostic logging affects these measurements; they establish local work reduction, not a full handoff or E2E claim.

## Probe-free service guardrails

Actual safe implementation, ABBA service starts, four 256-request windows per service; first window discarded, three warmed windows retained. Same checkpoint, source dataset/order, seed 42, temperature 0.7, max_new_tokens 256 and independent 16-sample warmup. Six retained windows per mode/concurrency; no per-request mode-file reads or timing logs in these snapshots. Source enablement differs only in configured capture shapes.

| Concurrency | Eager median qps | Graph median qps | Eager median E2E p95 s | Graph median E2E p95 s |
|---:|---:|---:|---:|---:|
| 16 | 9.709 | 9.721 | 2.1430 | 2.1345 |
| 32 | 13.157 | 12.8945 | 3.2295 | 3.2900 |

c16 is essentially unchanged. c32 throughput is about 2% lower and p95 about 1.9% higher; retain this negative result. These are medians of window-level metrics, not pooled per-request quantiles or significance intervals. High-concurrency sampled audio lengths vary, so equal work is not assumed solely from QPS. The feature stays opt-in and its end-to-end benefit is deployment-dependent.

## Correctness and quality boundaries

The actual-replay c1 acceptance diagnostic recorded seven windows, 128 requests each plus independent warmup. Codec sequences matched the warmed eager window for all requests in every graph/eager window. Output token counts and audio durations matched. Earlier eager/eager PCM differences show waveform-level nondeterminism, so exact PCM equality is not claimed. A separate same-input P0/P1 quality study found identical Whisper-small transcripts over 32 long inputs, but that is not a graph-specific speaker-similarity or human quality gate.

Final related GPU/config/request regression: 76 passed. Added cases cover changing CUDA streams, 48 retained outputs, eager fallback, state-dict compatibility, gradient fallback, shape validation, typed config round-trip and inference request-builder graph eligibility. Existing packed alignment correction remains a separate commit. Relevant Python/style hooks passed previously; repository-wide Rust formatting remains blocked when cargo is unavailable.

## Reproduction

Run on an exclusively admitted CUDA GPU with a compatible serving environment and FFmpeg runtime. The retained experiment scripts carry checkpoint and dataset fingerprints and all raw failed/successful windows.

```bash
python -m pytest tests/unit_test/minicpm_o/test_sglang_talker.py tests/unit_test/minicpm_o/test_native_config.py tests/unit_test/minicpm_o/test_request_builders.py -q
python handoff-results/projection-graph-20261017/run_integrated_acceptance_v2.py
python handoff-results/projection-graph-20261017/run_final_c32.py
python handoff-results/projection-graph-20261017/run_final_c16.py
pre-commit run --all-files
```

Artifact runners use explicit local paths and are not production APIs. For normal serving set the typed factory field in a pipeline config; leave it empty to disable. GPU handoff remains independently controlled and opt-in. Projection graphs do not require P2 or alter the hidden transport protocol.

## Other investigated directions

Continuous hidden blocks and direct relay packing improved local work but not stable service throughput. Pool allocation fast path, multiple conditioning workers, token-ID metadata and compact reference-audio metadata failed repeat-consistency or guardrails and remain experiment snapshots only. Talker-first fanout reduced receive-ready handoff median in repeated diagnostics and preserved c1 codecs, but adds a text-branch scheduling trade-off and is not included here. P2 can overlap only stable hidden that exists before final commit; known-text TTS commonly provides one prefill block, limiting that window. No unsupported architecture-wide impossibility claim is made.
