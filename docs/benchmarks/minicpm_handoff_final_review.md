# MiniCPM-o handoff: curated changes and final experiment

## Delivery scope

This branch contains the independently justified packed-tensor alignment correction (7874d6c88), an inference-only Talker request-builder correction, their tests, and experimental evidence. It does not ship experimental concurrent fan-out, projection CUDA graphs, direct-to-pool packing, hidden-block redesign, worker-count changes or metadata transformations. Those candidates remain local until independently accepted. No claim of overall performance acceptance is made.

The no-grad correction prevents an inference request from retaining an autograd graph through its conditioning embeddings. The regression starts with gradients enabled and differentiable hidden input, verifies exact conditioning values and requires no grad_fn on the returned embeddings. Disabling autograd is useful independently of optional CUDA graph execution. No isolated percentage speedup is attributed to this decorator.

## Baseline cost and measurement boundaries

H100 NVL, torch 2.13.0+cu130, checkpoint revision 503e754207c94da6bb26850b4469f367c9ea3582. Typical hidden payload is 112–128 KiB. Isolated sequential cross-process GPU-ready RTT medians for 128 KiB: CPU/SHM P0 0.500 ms, pooled P1 0.680 ms, direct IPC 0.840 ms. These include control-plane overhead, not just memcpy.

Device-ready diagnostic handoff medians (median of three run medians) were 16.89 ms at c16 and 15.53 ms at c32. This includes serialization, scheduling and receive completion. First-prefill latency additionally includes conditioning and scheduler admission. Timings with probes are not production E2E timings.

## Final bounded experiment: independent fan-out

Hypothesis: Stage.route_result awaits decode send before Talker send. Running independent branches concurrently may reduce audio handoff blocking without simply swapping branch priority.

The isolated candidate changed only the experimental snapshot's Thinker multi-successor dispatch to asyncio.gather(return_exceptions=True), waited for every branch before propagating an error, and kept existing projections. It was not applied to this branch's production runtime. CPU contract suite: 52 passed, 11 CUDA skips, including blocked-branch independence and waiting for the surviving branch after an error. Generic opt-in configuration, request cancellation cleanup and broader ownership acceptance were not completed, so this is not merge-ready runtime code.

Each row is 256 measured requests after 16 disjoint warmup requests, fixed input/seed/generation settings, three alternating paired repetitions per concurrency, detailed host trace on both arms. All twelve trials completed. P1 was pooled for these small hidden tensors. Raw measured rows are in minicpm_handoff_final_results.json.

| c | repeat | reference handoff median ms | fan-out median ms | reference handoff p95 ms | fan-out p95 ms | reference qps | fan-out qps | reference E2E p95 s | fan-out E2E p95 s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 16.275 | 13.282 | 23.227 | 19.150 | 8.380 | 8.989 | 2.692 | 2.500 |
| 16 | 2 | 16.347 | 13.891 | 22.965 | 20.515 | 8.681 | 8.863 | 2.552 | 2.536 |
| 16 | 3 | 16.494 | 14.143 | 23.547 | 20.059 | 8.516 | 8.649 | 2.530 | 2.698 |
| 32 | 1 | 20.528 | 14.831 | 34.997 | 29.549 | 10.412 | 11.076 | 4.848 | 4.404 |
| 32 | 2 | 20.949 | 16.415 | 36.447 | 34.821 | 10.176 | 10.516 | 4.868 | 4.955 |
| 32 | 3 | 22.067 | 17.130 | 37.919 | 37.796 | 10.422 | 10.107 | 4.972 | 4.902 |

Handoff medians and p95 improved in all paired repetitions. However decode receive medians increased from about 7.4–7.8 ms to 8.8–9.9 ms. E2E p95 regressed in one c16 and one c32 round; c32 throughput regressed in the third round. Audio durations varied at concurrency. Therefore concurrent fan-out was **not accepted for the product branch**. These are trace-enabled screening results, not uninstrumented proof of throughput gain.

## Prior candidates and what is retained

- Alignment: real checkpoint frozen GPU math and full 1060-sample odd/even duration comparison support the fix. Exact PCM determinism remains a separately documented limitation.
- Talker-first successor order: reduced audio handoff but delayed text decode and had inconsistent tail effects; not shipped.
- Exact-shape projection graph: frozen math can remain bit-exact; observed host projection reduction from roughly 178–235 us to 111–149 us. Absolute savings are tens of microseconds. Safety, memory, configuration and executed-path checks are necessary; graph code remains uncommitted locally, not part of this delivery.
- Some integrated graph windows actually executed eager because an empty typed factory setting overrode a function default. They are retained as invalid comparisons/A-A evidence, not counted as graph gains. Subsequent explicit-config/no-grad windows proved graph execution. This prevents attributing startup noise to a flag.
- Conditioning worker overlap, token-ID metadata, reference-audio byte encoding and allocator fast path did not demonstrate consistent guardrail-safe benefit; not shipped.
- Large-hidden isolated transport crossover: pooled P1 beat P0 from 512 KiB under the tested conditions, but expanded long-text c16 trials regressed. No universal size threshold or long-form quality claim follows.
- Batched projection saved local time but changed bf16 results when GEMM shapes changed; not implemented in the service.
- P2 pretransfer was investigated, not implemented: known-text hidden originates in a single prefill; a useful overlap window was not demonstrated. Free-generation staging needs completion coverage, COMMIT, epoch/cancellation ownership and backpressure.

## Validation and reproduction

Run in an environment matching the pinned repository dependencies:

```bash
python -m pytest tests/unit_test/minicpm_o/test_sglang_talker.py -q
python -m pytest tests/unit_test/minicpm_o/test_request_builders.py -q
python -m pytest tests/unit_test/pipeline/test_stage.py -q
pre-commit run --all-files
```

GPU cases must execute on CUDA; two-GPU topology was unavailable. Prior full GPU suite: 164 passed, two cross-GPU skips; prior CPU suite: 126 passed, 35 CUDA-dependent skips. Extended experimental graph/config suite: 44 passed before later local additions; it validates local graph work, not a graph feature in this commit. The isolated no-grad inference contract passed. Full pre-commit previously passed all available Python/project checks but Rust formatting was blocked by missing cargo. The exact curated source snapshot passed 81 CPU cases with 13 CUDA-dependent skips. Changed-file pre-commit passed. A fresh exact-source GPU regression was queued behind a different session's GPU lock and was still pending at publication; earlier GPU results must not be presented as a fresh run of this curated commit. Commit creation bypassed automatic Git hooks after explicit checks; no all-files Rust-format success is claimed.

Local artifacts are under handoff-results/final-fanout and earlier dated experiment directories. They retain source snapshots/patches, logs, speed_results.json, per-request outputs and WAVs. Only a compact numeric summary is published here; raw audio, credentials, machine-specific launch scripts and unaccepted implementations are intentionally not pushed. Re-running the final experimental fan-out requires reconstructing the described snapshot change; it is not a supported production CLI option.

## Conclusion

Ship the correctness and inference-contract improvements separately from speculative performance features. The experiments show millisecond-scale scheduling opportunities, but the final concurrent fan-out transfers some cost to text and has inconsistent service guardrails. This branch is evidence for review, not a declaration that the broader optimization goal has been achieved.
