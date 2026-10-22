# Drama 3 reference encoding: formal factory gate

**Decision (2026-10-22): stop performance acceptance.** The opt-in two-stream factory improves cold-reference output throughput in both visit pairings, but regresses hot throughput and E2E p95, with a reverse-pair hot continuity regression. Do not proceed to full-1,088-prompt generation, quality scoring, retries or tuning on this evidence. The implementation is submitted for draft review at the author's request, not for merge or production deployment; `reference_encode_cuda_streams` remains default `0`, and there is no accepted production performance or quality claim.

## Protocol and provenance

The formal artifact label is `drama3-production-smallab-prepare/prepared-v2`. Read `plan.json`, all eight measured `<visit>/{cold,hot}/verification.json` files, four discarded-warmup verifications, all four `<visit>/cleanup.json` files, `status.json` and the final `formal-summary.json`. The [compact evidence JSON](drama3_reference_encode_formal_factory_gate.json) retains every phase summary, finish reason, cap ID, nonzero underrun ID, source pin and sampled resource peak without machine-specific paths. These are locally retained artifacts, not repository-distributed raw traces or audio.

- Four fresh servers in order **A1/B1/B2/A2**; A uses formal option `--preprocessing.factory.reference_encode_cuda_streams 0`, B uses `2`, with no monkeypatch.
- Both arms use the same two H100 NVL GPUs: preprocessing on GPU 1, AR and vocoder on GPU 0. Client concurrency, engine capacity and decode graph maximum batch are all **16**. Server affinity is `1-12`, client affinity `13-15,24`, with `OMP_NUM_THREADS=4`.
- Each visit discards 32 warmup requests, then measures 64 selected references cold and the same 64 hot. There are 64 distinct measured references and 20 distinct warmup references. Cold means first selected-reference use within a visit, not a new process per request.
- Streaming generation uses temperature `1.0`, top-p `0.9`, maximum `500` new tokens, no fixed seed or request top-k. Output lengths differ; completion is not quality or normal-termination proof.
- Model: `fishaudio/drama-3@b2b5ee998034b2975720906c815e7fb0e9f007bd`; codec: `fishaudio/s2-pro@1de9996b6be38b745688de084d87a5633f714e4e`; source English corpus: `zhaochenyang20/seed-tts-eval-arrow@27f4c1adee83b5b29b7c4b375f6b976324bda308`. Software: PyTorch `2.13.0+cu130`, CUDA `13.0`, SGLang `0.5.21`.

Saved command records invoke `python -m sglang_omni.cli serve` with `--tts_engine.gpu 0 --vocoder.gpu 0 --preprocessing.gpu 1 --tts_engine.factory.max_running_requests 16 --tts_engine.engine.cuda_graph_max_bs 16` and the arm-specific factory option above. Each client invokes `python -m benchmarks.eval.benchmark_tts_seedtts --use-existing-server --generate-only --concurrency 16 --max-running-requests 16 --cuda-graph-max-bs 16 --warmup 0 --repeats 1 --stream --collect-stream-outcomes --temperature 1.0 --top-p 0.9 --max-new-tokens 500`, using `warmup.lst` for 32 discarded samples and `selected.lst` for each 64-sample measured cell. Model/codec snapshots, server URL and output directories are allocation-specific inputs, not hardcoded reproduction paths.

## Formal service results

Output throughput is **frames/s**, not aggregate residual-codebook codes/s. C50/C100 are percentages of requests whose maximum playback underrun is within 50/100 ms. Quantiles below are individual visit statistics, not pooled quantiles or confidence intervals.

| Visit | Regime | Completed / errors / caps | Frames/s | E2E p95 (s) | TTFA p95 (s) | C50 (%) | C100 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | cold | 64 / 0 / 0 | 322.3 | 4.507 | 3.5378 | 100 | 100 |
| A1 | hot | 64 / 0 / 0 | 625.1 | 3.338 | 0.6879 | 100 | 100 |
| B1 | cold | 64 / 0 / 0 | 483.7 | 3.664 | 1.9125 | 100 | 100 |
| B1 | hot | 64 / 0 / 0 | 579.6 | 3.450 | 0.6523 | 98.44 | 100 |
| B2 | cold | 64 / 0 / 0 | 476.9 | 3.853 | 1.9898 | 98.44 | 100 |
| B2 | hot | 64 / 0 / 1 | 531.4 | 3.558 | 0.6689 | 98.44 | 98.44 |
| A2 | cold | 64 / 0 / 0 | 372.6 | 4.067 | 2.9693 | 100 | 100 |
| A2 | hot | 64 / 0 / 0 | 668.9 | 3.008 | 0.5269 | 100 | 100 |

Cold frames/s rises **322.3→483.7 (+50.08%)**, with reverse pairing **372.6→476.9 (+27.99%)**. Hot frames/s falls **625.1→579.6 (−7.28%)** and **668.9→531.4 (−20.56%)**; hot E2E p95 rises **3.338→3.450 s** and **3.008→3.558 s**. Reverse-pair hot TTFA p95 also rises 26.95%. These observed hot and continuity regressions block escalation; the cold gains do not establish an overall pass, and their cause is not isolated by this campaign.

B2 hot retains capped ID `common_voice_en_17369337-common_voice_en_17369336` (`length`: 1, `stop`: 63); every other measured cell has 64 `stop` outcomes. Its 196.8 ms underrun belongs to `common_voice_en_17444666-common_voice_en_17444667`, a different ID. All eight nonzero underrun records remain in the JSON, including baseline A1 hot; none is filtered from the summaries.

All **640 requests completed: 512 measured plus 128 discarded warmups**, with zero recorded errors. All four servers exited `0`, without forced shutdown; process groups, descendants and resource samplers were gone. The saved final GPU state has no compute processes, 0 MiB used and 0% utilization on both GPUs. Sampled GPU 1 peaks were A1/B1/B2/A2 = 4,015/4,307/4,183/3,873 MiB; these sampled service measurements are separate from the allocator gate below and are not continuous peak guarantees.

## Correctness, memory and lifecycle gates are narrower

Supporting artifact labels are under `drama3-profile-444026`; they do not replace the formal service comparison.

| Gate | Evidence and boundary |
|---|---|
| Real reference cohort | `two-stream-cohort128-drama-validation/{A,B}.json`: 128 real references, 384 calls per arm; cold and repeat outputs exact across all ten codebooks, including cross-arm comparison. Executors drained. This is finite reference-code equivalence, not generated-audio quality. |
| Serial duration/boundary | `reference-long-budget-validation`: 1/10/30/60-second inputs and 10-second boundary neighbors (440,559/441,441 samples at 44.1 kHz) completed; repeat comparisons across all ten codebooks passed. Inputs repeat/crop a real waveform. Cold records are retained but were not the long-input parity acceptance gate. |
| Actual mixed factory | `reference-production-factory-validation-v2/{factory0,factory2}.json`: A=18 and B=30 calls, including nine warmup calls per arm and **21 measured B calls**. All ten codebooks match the baseline envelope exactly. Six short-call host-interval overlaps were observed; long-call overlaps were zero. Host interval overlap is not proof of simultaneous GPU execution. Executors drained and the B owner closed. |
| Memory limit | Factory B peak reserved **5.9921875 GiB**, only **8 MiB below 6 GiB** (peak allocated 5,172,883,968 bytes). This is a PyTorch allocator budget, excludes CUDA context and other non-allocator memory, and does not guarantee arbitrary 60-second concurrent workloads fit. No rescue or `empty_cache` is credited. |
| CPU/lifecycle verification | Recorded owner tests: **13 passed**; configuration/stage tests: **56 passed, 3 GPU cases deselected**. Independent review verified the P1 shutdown fix. These are existing verification results, not tests rerun for this report. |
| Tooling | Recorded full pre-commit completed the Python checks, but Rust formatting failed because `cargo` was absent, exit **127**. The full pre-commit run did **not** pass. |

Earlier invalid indexed-device and mismatched-TF32 harness attempts remain retained as harness errors, not candidate correctness failures or successful acceptance evidence. The corrected factory validation is explicitly `v2`.

## Historical evidence and stop boundary

Earlier temporary-worker full-1,088 runs, approximately +10% throughput points and quality estimates describe different experimental implementations/protocols. They are not formal factory production proof and cannot override this hot/continuity gate. The existing historical report and cookbook modifications are preserved, not rewritten. No performance recommendation is added to the cookbook.

The initial stop decision preserved the opt-in implementation with default `0`, the capped output, underruns and failed harness attempts without launching full-corpus quality or tuning. The author subsequently requested a draft PR for review of this incomplete experiment. Draft publication does not waive the failed performance/continuity gates or authorize merging; no additional GPU work or quality scoring is implied.
