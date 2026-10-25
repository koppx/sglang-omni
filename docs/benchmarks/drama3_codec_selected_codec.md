# Drama 3: selected stateful codec optimizations

## Scope and attribution

This opt-in SGLang-Omni implementation combines stateful codec execution, GEMM quantizer upsampling, folded embedding/output-projection tables, Snake/history-input fusion and compiled window attention. Existing channels-last decoding, tower fusion, device-resident AR inputs and sampling are not credited as new contributions. The selected production scope excludes compact LM head, mixed GDN, request-stable sampling, async lookahead, transport packing and admission coalescing experiments.

Each stream owns a generation-checked arena lease, convolution histories, unbiased transposed-convolution carry and window-attention state. Exact delta lengths 1–64 have shared CUDA graphs; the scheduler splits longer arrivals into supported deltas, and direct decoder calls above 64 frames are rejected. Graph submissions are serialized, outputs are independently owned, and release waits for submitted work. Capacity follows bounded end-to-end admission, not the vocoder batch size. Stateful mode requires fixed routing, single replicas, a dedicated TP1 vocoder process and no crossfade. Existing stateless mode remains the default.

The attention compiler uses emulate_precision_casts=True to retain eager low-precision boundaries. The original compiler configuration failed a retained 63-frame generation at the unchanged FP16 atol=rtol=0.02; that failure remains documented and is not attributed to baseline randomness. The precision-preserving implementation passes that input and the full retained 288-input cohort. A formatting-only source-hash change occurred during the latter component run and is explicitly recorded in the retained local experiment summary; the later service runs have unchanged frozen hashes.

## Measured end-to-end results

Hardware: one H100 NVL, Torch 2.13 and SGLang 0.5.21. The headline below uses the exact codec-only staged publication tree, not the broader experimental tree. Each comparison is an independent fresh-boot A1/B1/B2/A2 experiment with 32 discarded warmups and 384 measured requests per boot: C1 64, C16 64, C32-hot 128 and arrival 128. Arrival rate is 16 requests/second with a fixed dispatch seed. Native temperature 1, top_p 0.9, top_k unset, seed unset and max_new_tokens 500 are unchanged. Original window graphs 10/20/32/64/96, FP16 codec, FP32 GDN state, original chunk ramp, async off, coalescing off and MPS off are retained. Percentages compare arm boot means and are not additive.

| Control | Workload | Mean E2E | Mean first audio | Audio throughput | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Mature window A | C1 | -1.74% | +1.39% | +1.53% | -0.36% |
| Mature window A | C16 | +0.10% | +3.24% | +2.02% | +2.02% |
| Mature window A | C32-hot | -4.27% | +1.71% | +8.85% | -4.03% |
| Mature window A | Arrival | -7.73% | -30.40% | +7.72% | +0.99% |
| Original stateful S | C1 | -3.18% | -2.35% | +3.43% | +0.04% |
| Original stateful S | C16 | -1.44% | -1.28% | +5.56% | +2.41% |
| Original stateful S | C32-hot | -6.26% | -7.83% | +0.94% | +2.36% |
| Original stateful S | Arrival | -6.92% | -2.11% | +4.81% | +0.62% |

Both comparisons complete all 1,536 measured requests without HTTP failures and with clean service/resource teardown and unchanged frozen source hashes. The window comparison retains one control A1 C32-hot length cap; the stateful comparison retains one candidate B2 C32-hot cap. C1/C16/C32 first-audio regressions against A and approximately neutral C16 E2E are real tradeoffs. Window-arrival C50/C100 continuity falls by 18.10%/16.82% relatively despite lower first-audio latency; this prevents claiming uniformly improved streaming continuity. C32 throughput is confounded by caps, output length and tail duration. Work varies stochastically, and two boots per arm do not establish precise causal attribution to individual features. The combination is a high-concurrency opt-in, not a universally better default.

The separate GF-controlled production ablation measures the incremental Snake plus precision-preserving compiled attention combination: C1/C16/C32-hot/arrival E2E -1.41%/-8.25%/-2.85%/-3.56%, with no failures/caps and near-equal token work. This is not transferred to Snake alone. Earlier GF-only E2E results were mixed; component speedups alone did not justify its independent selection.

## Quality and numerical limitations

All 3,072 measured A/S comparison outputs were scored with pinned Qwen3-ASR-1.7B revision 7278e1e70fe206f11671096ffdd38061171dd6e5 and UTMOS, with zero skips, regeneration or filtering. Both scoring services exited zero. Values below are means of the two per-boot corpus WERs and per-boot UTMOS means; caps remain included.

| Control | Workload | WER control → candidate | UTMOS control → candidate |
| --- | --- | --- | --- |
| A | C1 | 0.7824% → 0.6401% | 3.8984 → 3.9302 |
| A | C16 | 1.2091% → 1.4225% | 3.9351 → 3.8912 |
| A | C32-hot | 1.5739% → 0.9151% | 3.8917 → 3.8819 |
| A | Arrival | 1.0615% → 1.1347% | 3.8839 → 3.8840 |
| S | C1 | 0.7824% → 0.9246% | 3.9121 → 3.9045 |
| S | C16 | 0.9957% → 0.7112% | 3.9347 → 3.9102 |
| S | C32-hot | 0.9517% → 0.8419% | 3.8799 → 3.8804 |
| S | Arrival | 0.6589% → 1.0981% | 3.9052 → 3.8764 |

These small stochastic cohorts show mixed changes, not formal quality noninferiority or proven baseline attribution for individual anomalies. Independent same-code reconstruction is the numerical gate, not service WER. The production precision component gate passes 281 numerical/lifecycle comparisons, including irregular chunks, interleaving, ownership, slot reset and graph replay, at the original FP16 tolerance. Its three alternating complete-set timing medians are 0.70530 seconds mature window versus 0.37116 seconds selected codec, including copies/D2H/release; this combined component comparison does not isolate individual features.

## Reproduction and evidence

Use the normal Drama3 serving command and configure the following stage-owned flags; supply local model/codec paths and the normal deployment memory budgets. A finite admission bound must be set before enabling stateful mode.

```bash
python -m sglang_omni.cli serve --model-path "$DRAMA3_CHECKPOINT" \
  --vocoder.factory.codec_path "$CODEC_CHECKPOINT" \
  --tts_engine.factory.max_running_requests 32 \
  --tts_engine.engine.max_running_requests 32 \
  --tts_engine.engine.max_queued_requests 32 \
  --tts_engine.engine.cuda_graph_max_bs 32 \
  --tts_engine.factory.enable_async_decode false \
  --vocoder.factory.stateful_decode true \
  --vocoder.factory.stateful_gemm_upsample true \
  --vocoder.factory.stateful_folded_codebooks true \
  --vocoder.factory.stateful_fuse_snake_history true \
  --vocoder.factory.stateful_compile_attention true
```

Use benchmarks.eval.benchmark_tts_seedtts with --use-existing-server --generate-only --stream --collect-stream-outcomes, the same metadata in both arms, temperature 1, top_p 0.9 and max_new_tokens 500. Retain every generated output; run --transcribe-only and --utmos-only afterward. The archived service summaries contain per-boot latency, work, continuity, caps, activation and cleanup, not private waveforms or target text. Exact local experiment drivers and full artifacts are preserved locally; the portable command above does not claim to recreate the private cohort selection automatically.

The per-boot reports, source freezes, worker activation records, quality scores and failed/incomplete experiments are retained locally, including the earlier broader-tree comparisons and compact-head negative results. They are not checked in as bulk JSON artifacts. The tables above report the exact publication-tree measurements, not the larger earlier experimental-tree gains. Mixed graph debugging was stopped and is not a release prerequisite. Independent review attempts failed at the provider API before analysis and provide no review credit.

## Optimization inventory and disposition

| Optimization direction | Local disposition |
| --- | --- |
| One backbone/tower forward, paged tower attention, fused tower, device-resident decode inputs | Already present and exercised; not new gains in this PR. |
| Compact EOS/semantic head | Production-integrated locally and fully service-tested, but mixed high-concurrency/quality results; excluded from runtime publication. Component and production negative evidence retained. |
| Stateful codec, GEMM, folded tables, Snake/history, compiled attention | Selected opt-in combination with exact-tree evidence above; no independent feature percentages added. |
| Deferred convolution bias, fused RVQ, batched codec/split replay | Screened locally; negative E2E, failed numerical prerequisites or incomplete failed runs retained; excluded. |
| Extra graph buckets, channels-last reference encoding | Screened separately; modest mixed gains or slower/changed conditioning codes; excluded. Baseline graph family unchanged here. |
| Mixed GDN packed decode/chunked prefill, fused exp(g), startup warmup | Kernel/API probes and experimental integration retained; local mixed-graph replay failures unresolved and work stopped. No external-backend bug attribution. |
| BF16 GDN recurrent state | Separate precision experiment, not adopted into the unchanged-FP32 comparison. |
| Request-seeded draws and preemption history | Existing recovery and local seed probes exercised; no E2E speedup established, experimental implementation excluded. Native sampling unchanged. |
| Prefill admission | Existing mechanism screened; load-specific first-audio/work tradeoffs, excluded. |
| Whole-step output batching and tensor windows | Existing mechanisms exercised; new wire/packed transport experiments excluded. |
| Ready notifications, event-driven scheduling, first-chunk express, sampling-row buckets | Existing paths or local screens; no qualified incremental E2E gain, no sibling-model behavior changes. |
| Pre-encoded references | Existing validated references.N.vq_codes API; uncached encoder-cost screen is not generation E2E gain. |
| MPS and SM share | Existing deployment support screened; mixed C1/quality/continuity or negative settings, excluded. |
| Window ramp/smaller first windows | Lower TTFA but throughput/continuity tradeoffs, excluded; original ramp retained. |
| MoE backends | DeepGEMM negative screen; installed FlashInfer FP8 selection falls back to Triton, so no false CUTLASS timing claim. |
| Dense FP8 and prefix caching | Deliberately unadopted: unchanged dense precision and unsafe token-only audio-prefix hashing. |

The earlier detailed screening report and all failed/incomplete raw experiments remain preserved locally. This inventory is a disposition record, not a claim that every experimental direction completed E2E qualification or that source reading is a timed experiment.

## Regression and review status

The exact codec-only staged tree passes the complete Drama3 CPU suite (507 passed, four skipped, 143 accelerator cases deselected), the focused six-module CPU regression (160 passed, 11 accelerator cases deselected) and actual H100 codec CUDA regression (8 passed, 95 deselected). A broader Drama3 plus pipeline IPC CPU run reports 546 passed, four skipped, 143 deselected and one failure: test_mp_runner_startup_failure_includes_child_factory_traceback times out after 30 seconds without child startup progress rather than reaching the intentional factory error. An isolated selected-tree run reproduces that failure, and the unchanged pinned baseline reproduces the same 30-second TimeoutError. This baseline/environment startup-test limitation is retained, not fixed by weakening the test or reported as a complete regression pass. Real service boots in the performance experiments complete successfully. Full repository pre-commit passes, including Rust formatting; staged Python files compile and the staged private-path scan finds no private host paths or waveform/target payloads. Independent review attempts fail with provider 400/403 before analysis, so no independent approval is claimed. Human review, broader hardware coverage and formal quality noninferiority remain outstanding for this Draft.
