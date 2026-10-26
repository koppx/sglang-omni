# Drama 3 incremental streaming codec

## Scope

The decoder reuses per-request causal convolution histories, unbiased transposed-convolution carries and window-attention state, processing only new codec frames. GEMM upsampling, folded codebooks, Snake/history fusion and precision-preserving compiled attention run with exact-length CUDA graphs for 1–64 new frames. Longer arrivals are split. Outputs own their storage and slot release waits for submitted GPU work. Compatible CUDA deployments automatically enable the full bundle when the stateful options are unset, for both FP16 and FP32 decoding. Selection checks Triton and the platform graph backend, without a GPU-name allowlist. It also requires window CUDA graphs, a vocoder memory budget and the topology constraints below. Missing capabilities and incompatible configurations retain the window decoder. Explicitly set --vocoder.factory.stateful_decode false to keep the window path. An explicit true enables the same default bundle, with individual optimization overrides preserved.

The service and GPU reconstruction evidence below remains specific to H100 FP16. Capability-based selection does not establish accuracy or performance on other GPUs or GPU FP32. Historical performance tables use explicit codec and admission settings, not the automatic admission policy.

This change preserves main's compact semantic head and shared convolution bias/history contracts. Compact-head execution, existing tower fusion and existing sampling are not new contributions of this PR. Mixed recurrent kernels, recurrent-state precision changes, MPS and additional scheduling changes are outside scope.

Stateful mode requires bounded end-to-end admission, fixed routing, single replicas, a dedicated TP1 vocoder process and zero crossfade. Cancellation acknowledgments retain admission until cleanup completes; stale acknowledgments cannot retire a newer cancellation generation. Cleanup exceptions fail the stage instead of silently disabling cancellation. Failed slot release retains its owner for recovery.

## Main-integrated service evidence

Two independent fresh-boot ABBA experiments used one H100 NVL, Torch 2.13, SGLang 0.5.21, the same checkpoint, FP16 codec and FP32 recurrent state. Both arms retained main's compact semantic head, native temperature 1, top_p 0.9, unset seed, max_new_tokens 500, original window ramp and baseline graph buckets 10/20/32/64/96. Async decode and MPS were off. Each boot discarded 32 warmups and measured C1 64, C16 64, C32-hot 128 and arrival=16 requests/s 128 requests. All 3,072 measured requests completed without HTTP failures; two first-round length caps remain included. Output work varies stochastically.

The service baseline was 85d309fbc and the measured candidate was 86da9dfd4. The later main integration at 694461c2b changes MLX code, not this Python serving path. Subsequent fixes concern cleanup exception ordering and tests, not successful decoding arithmetic. These old measurements retain their original versions; they are not relabeled as measurements of later commits.

| Workload | Mean E2E change, round 1 / 2 | Audio throughput change, round 1 / 2 | Mean RTF change, round 1 / 2 |
| --- | ---: | ---: | ---: |
| C1 | +2.60% / -3.49% | -0.52% / +1.67% | Not a high-concurrency claim |
| C16 | -4.00% / -6.51% | +3.60% / +6.76% | -3.41% / -6.50% |
| C32-hot | -4.57% / -2.13% | +2.98% / +0.29% | -5.07% / -3.66% |
| Arrival 16/s | -11.06% / -18.42% | +10.03% / +18.39% | -9.84% / -17.44% |

C16 and arrival arm ranges separate in these finite repetitions. C32 throughput ranges overlap and the second-round throughput change is small; caps, output length and tail work prevent a strong throughput claim there. Component acceleration is not an end-to-end acceleration percentage.

## First-audio and continuity tradeoffs

The original 5% first-audio requirement did not specify a statistic. Mean first audio changes +0.31%/-3.26%, but the means of per-boot p50 values increase 15.80%/12.27%; p95 and p99 arm means improve. A later bounded diagnostic did not reproduce the p50 regression, which does not invalidate the original unfavorable boots or fully explain their cause. No universal first-audio improvement is claimed.

Main-integrated arrival C50/C100 improve in both rounds. Earlier publication-tree experiments had an arrival continuity regression despite faster first audio; that historical result remains a limitation rather than being transferred to this configuration.

In the first C128 capacity experiment, all four boots have 128 valid multi-chunk requests, no N/A values and C50/C100=0. Maximum playback-underrun p50/p95/p99 seconds are A1 0.6689/0.7253/0.8127, A2 0.7007/0.7419/0.7991, B1 0.2954/0.3322/0.4298 and B2 0.3281/0.3512/0.3558. The candidate improves this distribution, but neither arm meets continuous-playback requirements at this load. C128 smooth playback and sustained arrival capacity are not validated. Two additional independent capacity ABBA experiments completed on the review runtime, with identical settings and 2,048 additional measured outputs. All outputs have valid full-target WER/UTMOS and verified waveform hashes; no HTTP failures occurred. One round-2 C64 length cap remains included. All C128 boots still have C50/C100=0.

The three capacity experiments use two fresh boots per arm each; they are three independent ABBA experiments, not six independent experiments. The first retains its original baseline/candidate versions; rounds 2 and 3 use main 694461c2b and the review runtime 2d6ee725c. Local dependency placement is identical between arms in the follow-up runs, but differs from the earlier shared-filesystem placement, so rounds are reported separately.

| Workload | Round | Mean E2E change | Audio throughput change | Mean RTF change | WER A → B | UTMOS A → B |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| C64 | 1 | -14.16% | +15.18% | -15.25% | 1.0981% → 0.5857% | 3.8724 → 3.8858 |
| C64 | 2 | -11.42% | +24.72% | -12.09% | 1.0249% → 0.8785% | 3.8707 → 3.8860 |
| C64 | 3 | -9.77% | +19.85% | -10.12% | 1.2079% → 0.8419% | 3.8693 → 3.8726 |
| C128 | 1 | -19.59% | +21.90% | -18.52% | 0.6589% → 0.8053% | 3.8769 → 3.8893 |
| C128 | 2 | -30.10% | +34.10% | -30.37% | 0.8053% → 1.0981% | 3.8670 → 3.8773 |
| C128 | 3 | -17.32% | +18.61% | -16.58% | 0.9517% → 0.8419% | 3.8770 → 3.8598 |

The capacity gains repeat in direction, while quality changes remain mixed. C128 round-3 mean first audio increases 0.92% despite faster completion. No additional quality noninferiority claim follows from these measurements. Service shutdown exits zero and GPU process checks clear, but Python resource_tracker reports leaked-semaphore cleanup warnings in both arms; these warnings remain recorded and are not described as warning-free teardown.

## Full-output quality

The two main service experiments and first capacity experiment contain 4,096 measured outputs. Every output was checked against its run, request, full target text, waveform hash and scoring record; all have valid WER and UTMOS, without filtering, skips or regeneration. The truncated display-text mistake occurred only in a later diagnostic's initial scoring attempt; that attempt was retained and all affected diagnostic WER scores were recomputed using complete targets. It does not affect these 4,096 original scores.

| Workload | WER A → B, round 1 / 2 | UTMOS A → B, round 1 / 2 |
| --- | --- | --- |
| C1 | 0.7112% → 0.9246% / 0.8535% → 0.7824% | 3.9003 → 3.9083 / 3.9250 → 3.8978 |
| C16 | 1.1380% → 1.3514% / 1.0669% → 0.7112% | 3.8747 → 3.8864 / 3.9277 → 3.9189 |
| C32-hot | 1.1713% → 1.4275% / 0.8053% → 1.0615% | 3.8524 → 3.8830 / 3.8862 → 3.8638 |
| Arrival 16/s | 0.8419% → 0.8419% / 1.0615% → 1.2811% | 3.8948 → 3.8835 / 3.8778 → 3.8868 |

C32 contains 1,366 reference words per boot. Candidate totals increase by seven error words per ABBA experiment, or 0.2562 percentage points in the arm mean, in both rounds. Unfavorable long generations remain included. UTMOS improvements do not offset WER deterioration. There is no predeclared numerical noninferiority margin, and formal service-quality noninferiority is not established.

A selected diagnostic captured 32 service code sequences per arm and replayed every sequence through both codecs and native whole decoding. For A-source codes, both codecs have 2/363 error words; for B-source codes, finite-window decoding has 5/363 and stateful/native decoding 3/363. All candidate reconstructions pass the unchanged FP16 atol=rtol=0.02 against native whole decoding. This finite selected cohort supports codec correctness, not unbiased full-service quality acceptance. The original 500/483-frame anomalies were not reproduced, and their cause remains incompletely explained. No independent listening assessment is claimed.

## Regression and reproduction

The previously validated runtime passed 967 pipeline tests with six skips and 15 deselections; fixed main passed 956 with the same skips/deselections. All 962 common test outcomes matched, and 11 additional cleanup cases passed. Drama3 plus shared convolution passed 515 tests with four skips and 167 deselections. Actual H100 graph replay passed both optimized/unoptimized cases, including 63/64-frame history, a one-frame tail, interleaving, slot reuse, output ownership and a 193-frame scheduler split. Full repository pre-commit passed. These results predate capability-based automatic selection and the TP follower acknowledgment guard; they are not fresh GPU validation of those changes.

Enable the existing stage-owned options explicitly; keep all other deployment conditions identical in the control:

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

Use benchmarks.eval.benchmark_tts_seedtts with --use-existing-server --generate-only --stream --collect-stream-outcomes and a fixed metadata list, retaining every output and cap. Score all outputs with the same pinned ASR and UTMOS assets. Private audio, target text and machine-specific paths are not published. Human review and acceptance of the documented quality/latency tradeoffs remain required. Automatic and explicitly enabled modes derive end-to-end admission and slot capacity from resolved engine running plus queued limits. If engine limits are absent, the engine factory's max_running_requests supplies the running limit and queued capacity defaults to zero. Explicit engine limits take precedence; slot capacity and cleanup acknowledgments use that same resolved selection. This admission policy differs from earlier explicitly enabled benchmark configurations, whose original measurements remain labeled separately.

## Portable identical-code reproduction

Save an integer NumPy array shaped [codebooks, frames], with no semantic text-token column. Supply the codec checkpoint and choose a fresh output directory:

```bash
python -m benchmarks.eval.benchmark_drama3_codec \
  --codec-path "$CODEC_CHECKPOINT" \
  --codes "$CODES_NPY" \
  --output-dir "$RESULT_DIR" \
  --device cuda:0
```

The command writes finite-window, all-enabled stateful graph and native whole waveforms, input/output hashes, source fingerprints, exact frame partitions, length checks and numerical differences. It enforces the unchanged 0.02 candidate/native tolerance and retains failure status. It never overwrites an existing output directory. A synthetic array can exercise the numerical path, but its audio has no intelligibility reference; use actual codec codes and their corresponding full transcript for WER. This component command is neither an E2E benchmark nor a quality noninferiority test. Service throughput and continuity must still be measured with the existing SeedTTS benchmark under matched deployment conditions.

## Sampling attribution boundary

The existing Triton sampler uses a process seed together with a global decode-step counter, batch row and draw-stream ID. A request occupying a different row or global step can therefore receive different random draws even with identical logits. A fixed-logit diagnostic using the production sampler reproduced exact output for unchanged seed/step/row, but changed 32/32 tokens after row remapping and 29/32 after changing the step in one synthetic cohort. This demonstrates the mechanism, not the cause of the historical service WER difference. It does not justify adding a new sampling policy to this codec PR, treating per-request seeds as an equality guarantee, or discarding unfavorable service generations.
