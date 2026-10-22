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

## Follow-up diagnosis, minimal fix and postfix gate (2026-10-22)

**Decision: still not ready to merge.** This follow-up does not replace or rewrite the original failed gate above. The long-input memory fix has narrower supporting evidence, but the uninstrumented postfix service run fails three of 32 frozen engineering criteria. Full-1,088-prompt generation and quality scoring were **NOT RUN** because the guards failed; default `reference_encode_cuda_streams=0` remains unchanged.

### Cache-phase diagnosis, not acceptance

`drama3-cache-phase-diagnostic-20261022/diagnostic-summary.json` records six fresh boots (Acal1/Acal2/A1/B1/B2/A2), 1,152 measured requests plus 192 discarded warmups. Every measured hot phase has exactly 64 cache hits and zero encoder leaders; available reference-code multisets match exactly, including multiplicities. Direct reference-encoder GPU speedup therefore cannot explain the hot contrasts. Instrumented get-or-encode wall intervals include observer work and are not isolated encoder or full-stage timings.

Acal2 baseline C100 is 95.31%. A/A variation is descriptive evidence of instability, not a noise bound or proof that the earlier regression was noise. Unseeded downstream generation, output lengths, scheduling, clocks, power and host load remain possible confounds; the cause of the hot differences is unknown. Identical cached codes do not make downstream generated work identical.

### Minimal long-input stream fix and four-arm memory evidence

The recorded runtime change touches only 20 lines: long encoding reuses one existing stable execution stream rather than inheriting the calling worker's stream. It adds no execution stream and changes no hot-cache code. New CPU hot-cache cases cover factory streams 0 and 2; CPU device-boundary tests were fixed. Recorded verification is 198 passed, four accelerator cases deselected, with independent review finding no verified blocker. These are supplied verification results, not tests rerun for this documentation update. The coordinator reports that the full pre-commit run passed Python/style checks but Rust formatting again failed because `cargo` was missing (exit 127); full pre-commit did not pass, and it was not rerun for this documentation update.

Memory evidence is `drama3-profile-444026/reference-production-memory-{recovery-v1,fixed-long-v1,stable-stream-v1}/memory*.json`. The four arms below are distinct controls, not interchangeable implementations. Retry counters are cumulative after each round's 60-second call.

| Memory arm | End-of-round allocation retries | Peak reserved (GiB) | Peak allocated (GiB) |
|---|---|---:|---:|
| baseline0 | 0 / 0 / 0 / 0 / 0 / 0 | 5.826172 | 4.788034 |
| old_B_alternating | 1 / 1 / 2 / 2 / 3 / 3 | 5.990234 | 4.817624 |
| fixed_worker | 1 / 1 / 1 / 1 / 1 / 1 | 5.863281 | 4.817624 |
| stable_execution_stream | 1 / 1 / 1 / 1 / 1 / 1 | 5.863281 | 4.847897 |

The new stable-stream arm alternates two host threads across six long calls but uses the same execution stream. All 18 calls match the baseline envelope across all ten codebooks. All arms stay within the 6 GiB allocator budget with no OOM and no `empty_cache` rescue. New peak reserved is 5.863 GiB versus old B's 5.990 GiB; peak allocated is 4.848 versus 4.818 GiB. This is allocator accounting, not total device-memory accounting.

The stable-stream arm has a one-off **+32 MiB live allocation**, then a **1.671465 GiB** plateau through the later rounds and shutdown checkpoint. Ownership of that retained allocation is unknown. Stable retry counts do not prove zero extra overhead, no retained tensors, or arbitrary long-workload safety.

### Uninstrumented postfix acceptance

`drama3-reference-postfix-acceptance-v1/formal-postfix-summary.json` records four fresh A1/B1/B2/A2 services without cache observers or `sitecustomize`: all **896 requests completed**, comprising **768 measured plus 128 discarded warmups**, with **zero errors**. Each visit measures cold, hot1 and hot2, 64 requests each, after 32 discarded warmups. The frozen criteria hash, integrity checks, compact cells, cap IDs and every nonzero underrun are retained under `followup` in the adjacent JSON; massive per-request comparison arrays are not copied.

| Visit | Regime | Completed / errors / caps | Frames/s | E2E p95 (s) | TTFA p95 (s) | C50 (%) | C100 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | cold | 64 / 0 / 0 | 329.8 | 5.112 | 3.4704 | 100.0 | 100.0 |
| A1 | hot1 | 64 / 0 / 0 | 584.8 | 3.401 | 0.6798 | 98.44 | 100.0 |
| A1 | hot2 | 64 / 0 / 0 | 544.2 | 3.38 | 0.6297 | 95.31 | 98.44 |
| B1 | cold | 64 / 0 / 1 | 540.5 | 3.715 | 1.5593 | 100.0 | 100.0 |
| B1 | hot1 | 64 / 0 / 0 | 627.4 | 3.004 | 0.5291 | 100.0 | 100.0 |
| B1 | hot2 | 64 / 0 / 0 | 640.4 | 3.288 | 0.5398 | 100.0 | 100.0 |
| B2 | cold | 64 / 0 / 0 | 570.6 | 3.289 | 1.5193 | 100.0 | 100.0 |
| B2 | hot1 | 64 / 0 / 0 | 627.9 | 2.984 | 0.5254 | 100.0 | 100.0 |
| B2 | hot2 | 64 / 0 / 0 | 628.6 | 3.322 | 0.5118 | 100.0 | 100.0 |
| A2 | cold | 64 / 0 / 0 | 381.7 | 4.481 | 3.0139 | 100.0 | 100.0 |
| A2 | hot1 | 64 / 0 / 0 | 643.7 | 3.015 | 0.5442 | 100.0 | 100.0 |
| A2 | hot2 | 64 / 0 / 0 | 617.6 | 3.138 | 0.4848 | 100.0 | 100.0 |

Cold throughput improves 329.8→540.5 and 381.7→570.6 frames/s. Hot throughput is mixed: A1→B1 hot1 584.8→627.4 and hot2 544.2→640.4; A2→B2 hot1 643.7→627.9 (a decrease) and hot2 617.6→628.6. All hot throughput guards pass the frozen ≥0.95 ratio, but this is not uniform improvement and does not establish that the long-input fix caused any hot benefit. All measured B cells have C100=100%; baseline A1 hot2 is 98.44%.

The three failures are B1 cold cap count **1 versus A1's 0**, B2/A2 hot2 E2E p95 **3.322/3.138 s (+5.8636%)**, and hot2 TTFA p95 **0.5118/0.4848 s (+5.5693%)**; both latency limits are +5%. B1 cold capped ID is `common_voice_en_17263752-common_voice_en_17263750` (63 `stop`, one `length`). Postfix nonzero underruns are A1 hot1 `common_voice_en_17444666-common_voice_en_17444667` (0.0682 s), and A1 hot2 that same ID (0.1641 s), `common_voice_en_17784-common_voice_en_17785` (0.0635 s), and `common_voice_en_177895-common_voice_en_177894` (0.0634 s). No cap or underrun is filtered, retried away, or substituted for the historical records above.

Only two independent server pairs exist; hot1/hot2 repeats are not independent server replicates. Frozen engineering thresholds are not statistical non-inferiority, confidence intervals or causal-effect estimates. Improved throughput in some new cells does not erase the original failed hot/continuity gate or solve the remaining postfix failures. No additional GPU work, runtime edits, quality evaluation or merge approval is implied.

## Full formal 1,088-request joint validation (2026-10-22)

This is a later, explicitly authorized **formal production-factory** campaign, artifact label `drama3-formal-full1088-final`, not a substitution of historical temporary-hook results. The original failed small gate and all three postfix failures above remain unchanged historical evidence; their “NOT RUN” statements describe those earlier stopping points. This section records the subsequently executed full campaign, not a readiness or merge verdict. Review is ongoing, with no verified blocker yet reported; default `reference_encode_cuda_streams=0` is unchanged.

### Protocol, integrity and retained verifier failure

The actual generation metadata, `source-pins.json`, `correctness-evidence-pins.json`, `reference-isolated-full1088/source-manifest.json`, input manifest, per-arm verification and cleanup records, and quality manifest/status/post-verification form the evidence chain. The adjacent JSON adds only `full_corpus_validation`, with compact hashes, source pins, metrics, failure/cap IDs and every nonzero underrun; no audio or giant per-request arrays are copied. Generation source HEAD is `34e42dc8cab8ecf026648dcd7b010d357bdd37ad`. The source manifest pins the English corpus revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`; model and codec revisions are the same as the original protocol above. Pins distinguish executed artifacts from later local benchmark corrections.

One fresh A factory-0 service and one fresh B factory-2 service use the formal option, GPU split and concurrency/capacity 16 described above. There are **no CUDA-event, Torch-profiler, Pyspy or sitecustomize hooks**. Each arm discards 32 warmup requests and then attempts all 1,088 IDs once. Warmup has 20 distinct content references, all overlapping the full cohort, which has **666 distinct content references**. This is **full mixed-reference**, neither pure cold nor pure hot; the full pass includes first uses, repeats and warmed references. Sampling remains temperature 1.0, top-p 0.9, cap 500, without fixed seed/top-k. Pairing by sample ID is not common-RNG or identical generated work.

A produced 1,087 audio outputs and one HTTP 500/no-audio failure before the external verifier aborted on a null underrun (`None > 0`). `status.baseline-validator-failed.json` preserves that failure. The corrected verifier treats failed rows explicitly, validates successful audio, and does not reinterpret completion validity as all-request success. Original A metadata/audio were retained without rerun or rescue; the **sole B continuation** used the original 90-minute generation deadline, not a reset budget. Both arms remain `all_requests_succeeded=false`.

### Actual full-corpus performance and adverse outcomes

| Arm | Attempted / audio successes / HTTP errors / caps | Frames/s | E2E p95 (s) | TTFA p95 (s) | RTF p95 | C50 (%) | C100 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A, factory 0 | 1,088 / 1,087 / 1 / 0 | 611.6 | 3.532 | 0.9798 | 0.7203 | 100 | 100 |
| B, factory 2 | 1,088 / 1,087 / 1 / 1 | 666.0 | 3.338 | 0.6232 | 0.6351 | 99.91 | 100 |

B/A output throughput improves **8.895%**, E2E p95 changes **−5.4926%**, and TTFA p95 **−36.3952%**. These new full mixed-reference performance observations are within the 5% adverse latency/throughput guards. They do not make every guard pass: B has one cap versus A's zero and lower C50. Continuity denominators are the 1,087 successful audio outputs per arm, not all attempted IDs. Output frame totals differ, 98,350 versus 98,736. This is one sequential server pair, not reverse-order replication or formal hot-path non-inferiority; the earlier incremental-hot uncertainty cannot be erased by a mixed-reference aggregate.

- A HTTP/no-audio failure: `common_voice_en_21499767-common_voice_en_21499784`.
- B HTTP/no-audio failure: `common_voice_en_588627-common_voice_en_588628`.
- B cap: `common_voice_en_30735432-common_voice_en_30735433`, `length`, 500 tokens, 23.22 seconds of audio. A has 1,087 `stop` outcomes plus one `unknown`; B has 1,086 `stop`, one `length` and one `unknown`.
- B's only underrun above 50 ms is `common_voice_en_591530-common_voice_en_591529`, **60.5 ms**. All six A and twelve B nonzero underruns, including smaller ones, remain in the compact JSON, not filtered out.

Both generation servers exited 0 without forced shutdown; groups, descendants and samplers were gone. Recorded post-scoring GPU process verification is empty, with all GPUs reported clean. This update ran no GPU work.

### Actual quality scoring and paired uncertainty

Quality evidence is `quality-isolated-full1088/paired-quality-results.json`, with `status.json`, `post-verification.json` and scorer/source/checkpoint pins in the manifest and asset provenance. All six WER/SIM/UTMOS scoring jobs exited **0**. Post-verification checked **2,174 generated audio records** and confirms unchanged metadata, generated audio, reference audio and source hashes. The ASR scorer revision is `7278e1e70fe206f11671096ffdd38061171dd6e5`; checkpoint provenance and safe-load audit are retained, not inferred from exit status alone. Scoring jobs completed successfully, but generation failure prevents a complete-1,088 quality-pass claim.

Only generation **FAIL** outputs are ineligible: each metric evaluates **1,087 A and 1,087 B** outputs. Paired comparison uses **1,086 IDs**, excluding exactly the two different failure IDs listed above. No cap, WER>50% output, poor SIM or low UTMOS is excluded for convenience.

| Eligible-arm aggregate | A (1,087) | B (1,087) | B−A |
|---|---:|---:|---:|
| Corpus WER (%) | 1.315789 | 1.399246 | +0.083456 percentage points (worse) |
| SIM (scorer units) | 64.180898 | 64.126950 | −0.053948 |
| UTMOS | 3.987976 | 3.988668 | +0.000692 |

Corpus WER is summed edits / summed reference words: A **157/11,932**, B **167/11,935**, not an unweighted mean of per-utterance percentages. The paired estimates below use the common 1,086-ID subset, so their deltas differ from the eligible-arm aggregate deltas above. The actual paired bootstrap uses 10,000 replicates, seed 20261021, percentile 95% intervals.

| Paired B−A estimate | Difference | 95% paired interval |
|---|---:|---:|
| WER (percentage points) | +0.083864475 | [−0.184362692, +0.397326770] |
| SIM (scorer units) | −0.057637466 | [−0.498213487, +0.387069964] |
| UTMOS | −0.000460958 | [−0.017019436, +0.015719542] |

The source JSON stores paired WER delta **0.0008386447500838643 as a ratio**, with interval [−0.0018436269169529885, +0.0039732676966225495]; multiplying by 100 gives the percentage-point table. **No non-inferiority margin was predeclared. Intervals crossing zero are not proof of non-inferiority.** Neither unchanged failure/cap frequency nor a complete-1,088 quality pass is established.

The B capped ID remains scored despite its **WER 100%, SIM 26.876322, UTMOS 1.362**; the same ID's A scores are WER 0%, SIM 59.199154 and UTMOS 3.5388. The A 100%-WER ID `common_voice_en_21538043-common_voice_en_21538044` is also retained. The user-approved scope excludes the confirmed baseline 1+15/EOS shared mechanism from this optimization's blocker accounting, but does not delete those outcomes or prove that the candidate leaves their frequency unchanged. Shared-mechanism diagnosis cannot settle incremental hot-path uncertainty.

### CPU verification and benchmark correlation boundary

Recorded verification supplied for the combined work is **198 core CPU tests** and **88 benchmark CPU tests** passing, scoped hooks passing, and independent review of the P2 first-fragment timestamp correction. These are recorded results, not rerun GPU/component evidence for this documentation update. The prior whole pre-commit attempt failed Rust formatting because `cargo` was missing (exit 127), despite passing Python/style checks; it was not a complete pre-commit pass. Quality scoring exit-0 evidence is separate from repository CI/tooling status.

The current local benchmark-only correlation fix adds typed optional epoch-nanosecond fields for client send start, first nonempty PCM and request end, plus the server speech outcome ID in saved rows (including HTTP-error responses). First-PCM epoch time follows the **first nonempty fragment**, not delayed HTTP-chunk assembly. Focused tests cover empty markers, fragmented chunks/end-of-stream buffering, JSON persistence, error/no-audio cases and unchanged monotonic timing/continuity behavior. These fields enable joining client and server evidence; E2E/TTFA still use the existing monotonic clock, with **no metric-semantic change**. Do not confuse the later reviewed correction with the pinned implementation used during generation.

The joint evidence therefore consists of finite reference-code correctness, bounded allocator/lifecycle results, improved full mixed-reference service metrics, retained adverse outcomes and descriptive quality uncertainty. It does not overwrite the failed small/postfix guards, establish formal hot or quality non-inferiority, authorize production/default changes, or supply a ready/merge verdict. The main reviewer owns final status.

Documentation sanity check: the current whole-repository pre-commit run again passed Python/style and other available hooks, but Rust formatting failed with missing `cargo` (exit 127); the whole run did not pass. The coordinator reports prior Rust quality CI passes separately. Final independent review reports no verified PR-new-code blocker; this does not change the non-inferiority limits or grant merge approval. Publication and any review-status transition remain the main reviewer’s responsibility.

## H100-only strategy and deployment-cost supplement (2026-10-22)

> **Latest decision: H100-only design validation, not production readiness. Current PR does not satisfy all guards.** Keep the implementation opt-in with default `reference_encode_cuda_streams=0`. Single-card B2 fails continuity; dual-card hot2 fails throughput in both pairings and E2E p95 forward. No unique optimum, universal best-two-stream strategy, portability, cost-win or energy-win claim is supported. This supplements rather than deletes the historical evidence above.

### Scope and fair strategy screen

Only the exact **two H100 NVL** cards were tested; no other GPU was tested. Source HEAD is `cc6bfa2f9f2267579098ccea489aff8294396751`. The adjacent JSON adds only `h100_strategy_cost_supplement`, retaining raw-source/script/cohort hashes, public asset revisions, all phase guards, compact costs and adverse-outcome IDs without machine paths or audio/code arrays. Production code/configuration is unchanged.

`drama3-h100-strategy-fair-20261022/verification.json` repairs the comparison with fair FIFO admission: 128 real references × cold/repeat1/repeat2 = **384 calls per n**, n=1–4, all outputs exact across **all ten codebooks** against the retained original baseline codes. **INVALID old timings** from `drama3-h100-strategy-20261022` are excluded from every ranking: original0 had CPU-overlap contamination and non-FIFO admission caused severe starvation. Original0 is a code baseline only, not a clean timing control.

| n | Cold frames/s | Pooled hot frames/s | Hot repeat1 / repeat2 p95 (ms) |
|---|---:|---:|---:|
| 1 | 984 | 3879 | 232.1 / 228.5 |
| 2 | 1451 | 3602 | 485.4 / 264.1 |
| 3 | 1665 | 3144 | 546.4 / 324.5 |
| 4 | 1523 | 3066 | 563.9 / 453.9 |

n=1 has higher pooled hot throughput and better p95 than n=2 in **both** repeats; cold n=3 is fastest. Material repeat variance and only two hot passes preclude a unique optimum. Timings include CPU resampling, admission, H2D, encode, D2H, synchronization, hashing and diagnostic bookkeeping/serialization (explicit final JSON save excluded from phase wall): **neither GPU-only compute nor full file decode**. Fixed eight outstanding waveforms aligns with eight callers, not a tested optimal queue depth.

Natural equal-length batch2/batch4 fail exact parity at **104/217 code indices**, with differences across all ten books; eligible coverage is only **14/128 and 4/128**. They stopped before timed or mixed-length gates. This rejects the exact-code contract, not batching as inherently incorrect or a generated-quality claim.

n=3/n=4 boundary runs actually exercise **3/4 concurrent 10-second short calls** and exclusive long calls, with exact codes and no OOM. Peak reserved is **5.969/5.992 GiB**, allocator retries **5/8**. Long jobs reserve all permits but choose a **free worker**, not production’s stable `long_stream`; no per-transition memory history exists. This is not stable-long-stream validation, proof of leak freedom, or evidence that stream count caused retries. The 10-second cutoff remains conservative under the existing 6 GiB allocator baseline; repeated/cropped loops of one public waveform are not varied real long-input coverage.

### Service deployment and individual guard outcomes

`drama3-h100-service-cost-20261022/service-summary.json` records **eight fresh boots, 1,664 completed requests: 256 discarded warmups + 1,408 measured, zero errors, three baseline-only caps**. Each completed measured cell has 64 requests. Single-card uses one physical GPU; dual-card keeps preprocessing isolated on the second GPU. Single B uses test-only placement injection into the real Factory2 on the same card, not a production configuration change. Production isolation remains intact. ABBA order is A1/B1/B2/A2 per topology; sampling/concurrency follow the formal service protocol above.

| Topology / visit | Phase | Frames/s | E2E p95 (s) | TTFA p95 (s) | C100 (%) | Caps |
|---|---|---:|---:|---:|---:|---:|
| single-A1 | cold | 368.0 | 4.181 | 2.6710 | 100.00 | 0 |
| single-A1 | hot1 | 638.0 | 3.426 | 0.5322 | 100.00 | 0 |
| single-A1 | hot2 | 564.9 | 3.257 | 0.4507 | 100.00 | 1 |
| single-B1 | cold | 500.9 | 3.716 | 1.7331 | 100.00 | 0 |
| single-B1 | hot1 | 654.4 | 2.885 | 0.5351 | 100.00 | 0 |
| single-B1 | hot2 | 693.5 | 2.946 | 0.4542 | 100.00 | 0 |
| single-B2 | cold | 505.5 | 4.066 | 1.4961 | 96.88 | 0 |
| single-B2 | hot1 | NOT RUN | — | — | — | — |
| single-B2 | hot2 | NOT RUN | — | — | — | — |
| single-A2 | cold | 385.6 | 4.638 | 3.0961 | 100.00 | 1 |
| single-A2 | hot1 | 612.3 | 3.375 | 0.5273 | 100.00 | 0 |
| single-A2 | hot2 | 627.7 | 3.218 | 0.5200 | 100.00 | 0 |
| dual-A1 | cold | 373.6 | 3.686 | 2.5081 | 100.00 | 0 |
| dual-A1 | hot1 | 601.4 | 3.207 | 0.5303 | 100.00 | 0 |
| dual-A1 | hot2 | 617.8 | 2.966 | 0.5227 | 100.00 | 0 |
| dual-B1 | cold | 542.2 | 3.411 | 1.4374 | 100.00 | 0 |
| dual-B1 | hot1 | 642.8 | 3.027 | 0.5326 | 100.00 | 0 |
| dual-B1 | hot2 | 563.8 | 3.133 | 0.5178 | 100.00 | 0 |
| dual-B2 | cold | 550.9 | 3.349 | 1.5622 | 100.00 | 0 |
| dual-B2 | hot1 | 633.7 | 3.179 | 0.5235 | 100.00 | 0 |
| dual-B2 | hot2 | 607.0 | 3.114 | 0.5159 | 100.00 | 0 |
| dual-A2 | cold | 407.4 | 4.425 | 2.8764 | 100.00 | 1 |
| dual-A2 | hot1 | 598.1 | 3.094 | 0.5253 | 100.00 | 0 |
| dual-A2 | hot2 | 645.5 | 3.306 | 0.5000 | 100.00 | 0 |

Single B1 passes its observed pair but does not generalize: **single-B2 cold C100=96.88% versus both A visits’ 100%**, triggering the stop after warmup+cold; B2 hot1/hot2 were **NOT RUN**, with no repeat rescue. Retained >100 ms underruns are `common_voice_en_17444666-common_voice_en_17444667` (**142.5 ms**) and `common_voice_en_17519982-common_voice_en_17519985` (**158.4 ms**). Every smaller nonzero underrun and cap ID is also retained in JSON; the prior full-corpus 18 underruns are unchanged.

For every pair/phase below, throughput uses the −5% adverse guard, latency/TTFA +5%, C100 must not fall and errors/caps must not increase. Cold additionally improves throughput. No pooling or aggregate pass replaces an individual failure.

| Pair / phase | Throughput Δ (%) | E2E p95 Δ (%) | TTFA p95 Δ (%) | Individual verdict |
|---|---:|---:|---:|---|
| single forward / cold | +36.1141 | -11.1217 | -35.1142 | Listed guards pass |
| single forward / hot1 | +2.5705 | -15.7910 | +0.5449 | Listed guards pass |
| single forward / hot2 | +22.7651 | -9.5487 | +0.7766 | Listed guards pass |
| single reverse / cold | +31.0944 | -12.3329 | -51.6779 | FAIL: c100 |
| dual forward / cold | +45.1285 | -7.4607 | -42.6897 | Listed guards pass |
| dual forward / hot1 | +6.8839 | -5.6127 | +0.4337 | Listed guards pass |
| dual forward / hot2 | -8.7407 | +5.6305 | -0.9374 | FAIL: throughput, e2e_p95 |
| dual reverse / cold | +35.2234 | -24.3164 | -45.6891 | Listed guards pass |
| dual reverse / hot1 | +5.9522 | +2.7473 | -0.3427 | Listed guards pass |
| dual reverse / hot2 | -5.9644 | -5.8076 | +3.1800 | FAIL: throughput |

All dual cells have C100=100%, but forward hot2 throughput **−8.7407%** and E2E p95 **+5.6305%** fail; reverse hot2 throughput **−5.9644%** also fails. Cold gains **+45.13%/+35.22%** do not cancel these failures. JSON also retains the frozen-stop-reference comparisons separately; their passing cells do not supersede reverse-pair failures. All candidate caps/errors are zero; three caps belong to single-A1 hot2, single-A2 cold and dual-A2 cold.

### Physical reservation cost and limits

Cold single-default A1/A2 delivers **368.06/385.64 frames/s per reserved GPU**, versus dual-candidate B1/B2 **271.17/275.50**: **-26.32%/-28.56%** normalized efficiency. Active-window GPU-seconds rise **13.18/13.73 → 18.25/18.30**, or **+38.44%/+33.34%**. Both reserved physical GPUs count throughout dual windows, even when preprocessing is idle. Request cohorts match but output frames differ stochastically (4,852/5,293 versus 4,949/5,042); use the per-frame metric, not a same-generated-work claim. Full compact costs for every executed phase, including warmups and hot passes, are in JSON.

Power has only **5–14 samples per phase, ALL <20**. Integrated energy covers only first-to-last active samples: partial covered-time device energy, not exclusive application energy, full-window energy or an energy win. Thirty-second idle baselines exist raw; no idle adjustment is claimed. Completed service/whole elapsed accounting is **31.52/40.31 minutes**, from the original component-start budget with no deadline reset; the JSON separately preserves later summary-observation elapsed values. All eight cleanup records report exit 0, no forced shutdown, groups/descendants gone and samplers stopped.

No new generated-audio quality run was performed. The original full-1,088 mixed-reference quality evidence remains valid in its separate regime; it does not override these new hot/continuity failures. Recommendation: **keep opt-in, make no optimal-strategy claim, and do not describe the current PR as production-ready with all guards satisfied**. No code fix is proposed here. PR-body/status decisions remain the coordinator’s responsibility; this update makes no GitHub change.
