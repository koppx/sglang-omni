# MiniCPM-o handoff acceptance investigation

## Verdict

The packed-tensor alignment correctness fix is supported by GPU and real-service evidence. Stable end-to-end performance improvement is **not accepted**. GPU handoff remains opt-in. The long-workload concurrency-16 experiment regressed in all three repetitions. This report records negative results rather than labeling the optimization goal successful.

## Source and reproduction

Parent: 067816ed8125776d022cdead6f3fbbea4dc78dc8. The existing uncommitted candidate contained continuous hidden blocks, direct-to-allocation SHM/CUDA packing and receiver allocation changes. These were preserved. Experiments used immutable source snapshots and saved candidate patches, not a claim that every candidate change belongs to this correctness commit.

Artifacts are under handoff-results/followup-20261015, critical-path-20261015 and acceptance-20261015. Commands below run on an allocated idle CUDA GPU with the experiment Python environment and FFmpeg shared libraries available. Experiment scripts and raw files are retained locally; their artifact paths are not dependencies of production code.

```bash
python handoff-results/critical-path-20261015/hidden_size_sweep_v2.py
python handoff-results/critical-path-20261015/hidden_size_sweep_fresh.py
python handoff-results/acceptance-20261015/run_acceptance.py
python handoff-results/acceptance-20261015/quality_expanded.py
python -m pytest tests/unit_test/minicpm_o/test_sglang_talker.py tests/unit_test/pipeline/test_stage.py tests/unit_test/relay/test_cuda_ipc_relay.py tests/unit_test/relay/test_shm_relay.py -q
pre-commit run --all-files
```

Checkpoint: MiniCPM-o-4_5 revision 503e754207c94da6bb26850b4469f367c9ea3582. H100 NVL, torch 2.13.0+cu130. Measurement lists, seeds, request order and warmup lists are frozen in each artifact directory. Long-workload arms use temperature 0.7, seed 42 and max_new_tokens 4096; these results must not be merged with the original 256-token experiment. Audio was retained for comparison. Performance trials disable detailed probes.

## Alignment correctness

An odd int64 prompt places the next bf16 tensor at an eight-byte residue modulo 16 under dtype-only packing. A GPU buffer slice retains that address residue. Aligning tensor offsets to 256 bytes fixes the layout without a new receiver clone.

Original full c1 records independently reproduced 493/520 duration differences for odd prompts and 0/540 for even prompts. Frozen real-checkpoint semantic-projector tests covered prompt lengths 35/36/37/51 and hidden rows 16/37/64. All aligned outputs matched standalone allocations bitwise; odd old-layout outputs differed by up to 0.00146484375. This verifies an alignment-dependent numerical effect, not the identity of a particular CUDA kernel.

After alignment, candidate P0 and pooled P1 had zero duration differences against baseline over all 1060 samples (520 odd, 540 even). Exact PCM equality remains unresolved: repeated baseline differed on 3/32 samples; full candidate P0/P1 differed on 75/84 samples, with equal frame counts. Among differing samples median RMS was 0.097/0.079 int16 LSB, maximum 17 LSB. No claim of full waveform bit-exactness or human listening acceptance is made.

## Why existing P0/P1 gains were unproven

The completed original 54-trial experiment showed P1 throughput changes versus baseline at c1/2/4/8/16/32 of -0.06/-0.60/-2.75/+0.37/-2.72/-0.45 percent. Small deltas did not exceed repeat variation, and output differences invalidated a simple equal-work interpretation.

Identical c32 trace snapshots measured host-observed serialization medians around 0.65–0.68 ms and conditioning enqueue around 1.08–1.14 ms. Pooled put was 3.59 ms versus SHM 0.70–0.86 ms; pooled acquisition alone was about 2.50 ms. Receiver copy wait included GPU queueing and host resume; it is not equivalent to isolated copy bandwidth. Host trace overhead is not production latency.

Continuous hidden blocks reduced isolated 16-row clone/assembly from 80.01 to 24.61 microseconds, but such savings alone cannot justify percent-level E2E gains. Four-arm c32 repeated medians were baseline 12.256, previous P0 11.396, candidate P0 11.580, candidate P1 11.885 req/s, with large repeat variation. No stable win was accepted.

## Hidden-size dependence

Sequential same-GPU cross-process transport, three rotated repetitions, five warmup and twenty measured transfers per arm; fresh source tensor creation excluded equally. Values and alignment checked outside the timed GPU-ready response.

| Hidden | P0 RTT ms | Pooled P1 RTT ms | Direct IPC RTT ms |
|---|---:|---:|---:|
| 128 KiB | 0.500 | 0.680 | 0.840 |
| 512 KiB | 0.917 | 0.723 | 0.886 |
| 2 MiB | 2.076 | 0.853 | 0.971 |
| 8 MiB | 5.851 | 0.865 | 0.968 |

At 512 KiB pooled latency improved 12.7/21.2/12.9 percent; at 128 KiB it regressed 36.0/34.2/41.6 percent. The original diagnostic workload's hidden min/median/p95/max was 56/112/176/240 KiB. Payload size explains a transport crossover, not service acceptance. These experiments include Pipe overhead, hot allocation/cache, no competing model computation and one in-flight request. Fresh tensor objects do not guarantee fresh CUDA storage handles. Direct IPC did not beat pooling at any sampled size, so 8 MiB is not validated as an optimal universal threshold.

The first sweep included forced GC before acknowledgement, contaminating RTT. Its raw results were preserved and excluded; corrected v2 and fresh-source experiments remove that contamination.

## Expanded real-service acceptance

Eight repeated-sentence inputs first showed positive three-repeat p95 changes, but Whisper-small WER was 0.336–0.980 on both arms. That set was rejected as sole quality acceptance evidence, not silently pruned.

A second set used 32 deterministic paragraphs made from distinct dataset sentences, four length strata, identical references and generation settings. All 18 trials completed successfully.

| Concurrency | Repeat | P0 qps | P1 qps | P0 p95 s | P1 p95 s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.188 | 0.190 | 9.149 | 9.270 |
| 1 | 2 | 0.194 | 0.190 | 8.929 | 9.481 |
| 1 | 3 | 0.197 | 0.197 | 8.864 | 8.881 |
| 16 | 1 | 0.928 | 0.867 | 27.403 | 29.489 |
| 16 | 2 | 1.050 | 1.027 | 24.198 | 24.607 |
| 16 | 3 | 1.073 | 0.961 | 23.314 | 25.387 |
| 32 | 1 | 1.813 | 1.705 | 17.279 | 18.676 |
| 32 | 2 | 1.725 | 1.789 | 18.451 | 17.373 |
| 32 | 3 | 1.776 | 1.794 | 17.473 | 17.584 |

c1 average audio duration was identical at 47.235 seconds for all six trials. At higher concurrency audio lengths varied, so QPS alone does not establish equal-work improvement. c16 regressed throughput 6.57/2.19/10.44 percent and worsened p95 in all repetitions. c32 was inconsistent. The performance gate fails.

Whisper-small c1 quality evaluation processed all 30-second windows rather than truncating long audio. P0/P1 produced identical transcriptions for all 32 inputs; aggregate WER was 17.55 percent and per-sample median 4.74 percent. This is an ASR quality proxy only: no speaker-similarity/MOS or human listening acceptance was performed. Long paragraphs still contain poor outputs; no claim of absolute long-form quality acceptance follows from equal WER.

## Additional rejected experiment

A free-slot acquisition fast path avoided creating two asyncio tasks when allocation could complete without yielding. CPU failure/blocked-allocation regressions passed. Three paired c32 rounds changed QPS -0.70/+1.53/+2.19 percent. It failed the direction-consistency criterion and was removed from production code. Snapshot and raw results remain under acceptance-20261015/fast-slot-screen.

## P2 assessment and next work

Qwen3-Omni buffering/deferred readiness is a useful scheduling reference, not a drop-in protocol. Known-text MiniCPM hidden arrives from one prefill, giving little pretransfer overlap. Free-generation staged hidden transfer would need complete-coverage COMMIT, copy-completion notification, cancellation, epoch ownership and backpressure; arbitrary hidden prefixes must not start codec generation. No evidence yet justifies adding that complexity for this workload.

Further work should attribute the 2.5 ms pooled acquire scheduling delay with a GPU/host timeline under contention, test a dedicated transfer event loop or safe batching, and measure handoff-to-first-compute rather than non-streaming completion alone. No architecture-wide impossibility claim is made: the current experiments disprove acceptance of these candidates, not every future optimization.

## Validation and limitations

CPU suite: 126 passed, 35 CUDA-dependent skips. GPU suite: 164 passed, 2 cross-GPU skips. Added frozen-conditioning GPU regression: four passed. All available repository pre-commit checks passed; Rust formatting blocked because cargo is unavailable. CUDA IPC teardown emitted a producer-terminated/shared-tensor warning requiring lifecycle follow-up. Two-GPU behavior, exact waveform determinism, speaker quality and statistically powered large-corpus tails remain unaccepted.

The deliverable is a justified alignment correction plus honest experimental evidence. It is not a claim that the >=2 percent end-to-end goal was achieved.
