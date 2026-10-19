# Binary stage-payload headers

## Change and scope

The inter-stage control plane is MessagePack and natively supports binary values.
Stage payload headers were pickled, base64 encoded, converted to text, packed into
MessagePack, then decoded in reverse at the receiver. Carry pickle bytes directly
instead. Tensor packing, transport selection, CUDA ordering and request sampling
are unchanged.

New receivers accept both legacy base64 strings and binary headers. Old receivers
cannot accept new binary headers. Deploy homogeneous revisions or upgrade
receivers before enabling new senders; this is not bidirectional mixed-version
negotiation. The existing trusted-process pickle boundary is unchanged; bytes
must not be accepted from untrusted callers as executable payloads.

## Repeatable serialization measurement

Three alternating-order CPU runs used 32 reconstructed request headers containing
actual reference audio from the fixed English evaluation set. Each run discarded
20 warmup iterations and retained 100 iterations, measuring pickle, optional
base64, actual MessagePack packing/unpacking and pickle restoration. All restored
fields were checked equal. This is an isolated metadata benchmark, not GPU time
or E2E service performance.

| Repeat | Base64 round trip us/header | Binary round trip us/header |
|---:|---:|---:|
| 1 | 1114.29 | 141.73 |
| 2 | 1101.90 | 141.10 |
| 3 | 1161.48 | 146.00 |

Median representative MessagePack message sizes were 390037 versus 292530 bytes:
25% fewer header wire bytes. Roughly 0.96 ms/header of CPU work was removed on this
host. The magnitude depends on metadata size; small headers will save less.

## Service evidence and limitations

H100 NVL, torch 2.13.0+cu130, MiniCPM-o checkpoint revision
503e754207c94da6bb26850b4469f367c9ea3582, fixed English input order, seed 42,
temperature 0.7 and max_new_tokens 256. Same-service mode-switch diagnostics
preserved all 128 c1 codec sequences in seven windows including reference repeats.
These snapshots used a diagnostic switch that is not part of product code.

Probe-free full-input ABBA starts used 1060 requests per window, independent
16-request warmup, four windows per service with the first excluded. Results:

| c | Reference median req/s | Binary median req/s | Reference median E2E p95 s | Binary median E2E p95 s |
|---:|---:|---:|---:|---:|
| 16 | 9.542 | 10.112 | 2.2900 | 2.1300 |
| 32 | 13.300 | 13.1995 | 3.2170 | 3.2715 |

Do not claim the c16 positive delta as causal: the final reference start was
materially slower, indicating drift. c32 is approximately -0.76% throughput and
+1.69% p95. These guardrails do not establish service acceleration, and the
serialization improvement must not be described as an E2E speedup. Quantiles are
medians of per-window p95 values, not pooled request quantiles.

Raw successful and negative runs remain under
handoff-results/binary-header-20261019. Initial independent short screens and
same-service diagnostic windows are retained, not selected away. Tensor data and
model output behavior are covered by control-plane round-trip and GPU tests;
PCM-level determinism and speaker/MOS quality acceptance are not inferred merely
from codec equality.

## Correctness checks

Tests cover binary and legacy strings through the actual MessagePack control
plane, mixed tensor/payload reconstruction, unchanged request parameters and
binary reference audio, and invalid header type rejection. Run:

```bash
python -m pytest tests/unit_test/pipeline/test_stage.py tests/unit_test/pipeline/test_payload_device_restore.py tests/unit_test/minicpm_o/test_sglang_talker.py -q
pre-commit run --files sglang_omni/comm/data_ref.py sglang_omni/comm/stage_io.py tests/unit_test/pipeline/test_stage.py
```

The exact staged-commit snapshot passed 101 GPU-enabled tests in 41.98 seconds.
It emitted the existing producer-terminated/shared-CUDA-tensor teardown warning;
passing assertions do not establish clean IPC shutdown. The first launcher used
stdin and failed one spawned-process case because its main module could not be
imported; the importable file launcher passed, with both logs retained. Prior
working-tree run passed 109 tests but included other local changes and is not
presented as the exact committed-code test result.

## Relation to staged handoff research

Qwen3-Omni-inspired block buffering and terminal readiness were separately
investigated. GPU-resident hidden pretransfer reduced terminal-to-conditioning
median latency in several repeated full-input studies, while preserving fixed-seed
c1 codec sequences. Ablation showed that terminal-only separated data delivery
can account for much of that improvement; overlap alone is not the whole cause.
Those prototypes still have lifecycle and service-guardrail limitations and are
not included in this change. This commit addresses redundant serialization only,
not a claim that production staged transfer or every CPU/GPU round trip is solved.
