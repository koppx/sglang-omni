# Keep small host tensors out of CUDA payload buffers

## Mechanism

A mixed Thinker-to-Talker payload includes CPU prompt token IDs and GPU hidden
states. Previously all extracted tensors were concatenated on the CUDA relay
device. CPU token IDs therefore took H2D on send and D2H on receive, although the
Talker span parser expects them on CPU. Avoid that detour: store small CPU tensor
contents, dtype and shape as raw bytes in the trusted payload header; pack GPU
tensors through the existing relay unchanged. CPU values are reconstructed into
owned writable storage, retaining dtype/shape and CPU placement. Values are not
converted into Python token lists or passed through PyTorch tensor pickle.

The total inline CPU contents budget is 64 KiB per payload. Larger CPU tensors
remain on the previous relay path. SHM, direct torch IPC, sampling parameters,
CUDA fences and relay lease/reclamation logic are not changed. Existing payload
headers remain readable. Older receivers do not understand the new mixed-device
header class, so homogeneous deployment is required. This does not add untrusted
pickle support; the existing trusted-process pickle boundary remains.

## Exact-source correctness gates

The selectively staged source snapshot excludes the working tree's unrelated
direct-to-allocation relay changes. Its import path was asserted before testing.
Stage and CUDA relay suite: 78 passed, one cross-GPU topology skip, 61.05 seconds.
The existing producer-terminated/shared-CUDA-tensor warning occurred at teardown;
this does not establish clean IPC lifetime shutdown. Tests cover scalar, empty
and noncontiguous CPU tensors, exact tensor equality, preserved placement,
aggregate inline budget, oversized header rejection and malformed byte lengths.

A same-service c1 study of the bounded typed implementation used seven windows
of 256 measurement requests plus 16 independent warmup requests. Every measured
codec sequence matched reference across all c1 windows, including reference
repeats. This is not a claim of PCM bitwise determinism, speaker similarity or
human quality acceptance. Earlier source checks passed 97 GPU-enabled cases with
two unavailable cross-GPU cases skipped, but those included other local changes
and are not the exact curated-source gate.

## Repeated cross-process transport measurement

H100 NVL, torch 2.13.0+cu130. The exact staged module ran in the same live sender
and receiver processes, using the same CUDA relay code and binary header format.
The only toggle was the inline host budget (zero versus 64 KiB). Each payload
had a 37-element CPU int64 prompt and a bf16 hidden matrix of width 4096.
Three repetitions alternated mode order, with five warmup and twenty measured
transfers per mode/size. Receiver values, CPU placement and GPU alignment were
verified after a timed GPU-ready acknowledgement. RTT includes Pipe notification,
serialization and receiver readiness; value validation was outside its endpoint.
These are isolated sequential transfers, not service E2E or pure CUDA memcpy time.

| Hidden | Repeat | Packed CPU token RTT ms | Inline host token RTT ms |
|---|---:|---:|---:|
| 128 KiB | 1 | 1.0321 | 0.7773 |
| 128 KiB | 2 | 0.9873 | 0.7748 |
| 128 KiB | 3 | 0.9892 | 0.7668 |
| 512 KiB | 1 | 0.9878 | 0.7691 |
| 512 KiB | 2 | 1.0147 | 0.7703 |
| 512 KiB | 3 | 0.9945 | 0.7834 |
| 2 MiB | 1 | 0.9976 | 0.7780 |
| 2 MiB | 2 | 0.9986 | 0.7693 |
| 2 MiB | 3 | 1.0065 | 0.7624 |

All paired directions improved, roughly 21–25% RTT under these conditions. This
measures elimination of a host-metadata device detour, not a general acceleration
for all payloads. Allocation/cache temperature and one in-flight request differ
from a contended model server. Earlier experiments that used different processes
for the two layouts were order-dependent; they are retained but not substituted
for this paired evidence.

## Probe-free service guardrails and limits

Matched-relay ABBA starts at c32, four full 1060-request windows per service,
first window excluded, six retained windows per arm. Same checkpoint revision
503e754207c94da6bb26850b4469f367c9ea3582, input order, references, seed 42,
temperature 0.7 and max_new_tokens 256. Both arms used identical relay packing;
only the CPU-inline byte budget differed. No mode-file/per-request timing probes
were installed in these snapshots.

Reference median throughput was 12.3495 req/s, candidate 12.363; E2E p95 medians
were 3.502 and 3.4895 seconds. This is essentially unchanged, not proof of E2E
speedup. Window-level quantiles are not pooled per-request quantiles. Sampled
high-concurrency audio lengths vary. Earlier same-service c16 diagnostic windows
were mixed: the full handoff median did not improve consistently, and individual
throughput/p95 regressions remain in raw artifacts. Do not claim complete
c16/c32 production latency acceptance from the isolated RTT reduction.

A first formal launch mixed unrelated relay changes across arms and was stopped
before measurement, preserved as INVALID_COMPARISON.txt. The corrected matched
relay run is formal-c32-v2. Raw artifacts are under
handoff-results/host-bytes-20261019, including curated-paired.json,
curated-source-verified-regression.log, resident-typed and formal-c32-v2.

## Reproduction

Use an exclusively allocated CUDA GPU and the repository's pinned environment:

```bash
python -m pytest tests/unit_test/pipeline/test_stage.py tests/unit_test/relay/test_cuda_ipc_relay.py -q
python handoff-results/host-bytes-20261019/benchmark_curated_pair.py --source /path/to/source-snapshot --output results.json
pre-commit run --files sglang_omni/comm/stage_io.py tests/unit_test/pipeline/test_stage.py
```

The retained experiment runner uses local checkpoint-independent synthetic
hidden tensors; it must use this source revision for both child processes.
Machine-specific launch scripts and generated audio are not production APIs.

## Relation to staged transfer

Qwen3-Omni-style hidden buffering/deferred readiness was also prototyped. GPU
hidden chunks can overlap generation and reduce terminal-to-conditioning tails,
while retaining COMMIT coverage, epochs, receive expiry, byte reservations and
producer/receiver CUDA fences. However production-safe refactors had mixed E2E
and tail guardrails. Balanced ablation showed much of the reduction also occurs
with a terminal-only separated data channel, so overlap alone was not the cause.
Those experimental protocols are not part of this patch and are not claimed to
be fully accepted. This change independently removes CPU token H2D/D2H detours;
the binary-header change independently removes redundant base64 serialization.
