# MiniCPM-o: bound offline audio feature padding

## Bottleneck and change

The checkpoint processor calls its audio feature extractor with thirty-second padding for every audio segment, then slices the mel features back to the valid lengths. Interleaved video audio can contain dozens of subsecond or one-second segments: the frontend computes and discards most of the STFT and mel frames even when the audio encoder hits its content cache.

The offline MiniCPM preprocessor now retains the checkpoint extractor behind a small adapter. Its padding bound is the longest segment plus one FFT window, capped at the checkpoint's original maximum. The extra zeros preserve the original STFT right boundary and include the zero-tail floor used by dynamic log normalization. The checkpoint still performs feature extraction, normalization, attention-mask construction and truncation. Extractors with dither retain the original padding because shortening their noise would change both the RNG sequence and normalization. Native duplex streaming is unchanged.

This does not optimize Thinker–Talker handoff, lower generation limits, skip input audio, alter frame sampling or change model weights.

## Environment and workload

- Baseline: `b658fa671`; candidate contains only the bounded audio adapter, its preprocessing integration, tests and documentation.
- One H100 NVL, GPU0, same twelve allocated CPU cores, OMP_NUM_THREADS4; torch2.13.0+cu130, sglang0.5.21, transformers5.12.1.
- Checkpoint: `openbmb/MiniCPM-o-4_5`, revision `503e754207c94da6bb26850b4469f367c9ea3582`.
- Three public videos from `zhaochenyang20/Video_MME_ci`: `qvwfWyU9Gfc.mp4`, `24i4ncHuf6A.mp4`, `N1cdUjctpG8.mp4`.
- Ordered inline video with `use_audio_in_video=true`, fps1, max64frames, max401408pixels; request asks for a one-sentence description; temperature0, seed42, max_tokens32 in both arms.
- Four fresh services, A1→B1→B2→A2 on the same GPU. Each performs three initial warmup requests, two first-use requests for the other videos, two three-video sequential windows, six requests at concurrency3, one speech-output request and two single-audio protection requests.
- All seventeen measured requests per service are retained, including cold and first-use outliers:68/68 succeeded. Twelve initial warmup requests are excluded from the table.

This targets interleaved source audio, not the default Video-AMME benchmark's independent spoken question. Repeated videos create an encoder-cache-hit workload; that scope is intentional and must not be generalized to every video or TTS deployment. Serving is non-streaming; latency is complete-response latency, not first-audio latency.

## Exact-code performance

| Metric | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| Warm sequential mean, three videos (s) |2.4529|1.2786|1.2045|2.2089|
| c3 throughput, six requests (req/s) |0.4626|0.9332|0.9678|0.5011|
| c3 p95 (s), descriptive only |6.3003|3.5862|3.5391|6.4043|
| End-of-run GPU memory (MiB) |63513|62509|63501|62513|

Both optimized services exceed both baselines. Mean c3 throughput is approximately97% higher; warm sequential latency is approximately47% lower. These are small, mechanism-specific repeated workloads, not a large benchmark suite. Six-request p95 is descriptive, not a statistically established production tail guarantee; the throughput criterion is the relevant acceptance signal. GPU figures are end-of-run allocations, not peak-memory proofs; they show no persistent increase in this comparison.

An earlier same-service ABBA on the first real video independently measured median latency2.358/2.359s for A and1.059/1.040s for B, with four requests per cell. A subsequent candidate-only service without experiment hooks measured0.82–1.01s. Final table above compares clean baseline source with production candidate source, not a monkeypatch implementation.

### Cold and protection results

First-use video request latencies were4.915/2.999s (A1),6.001/2.179s (B1),2.107/2.062s (B2),7.468/2.757s (A2). First-use initialization varies; the B1 outlier is retained. No stable cold-request gain is claimed.

The second single-audio request was0.275/0.264/0.265/0.268s across A1/B1/B2/A2. First single-audio requests were1.014/1.019/1.271/2.292s. No broad single-audio speedup is claimed.

The first speech-output request in each service incurred substantial lazy compilation/plan initialization (31.51/32.70/21.91/61.19s). These are not comparable steady-state speech latency measurements. All four returned the same transcript and a valid24kHz mono16-bit WAV with156480frames. No steady-state speech-performance or perceptual quality score is claimed.

## Numerical and functional validation

- All final per-request text responses and token-use records match across the four services. The three videos produce the same text and prompt/completion counts in every matching window, including concurrency3 and speech output.
- Checkpoint audio extractor differential screen:33 cases covering silence, random input, tail impulses, single-sample/hop boundaries, mixed lengths and near/full thirty-second audio; valid frames matched bitwise.
- Unit tests compare valid features and lengths, preserve thirty-second truncation, and preserve dither output with identical RNG state.
- Checkpoint-backed full processor regression compares token IDs, audio bounds, valid mel features and feature lengths before/after wrapping the original extractor.
- The existing prompt/video, Thinker and Talker regression suites are run with the pinned checkpoint. See the accompanying results JSON for exact commands and final counts.

Feature-extractor CPU screen, four threads, checkpoint extractor, alternating order (not E2E):16×1s segments92.67→3.89ms;64×1s586.57→11.87ms. Single1s and5s segments save only about2–3ms. The actual service benefit includes eliminating repeated preprocessing and queueing; local ratios are not presented as E2E ratios.

## Reproduction

Install pinned project dependencies including decord. Pin the model revision and download the three video files listed above. Run baseline and candidate on the same GPU and CPU affinity with identical flags:

```bash
python -m sglang_omni.cli serve \
  --model-path "$CHECKPOINT" --host 127.0.0.1 --port 18129 \
  --thinker.factory.max_seq_len 32768 \
  --thinker.engine.mem_fraction_static 0.55 \
  --talker.engine.mem_fraction_static 0.15
```

Send identical `/v1/chat/completions` requests containing an inline `video_url`, `use_audio_in_video=true`, `video_fps=1`, `video_max_frames=64`, `video_max_pixels=401408`, and the generation parameters above. Warm independently before measuring repeated-input windows; include first-use results separately. For the throughput cell submit the three videos twice using three client workers, measure total wall time and divide six completed requests by that wall time. Reverse service order instead of reporting a single best visit.

```bash
MINICPMO_CHECKPOINT="$CHECKPOINT" python -m pytest \
  tests/unit_test/minicpm_o/test_preprocessor_prompt.py \
  tests/unit_test/minicpm_o/test_preprocessor_video.py \
  tests/unit_test/minicpm_o/test_sglang_talker.py \
  tests/unit_test/minicpm_o/test_thinker_model_runner.py -q
pre-commit run --all-files
```

## Limitations and rejected alternatives

No new knob, protocol, checkpoint format or numerical tolerance is introduced. The adapter belongs to the offline preprocessor; native full-duplex streaming remains on its existing processor. A fixed-seed mel differential test is not a substitute for arbitrary dither equivalence, hence dither stays on the old padding path.

Prior candidates were explicitly rejected rather than bundled: speech Thinker lookahead did not retain its apparent gain under resident reverse-order A/B; graph-launch fusion saves only about3.6μs per synthetic step; HiFT F0/source/STFT together are under0.9ms; the Flow Conformer alone is about6.9ms; legacy float-video conversion is bypassed by the modern ordered video-audio route. Existing Talker, DiT and batching PRs were not duplicated.

Full SeedTTS validation was not run: text-only TTS does not exercise the optimized input path. Detailed raw responses, WAVs, video inputs, trace files and machine-specific launch scripts remain local rather than being included in the patch. A compact results JSON records all final measurements without audio payloads or local paths. Existing shutdown semaphore warnings and the missing local Rust `cargo` executable are reported separately from Python test results.
