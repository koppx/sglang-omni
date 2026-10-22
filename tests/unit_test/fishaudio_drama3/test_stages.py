# SPDX-License-Identifier: Apache-2.0
"""Drama 3 stage factories on CPU: codec path resolution and what each factory builds."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import Literal

import pytest
import torch
from tokenizers import Tokenizer

from sglang_omni.models.fishaudio_drama3 import stages
from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_CODEC_PATH,
    DEFAULT_MAX_NEW_TOKENS_CAP,
    Drama3VocoderFactoryArgs,
)
from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
from sglang_omni.models.fishaudio_drama3.reference_encode import (
    DeviceEncode,
    ReferenceEncodeOwner,
)
from sglang_omni.models.fishaudio_drama3.speech_request import Drama3SpeechPreprocessor
from sglang_omni.models.fishaudio_drama3.vocoder import Drama3VocoderScheduler
from sglang_omni.models.fishaudio_s2_pro.stages import FishReferenceEncodeHook
from sglang_omni.models.fishaudio_s2_pro.streaming_vocoder import WindowDecoder
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService
from tests.unit_test.fishaudio_drama3.test_prompt import (
    DEFAULT_SYSTEM_TURN,
    END,
    USER,
    VOICE_TURN,
    build_toy_tokenizer,
    expected_rows,
)
from tests.unit_test.fishaudio_drama3.test_speech_request import (
    generation_input_of,
    lowered,
    refused,
    toy_config,
)
from tests.unit_test.fishaudio_drama3.test_vocoder import FrameNumberCodec

COMMIT = "1de9996b6be38b745688de084d87a5633f714e4e"


class HubDownloads:
    """Stands in for hf_hub_download and records each call."""

    def __init__(self, local_file: Path) -> None:
        self.local_file = local_file
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, repo_id: str, filename: str, *, revision: str) -> str:
        self.calls.append((repo_id, filename, revision))
        return str(self.local_file)


@pytest.fixture
def hub_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HubDownloads:
    downloads = HubDownloads(tmp_path / "snapshot" / "codec.pth")
    monkeypatch.setattr(stages, "hf_hub_download", downloads)
    return downloads


def test_a_local_codec_file_or_directory_resolves_without_the_hub(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    codec_file = tmp_path / "codec.pth"
    codec_file.write_bytes(b"")

    assert stages.resolve_codec_directory(str(codec_file)) == str(tmp_path)
    assert stages.resolve_codec_directory(str(tmp_path)) == str(tmp_path)
    assert hub_downloads.calls == []


def test_a_hub_codec_downloads_only_codec_pth_at_its_commit(
    hub_downloads: HubDownloads,
) -> None:
    assert stages.resolve_codec_directory(DEFAULT_CODEC_PATH) == str(
        hub_downloads.local_file.parent
    )
    assert hub_downloads.calls == [("fishaudio/s2-pro", "codec.pth", COMMIT)]


@pytest.mark.parametrize(
    "codec_path",
    [
        pytest.param("fishaudio/s2-pro", id="no-revision"),
        pytest.param("fishaudio/s2-pro@main", id="branch"),
        pytest.param(f"fishaudio/s2-pro@{COMMIT[:7]}", id="short-commit"),
        pytest.param(f"fishaudio/s2-pro@{COMMIT.upper()}", id="upper-case-commit"),
        pytest.param("missing/codec.pth", id="missing-file"),
    ],
)
def test_a_codec_path_that_could_drift_or_does_not_exist_is_refused(
    hub_downloads: HubDownloads, codec_path: str
) -> None:
    with pytest.raises(ValueError, match="names no codec.pth file"):
        stages.resolve_codec_directory(codec_path)
    assert hub_downloads.calls == []


def test_a_codec_file_of_another_name_is_refused(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    renamed_codec_file = tmp_path / "renamed-codec.pth"
    renamed_codec_file.write_bytes(b"")

    with pytest.raises(ValueError, match="names no codec.pth file"):
        stages.resolve_codec_directory(str(renamed_codec_file))


def test_a_directory_without_codec_pth_is_refused(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="names no codec.pth file"):
        stages.resolve_codec_directory(str(tmp_path))


class CodecLoads:
    """Stands in for load_codec and records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, checkpoint_dir: str, device: str) -> FrameNumberCodec:
        self.calls.append((checkpoint_dir, device))
        return FrameNumberCodec()


@pytest.fixture
def codec_loads(monkeypatch: pytest.MonkeyPatch) -> CodecLoads:
    loads = CodecLoads()
    monkeypatch.setattr(stages, "load_codec", loads)
    return loads


def write_toy_checkpoint(tmp_path: Path) -> Tokenizer:
    tokenizer = build_toy_tokenizer()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
    toy_config(tokenizer).save_pretrained(str(checkpoint_dir))
    (tmp_path / "codec.pth").write_bytes(b"")
    return tokenizer


def test_preprocessing_reads_the_checkpoint_and_loads_the_codec_on_cpu(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    tokenizer = write_toy_checkpoint(tmp_path)
    rows = expected_rows(
        tokenizer, DEFAULT_SYSTEM_TURN + [USER, "<|speaker:0|> hello", END, VOICE_TURN]
    )

    scheduler = stages.create_preprocessing_executor(
        str(tmp_path / "checkpoint"),
        device="cpu",
        max_concurrency=2,
        max_prompt_rows=rows.shape[0],
        codec_path=str(tmp_path / "codec.pth"),
        reference_encode_dtype="float32",
        reference_encode_cuda_streams=0,
        max_reference_audio_seconds=30.0,
    )

    preprocessor = scheduler.fn
    assert isinstance(preprocessor, Drama3SpeechPreprocessor)
    assert scheduler.max_concurrency == 2
    assert codec_loads.calls == [(str(tmp_path), "cpu")]
    hook = preprocessor.reference_encoder.hook
    assert isinstance(hook, FishReferenceEncodeHook)
    assert (hook.device, hook.autocast_dtype, hook.max_reference_audio_seconds) == (
        torch.device("cpu"),
        None,
        30.0,
    )
    assert torch.equal(
        generation_input_of(preprocessor(lowered({"input": "hello"}))).prompt_rows,
        rows,
    )
    assert "maximum allowed length" in refused(
        preprocessor, lowered({"input": "hello there"})
    )


def test_a_bfloat16_reference_encode_on_the_cpu_is_refused(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    write_toy_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="reference_encode_dtype to float32"):
        stages.create_preprocessing_executor(
            str(tmp_path / "checkpoint"),
            device="cpu",
            max_concurrency=1,
            max_prompt_rows=64,
            codec_path=str(tmp_path / "codec.pth"),
            reference_encode_dtype="bfloat16",
            reference_encode_cuda_streams=0,
            max_reference_audio_seconds=60.0,
        )

    assert codec_loads.calls == []


class EncodeRecordingCodec:
    sample_rate: int = 44100

    def __init__(self) -> None:
        self.encoded_shapes: list[tuple[int, ...]] = []

    def encode(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoded_shapes.append(tuple(audios.shape))
        return torch.zeros((1, 10, 1), dtype=torch.long), audio_lengths


@pytest.mark.parametrize(
    ("device", "reference_encode_dtype", "expected_encoded_shapes"),
    [
        pytest.param("cpu", "float32", [], id="cpu"),
        pytest.param(
            "cuda",
            "bfloat16",
            [(1, EncodeRecordingCodec.sample_rate)],
            id="cuda",
            marks=pytest.mark.accelerator,
        ),
    ],
)
def test_preprocessing_warms_the_encoder_up_once_on_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    device: str,
    reference_encode_dtype: Literal["float32", "bfloat16"],
    expected_encoded_shapes: list[tuple[int, ...]],
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    write_toy_checkpoint(tmp_path)
    codec = EncodeRecordingCodec()
    monkeypatch.setattr(stages, "load_codec", lambda checkpoint_dir, device: codec)

    stages.create_preprocessing_executor(
        str(tmp_path / "checkpoint"),
        device=device,
        max_concurrency=1,
        max_prompt_rows=64,
        codec_path=str(tmp_path / "codec.pth"),
        reference_encode_dtype=reference_encode_dtype,
        reference_encode_cuda_streams=0,
        max_reference_audio_seconds=60.0,
    )

    assert codec.encoded_shapes == expected_encoded_shapes


def test_the_engine_factory_hands_its_knobs_to_the_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(
        builder: Drama3EngineBuilder,
        model_path: str,
        *,
        device: str | None,
        gpu_id: int | None,
        server_args_overrides: Mapping[str, object] | None,
        total_gpu_memory_fraction: float | None,
    ) -> str:
        assert (
            model_path,
            device,
            gpu_id,
            server_args_overrides,
            total_gpu_memory_fraction,
        ) == ("checkpoint", None, 1, {"mem_fraction_static": 0.5}, None)
        assert (
            builder.context_length,
            builder.max_running_requests,
            builder.max_prompt_rows,
            builder.max_new_tokens_cap,
            builder.enable_async_decode,
            builder.async_decode_min_batch_size,
        ) == (4096, 4, 1024, 512, True, 3)
        return "engine"

    monkeypatch.setattr(Drama3EngineBuilder, "build", build)

    engine = stages.create_tts_engine_executor(
        "checkpoint",
        gpu_id=1,
        context_length=4096,
        max_running_requests=4,
        max_new_tokens=512,
        max_prompt_rows=1024,
        server_args_overrides={"mem_fraction_static": 0.5},
        enable_async_decode=True,
        async_decode_min_batch_size=3,
    )

    assert engine == "engine"


@pytest.mark.parametrize(
    (
        "process_total_gpu_memory_fraction",
        "server_args_overrides",
        "kv_cache_budget",
        "expected_overrides",
    ),
    [
        pytest.param(None, None, 0.7, {"mem_fraction_static": 0.7}, id="own-share"),
        pytest.param(0.8, None, 0.8, {"mem_fraction_static": 0.7}, id="process-share"),
        pytest.param(
            None,
            {"mem_fraction_static": 0.7},
            0.7,
            {"mem_fraction_static": 0.7},
            id="equal-static-fraction",
        ),
    ],
)
def test_the_engine_sizes_its_kv_cache_from_its_declared_share(
    monkeypatch: pytest.MonkeyPatch,
    process_total_gpu_memory_fraction: float | None,
    server_args_overrides: Mapping[str, object] | None,
    kv_cache_budget: float,
    expected_overrides: Mapping[str, object],
) -> None:
    def build(
        builder: Drama3EngineBuilder,
        model_path: str,
        *,
        device: str | None,
        gpu_id: int | None,
        server_args_overrides: Mapping[str, object] | None,
        total_gpu_memory_fraction: float | None,
    ) -> tuple[float | None, Mapping[str, object] | None]:
        return total_gpu_memory_fraction, server_args_overrides

    monkeypatch.setattr(Drama3EngineBuilder, "build", build)

    engine = stages.create_tts_engine_executor(
        "checkpoint",
        gpu_id=0,
        total_gpu_memory_fraction=0.7,
        process_total_gpu_memory_fraction=process_total_gpu_memory_fraction,
        server_args_overrides=server_args_overrides,
    )

    assert engine == (kv_cache_budget, expected_overrides)


def test_a_static_fraction_other_than_the_stage_share_is_refused_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds: list[str] = []

    def build(
        builder: Drama3EngineBuilder,
        model_path: str,
        *,
        device: str | None,
        gpu_id: int | None,
        server_args_overrides: Mapping[str, object] | None,
        total_gpu_memory_fraction: float | None,
    ) -> str:
        builds.append(model_path)
        return "engine"

    monkeypatch.setattr(Drama3EngineBuilder, "build", build)

    with pytest.raises(ValueError, match="--tts_engine.gpu_memory_fraction"):
        stages.create_tts_engine_executor(
            "checkpoint",
            gpu_id=0,
            total_gpu_memory_fraction=0.7,
            server_args_overrides={"mem_fraction_static": 0.5},
        )

    assert builds == []


def cpu_vocoder(factory_args: Drama3VocoderFactoryArgs) -> Drama3VocoderScheduler:
    return stages.create_vocoder_executor(
        device="cpu",
        codec_path=factory_args.codec_path,
        stream_stride=factory_args.stream_stride,
        stream_followup_stride_ramp=factory_args.stream_followup_stride_ramp,
        stream_followup_stride=factory_args.stream_followup_stride,
        stream_overlap_tokens=factory_args.stream_overlap_tokens,
        stream_crossfade_samples=factory_args.stream_crossfade_samples,
        max_batch_size=factory_args.max_batch_size,
        max_batch_wait_ms=factory_args.max_batch_wait_ms,
        window_cuda_graphs=factory_args.window_cuda_graphs,
        decode_dtype=factory_args.decode_dtype,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS_CAP,
    )


def test_the_vocoder_factory_loads_the_codec_on_its_device_with_the_windows(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    codec_file = tmp_path / "codec.pth"
    codec_file.write_bytes(b"")

    vocoder = cpu_vocoder(
        Drama3VocoderFactoryArgs(codec_path=str(codec_file)).model_copy(
            update={"window_cuda_graphs": False, "decode_dtype": "float32"}
        )
    )

    assert isinstance(vocoder, Drama3VocoderScheduler)
    assert codec_loads.calls == [(str(tmp_path), "cpu")]
    assert isinstance(vocoder.codec, FrameNumberCodec)
    assert vocoder.codec.is_weight_norm_folded
    assert vocoder.decode_window == vocoder.codec.from_indices
    assert (
        vocoder.stream_stride,
        vocoder.stream_followup_stride_ramp,
        vocoder.stream_followup_stride,
        vocoder.stream_overlap_tokens,
        vocoder.stream_crossfade_samples,
        vocoder.max_batch_size,
    ) == (10, (10, 12, 32), 64, 32, 0, 8)
    assert vocoder.max_padded_frames_per_decode is None


def test_window_cuda_graphs_on_a_cpu_device_are_refused(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    codec_file = tmp_path / "codec.pth"
    codec_file.write_bytes(b"")

    with pytest.raises(RuntimeError, match="window_cuda_graphs to false"):
        cpu_vocoder(
            Drama3VocoderFactoryArgs(codec_path=str(codec_file)).model_copy(
                update={"decode_dtype": "float32"}
            )
        )


def bytes_per_frame_decoder(bytes_per_frame: int) -> WindowDecoder:
    def decode(indices: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (indices.shape[0], 1, indices.shape[-1] * bytes_per_frame // 4),
            dtype=torch.float32,
            device=indices.device,
        )

    return decode


@pytest.mark.accelerator
def test_the_padded_frame_limit_is_the_share_left_after_held_memory() -> None:
    device = torch.device("cuda")
    bytes_per_frame = 4096
    share_bytes = int(0.01 * torch.cuda.get_device_properties(device).total_memory)
    torch.cuda.empty_cache()
    held_before = torch.cuda.memory_reserved(device)

    max_padded_frames_per_decode = stages.measure_max_padded_frames_per_decode(
        bytes_per_frame_decoder(bytes_per_frame),
        device=device,
        num_codebooks=4,
        total_gpu_memory_fraction=0.01,
        max_new_tokens=16,
    )

    held_after = torch.cuda.memory_reserved(device)
    assert (
        (share_bytes - held_after) // bytes_per_frame
        <= max_padded_frames_per_decode
        <= (share_bytes - held_before) // bytes_per_frame
    )


@pytest.mark.accelerator
def test_a_share_that_cannot_hold_one_longest_request_is_refused() -> None:
    with pytest.raises(ValueError, match="--vocoder.gpu_memory_fraction"):
        stages.measure_max_padded_frames_per_decode(
            bytes_per_frame_decoder(4096),
            device=torch.device("cuda"),
            num_codebooks=4,
            total_gpu_memory_fraction=0.01,
            max_new_tokens=10**9,
        )


class PreparedWaveformCodec:
    sample_rate: int = 44100

    def encode(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not torch.is_grad_enabled()
        assert torch.is_autocast_enabled("cpu")
        assert audios.shape == (1, self.sample_rate)
        assert audio_lengths.tolist() == [self.sample_rate]
        assert audios.max().item() == 1.0
        assert audios.min().item() == 1.0
        return torch.arange(20).reshape(1, 10, 2), audio_lengths


def test_reference_preparation_preserves_device_context_and_output() -> None:
    hook = FishReferenceEncodeHook(
        codec=PreparedWaveformCodec(),
        device="cpu",
        checkpoint_id="codec",
        autocast_dtype=torch.bfloat16,
        max_reference_audio_seconds=1.0,
    )
    stereo_audio = torch.stack((torch.ones(22050), torch.full((22050,), 3.0)))
    codes = hook.encode_reference_waveform(stereo_audio, 22050)
    assert codes.device.type == "cpu"
    assert torch.equal(codes, torch.arange(20).reshape(10, 2))
    with pytest.raises(ValueError, match="longer than"):
        hook.encode_reference_waveform(torch.ones(2, 22051), 22050)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_two_stream_factory_rejects_unsupported_device_or_precision_before_loading(
    codec_loads: CodecLoads,
    device: str,
) -> None:
    with pytest.raises(ValueError, match="requires CUDA and bfloat16"):
        stages.create_preprocessing_executor(
            "unused-checkpoint",
            device=device,
            gpu_id=0,
            max_concurrency=2,
            max_prompt_rows=64,
            codec_path="unused-codec",
            reference_encode_dtype="float32",
            reference_encode_cuda_streams=2,
            max_reference_audio_seconds=60.0,
        )
    assert codec_loads.calls == []


class CPUOnlyStream:
    def __init__(self, device: torch.device) -> None:
        assert device == torch.device("cuda:0")

    def synchronize(self) -> None:
        pass


@pytest.fixture
def cpu_cuda_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    def set_device(device: torch.device) -> None:
        assert device == torch.device("cuda:0")

    def device_context(device: torch.device) -> nullcontext[None]:
        assert device == torch.device("cuda:0")
        return nullcontext()

    def stream_context(stream: CPUOnlyStream) -> nullcontext[None]:
        assert isinstance(stream, CPUOnlyStream)
        return nullcontext()

    def autocast_context(
        device_type: str,
        *,
        enabled: bool = True,
        dtype: torch.dtype | None = None,
        cache_enabled: bool = True,
    ) -> nullcontext[None]:
        assert device_type == "cuda"
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch.cuda, "synchronize", set_device)
    monkeypatch.setattr(torch.cuda, "Stream", CPUOnlyStream)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch, "autocast", autocast_context)


class FoldableCPUCodec(torch.nn.Module):
    sample_rate: int = 44100

    def __init__(self) -> None:
        super().__init__()
        self.folded = False
        self.eval()

    def fold_weight_norm(self) -> None:
        self.folded = True


def test_factory_warmup_failure_closes_and_drains_real_owner(
    cpu_cuda_boundary: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed startup must not leak the other stream's in-flight warmup."""
    write_toy_checkpoint(tmp_path)
    codec = FoldableCPUCodec()
    both_workers = Barrier(2)
    slow_worker_entered = Event()
    release_worker = Event()
    failed_worker = Event()
    owners: list[ReferenceEncodeOwner] = []
    lock = Lock()
    calls = 0

    def encode_prepared(
        hook: stages.Drama3ReferenceEncodeHook,
        audios: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        with lock:
            calls += 1
            call_number = calls
        both_workers.wait(timeout=5)
        if call_number == 1:
            failed_worker.set()
            raise ValueError("warmup failed")
        else:
            slow_worker_entered.set()
            assert release_worker.wait(5)
            return torch.zeros(10, 1, dtype=torch.long)

    def create_owner(
        device: torch.device, device_encode: DeviceEncode
    ) -> ReferenceEncodeOwner:
        owner = ReferenceEncodeOwner(device, device_encode)
        owners.append(owner)
        return owner

    def load_codec(checkpoint_dir: str, device: str) -> FoldableCPUCodec:
        assert device == "cuda:0"
        return codec

    def create_factory() -> stages.ThreadedSimpleScheduler:
        return stages.create_preprocessing_executor(
            str(tmp_path / "checkpoint"),
            device="cuda",
            gpu_id=0,
            max_concurrency=2,
            max_prompt_rows=64,
            codec_path=str(tmp_path / "codec.pth"),
            reference_encode_dtype="bfloat16",
            reference_encode_cuda_streams=2,
            max_reference_audio_seconds=60.0,
        )

    monkeypatch.setattr(stages, "load_codec", load_codec)
    monkeypatch.setattr(stages, "ReferenceEncodeOwner", create_owner)
    monkeypatch.setattr(
        stages.Drama3ReferenceEncodeHook, "encode_prepared", encode_prepared
    )
    with ThreadPoolExecutor(max_workers=1) as caller:
        result = caller.submit(create_factory)
        try:
            assert slow_worker_entered.wait(5)
            assert failed_worker.wait(5)
            assert not result.done()
        finally:
            release_worker.set()
        with pytest.raises(ValueError, match="warmup failed"):
            result.result(timeout=5)
    assert codec.folded
    assert owners[0].is_closed
    assert owners[0].outstanding_jobs == 0
    with pytest.raises(RuntimeError, match="closed"):
        owners[0].encode(torch.zeros(1), torch.ones(1), duration_seconds=1)


class LifecyclePreprocessor:
    def __init__(self, hook: stages.Drama3ReferenceEncodeHook) -> None:
        self.reference_encoder = ReferenceEncodeService(hook)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return self.reference_encoder.hook.encode_prepared_waveform(
            audio, torch.tensor([1])
        )


def test_scheduler_stop_cancels_pending_before_closing_and_drains_running_owner(
    cpu_cuda_boundary: None,
) -> None:
    """Stopping intake cancels queued preprocessing and waits for active GPU work."""
    entered = Event()
    release = Event()
    close_entered = Event()
    pending: Future[torch.Tensor]

    def encode(audios: torch.Tensor, audio_lengths: torch.Tensor) -> torch.Tensor:
        entered.set()
        assert release.wait(5)
        return audios.long()

    class ObservedHook(stages.Drama3ReferenceEncodeHook):
        def close(self) -> None:
            assert pending.cancelled()
            close_entered.set()
            super().close()

    hook = ObservedHook(codec=PreparedWaveformCodec(), checkpoint_id="codec")
    hook.owner = ReferenceEncodeOwner(torch.device("cuda:0"), encode)
    scheduler = stages.Drama3PreprocessingScheduler(
        LifecyclePreprocessor(hook), max_concurrency=1
    )
    running = scheduler.executor.submit(scheduler.run_one, torch.ones(1, 1))
    assert entered.wait(5)
    pending = scheduler.executor.submit(scheduler.run_one, torch.ones(1, 1))
    with ThreadPoolExecutor(max_workers=1) as closer:
        stopped = closer.submit(scheduler.stop)
        try:
            assert close_entered.wait(5)
            assert not stopped.done()
            assert not running.done()
        finally:
            release.set()
        stopped.result(timeout=5)
    assert torch.equal(running.result(timeout=5), torch.ones(1, 1, dtype=torch.long))
    assert hook.owner.is_closed
    scheduler.stop()
    scheduler.executor.shutdown(wait=True)


def test_scheduler_start_finally_closes_service_when_dispatch_fails(
    cpu_cuda_boundary: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scheduler loop exception must close its reference owner, not just its CPU pool."""
    hook = stages.Drama3ReferenceEncodeHook(
        codec=PreparedWaveformCodec(), checkpoint_id="codec"
    )

    def encode(audios: torch.Tensor, audio_lengths: torch.Tensor) -> torch.Tensor:
        return audios.long()

    hook.owner = ReferenceEncodeOwner(torch.device("cuda:0"), encode)
    scheduler = stages.Drama3PreprocessingScheduler(
        LifecyclePreprocessor(hook), max_concurrency=1
    )

    def fail_start() -> None:
        raise ValueError("dispatch failed")

    monkeypatch.setattr(scheduler, "wait_for_capacity", fail_start)
    with pytest.raises(ValueError, match="dispatch failed"):
        scheduler.start()
    assert hook.owner.is_closed
    with pytest.raises(RuntimeError, match="shutdown"):
        scheduler.executor.submit(encode, torch.ones(1), torch.ones(1))
