# SPDX-License-Identifier: Apache-2.0
"""CPU admission and lifecycle coverage with only CUDA boundaries replaced."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, Event, Lock, get_ident
from typing import Protocol

import pytest
import torch

from sglang_omni.models.fishaudio_drama3.reference_encode import ReferenceEncodeOwner

TIMEOUT_SECONDS = 5


class DeviceEncodeWarmup(Protocol):
    def __call__(self) -> torch.Tensor: ...


class FakeStream:
    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device
        self.thread_id: int = get_ident()

    def synchronize(self) -> None:
        assert get_ident() == self.thread_id


class GatedEncode:
    def __init__(self) -> None:
        self.entered: list[Event] = [Event() for _ in range(12)]
        self.release: list[Event] = [Event() for _ in range(12)]
        self.order: list[int] = []
        self.lock: Lock = Lock()
        self.failure_ids: set[int] = set()
        self.thread_ids: set[int] = set()
        self.warmup_barrier: Barrier | None = None

    def __call__(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        assert audio_lengths.device.type == "cpu"
        request_id = int(audios.item())
        with self.lock:
            self.order.append(request_id)
            self.thread_ids.add(get_ident())
        self.entered[request_id].set()
        if self.warmup_barrier is not None:
            self.warmup_barrier.wait(timeout=TIMEOUT_SECONDS)
        else:
            pass
        assert self.release[request_id].wait(TIMEOUT_SECONDS)
        if request_id in self.failure_ids:
            raise ValueError("encode failed")
        else:
            return audios.long()


@pytest.fixture
def cuda_boundary(monkeypatch: pytest.MonkeyPatch) -> list[torch.device]:
    devices: list[torch.device] = []

    def set_device(device: torch.device) -> None:
        devices.append(device)

    def device_context(device: torch.device) -> nullcontext[None]:
        assert device == torch.device("cuda:3")
        return nullcontext()

    def stream_context(stream: FakeStream) -> nullcontext[None]:
        assert stream.thread_id == get_ident()
        return nullcontext()

    def autocast_context(
        device_type: str, *, dtype: torch.dtype, cache_enabled: bool
    ) -> nullcontext[None]:
        assert device_type == "cuda"
        assert dtype == torch.bfloat16
        assert cache_enabled is False
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch, "autocast", autocast_context)
    return devices


def test_short_overlap_long_fifo_and_boundaries(
    cuda_boundary: list[torch.device],
) -> None:
    """A queued exclusive reference cannot be overtaken by later short inputs."""
    callback = GatedEncode()
    owner = ReferenceEncodeOwner(torch.device("cuda:3"), callback)
    lengths = torch.tensor([1])
    try:
        with ThreadPoolExecutor(max_workers=4) as callers:
            first = callers.submit(
                owner.encode, torch.tensor([0]), lengths, duration_seconds=9.99
            )
            second = callers.submit(
                owner.encode, torch.tensor([1]), lengths, duration_seconds=10.0
            )
            assert callback.entered[0].wait(TIMEOUT_SECONDS)
            assert callback.entered[1].wait(TIMEOUT_SECONDS)
            long = callers.submit(
                owner.encode, torch.tensor([2]), lengths, duration_seconds=10.01
            )
            with owner.condition:
                assert owner.condition.wait_for(
                    lambda: owner.outstanding_jobs == 3, TIMEOUT_SECONDS
                )
            later = callers.submit(
                owner.encode, torch.tensor([3]), lengths, duration_seconds=1
            )
            with owner.condition:
                assert owner.condition.wait_for(
                    lambda: owner.outstanding_jobs == 4, TIMEOUT_SECONDS
                )
            callback.release[0].set()
            assert first.result(TIMEOUT_SECONDS).item() == 0
            assert not callback.entered[2].is_set()
            assert not callback.entered[3].is_set()
            callback.release[1].set()
            assert second.result(TIMEOUT_SECONDS).item() == 1
            assert callback.entered[2].wait(TIMEOUT_SECONDS)
            assert not callback.entered[3].is_set()
            callback.release[2].set()
            assert long.result(TIMEOUT_SECONDS).item() == 2
            assert callback.entered[3].wait(TIMEOUT_SECONDS)
            callback.release[3].set()
            assert later.result(TIMEOUT_SECONDS).item() == 3
    finally:
        for release in callback.release:
            release.set()
        owner.close()
    assert cuda_boundary == [torch.device("cuda:3")] * 2


def test_overflow_close_wakes_pending_and_drains_running(
    cuda_boundary: list[torch.device],
) -> None:
    """Closing releases queued callers but waits for already running callbacks."""
    callback = GatedEncode()
    owner = ReferenceEncodeOwner(torch.device("cuda:3"), callback)
    lengths = torch.tensor([1])
    with ThreadPoolExecutor(max_workers=10) as callers:
        running = callers.submit(
            owner.encode, torch.tensor([0]), lengths, duration_seconds=11
        )
        assert callback.entered[0].wait(TIMEOUT_SECONDS)
        pending = [
            callers.submit(owner.encode, torch.tensor([i]), lengths, duration_seconds=1)
            for i in range(1, 8)
        ]
        with owner.condition:
            assert owner.condition.wait_for(
                lambda: owner.outstanding_jobs == 8, TIMEOUT_SECONDS
            )
        with pytest.raises(RuntimeError, match="eight outstanding"):
            owner.encode(torch.tensor([8]), lengths, duration_seconds=1)
        closing = callers.submit(owner.close)
        closing_again = callers.submit(owner.close)
        try:
            for pending_future in pending:
                with pytest.raises(RuntimeError, match="closed"):
                    pending_future.result(TIMEOUT_SECONDS)
            assert not closing.done()
            assert not closing_again.done()
            with pytest.raises(RuntimeError, match="closed"):
                owner.encode(torch.tensor([9]), lengths, duration_seconds=1)
        finally:
            callback.release[0].set()
        assert running.result(TIMEOUT_SECONDS).item() == 0
        closing.result(TIMEOUT_SECONDS)
        closing_again.result(TIMEOUT_SECONDS)
    owner.close()
    assert callback.order == [0]


def test_callback_failure_reclaims_exclusive_admission(
    cuda_boundary: list[torch.device],
) -> None:
    callback = GatedEncode()
    callback.failure_ids.add(0)
    callback.release[0].set()
    callback.release[1].set()
    owner = ReferenceEncodeOwner(torch.device("cuda:3"), callback)
    try:
        with pytest.raises(ValueError, match="encode failed"):
            owner.encode(torch.tensor([0]), torch.tensor([1]), duration_seconds=60)
        assert (
            owner.encode(
                torch.tensor([1]), torch.tensor([1]), duration_seconds=60
            ).item()
            == 1
        )
    finally:
        owner.close()


def test_warmup_executes_both_workers_and_releases_failure(
    cuda_boundary: list[torch.device],
) -> None:
    callback = GatedEncode()
    callback.release[0].set()
    callback.warmup_barrier = Barrier(2)
    callback.failure_ids.add(0)
    owner = ReferenceEncodeOwner(torch.device("cuda:3"), callback)
    try:
        with pytest.raises(ValueError, match="encode failed"):
            owner.warmup(torch.tensor([0]), torch.tensor([1]))
        assert len(callback.thread_ids) == 2
        callback.warmup_barrier = None
        callback.release[1].set()
        assert (
            owner.encode(
                torch.tensor([1]), torch.tensor([1]), duration_seconds=1
            ).item()
            == 1
        )
    finally:
        owner.close()


def test_failed_gpu_work_drains_before_releasing_admission(
    cuda_boundary: list[torch.device], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An encode exception cannot admit new work before its stream completes."""
    synchronize_entered = Event()
    synchronize_release = Event()

    def synchronize(stream: FakeStream) -> None:
        assert stream.thread_id == get_ident()
        synchronize_entered.set()
        assert synchronize_release.wait(TIMEOUT_SECONDS)

    monkeypatch.setattr(FakeStream, "synchronize", synchronize)
    callback = GatedEncode()
    callback.failure_ids.add(0)
    callback.release[0].set()
    callback.release[1].set()
    owner = ReferenceEncodeOwner(torch.device("cuda:3"), callback)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            failing = callers.submit(
                owner.encode, torch.tensor([0]), torch.tensor([1]), duration_seconds=60
            )
            assert synchronize_entered.wait(TIMEOUT_SECONDS)
            later = callers.submit(
                owner.encode, torch.tensor([1]), torch.tensor([1]), duration_seconds=60
            )
            try:
                with owner.condition:
                    assert owner.condition.wait_for(
                        lambda: owner.outstanding_jobs == 2, TIMEOUT_SECONDS
                    )
                assert not callback.entered[1].is_set()
                assert not failing.done()
            finally:
                synchronize_release.set()
            with pytest.raises(ValueError, match="encode failed"):
                failing.result(TIMEOUT_SECONDS)
            assert later.result(TIMEOUT_SECONDS).item() == 1
    finally:
        synchronize_release.set()
        owner.close()


def test_close_cancels_submitted_job_without_hanging_caller(
    cuda_boundary: list[torch.device],
) -> None:
    """A shutdown-cancelled queued future used to hang admission cleanup forever."""
    callback = GatedEncode()
    owner = ReferenceEncodeOwner(torch.device("cuda:3"), callback)
    workers_entered = Barrier(3)
    release_workers = Event()

    def occupy_worker() -> None:
        workers_entered.wait(timeout=TIMEOUT_SECONDS)
        assert release_workers.wait(TIMEOUT_SECONDS)

    blockers = [owner.executor.submit(occupy_worker) for _ in range(2)]
    workers_entered.wait(timeout=TIMEOUT_SECONDS)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            queued = callers.submit(
                owner.encode, torch.tensor([0]), torch.tensor([1]), duration_seconds=1
            )
            with owner.condition:
                assert owner.condition.wait_for(
                    lambda: owner.active_permits == 1, TIMEOUT_SECONDS
                )
            closing = callers.submit(owner.close)
            try:
                with pytest.raises(CancelledError):
                    queued.result(TIMEOUT_SECONDS)
                assert not closing.done()
                assert not callback.order
                with owner.condition:
                    assert owner.outstanding_jobs == 0
                    assert owner.active_permits == 0
            finally:
                release_workers.set()
            closing.result(TIMEOUT_SECONDS)
    finally:
        release_workers.set()
        for blocker in blockers:
            blocker.result(TIMEOUT_SECONDS)
        owner.close()


def test_partial_warmup_submission_failure_drains_first_worker(
    cuda_boundary: list[torch.device], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second submit failure must not strand the first warmup at its barrier."""
    callback = GatedEncode()
    owner = ReferenceEncodeOwner(torch.device("cuda:3"), callback)
    original_submit = owner.executor.submit
    submitted: list[Future[torch.Tensor]] = []

    def submit_warmup(warmup_encode: DeviceEncodeWarmup) -> Future[torch.Tensor]:
        if submitted:
            raise RuntimeError("second submit failed")
        else:
            warmup_future = original_submit(warmup_encode)
            submitted.append(warmup_future)
            return warmup_future

    monkeypatch.setattr(owner.executor, "submit", submit_warmup)
    try:
        with pytest.raises(RuntimeError, match="second submit failed"):
            owner.warmup(torch.tensor([0]), torch.tensor([1]))
        assert len(submitted) == 1
        assert submitted[0].done()
        with owner.condition:
            assert owner.outstanding_jobs == 0
            assert owner.active_permits == 0
    finally:
        owner.close()


def test_worker_initialization_failure_is_propagated(
    cuda_boundary: list[torch.device], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_device(device: torch.device) -> None:
        raise RuntimeError(f"cannot initialize {device}")

    monkeypatch.setattr(torch.cuda, "set_device", fail_device)
    with pytest.raises(RuntimeError):
        ReferenceEncodeOwner(torch.device("cuda:3"), GatedEncode())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_cuda_device_required(device: str) -> None:
    with pytest.raises(ValueError, match="explicit CUDA"):
        ReferenceEncodeOwner(torch.device(device), GatedEncode())


@pytest.mark.parametrize("duration_seconds", [-1, float("inf"), float("nan")])
def test_invalid_duration_does_not_consume_admission(
    cuda_boundary: list[torch.device], duration_seconds: float
) -> None:
    callback = GatedEncode()
    callback.release[0].set()
    owner = ReferenceEncodeOwner(torch.device("cuda:3"), callback)
    try:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            owner.encode(
                torch.tensor([0]), torch.tensor([1]), duration_seconds=duration_seconds
            )
        assert (
            owner.encode(
                torch.tensor([0]), torch.tensor([1]), duration_seconds=1
            ).item()
            == 0
        )
    finally:
        owner.close()
