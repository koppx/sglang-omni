# SPDX-License-Identifier: Apache-2.0
"""Bounded, stream-owning reference encoding for the opt-in Drama pipeline."""

from __future__ import annotations

import math
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier, Condition, Event, local
from typing import Protocol

import torch

WORKER_COUNT = 2
MAX_OUTSTANDING_JOBS = 8
SHORT_REFERENCE_SECONDS = 10.0
WORKER_START_TIMEOUT_SECONDS = 10.0


class DeviceEncode(Protocol):
    def __call__(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor: ...


class WorkerState(local):
    stream: torch.cuda.Stream


@dataclass(eq=False, kw_only=True)
class Admission:
    permits: int


def drain_future(encode_future: Future[torch.Tensor]) -> None:
    """Wait for completion even when shutdown cancels a queued future."""
    completed = Event()

    def notify_completed(completed_future: Future[torch.Tensor]) -> None:
        assert completed_future.done()
        completed.set()

    encode_future.add_done_callback(notify_completed)
    completed.wait()


class ReferenceEncodeOwner:
    """Run prepared CPU audio through two explicitly placed CUDA workers.

    The factory must fold an eval-only, FP32, stateless codec and synchronize it before construction.
    """

    def __init__(self, device: torch.device, device_encode: DeviceEncode) -> None:
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                "reference encoding requires an explicit CUDA device index"
            )
        else:
            pass
        self.device: torch.device = device
        self.device_encode: DeviceEncode = device_encode
        self.worker_state: WorkerState = WorkerState()
        self.condition: Condition = Condition()
        self.waiting: deque[Admission] = deque()
        self.outstanding_jobs: int = 0
        self.active_permits: int = 0
        self.is_closing: bool = False
        self.is_closed: bool = False
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=WORKER_COUNT,
            thread_name_prefix="drama-reference",
            initializer=self.initialize_worker,
        )
        startup_barrier = Barrier(WORKER_COUNT)

        def await_workers() -> None:
            startup_barrier.wait(timeout=WORKER_START_TIMEOUT_SECONDS)

        try:
            startup_futures = [
                self.executor.submit(await_workers) for _ in range(WORKER_COUNT)
            ]
            for startup_future in startup_futures:
                startup_future.result(timeout=2 * WORKER_START_TIMEOUT_SECONDS)
        except BaseException:
            startup_barrier.abort()
            self.close()
            raise

    def initialize_worker(self) -> None:
        torch.cuda.set_device(self.device)
        self.worker_state.stream = torch.cuda.Stream(device=self.device)

    def encode(
        self,
        audios: torch.Tensor,
        audio_lengths: torch.Tensor,
        *,
        duration_seconds: float,
    ) -> torch.Tensor:
        """Reject overflow immediately; otherwise admit FIFO and wait for CPU codes."""
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError(
                "reference duration_seconds must be finite and nonnegative"
            )
        elif audios.device.type != "cpu" or audio_lengths.device.type != "cpu":
            raise ValueError("reference encoding expects prepared CPU tensors")
        else:
            pass
        admission = Admission(
            permits=1 if duration_seconds <= SHORT_REFERENCE_SECONDS else WORKER_COUNT
        )
        with self.condition:
            if self.is_closing:
                raise RuntimeError("reference encoder is closed")
            elif self.outstanding_jobs >= MAX_OUTSTANDING_JOBS:
                raise RuntimeError("reference encoder has eight outstanding jobs")
            else:
                self.outstanding_jobs += 1
                self.waiting.append(admission)
                self.condition.notify_all()
            try:
                self.condition.wait_for(
                    lambda: self.is_closing
                    or (
                        self.waiting[0] is admission
                        and self.active_permits + admission.permits <= WORKER_COUNT
                    )
                )
                if self.is_closing:
                    raise RuntimeError("reference encoder is closed")
                else:
                    encode_future = self.executor.submit(
                        self.encode_on_worker, audios, audio_lengths
                    )
                    self.waiting.popleft()
                    self.active_permits += admission.permits
                    self.condition.notify_all()
            except BaseException:
                self.waiting.remove(admission)
                self.outstanding_jobs -= 1
                self.condition.notify_all()
                raise
        try:
            return encode_future.result()
        finally:
            drain_future(encode_future)
            with self.condition:
                self.active_permits -= admission.permits
                self.outstanding_jobs -= 1
                self.condition.notify_all()

    def warmup(self, audios: torch.Tensor, audio_lengths: torch.Tensor) -> None:
        """Exercise the real encode on both streams before publishing the owner."""
        warmup_barrier = Barrier(WORKER_COUNT)

        def warmup_on_worker() -> torch.Tensor:
            warmup_barrier.wait(timeout=WORKER_START_TIMEOUT_SECONDS)
            return self.encode_on_worker(audios, audio_lengths)

        warmup_futures: list[Future[torch.Tensor]] = []
        with self.condition:
            if self.is_closing:
                raise RuntimeError("reference encoder is closed")
            elif self.outstanding_jobs:
                raise RuntimeError("reference warmup requires an idle owner")
            else:
                self.outstanding_jobs = 1
                self.active_permits = WORKER_COUNT
        try:
            with self.condition:
                for _ in range(WORKER_COUNT):
                    warmup_futures.append(self.executor.submit(warmup_on_worker))
            for warmup_future in warmup_futures:
                warmup_future.result()
        finally:
            warmup_barrier.abort()
            for warmup_future in warmup_futures:
                drain_future(warmup_future)
            with self.condition:
                self.active_permits = 0
                self.outstanding_jobs -= 1
                self.condition.notify_all()

    def encode_on_worker(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        stream = self.worker_state.stream
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(stream),
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            try:
                return self.device_encode(audios, audio_lengths).cpu()
            finally:
                # note (koppx): Failed encodes can still leave queued GPU work using shared weights.
                stream.synchronize()

    def close(self) -> None:
        """Wake admission waiters, cancel pending work and drain active streams."""
        with self.condition:
            if self.is_closing:
                self.condition.wait_for(lambda: self.is_closed)
                return
            else:
                self.is_closing = True
                self.condition.notify_all()
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            with self.condition:
                self.is_closed = True
                self.condition.notify_all()
