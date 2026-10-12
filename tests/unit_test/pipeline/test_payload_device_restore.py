# SPDX-License-Identifier: Apache-2.0
"""What stage_io fixes up when a payload crosses a process boundary.

An accelerator-origin tensor relayed over host shm must land on the receiver's
card: only the sender's device string travels with the payload, so the receiver
has to supply its own device, and using the sender's index would target the wrong
card. Device events must not travel at all; they only order readers inside the
process that recorded them.
"""

from __future__ import annotations

import asyncio
import gc
import multiprocessing
import threading
import weakref
from multiprocessing.connection import Connection
from typing import Literal

import pytest
import torch

import sglang_omni.comm.stage_io as stage_io
from sglang_omni.comm.stage_io import (
    read_direct_cuda_ipc_payload,
    restore_tensor_device,
    serialize_direct_cuda_ipc_payload,
    strip_process_local_metadata,
)
from sglang_omni.proto.request import OmniRequest, StagePayload


class BackendEvent(torch.Event):
    """A vendor event class, as torch.xpu.Event and torch.npu.Event are."""


def test_host_origin_tensor_stays_on_the_host() -> None:
    restored = restore_tensor_device(torch.empty(2), "cpu", "xpu:1")

    assert restored.device.type == "cpu"


def test_already_resident_tensor_is_left_alone() -> None:
    """A cuda_ipc payload arrives on the accelerator, so it must not be copied."""
    tensor = torch.empty(2, device="meta")

    restored = restore_tensor_device(tensor, "meta:0", "meta:1")

    assert restored is tensor


def test_accelerator_origin_tensor_moves_to_the_receiver_device() -> None:
    """The receiver's index wins over the sender's."""
    restored = restore_tensor_device(torch.empty(2), "meta:3", "meta:1")

    assert restored.device.type == "meta"


def test_host_only_stage_keeps_an_accelerator_origin_tensor_on_the_host() -> None:
    """A stage with no card assigned consumes the host copy."""
    restored = restore_tensor_device(torch.empty(2), "meta:3", None)

    assert restored.device.type == "cpu"


def test_a_raw_stream_ref_carries_its_source_device() -> None:
    """A raw DataRef had nowhere to record residency, so a host-shm hop lost it.
    Qwen3-TTS emits device-resident codec chunks, and a process-isolated vocoder
    (--vocoder.process vocoder) would otherwise feed CPU codes to an accelerator decoder.
    """
    from sglang_omni.comm.data_ref import (
        BackendRef,
        DataKind,
        DataLayout,
        DataRef,
        TransportKind,
    )

    ref = DataRef(
        version=1,
        kind=DataKind.STREAM_CHUNK,
        object_id="req:stream:talker:vocoder:0",
        transport=TransportKind.SHM,
        layout=DataLayout.RAW_TENSOR,
        buffer=BackendRef(transport=TransportKind.SHM, info={}, length=16),
        shape=(2,),
        dtype="torch.int64",
        device="meta:3",
        offset=0,
    )

    assert DataRef.from_dict(ref.to_dict()).device == "meta:3"


def test_a_stream_ref_without_a_device_stays_wire_compatible() -> None:
    """An older ref records no device, so from_dict yields None and nothing moves."""
    from sglang_omni.comm.data_ref import (
        BackendRef,
        DataKind,
        DataLayout,
        DataRef,
        TransportKind,
    )

    payload = DataRef(
        version=1,
        kind=DataKind.STREAM_CHUNK,
        object_id="req:stream:talker:vocoder:0",
        transport=TransportKind.SHM,
        layout=DataLayout.RAW_TENSOR,
        buffer=BackendRef(transport=TransportKind.SHM, info={}, length=16),
        shape=(2,),
        dtype="torch.int64",
        offset=0,
    ).to_dict()

    assert "device" not in payload
    assert DataRef.from_dict(payload).device is None


async def stream_round_trip(local_device: str | None, *, with_metadata: bool):
    """Write a chunk through a host-shm relay and read it back."""
    from sglang_omni.comm import stage_io
    from sglang_omni.comm.data_ref import TransportKind
    from tests.unit_test.fixtures.pipeline_fakes import FakeRelay

    relay = FakeRelay(device="cpu")
    codes = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    metadata = (
        {"ref_code": torch.tensor([7, 8], dtype=torch.long)} if with_metadata else None
    )

    data_ref, _ = await stage_io.write_stream_chunk(
        relay,
        request_id="req-1",
        data=codes,
        target_stage="vocoder",
        from_stage="talker",
        chunk_id=0,
        metadata=metadata,
        transport=TransportKind.SHM,
    )
    return data_ref, await stage_io.read_stream_chunk(relay, data_ref, local_device)


def test_a_metadata_bearing_chunk_keeps_its_source_device() -> None:
    """Qwen3-TTS attaches metadata to every codec chunk, and the metadata rebuild
    used to drop the outer device, so an isolated vocoder got the shm CPU tensor.
    """
    import asyncio

    data_ref, (data, metadata) = asyncio.run(
        stream_round_trip(None, with_metadata=True)
    )

    assert data_ref.device == "cpu"
    assert data.tolist() == [1, 2, 3, 4]
    assert metadata is not None and metadata["ref_code"].tolist() == [7, 8]


def test_a_metadata_tensor_records_its_own_source_device() -> None:
    """Nested refs are restored from their own recorded device, not the outer one."""
    import asyncio

    data_ref, _ = asyncio.run(stream_round_trip(None, with_metadata=True))

    assert data_ref.metadata_tensors
    assert all(ref.ref.device == "cpu" for ref in data_ref.metadata_tensors)


def test_a_chunk_without_metadata_still_records_its_device() -> None:
    import asyncio

    data_ref, _ = asyncio.run(stream_round_trip(None, with_metadata=False))

    assert data_ref.device == "cpu"


def test_a_backend_event_is_stripped_and_the_rest_survives() -> None:
    """A device event means nothing in the receiving process, and the transport
    establishes readiness itself once the payload crosses over."""
    metadata = {
        "codes_ready_event": BackendEvent(),
        "ref_code_len": 7,
        "sample_rate": 24000,
    }

    stripped = strip_process_local_metadata(metadata)

    assert stripped == {"ref_code_len": 7, "sample_rate": 24000}


def test_metadata_without_events_is_returned_whole() -> None:
    assert strip_process_local_metadata({"ref_code_len": 0}) == {"ref_code_len": 0}
    assert strip_process_local_metadata(None) is None


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
@pytest.mark.parametrize("fail_after_release", [False, True])
def test_ipc_receive_interrupt_discards_late_result_safely(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    interruption: Literal["timeout", "cancel"],
    fail_after_release: bool,
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    payload_references: list[weakref.ReferenceType[StagePayload]] = []

    def deserialize(data_ref: dict[str, object]) -> StagePayload:
        assert data_ref["materialize_cuda_tensors"] is True
        payload = StagePayload(
            request_id="late", request=OmniRequest(inputs=None), data={}
        )
        payload_references.append(weakref.ref(payload))
        started.set()
        try:
            assert release.wait(5)
            if fail_after_release:
                raise RuntimeError("late receive failure")
            else:
                return payload
        finally:
            finished.set()

    monkeypatch.setattr(stage_io, "deserialize_direct_cuda_ipc_payload", deserialize)

    async def run() -> None:
        receive_task = asyncio.create_task(
            read_direct_cuda_ipc_payload(
                {"materialize_cuda_tensors": True},
                timeout_seconds=0.1 if interruption == "timeout" else 5.0,
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            if interruption == "cancel":
                receive_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await receive_task
                receive_task.cancel()
            else:
                with pytest.raises(TimeoutError, match="CUDA IPC receive exceeded"):
                    await receive_task
            assert payload_references[0]() is not None
            assert not finished.is_set()
        finally:
            release.set()
        assert await asyncio.to_thread(finished.wait, 3)
        deadline = asyncio.get_running_loop().time() + 3
        while payload_references[0]() is not None:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
            gc.collect()
        if fail_after_release:
            assert "Abandoned CUDA IPC receive failed" in caplog.text
        else:
            pass

    asyncio.run(run())


@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_ipc_receive_preserves_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, materialize: bool, fail: bool
) -> None:
    payload = StagePayload(
        request_id="ready", request=OmniRequest(inputs=None), data={}
    )

    def deserialize(data_ref: dict[str, object]) -> StagePayload:
        assert data_ref["materialize_cuda_tensors"] is materialize
        if fail:
            raise ValueError("invalid IPC payload")
        else:
            return payload

    monkeypatch.setattr(stage_io, "deserialize_direct_cuda_ipc_payload", deserialize)
    if fail:
        with pytest.raises(ValueError, match="invalid IPC payload"):
            asyncio.run(
                read_direct_cuda_ipc_payload(
                    {"materialize_cuda_tensors": materialize}, timeout_seconds=1
                )
            )
    else:
        assert (
            asyncio.run(
                read_direct_cuda_ipc_payload(
                    {"materialize_cuda_tensors": materialize}, timeout_seconds=1
                )
            )
            is payload
        )


@pytest.mark.parametrize("timeout_seconds", [0, -1, float("nan"), float("inf")])
def test_ipc_receive_rejects_invalid_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        asyncio.run(read_direct_cuda_ipc_payload({}, timeout_seconds=timeout_seconds))


def receive_delayed_cuda_payloads(connection: Connection) -> None:
    """Import only after the producer has dropped its Python tensor reference."""
    try:
        for _ in range(8):
            data_ref = connection.recv()
            connection.send("waiting")
            assert connection.recv() == "receive"
            payload = asyncio.run(
                read_direct_cuda_ipc_payload(data_ref, timeout_seconds=30.0)
            )
            consumer = torch.cuda.Stream()
            with torch.cuda.stream(consumer):
                hidden_states = payload.data["hidden_states"]
                hidden_states.record_stream(consumer)
                result = hidden_states.sum()
            del payload, hidden_states
            with torch.cuda.stream(consumer):
                connection.send(result.item())
        connection.send("done")
    finally:
        connection.close()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA IPC")
def test_cuda_payload_survives_delayed_cross_process_consumption() -> None:
    context = multiprocessing.get_context("spawn")
    sender, receiver = context.Pipe()
    process = context.Process(target=receive_delayed_cuda_payloads, args=(receiver,))
    process.start()
    receiver.close()
    producer = torch.cuda.Stream()
    export = torch.cuda.Stream()
    baseline_bytes = torch.cuda.memory_allocated()
    try:
        for value in range(8):
            with torch.cuda.stream(producer):
                payload = StagePayload(
                    request_id=f"ipc-{value}",
                    request=OmniRequest(inputs=None),
                    data={
                        "hidden_states": torch.full(
                            (64, 1024), float(value), device="cuda"
                        )
                    },
                )
                ready_event = torch.cuda.Event()
                ready_event.record(producer)
                payload.tensor_ready_events = [ready_event]
            with torch.cuda.stream(export):
                data_ref = serialize_direct_cuda_ipc_payload(payload)
            assert data_ref["materialize_cuda_tensors"]
            sender.send(data_ref)
            assert sender.poll(120)
            assert sender.recv() == "waiting"
            del payload
            gc.collect()
            sender.send("receive")
            assert sender.poll(120)
            assert sender.recv() == float(value * 64 * 1024)
        assert sender.poll(120)
        assert sender.recv() == "done"
        process.join(120)
        assert process.exitcode == 0
        gc.collect()
        torch.cuda.ipc_collect()
        assert torch.cuda.memory_allocated() <= baseline_bytes
    finally:
        sender.close()
        if process.is_alive():
            process.terminate()
            process.join(10)
        else:
            pass
