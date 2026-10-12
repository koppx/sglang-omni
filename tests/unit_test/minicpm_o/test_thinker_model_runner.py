# SPDX-License-Identifier: Apache-2.0
"""Thinker length penalty: scale EOS logits on the sync and lookahead paths."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

EOS_TOKEN_IDS = [1, 3]


def bare_runner() -> MiniCPMOThinkerModelRunner:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.eos_token_ids = EOS_TOKEN_IDS
    runner.eos_token_id_cache = None
    runner.pending_hidden = {}
    runner.enable_gpu_handoff = False
    return runner


def scheduler_request(params: dict[str, Any]) -> SimpleNamespace:
    payload = StagePayload(
        request_id="request-0",
        request=OmniRequest(
            inputs=None,
            params=params,
            metadata={"output_modalities": ["text"]},
        ),
        data=None,
    )
    return SimpleNamespace(data=SimpleNamespace(stage_payload=payload))


def penalized_request(length_penalty: float) -> SimpleNamespace:
    return scheduler_request(
        {"stage_params": {"thinker": {"length_penalty": length_penalty}}}
    )


def test_length_penalty_scales_eos_logits_per_request() -> None:
    original = torch.tensor(
        [
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
        ]
    )
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(
        logits_output,
        [
            penalized_request(1.0),
            penalized_request(2.0),
            penalized_request(1.0),
        ],
    )

    torch.testing.assert_close(logits_output.next_token_logits[0], original[0])
    torch.testing.assert_close(logits_output.next_token_logits[2], original[2])
    expected = original[1].clone()
    expected[EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_output.next_token_logits[1], expected)


def test_lookahead_samples_from_penalized_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.tensor([[0.5, 2.0, -1.0, -4.0]])
    logits_output = SimpleNamespace(next_token_logits=original.clone())
    logits_seen_by_parent = []

    def parent_sample_lookahead(self, logits_output, forward_batch, requests):
        logits_seen_by_parent.append(logits_output.next_token_logits.clone())
        return torch.tensor([0])

    monkeypatch.setattr(ThinkerModelRunner, "sample_lookahead", parent_sample_lookahead)

    bare_runner().sample_lookahead(
        logits_output, forward_batch=None, requests=[penalized_request(2.0)]
    )

    expected = original.clone()
    expected[0, EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_seen_by_parent[0], expected)


def test_missing_length_penalty_leaves_logits_unchanged() -> None:
    original = torch.tensor([[0.5, 2.0, -1.0, -4.0]])
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(logits_output, [scheduler_request({})])

    torch.testing.assert_close(logits_output.next_token_logits, original)


@pytest.mark.parametrize("enable_gpu_handoff", [False, True])
@pytest.mark.parametrize("known_tokens", [None, [11, 12], []])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires CUDA"
                ),
            ],
        ),
    ],
)
def test_finished_hidden_preserves_rows_and_dtype(
    enable_gpu_handoff: bool, known_tokens: list[int] | None, device: str
) -> None:
    runner = bare_runner()
    runner.enable_gpu_handoff = enable_gpu_handoff
    captured = torch.arange(20, dtype=torch.bfloat16, device=device).reshape(5, 4)
    runner.pending_hidden["speech"] = list(captured.unbind(0))
    payload = StagePayload(
        request_id="speech",
        request=OmniRequest(inputs=None),
        data={"prompt": {"known_tts_output_ids": known_tokens}},
    )
    request_data = SGLangARRequestData(stage_payload=payload)
    runner.on_request_finished("speech", request_data)
    hidden_states = request_data.extra_model_outputs["hidden_states_seq"]
    expected = (
        captured if known_tokens is None else captured[-len(known_tokens) - 1 : -1]
    )
    assert isinstance(hidden_states, torch.Tensor)
    assert hidden_states.is_contiguous()
    assert hidden_states.device.type == (device if enable_gpu_handoff else "cpu")
    assert hidden_states.dtype == captured.dtype
    assert torch.equal(hidden_states.cpu(), expected.cpu())
    assert bool(payload.tensor_ready_events) == (
        enable_gpu_handoff and device == "cuda"
    )
    assert "speech" not in runner.pending_hidden


def test_abort_discards_hidden_without_terminal_output() -> None:
    runner = bare_runner()
    runner.pending_hidden["aborted"] = [torch.ones(4)]
    runner.reset_request("aborted")
    request_data = SGLangARRequestData()
    runner.on_request_finished("aborted", request_data)
    assert request_data.extra_model_outputs == {}


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_hidden_survives_graph_replay_on_another_stream() -> None:
    runner = bare_runner()
    runner.enable_gpu_handoff = True
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        graph_input = torch.ones(1, 4, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=producer):
            graph_hidden = graph_input * 2
        graph.replay()
        payload = StagePayload(
            request_id="graph", request=OmniRequest(inputs=None), data={}
        )
        request_data = SimpleNamespace(
            stage_payload=payload,
            req=SimpleNamespace(inflight_middle_chunks=0),
            extra_model_outputs={},
        )
        runner.post_process_outputs(
            None,
            SimpleNamespace(
                requests=[SimpleNamespace(request_id="graph", data=request_data)]
            ),
            {"graph": SimpleNamespace(extra={"hidden_states": graph_hidden})},
        )
        runner.on_request_finished("graph", request_data)
        graph_input.fill_(7)
        graph.replay()
    with torch.cuda.stream(consumer):
        payload.wait_for_tensor_ready()
        received = request_data.extra_model_outputs["hidden_states_seq"]
        assert received.is_cuda
        assert torch.equal(received.cpu(), torch.full((1, 4), 2.0))
