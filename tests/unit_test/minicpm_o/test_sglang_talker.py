# SPDX-License-Identifier: Apache-2.0
"""Public talker contract: condition embeddings match the reference math."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.comm.stage_io import pack_tensors
from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.minicpm_o.components.sglang_talker import (
    MiniCPMOTalkerForCausalLM,
    MiniCPMTTSProjector,
)
from sglang_omni.models.minicpm_o.talker_model_runner import MiniCPMOTalkerModelRunner
from sglang_omni.models.minicpm_o.talker_request import make_talker_scheduler_adapters
from sglang_omni.proto.request import OmniRequest, StagePayload

HIDDEN = 8
LLM_DIM = 16
NUM_TEXT = 20
TEXT_EOS = 5
AUDIO_BOS = 6


def bare_model() -> MiniCPMOTalkerForCausalLM:
    model = object.__new__(MiniCPMOTalkerForCausalLM)
    nn.Module.__init__(model)
    model.text_eos_token_id = TEXT_EOS
    model.audio_bos_token_id = AUDIO_BOS
    model.normalize_projected_hidden = True
    model.emb_text = nn.Embedding(NUM_TEXT, HIDDEN)
    model.projector_semantic = MiniCPMTTSProjector(LLM_DIM, HIDDEN)
    return model


def test_condition_matches_reference_math():
    model = bare_model()
    tokens = torch.tensor([3, 7, 1], dtype=torch.long)
    hidden = torch.randn(3, LLM_DIM)

    condition = model.build_condition_embeddings(tokens, hidden)

    # reference: emb_text(t) + l2norm(projector(h)), then [text_eos, audio_bos]
    ref = model.emb_text(tokens) + F.normalize(
        model.projector_semantic(hidden), p=2, dim=-1
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, torch.cat([ref, boundary], dim=0))
    assert condition.shape == (5, HIDDEN)


def test_condition_empty_span_is_boundary_only():
    model = bare_model()
    condition = model.build_condition_embeddings(
        torch.empty(0, dtype=torch.long), torch.empty(0, LLM_DIM)
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, boundary)


def test_condition_length_mismatch_raises():
    model = bare_model()
    with pytest.raises(ValueError, match="length mismatch"):
        model.build_condition_embeddings(torch.tensor([1, 2]), torch.randn(3, LLM_DIM))


def test_talker_request_builder_returns_inference_conditioning() -> None:
    model = bare_model()
    builder, _ = make_talker_scheduler_adapters(
        model=model,
        codec_vocab_size=32,
        codec_eos_id=31,
        tts_bos_token_id=30,
        tts_eos_token_id=31,
    )
    hidden_states = torch.randn(2, LLM_DIM, requires_grad=True)
    payload = StagePayload(
        request_id="inference-conditioning",
        request=OmniRequest(inputs=None),
        data={
            "prompt": {
                "input_ids": torch.tensor([30, 11, 12, 31]),
                "known_tts_output_ids": [11, 12],
            },
            "thinker_out": {
                "extra_model_outputs": {"hidden_states_seq": hidden_states}
            },
        },
    )
    with torch.no_grad():
        expected = model.build_condition_embeddings(
            torch.tensor([11, 12]), hidden_states
        )
    with torch.enable_grad():
        request = builder(payload)
    torch.testing.assert_close(request.prefill_input_embeds, expected, rtol=0, atol=0)
    assert not request.prefill_input_embeds.requires_grad
    assert request.prefill_input_embeds.grad_fn is None


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_projector_graph_replay_preserves_owned_outputs_across_streams() -> None:
    projector = MiniCPMTTSProjector(32, 16).to(device="cuda", dtype=torch.bfloat16)
    reference = MiniCPMTTSProjector(32, 16).to(device="cuda", dtype=torch.bfloat16)
    reference.load_state_dict(projector.state_dict())
    projector.capture_graphs((2, 3))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs: list[torch.Tensor] = []
    expected: list[torch.Tensor] = []
    torch.cuda.synchronize()
    with torch.no_grad():
        for index, rows in enumerate([2, 2, 3, 3, 2, 3] * 8):
            with torch.cuda.stream(streams[index % 2]):
                hidden = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
                outputs.append(projector(hidden))
                expected.append(reference(hidden))
        fallback = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            projector(fallback), reference(fallback), rtol=0, atol=0
        )
    torch.cuda.synchronize()
    for actual, target in zip(outputs, expected):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    assert len({output.data_ptr() for output in outputs}) == len(outputs)
    assert len(projector.graph_replay.captures) == 2
    with pytest.raises(RuntimeError, match="already captured"):
        projector.capture_graphs((2,))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_talker_request_builder_disables_grad_for_graph_replay(monkeypatch) -> None:
    model = bare_model().to(device="cuda", dtype=torch.bfloat16)
    model.projector_semantic.capture_graphs((2,))
    calls: list[bool] = []
    replay = model.projector_semantic.graph_replay.replay

    def record_replay(hidden_states: torch.Tensor) -> torch.Tensor | None:
        result = replay(hidden_states)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(model.projector_semantic.graph_replay, "replay", record_replay)
    builder, _ = make_talker_scheduler_adapters(
        model=model,
        codec_vocab_size=32,
        codec_eos_id=31,
        tts_bos_token_id=30,
        tts_eos_token_id=31,
    )
    payload = StagePayload(
        request_id="graph-builder",
        request=OmniRequest(inputs=None),
        data={
            "prompt": {
                "input_ids": torch.tensor([30, 11, 12, 31]),
                "known_tts_output_ids": [11, 12],
            },
            "thinker_out": {
                "extra_model_outputs": {
                    "hidden_states_seq": torch.randn(
                        2, LLM_DIM, device="cuda", dtype=torch.bfloat16
                    )
                }
            },
        },
    )
    with torch.enable_grad():
        request = builder(payload)
    assert calls == [True]
    assert not request.prefill_input_embeds.requires_grad


def test_projector_graph_capture_rejects_cpu_weights() -> None:
    with pytest.raises(ValueError, match="CUDA weights"):
        MiniCPMTTSProjector(32, 16).capture_graphs((2,))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_projector_graph_preserves_training_fallback_and_state_dict() -> None:
    projector = MiniCPMTTSProjector(32, 16).to(device="cuda", dtype=torch.bfloat16)
    reference = MiniCPMTTSProjector(32, 16).to(device="cuda", dtype=torch.bfloat16)
    reference.load_state_dict(projector.state_dict())
    names = set(projector.state_dict())
    projector.capture_graphs((2, 2))
    assert set(projector.state_dict()) == names
    assert len(projector.graph_replay.captures) == 1
    hidden = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.enable_grad():
        actual = projector(hidden)
        expected = reference(hidden)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().sum().backward()
    assert hidden.grad is not None


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("row_counts", [(), (0,), (-1,), (2, 0)])
def test_projector_graph_capture_rejects_invalid_shapes(
    row_counts: tuple[int, ...],
) -> None:
    projector = MiniCPMTTSProjector(32, 16).to(device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="positive"):
        projector.capture_graphs(row_counts)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prompt_token_count", [35, 36, 37, 51])
def test_packed_hidden_preserves_frozen_conditioning(
    prompt_token_count: int,
) -> None:
    torch.manual_seed(42)
    model = bare_model().to(device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.randn(17, LLM_DIM, device="cuda", dtype=torch.bfloat16)
    token_ids = torch.arange(17) % NUM_TEXT
    packed, tensor_metadata = pack_tensors(
        {
            "prompt": torch.arange(prompt_token_count),
            "hidden": hidden_states,
        },
        device="cuda",
    )
    hidden_metadata = tensor_metadata[1]
    packed_hidden = (
        packed[hidden_metadata.offset : hidden_metadata.offset + hidden_metadata.size]
        .view(torch.bfloat16)
        .reshape_as(hidden_states)
    )
    with torch.no_grad():
        expected = model.build_condition_embeddings(token_ids, hidden_states)
        actual = model.build_condition_embeddings(token_ids, packed_hidden)
    assert packed_hidden.data_ptr() % 16 == 0
    torch.testing.assert_close(packed_hidden, hidden_states, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gpu_conditioning_matches_cpu_handoff_across_streams(
    dtype: torch.dtype,
) -> None:
    producer = torch.cuda.Stream()
    builder = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    hidden_reference = torch.arange(2 * LLM_DIM, dtype=dtype).reshape(2, LLM_DIM) / 16
    with torch.cuda.stream(producer):
        payload = StagePayload(
            request_id="conditioning",
            request=OmniRequest(inputs=None),
            data={
                "prompt": {
                    "input_ids": torch.tensor([30, 11, 12, 31]),
                    "known_tts_output_ids": [11, 12],
                },
                "thinker_out": {
                    "extra_model_outputs": {
                        "hidden_states_seq": hidden_reference.to("cuda")
                    }
                },
            },
        )
        ready_event = torch.cuda.Event()
        ready_event.record(producer)
        payload.tensor_ready_events = [ready_event]

    with torch.no_grad(), torch.cuda.stream(builder):
        model = bare_model().to(device="cuda", dtype=dtype)
        model.emb_code = nn.Embedding(32, HIDDEN, device="cuda", dtype=dtype)
        request_builder, result_adapter = make_talker_scheduler_adapters(
            model=model,
            codec_vocab_size=32,
            codec_eos_id=31,
            tts_bos_token_id=30,
            tts_eos_token_id=31,
        )
        request_data = request_builder(payload)
        request_data.req.set_extend_range(0, 4)

    with torch.no_grad(), torch.cuda.stream(consumer):
        runner = object.__new__(MiniCPMOTalkerModelRunner)
        runner.model = model
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.long, device="cuda"),
            replace_embeds=None,
        )
        runner.before_prefill(forward_batch, None, [SimpleNamespace(data=request_data)])
        prefill = get_omni_prefill_inputs(forward_batch)
        expected = model.build_condition_embeddings(
            torch.tensor([11, 12]), hidden_reference
        )
        assert prefill is not None
        assert prefill.input_embeds.dtype == dtype
        assert torch.equal(prefill.input_embeds, expected)

    request_data.output_ids.extend([7, 31])
    result = result_adapter(request_data)
    assert result.data["engine_outputs"]["talker"]["codec_tokens"].tolist() == [7]
    assert result.tensor_ready_events == []
