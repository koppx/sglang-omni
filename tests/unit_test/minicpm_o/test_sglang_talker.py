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
