# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker request construction."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.minicpm_o.components.talker import build_tts_condition
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_sglang_thinker_request
from sglang_omni.models.minicpm_o.routing import (
    project_thinker_to_decode,
    project_thinker_to_talker,
    resolve_thinker_next_stages,
)
from sglang_omni.models.minicpm_o.talker_request import build_talker_request
from sglang_omni.proto.request import OmniRequest, StagePayload

VOCAB_SIZE = 256


def build(params: dict) -> None:
    build_sglang_thinker_request(
        MiniCPMOPipelineState(prompt={"input_ids": torch.tensor([1, 2, 3])}),
        params=params,
        tokenizer=None,
        vocab_size=32000,
    )


def thinker_params(length_penalty: object) -> dict:
    return {"stage_params": {"thinker": {"length_penalty": length_penalty}}}


@pytest.mark.parametrize(
    "params",
    [{}, {"stage_params": {"thinker": None}}, thinker_params(1.0), thinker_params(1.3)],
)
def test_missing_or_positive_length_penalty_builds(params: dict) -> None:
    build(params)


@pytest.mark.parametrize(
    ("length_penalty", "error"),
    [
        (None, TypeError),
        ("1.3", TypeError),
        (0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
    ],
)
def test_invalid_length_penalty_fails_at_build(
    length_penalty: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        build(thinker_params(length_penalty))


def test_prompt_ids_are_checked_before_media_placeholders_are_remapped() -> None:
    media_state = MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor([10, 255, 255, 11])},
        mm_inputs={
            "image": {"bounds": torch.tensor([[1, 3]]), "cache_key": "image:cache"}
        },
        thinker_inputs={"model_inputs": {"pixel_values": torch.ones(1)}},
    )
    text_state = MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor([VOCAB_SIZE - 1, VOCAB_SIZE])}
    )

    media_request = build_sglang_thinker_request(
        media_state,
        params={"max_new_tokens": 3},
        tokenizer=None,
        vocab_size=VOCAB_SIZE,
        request_id="minicpm-o-media",
    )
    with pytest.raises(ValueError, match=f"token id {VOCAB_SIZE} at position 1"):
        build_sglang_thinker_request(
            text_state,
            params={"max_new_tokens": 3},
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
            request_id="minicpm-o-text",
        )

    assert media_request.req.origin_input_ids[1] >= VOCAB_SIZE


@pytest.mark.parametrize("legacy_rows", [False, True])
@pytest.mark.parametrize(
    (
        "prompt_ids",
        "output_ids",
        "row_count",
        "known_tokens",
        "expected_tokens",
        "expected_rows",
    ),
    [
        ([10, 30], [11, 12, 31], 3, None, [11, 12], [1, 2]),
        ([10, 30], [11, 12], 2, None, [11], [1]),
        ([31, 9, 30], [11, 12], 3, None, [11, 12], [1, 2]),
        ([30, 8, 31, 9, 30], [11, 31], 2, None, [11], [1]),
        ([1], [11], 1, None, [], []),
        ([30], [31], 1, None, [], []),
        ([30], [11], 0, None, [], []),
        ([30, 11, 12, 31], [99], 2, [11, 12], [11, 12], [0, 1]),
    ],
)
def test_talker_handoff_preserves_frozen_conditioning(
    legacy_rows: bool,
    prompt_ids: list[int],
    output_ids: list[int],
    row_count: int,
    known_tokens: list[int] | None,
    expected_tokens: list[int],
    expected_rows: list[int],
) -> None:
    hidden_states = torch.arange(row_count * 4, dtype=torch.float32).reshape(-1, 4)
    state = MiniCPMOPipelineState(
        prompt={
            "input_ids": torch.tensor(prompt_ids),
            "known_tts_output_ids": known_tokens,
        },
        thinker_out={
            "output_ids": output_ids,
            "extra_model_outputs": {
                "hidden_states_seq": (
                    list(hidden_states.unbind(0)) if legacy_rows else hidden_states
                )
            },
        },
    )
    payload = StagePayload(
        request_id="frozen", request=OmniRequest(inputs=None), data=state.to_dict()
    )
    projected = project_thinker_to_talker(payload)
    span = build_talker_request(
        MiniCPMOPipelineState.from_dict(projected.data),
        tts_bos_token_id=30,
        tts_eos_token_id=31,
    )
    assert span["tts_token_ids"].tolist() == expected_tokens
    if expected_rows:
        assert torch.equal(span["tts_hidden"], hidden_states[expected_rows])
    else:
        assert span["tts_hidden"].numel() == 0
    text_embedding = torch.nn.Embedding(32, 4)
    with torch.no_grad():
        text_embedding.weight.copy_(torch.arange(128).reshape(32, 4))
    condition_parameters = {
        "text_embedding": text_embedding,
        "semantic_projector": torch.nn.Identity(),
        "boundary_tokens": (0, 1),
        "normalize_projected_hidden": True,
    }
    actual = build_tts_condition(
        span["tts_token_ids"], span["tts_hidden"], **condition_parameters
    )
    expected = build_tts_condition(
        torch.tensor(expected_tokens, dtype=torch.long),
        hidden_states[expected_rows],
        **condition_parameters,
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("row_count", [0, 4])
def test_talker_projection_preserves_tensor_and_removes_unused_prompt(
    row_count: int,
) -> None:
    hidden_states = torch.ones(row_count, 8)
    state = MiniCPMOPipelineState(
        prompt={
            "input_ids": torch.tensor([30]),
            "attention_mask": torch.ones(1),
            "prompt_text": "unused",
        },
        thinker_out={"extra_model_outputs": {"hidden_states_seq": hidden_states}},
    )
    payload = StagePayload(
        request_id="projection", request=OmniRequest(inputs=None), data=state.to_dict()
    )
    projected = MiniCPMOPipelineState.from_dict(project_thinker_to_talker(payload).data)
    assert (
        projected.thinker_out["extra_model_outputs"]["hidden_states_seq"]
        is hidden_states
    )
    assert set(projected.prompt) == {"input_ids"}
    decoded = MiniCPMOPipelineState.from_dict(project_thinker_to_decode(payload).data)
    assert "extra_model_outputs" not in decoded.thinker_out


@pytest.mark.parametrize("stream", [False, True])
def test_hidden_handoff_remains_audio_only(stream: bool) -> None:
    payload = StagePayload(
        request_id="text",
        request=OmniRequest(
            inputs=None,
            params={"stream": stream},
            metadata={"output_modalities": ["text"]},
        ),
        data={},
    )
    assert resolve_thinker_next_stages("text", payload) == ["decode"]
    payload.request.metadata["output_modalities"] = ["audio"]
    assert resolve_thinker_next_stages("text", payload) == ["decode", "talker"]


def test_talker_handoff_rejects_malformed_hidden_shape() -> None:
    state = MiniCPMOPipelineState(
        prompt={"input_ids": torch.tensor([30])},
        thinker_out={"extra_model_outputs": {"hidden_states_seq": torch.ones(4)}},
    )
    with pytest.raises(ValueError, match="shape"):
        build_talker_request(state, tts_bos_token_id=30, tts_eos_token_id=31)
