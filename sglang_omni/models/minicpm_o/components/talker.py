# SPDX-License-Identifier: Apache-2.0
"""Shared MiniCPM-o TTS conditioning."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible thinker-hidden to talker-hidden projection."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.graph_replay: ProjectorGraphReplay | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.graph_replay is not None:
            replayed = self.graph_replay.replay(hidden_states)
            if replayed is not None:
                return replayed
            else:
                pass
        else:
            pass
        return self.linear2(self.relu(self.linear1(hidden_states)))

    def capture_graphs(self, row_counts: tuple[int, ...]) -> None:
        """Capture exact row counts after weights are loaded; other shapes stay eager."""
        if self.graph_replay is not None:
            raise RuntimeError("projector graphs are already captured")
        elif not self.linear1.weight.is_cuda:
            raise ValueError("projector graphs require CUDA weights")
        elif not row_counts or any(rows <= 0 for rows in row_counts):
            raise ValueError("projector graph row counts must be positive")
        else:
            self.graph_replay = ProjectorGraphReplay(self, row_counts)


@dataclass(kw_only=True)
class ProjectorGraphCapture:
    hidden_states: torch.Tensor
    projected_hidden: torch.Tensor
    graph: torch.cuda.CUDAGraph


class ProjectorGraphReplay:
    """Own static buffers and serialize shared-pool replay across CUDA streams."""

    def __init__(
        self, projector: MiniCPMTTSProjector, row_counts: tuple[int, ...]
    ) -> None:
        self.captures: dict[int, ProjectorGraphCapture] = {}
        self.lock = threading.Lock()
        self.previous_done: torch.cuda.Event | None = None
        self.previous_stream: int | None = None
        weight = projector.linear1.weight
        self.device = weight.device
        self.dtype = weight.dtype
        self.input_size = projector.linear1.in_features
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.device(self.device), torch.cuda.stream(stream), torch.no_grad():
            for rows in sorted(set(row_counts)):
                hidden_states = torch.zeros(
                    rows, self.input_size, device=self.device, dtype=self.dtype
                )
                for _ in range(3):
                    projector(hidden_states)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, pool=pool):
                    projected_hidden = projector(hidden_states)
                self.captures[rows] = ProjectorGraphCapture(
                    hidden_states=hidden_states,
                    projected_hidden=projected_hidden,
                    graph=graph,
                )
        stream.synchronize()

    def replay(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != self.input_size
            or hidden_states.device != self.device
            or hidden_states.dtype != self.dtype
            or torch.is_grad_enabled()
        ):
            return None
        else:
            capture = self.captures.get(hidden_states.shape[0])
        if capture is None:
            return None
        else:
            pass
        with self.lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if (
                self.previous_done is not None
                and self.previous_stream != stream.cuda_stream
            ):
                stream.wait_event(self.previous_done)
            else:
                pass
            capture.hidden_states.copy_(hidden_states)
            capture.graph.replay()
            # note (koppx): shared graph storage is reused by the next request.
            result = capture.projected_hidden.clone()
            if self.previous_done is None:
                self.previous_done = torch.cuda.Event()
            else:
                pass
            self.previous_done.record(stream)
            self.previous_stream = stream.cuda_stream
            return result


def build_tts_condition(
    token_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    text_embedding: nn.Embedding,
    semantic_projector: nn.Module,
    boundary_tokens: tuple[int, ...],
    normalize_projected_hidden: bool,
) -> torch.Tensor:
    """Compose token-aligned speech conditions with explicit turn/unit boundaries."""
    device, dtype = text_embedding.weight.device, text_embedding.weight.dtype
    if hidden_states.is_cuda:
        hidden_states.record_stream(torch.cuda.current_stream(hidden_states.device))
    else:
        pass
    tokens = token_ids.to(device=device, dtype=torch.long).reshape(-1)
    hidden = hidden_states.to(device=device, dtype=dtype)
    boundary = text_embedding(torch.tensor(boundary_tokens, device=device))
    if not tokens.numel():
        return boundary
    else:
        if hidden.ndim != 2 or hidden.shape[0] != tokens.numel():
            raise ValueError(
                "talker condition length mismatch: token ids and hidden states must be position-aligned"
            )
        else:
            pass
        projected = semantic_projector(hidden)
        if normalize_projected_hidden:
            projected = F.normalize(projected, p=2, dim=-1)
        else:
            pass
        condition = text_embedding(tokens) + projected
        return torch.cat((condition, boundary), dim=0)
