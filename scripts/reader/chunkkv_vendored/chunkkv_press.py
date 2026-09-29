# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from NVIDIA/kvpress (kvpress/presses/chunkkv_press.py).
# Only the imports for BasePress and ScorerPress were rewritten to relative form.
#
# Paper: Liu et al., "ChunkKV: Semantic-Preserving KV Cache Compression
# for Efficient Long-Context LLM Inference" (NeurIPS 2025, arXiv 2502.00299).

from dataclasses import dataclass

import torch
from torch import nn

from .base_press import BasePress
from .scorer_press import ScorerPress


@dataclass
class ChunkKVPress(BasePress):
    """
    ChunkKV: Semantic-preserving compression with chunk-wise token selection.

    Wraps a ScorerPress: computes global importance scores, then keeps the
    top chunks (chunks of `chunk_length` consecutive tokens) by mean score.
    Preserves complete linguistic structures rather than scattered tokens.

    Parameters
    ----------
    press : ScorerPress
        Underlying scoring method (paper uses SnapKVPress).
    chunk_length : int, default=20
        Length of each chunk for token selection.
    """

    press: ScorerPress
    chunk_length: int = 20

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "ChunkKVPress requires a ScorerPress as input"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.press.compression_ratio == 0:
            return keys, values

        # Original kvpress asserted attentions is None (transformers 4.x with
        # output_attentions=False returns None). transformers >=5.x eager
        # attention always materialises the attention tensor in output[1],
        # which SnapKVPress.score() handles via its `attentions is not None`
        # branch (slices last window_size queries × earlier keys). Drop the
        # assertion so the same code path works on both stacks.
        kv_len = keys.shape[2]

        # 1. Calculate global scores first
        global_scores = self.press.score(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
        )

        # 2. Calculate complete chunks and remaining tokens
        num_complete_chunks = kv_len // self.chunk_length
        remaining_tokens = kv_len % self.chunk_length

        if num_complete_chunks == 0:
            return self.press.compress(module, hidden_states, keys, values, attentions, kwargs)

        if num_complete_chunks > 0:
            main_scores = global_scores[..., : num_complete_chunks * self.chunk_length]
            main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, self.chunk_length)
            main_chunk_scores = main_chunk_scores.mean(dim=-1)
        else:
            main_chunk_scores = torch.empty((global_scores.shape[0], 0), device=global_scores.device)

        if remaining_tokens > 0:
            remaining_scores = global_scores[..., -remaining_tokens:]
            remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
            chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
        else:
            chunk_scores = main_chunk_scores

        # 3. Number of chunks to keep
        n_chunks_kept = max(1, int((num_complete_chunks + (remaining_tokens > 0)) * (1 - self.press.compression_ratio)))
        top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

        # 4. Build keep indices
        indices = []
        for chunk_idx in top_chunks.indices[0]:
            if chunk_idx < num_complete_chunks:
                start_idx = chunk_idx * self.chunk_length
                chunk_indices = torch.arange(start_idx, start_idx + self.chunk_length, device=keys.device)
            else:
                chunk_indices = torch.arange(num_complete_chunks * self.chunk_length, kv_len, device=keys.device)
            indices.append(chunk_indices)

        indices = torch.cat(indices).sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)

        # 5. Gather selected keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
