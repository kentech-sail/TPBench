# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from NVIDIA/kvpress (https://github.com/NVIDIA/kvpress) — files copied
# from `kvpress/presses/` and `kvpress/utils.py`. Imports rewritten to relative
# form so the package is self-contained inside this repository (no `kvpress`
# pip install required).
#
# Provenance: NVIDIA/kvpress @ main branch fetched 2026-04-27.
# Modifications: only `from kvpress.* import` lines were changed to `from .` form.
# Original method logic is unchanged.
#
# ChunkKV paper: Liu et al., "ChunkKV: Semantic-Preserving KV Cache Compression
# for Efficient Long-Context LLM Inference" (NeurIPS 2025, arXiv 2502.00299).
#
# Public exports below mirror the subset our reader uses.

from .base_press import BasePress
from .scorer_press import ScorerPress
from .snapkv_press import SnapKVPress
from .chunkkv_press import ChunkKVPress

__all__ = ["BasePress", "ScorerPress", "SnapKVPress", "ChunkKVPress"]
