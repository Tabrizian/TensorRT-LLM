# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Per-layer indexer k-cache mask (compact indexer pool) for DSA / GLM 5.2
cross-layer indexer sharing.

Covers:
1. Mask derivation from the model config (index_topk_freq/offset and
   index_topk_pattern schedules, MTP layers, dense fallbacks) -- CPU only.
2. Memory estimator arithmetic (get_cache_size_per_token) -- CPU only.
3. Compact pool allocation shape / layer->row mapping / shared-layer
   access guard on a real DSACacheManager -- requires GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    DSACacheManager,
    derive_indexer_k_cache_layer_mask,
)
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping

# GLM 5.2 cross-layer indexer sharing schedule: one full indexer every
# index_topk_freq layers, offset by index_skip_topk_offset, layers 0/1 full.
GLM52_INDEXER_SCHEDULE = {"index_topk_freq": 4, "index_skip_topk_offset": 2}


def _sparse_config(index_head_dim: int = 128) -> DeepSeekSparseAttentionConfig:
    return DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=index_head_dim,
        index_topk=2048,
    )


class TestMaskDerivation:
    """derive_indexer_k_cache_layer_mask against the model's schedule."""

    def test_glm52_schedule_yields_21_of_78_full_layers(self):
        pretrained_config = SimpleNamespace(num_hidden_layers=78, **GLM52_INDEXER_SCHEDULE)
        mask = derive_indexer_k_cache_layer_mask(_sparse_config(), pretrained_config, 78)
        assert len(mask) == 78
        assert sum(mask) == 21
        expected_full = {0, 1} | set(range(5, 78, 4))
        assert {i for i, m in enumerate(mask) if m} == expected_full

    def test_mask_matches_indexer_construction_source_of_truth(self):
        """The mask must agree layer-by-layer with what DSATrtllmAttention
        uses to decide whether a layer constructs an Indexer module."""
        pretrained_config = SimpleNamespace(num_hidden_layers=78, **GLM52_INDEXER_SCHEDULE)
        sparse_config = _sparse_config()
        mask = derive_indexer_k_cache_layer_mask(sparse_config, pretrained_config, 78)
        for layer_idx in range(78):
            sparse_params = sparse_config.to_sparse_params(
                pretrained_config=pretrained_config, layer_idx=layer_idx
            )
            assert mask[layer_idx] == sparse_params.is_full_indexer_layer

    def test_mtp_layer_beyond_num_hidden_layers_is_full(self):
        pretrained_config = SimpleNamespace(num_hidden_layers=78, **GLM52_INDEXER_SCHEDULE)
        mask = derive_indexer_k_cache_layer_mask(_sparse_config(), pretrained_config, 79)
        assert mask[78] is True
        assert sum(mask) == 22

    def test_index_topk_pattern_schedule(self):
        pretrained_config = SimpleNamespace(
            num_hidden_layers=4, index_topk_pattern=["F", "S", "F", "S"]
        )
        mask = derive_indexer_k_cache_layer_mask(_sparse_config(), pretrained_config, 4)
        assert mask == [True, False, True, False]

    def test_dense_config_without_schedule_is_all_full(self):
        """DeepSeek V3.2-style configs (no freq/pattern) keep the dense
        legacy layout: every layer owns an indexer k-cache row."""
        pretrained_config = SimpleNamespace(num_hidden_layers=8)
        mask = derive_indexer_k_cache_layer_mask(_sparse_config(), pretrained_config, 8)
        assert mask == [True] * 8

    def test_missing_pretrained_config_is_all_full(self):
        mask = derive_indexer_k_cache_layer_mask(_sparse_config(), None, 5)
        assert mask == [True] * 5


class TestEstimatorArithmetic:
    """get_cache_size_per_token charges indexer bytes for full layers only."""

    @staticmethod
    def _model_config(pretrained_extra: dict):
        pretrained_config = SimpleNamespace(
            num_hidden_layers=78,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            **pretrained_extra,
        )

        class FakeModelConfig:
            sparse_attention_config = _sparse_config()
            quant_config = SimpleNamespace(
                quant_mode=SimpleNamespace(has_fp8_kv_cache=lambda: True)
            )

            def __init__(self):
                self.pretrained_config = pretrained_config

            def get_num_attention_layers(self):
                return 78

        return FakeModelConfig()

    def test_glm52_geometry_47700_bytes_per_token(self):
        """78 x 576 B latent + 21 x 132 B indexer = 47,700 B/token (vs
        55,224 dense) -- the acceptance number for GLM 5.2."""
        model_config = self._model_config(GLM52_INDEXER_SCHEDULE)
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        size = DSACacheManager.get_cache_size_per_token(model_config, mapping)
        assert size == 78 * 576 + 21 * 132 == 47700

    def test_dense_geometry_unchanged_55224_bytes_per_token(self):
        model_config = self._model_config({})
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        size = DSACacheManager.get_cache_size_per_token(model_config, mapping)
        assert size == 78 * (576 + 132) == 55224

    def test_pp_ranks_split_full_indexer_layers(self):
        """Under PP the indexer term must count only this rank's full
        layers (mask sliced with the same pp_layers rule as the pool)."""
        model_config = self._model_config(GLM52_INDEXER_SCHEDULE)
        sizes = []
        for rank in (0, 1):
            mapping = Mapping(world_size=2, rank=rank, tp_size=1, pp_size=2)
            local_layers = mapping.pp_layers(78)
            full_local = {0, 1} | set(range(5, 78, 4))
            num_full = sum(1 for i in local_layers if i in full_local)
            size = DSACacheManager.get_cache_size_per_token(model_config, mapping)
            assert size == len(local_layers) * 576 + num_full * 132
            sizes.append(size)
        # Both ranks together account for all 21 full layers.
        assert sum(sizes) == 78 * 576 + 21 * 132

    def test_num_layers_override_counts_all_layers_full(self):
        """Draft/MTP managers pass an explicit num_layers; MTP layers always
        run their own indexer, so every override layer is charged."""
        model_config = self._model_config(GLM52_INDEXER_SCHEDULE)
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        size = DSACacheManager.get_cache_size_per_token(model_config, mapping, num_layers=1)
        assert size == 576 + 132


class TestDisaggExtractorGate:
    """The Python KV-transfer extractor must fail fast on a compact pool."""

    @staticmethod
    def _mock_manager(local_indexer_mask):
        from tensorrt_llm.bindings import DataType

        num_layers = len(local_indexer_mask)
        tokens_per_block = 8
        head_dim = 576
        kv_pool = torch.zeros((4, 1, tokens_per_block * head_dim), dtype=torch.float16)
        return SimpleNamespace(
            dtype=DataType.HALF,
            tokens_per_block=tokens_per_block,
            head_dim=head_dim,
            kv_factor=1,
            num_kv_heads_per_layer=[1] * num_layers,
            layer_offsets={i: i for i in range(num_layers)},
            _get_window_size_to_layers=lambda: {1024: list(range(num_layers))},
            kv_cache_pool_mapping=torch.zeros((num_layers, 2), dtype=torch.int32),
            kv_cache_pool_pointers=torch.zeros((1, 2), dtype=torch.int64),
            enable_indexer_k_cache=True,
            indexer_k_cache_local_layer_mask=list(local_indexer_mask),
            impl=SimpleNamespace(
                get_primary_pool_data=lambda layer_idx: kv_pool,
                get_indexer_k_cache_pool=lambda: torch.zeros(
                    (4, num_layers, 1, tokens_per_block * 132), dtype=torch.uint8
                ),
            ),
        )

    def test_compact_pool_rejected_before_touching_the_pool(self):
        from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table

        manager = self._mock_manager([True, False, True, False])
        # A fully-masked-out rank has no C++ pool at all, so the gate must
        # fire from the mask alone, never fetching the pool.
        manager.impl.get_indexer_k_cache_pool = None
        with pytest.raises(NotImplementedError, match="indexer k-cache"):
            build_page_table(manager)

    def test_dense_mask_passes_the_gate(self):
        from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table

        manager = self._mock_manager([True, True, True])
        page_table = build_page_table(manager)
        # Dense layout keeps the legacy behavior: indexer pool published.
        assert len(page_table.pool_groups[0].pools) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class TestCompactPoolAllocation:
    """Allocation-shape checks on a real DSACacheManager."""

    # 6 layers under the GLM 5.2 schedule -> full layers {0, 1, 5}.
    _PRETRAINED = dict(num_hidden_layers=6, **GLM52_INDEXER_SCHEDULE)
    _EXPECTED_MASK = [True, True, False, False, False, True]

    def _create(self, pretrained_config):
        # Lazy import: test_dsa_indexer pulls deep_gemm and other GPU-only
        # dependencies that must not gate the CPU-only tests above.
        from .test_dsa_indexer import create_dsa_cache_manager

        return create_dsa_cache_manager(
            batch_size=2,
            head_dim=128,
            tokens_per_block=64,
            max_seq_len=1024,
            num_layers=6,
            pretrained_config=pretrained_config,
        )

    def test_shared_layers_have_no_indexer_pool(self):
        cache_manager, _ = self._create(SimpleNamespace(**self._PRETRAINED))
        try:
            pools = cache_manager.indexer_k_cache_pool_per_layer
            assert [p is not None for p in pools] == self._EXPECTED_MASK

            # The C++ pool holds exactly one row per full-indexer layer.
            pool = cache_manager.impl.get_indexer_k_cache_pool()
            assert pool.shape[1] == 3

            # Local layer -> compact row mapping (shared layers -> -1).
            rows = [cache_manager.impl.get_indexer_k_cache_pool_layer_idx(i) for i in range(6)]
            assert rows == [0, 1, -1, -1, -1, 2]

            # Full-layer buffers alias distinct compact rows.
            for layer_idx, expected_row, fill in ((0, 0, 1), (1, 1, 2), (5, 2, 3)):
                buf = cache_manager.get_indexer_k_cache_buffers(layer_idx)
                buf.fill_(fill)
                row = pool[:, expected_row]
                assert (row.flatten() == fill).all(), (
                    f"layer {layer_idx} did not map to compact row {expected_row}"
                )

            # Shared layers must never hand out a buffer.
            with pytest.raises(AssertionError, match="shared-indexer"):
                cache_manager.get_indexer_k_cache_buffers(2)

            # Instance byte accounting matches the compact layout:
            # HALF latent (2 B x 128 head dim x 6 layers) + 3 x 132 B indexer.
            assert cache_manager.get_cache_bytes_per_token() == 2 * 6 * 128 + 3 * 132
        finally:
            cache_manager.shutdown()

    def test_dense_default_unchanged(self):
        """Without a sharing schedule every layer keeps its row and the
        mapping is the identity (DeepSeek V3.2 behavior)."""
        cache_manager, _ = self._create(None)
        try:
            pools = cache_manager.indexer_k_cache_pool_per_layer
            assert all(p is not None for p in pools)
            pool = cache_manager.impl.get_indexer_k_cache_pool()
            assert pool.shape[1] == 6
            rows = [cache_manager.impl.get_indexer_k_cache_pool_layer_idx(i) for i in range(6)]
            assert rows == list(range(6))
            assert cache_manager.get_cache_bytes_per_token() == 2 * 6 * 128 + 6 * 132
        finally:
            cache_manager.shutdown()
