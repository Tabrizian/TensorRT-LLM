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
"""Unit tests for ``CuteDslFusedMoE.prezero_moe_output`` and its handshake
with ``run_moe_nvfp4_impl``: the fused-finalize output buffer is zeroed on the
MoeOutputMemset aux stream ahead of the MoE chunk (overlapping routing /
quant / dispatch) and the later in-place memset is skipped for that buffer.

The method only touches ``use_fused_finalize``, ``event_dict``,
``aux_stream_dict`` and ``_prezeroed_moe_output_ptr``, so it is exercised on a
stub carrying exactly those attributes -- no experts / weights needed.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
from tensorrt_llm._torch.utils import AuxStreamType, EventType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _stub(use_fused_finalize: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        use_fused_finalize=use_fused_finalize,
        aux_stream_dict={AuxStreamType.MoeOutputMemset: torch.cuda.Stream()},
        event_dict={EventType.Main: torch.cuda.Event(), EventType.MoeOutputMemset: torch.cuda.Event()},
        _prezeroed_moe_output_ptr=None,
    )


def test_prezero_zeroes_on_aux_stream_and_records_pointer():
    stub = _stub()
    buf = torch.full((32768, 6144), 7.0, dtype=torch.bfloat16, device="cuda")  # EP4 x 8192 tokens payload
    CuteDslFusedMoE.prezero_moe_output(stub, buf)
    # Consumers wait on the MoeOutputMemset event before reading the buffer.
    stub.event_dict[EventType.MoeOutputMemset].wait()
    torch.cuda.synchronize()
    assert stub._prezeroed_moe_output_ptr == buf.data_ptr()
    assert not buf.any(), "buffer not zeroed"


def test_prezero_orders_after_main_stream_producer():
    """A write to the buffer queued on the main stream before prezero must be
    ordered before the zero-fill (previous layer's combine is the last reader,
    but any earlier writer must also be done), i.e. the aux stream waits on the
    Main event recorded by prezero."""
    stub = _stub()
    buf = torch.zeros((4096, 6144), dtype=torch.bfloat16, device="cuda")
    # Long-running main-stream producer that writes non-zero values.
    for _ in range(20):
        buf.add_(1.0)
    CuteDslFusedMoE.prezero_moe_output(stub, buf)
    stub.event_dict[EventType.MoeOutputMemset].wait()
    torch.cuda.synchronize()
    assert not buf.any(), "zero-fill raced ahead of the main-stream producer"


def test_prezero_is_noop_without_fused_finalize():
    stub = _stub(use_fused_finalize=False)
    buf = torch.ones((16, 64), dtype=torch.bfloat16, device="cuda")
    CuteDslFusedMoE.prezero_moe_output(stub, buf)
    torch.cuda.synchronize()
    assert stub._prezeroed_moe_output_ptr is None
    assert buf.all(), "buffer must be untouched when the finalize is not fused"


def test_prezero_pointer_matches_only_the_same_buffer():
    """run_moe_nvfp4_impl skips its memset only when data_ptr matches; a
    different buffer of the same shape must not be mistaken for the prezeroed one."""
    stub = _stub()
    a = torch.ones((64, 128), dtype=torch.bfloat16, device="cuda")
    b = torch.ones((64, 128), dtype=torch.bfloat16, device="cuda")
    CuteDslFusedMoE.prezero_moe_output(stub, a)
    torch.cuda.synchronize()
    assert stub._prezeroed_moe_output_ptr == a.data_ptr()
    assert stub._prezeroed_moe_output_ptr != b.data_ptr()
    assert b.all()
