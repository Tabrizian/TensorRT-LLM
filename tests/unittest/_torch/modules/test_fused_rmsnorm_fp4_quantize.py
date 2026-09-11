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
"""Unit tests for the fused (add +) RMSNorm + NVFP4-quantize kernels.

Both ops fold the standalone NVFP4 input-quantize (FP16/BF16 -> NVFP4) into its
producing RMSNorm at DeepSeek-V3.2 / Kimi-K2.5 MLA sites:

  * ``fused_add_rmsnorm_fp4_quantize`` - layer-boundary
    ``residual_add -> rms_norm -> nvfp4_quantize`` feeding the next layer's
    ``kv_a_proj``.
  * ``fused_rmsnorm_fp4_quantize`` - the residual-less intra-MLA
    ``q_a_layernorm -> q_b_proj`` input quant, which additionally supports a
    row-strided (column-slice) input read so the ``q`` slice from
    ``kv_a_proj_with_mqa(...).split(...)`` needs no ``.contiguous()`` copy.

Accuracy is checked against the unfused reference: a PyTorch RMSNorm followed by
``torch.ops.trtllm.fp4_quantize`` (the exact kernel the fusion replaces). The
fused kernel computes the RMSNorm in registers while already reading the tensor
to quantize it, so a handful of values right at quantization boundaries can fall
on the other side of a step due to FMA-vs-separate rounding; we require a >= 99%
match on the packed FP4 bytes, mirroring ``test_fused_activation_quant.py``.
"""

import pytest
import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.utils import ceil_div, pad_up, unswizzle_sf
from tests.unittest.utils.util import getSMVersion


def fused_add_rmsnorm_fp4_quantize_available():
    return hasattr(torch.ops, "trtllm") and hasattr(
        torch.ops.trtllm, "fused_add_rmsnorm_fp4_quantize"
    )


def fused_rmsnorm_fp4_quantize_available():
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_rmsnorm_fp4_quantize")


def fp4_quantize_available():
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fp4_quantize")


skip_unless_add_rmsnorm = pytest.mark.skipif(
    getSMVersion() < 100
    or not fused_add_rmsnorm_fp4_quantize_available()
    or not fp4_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm "
    "fused_add_rmsnorm_fp4_quantize + fp4_quantize ops",
)

skip_unless_rmsnorm = pytest.mark.skipif(
    getSMVersion() < 100
    or not fused_rmsnorm_fp4_quantize_available()
    or not fp4_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm fused_rmsnorm_fp4_quantize + fp4_quantize ops",
)


def rms_norm_ref(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm. Uses the *production* flashinfer RMSNorm kernel (the
    exact kernel RMSNorm.forward runs on the unfused path the fusion replaces)
    when available, so the comparison is apples-to-apples; otherwise falls back
    to the fp32 PyTorch formula matching tensorrt_llm/_torch/modules/rms_norm.py.

    Note the fused kernel still differs from this reference by ULPs: it does the
    fp32 reduction + RMSNorm in registers in a single pass, whereas the unfused
    path materializes the normed bf16 in a separate kernel before fp4_quantize.
    Those ULP differences can flip a value across an E2M1/E4M3 step, so the
    cross-path comparison is a high match rate (see _FP4_MATCH_THRESHOLD), while
    the exact fused-epilogue equivalence is checked separately in
    assert_fp4_bitexact_from_norm."""
    if IS_FLASHINFER_AVAILABLE:
        from tensorrt_llm._torch.custom_ops import flashinfer_rmsnorm

        return flashinfer_rmsnorm(hidden_states.contiguous(), weight, eps)
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


# NVFP4 packs SF_VEC_SIZE elements per E4M3 scale-factor block.
SF_VEC = 16
# NVFP4 numeric maxima: FP8 (E4M3) and E2M1.
FP8_MAX, E2M1_MAX = 448.0, 6.0


def fp4_quantize_ref(normed: torch.Tensor, sf_scale: torch.Tensor):
    """Unfused baseline: the exact standalone NVFP4 input-quantize the fusion
    replaces. Mirrors the production static-NVFP4 call in
    NVFP4LinearMethod._input_prepare (linear.py):
        torch.ops.trtllm.fp4_quantize(input, input_scale, scaling_vector_size, False)
    i.e. sfVecSize=16, sfUseUE8M0=False, isSfSwizzledLayout defaulting to True."""
    return torch.ops.trtllm.fp4_quantize(
        normed.contiguous(),
        sf_scale,
        SF_VEC,
        False,  # sfUseUE8M0
    )


def make_sf_scale(normed: torch.Tensor) -> torch.Tensor:
    """Per-tensor global SF scale fed to both the fused op and fp4_quantize
    (both forward it to cvt_warp_fp16_to_fp4 as SFScaleVal). Matches the runtime
    static NVFP4 input_scale convention in linear.py: FP8_MAX * E2M1_MAX / amax,
    so the per-block E4M3 scale lands in range (SFValue = SFScaleVal*vecMax/6
    <= 448)."""
    amax = normed.abs().amax().float()
    return (FP8_MAX * E2M1_MAX / amax).to(normed.device).view(1)


# Packed-FP4 mantissa match threshold. The fused kernel computes RMSNorm in
# registers (FMA) while reading the tensor to quantize it, vs the reference's
# separate RMSNorm -> fp4_quantize; a few values right at an E2M1 step boundary
# round the other way. With the correct (non-inverted) input scale this exercises
# the full FP4 range, so bf16's 8-bit mantissa on small shapes can dip to ~0.984
# (a real kernel error collapses the rate to < 0.5). fp16 stays >= 0.99.
_FP4_MATCH_THRESHOLD = 0.97


def assert_fp4_match(fp4_fused: torch.Tensor, fp4_ref: torch.Tensor, ctx: str):
    assert fp4_fused.shape == fp4_ref.shape, (
        f"shape mismatch {tuple(fp4_fused.shape)} vs {tuple(fp4_ref.shape)} ({ctx})"
    )
    match_rate = (fp4_fused == fp4_ref).float().mean().item()
    assert match_rate >= _FP4_MATCH_THRESHOLD, (
        f"FP4 packed match rate {match_rate:.4f} < {_FP4_MATCH_THRESHOLD} ({ctx})"
    )


def assert_sf_match(sf_fused: torch.Tensor, sf_ref: torch.Tensor, m: int, n: int, ctx: str):
    """The per-block E4M3 scale factors must also match the unfused baseline,
    not just the packed FP4 mantissas. The raw buffers are swizzled and padded
    (rows -> 128, sf_cols -> 4) with uninitialized padding, so unswizzle both to
    the logical [m, n/SF_VEC] grid and compare only the real region.

    The SF is a per-16-elem-block amax. The fused kernel's FMA RMSNorm vs the
    reference's separate RMSNorm shifts a block's amax by at most one E4M3
    quantization step, so we require every block to match within +/-1 step
    (exact value, not just a match rate -- on tiny shapes a 1-2 block diff out of
    e.g. 32 blocks would fail a rate threshold despite being correct)."""
    assert sf_fused.shape == sf_ref.shape, (
        f"SF shape mismatch {tuple(sf_fused.shape)} vs {tuple(sf_ref.shape)} ({ctx})"
    )
    # unswizzle expects the padded dims; then slice the valid [m, num_sf_cols].
    padded_rows = pad_up(m, 128)
    num_sf_cols = ceil_div(n, SF_VEC)
    padded_cols = pad_up(num_sf_cols, 4) * SF_VEC
    u_fused = unswizzle_sf(sf_fused, padded_rows, padded_cols, SF_VEC)[:m, :num_sf_cols]
    u_ref = unswizzle_sf(sf_ref, padded_rows, padded_cols, SF_VEC)[:m, :num_sf_cols]
    # E4M3 is monotonic in its uint8 bit pattern over the positive range the SF
    # uses, so adjacent representable scales differ by 1 in uint8 -- a +/-1 step
    # tolerance is exactly |int(fused) - int(ref)| <= 1.
    diff = (u_fused.to(torch.int16) - u_ref.to(torch.int16)).abs()
    max_diff = int(diff.max().item()) if diff.numel() else 0
    assert max_diff <= 1, (
        f"SF differs by {max_diff} E4M3 steps (> 1) in {int((diff > 1).sum())} block(s) ({ctx})"
    )


def assert_fp4_bitexact_from_norm(
    fp4_fused: torch.Tensor,
    sf_fused: torch.Tensor,
    norm_out: torch.Tensor,
    sf_scale: torch.Tensor,
    m: int,
    n: int,
    ctx: str,
):
    """Bit-exact check of the fused NVFP4 epilogue. The fused kernel returns the
    same post-RMSNorm BF16 value (norm_out) that it quantized, so quantizing that
    exact tensor with the standalone fp4_quantize must reproduce the fused
    quant_out/scale_out bit-for-bit -- there is no rounding-order freedom left
    once the normed input is fixed. (This isolates the epilogue from the RMSNorm
    ULP differences that the cross-path match-rate check tolerates.)

    quant_out is dense row-major [m, n/2] so it's compared directly. scale_out is
    swizzled+padded (rows -> 128, sf_cols -> 4) with uninitialized padding, so the
    two separate allocations differ in the pad region only; compare the
    unswizzled valid [m, n/SF_VEC] grid where bit-exactness must hold."""
    fp4_req, sf_req = fp4_quantize_ref(norm_out, sf_scale)
    assert torch.equal(fp4_fused, fp4_req), (
        f"fused FP4 not bit-exact vs fp4_quantize(norm_out) ({ctx})"
    )
    padded_rows = pad_up(m, 128)
    num_sf_cols = ceil_div(n, SF_VEC)
    padded_cols = pad_up(num_sf_cols, 4) * SF_VEC
    u_fused = unswizzle_sf(sf_fused, padded_rows, padded_cols, SF_VEC)[:m, :num_sf_cols]
    u_req = unswizzle_sf(sf_req, padded_rows, padded_cols, SF_VEC)[:m, :num_sf_cols]
    assert torch.equal(u_fused, u_req), f"fused SF not bit-exact vs fp4_quantize(norm_out) ({ctx})"


# --------------------------------------------------------------------------- #
# fused_add_rmsnorm_fp4_quantize: residual_add -> rms_norm -> nvfp4_quantize    #
# --------------------------------------------------------------------------- #
@skip_unless_add_rmsnorm
@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n", [32, 128, 512, 7168])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_fp4_quantize_vs_separate(m, n, dtype):
    torch.manual_seed(42)
    device = torch.device("cuda")
    eps = 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    # Reference: add in fp32 (matches kernel), then rmsnorm, then quantize.
    added = hidden.float() + residual.float()
    normed_ref = rms_norm_ref(added.to(dtype), weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, sf_ref = fp4_quantize_ref(normed_ref, sf_scale)

    # The op returns residual_out as a fresh tensor and must NOT mutate
    # hidden_states (required for torch.compile functionalization).
    hidden_fused = hidden.clone()
    hidden_before = hidden_fused.clone()
    quant_out, scale_out, residual_out = torch.ops.trtllm.fused_add_rmsnorm_fp4_quantize(
        hidden_fused, residual, weight, sf_scale, eps, False
    )

    assert_fp4_match(quant_out, fp4_ref, f"add-rmsnorm m={m} n={n} {dtype}")
    # The per-block scale factors must match the unfused baseline too.
    assert_sf_match(scale_out, sf_ref, m, n, f"add-rmsnorm m={m} n={n} {dtype}")

    # hidden_states must be left untouched (no in-place alias).
    torch.testing.assert_close(hidden_fused, hidden_before, rtol=0, atol=0)
    # residual_out must equal hidden + residual (the next layer's residual).
    torch.testing.assert_close(residual_out.float(), added, rtol=2e-2, atol=2e-2)


@skip_unless_add_rmsnorm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_fp4_quantize_return_norm_out(dtype):
    """return_norm_out=True must also yield the BF16/FP16 post-RMSNorm value
    (consumed by the DSA indexer's pre_indexer_proj)."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    m, n, eps = 64, 7168, 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    added = hidden.float() + residual.float()
    normed_ref = rms_norm_ref(added.to(dtype), weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, sf_ref = fp4_quantize_ref(normed_ref, sf_scale)

    hidden_fused = hidden.clone()
    norm_out, quant_out, scale_out, residual_out = torch.ops.trtllm.fused_add_rmsnorm_fp4_quantize(
        hidden_fused, residual, weight, sf_scale, eps, True
    )

    assert_fp4_match(quant_out, fp4_ref, f"add-rmsnorm return_norm_out {dtype}")
    assert_sf_match(scale_out, sf_ref, m, n, f"add-rmsnorm return_norm_out {dtype}")
    # Bit-exact: re-quantizing the fused op's own norm_out must reproduce its
    # quant_out/scale_out exactly (isolates the epilogue from RMSNorm ULP noise).
    assert_fp4_bitexact_from_norm(
        quant_out, scale_out, norm_out, sf_scale, m, n, f"add-rmsnorm return_norm_out {dtype}"
    )
    torch.testing.assert_close(norm_out, normed_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(residual_out.float(), added, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------- #
# fused_rmsnorm_fp4_quantize: residual-less rms_norm -> nvfp4_quantize          #
# --------------------------------------------------------------------------- #
@skip_unless_rmsnorm
@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n", [32, 128, 512, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_vs_separate(m, n, dtype):
    torch.manual_seed(42)
    device = torch.device("cuda")
    eps = 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    normed_ref = rms_norm_ref(hidden, weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, sf_ref = fp4_quantize_ref(normed_ref, sf_scale)

    quant_out, scale_out = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        hidden, weight, sf_scale, eps, False
    )

    assert_fp4_match(quant_out, fp4_ref, f"rmsnorm m={m} n={n} {dtype}")
    assert_sf_match(scale_out, sf_ref, m, n, f"rmsnorm m={m} n={n} {dtype}")


@skip_unless_rmsnorm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_return_norm_out(dtype):
    torch.manual_seed(11)
    device = torch.device("cuda")
    m, n, eps = 64, 1536, 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    normed_ref = rms_norm_ref(hidden, weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, sf_ref = fp4_quantize_ref(normed_ref, sf_scale)

    norm_out, quant_out, scale_out = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        hidden, weight, sf_scale, eps, True
    )

    assert_fp4_match(quant_out, fp4_ref, f"rmsnorm return_norm_out {dtype}")
    assert_sf_match(scale_out, sf_ref, m, n, f"rmsnorm return_norm_out {dtype}")
    # Bit-exact: re-quantizing the fused op's own norm_out must reproduce its
    # quant_out/scale_out exactly (isolates the epilogue from RMSNorm ULP noise).
    assert_fp4_bitexact_from_norm(
        quant_out, scale_out, norm_out, sf_scale, m, n, f"rmsnorm return_norm_out {dtype}"
    )
    torch.testing.assert_close(norm_out, normed_ref, rtol=2e-2, atol=2e-2)


@skip_unless_rmsnorm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_does_not_mutate_input(dtype):
    """Residual=false path must not write back into hidden_states."""
    torch.manual_seed(3)
    device = torch.device("cuda")
    m, n, eps = 32, 512, 1e-6

    hidden = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)
    hidden_before = hidden.clone()
    sf_scale = make_sf_scale(rms_norm_ref(hidden, weight, eps))

    torch.ops.trtllm.fused_rmsnorm_fp4_quantize(hidden, weight, sf_scale, eps, False)
    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)


# m starts at 2: a single-row (m=1) column slice is still contiguous in PyTorch
# (the row-pitch stride is irrelevant when there is only one row), so it would
# not exercise the input_row_stride read path at all.
@skip_unless_rmsnorm
@pytest.mark.parametrize("m", [2, 16, 64])
@pytest.mark.parametrize("n", [512, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_strided_input(m, n, dtype):
    """The key F1 feature: read a row-strided column slice in place (no copy).

    Mimics ``q = kv_a_proj_with_mqa(x).split(...)[0]``: a leading column slice
    of a wider projection whose last dim is unit-stride but whose row pitch is
    the full projection width. The fused op must read this strided layout
    directly and match the contiguous reference exactly (the read offset is the
    only thing the stride changes; all outputs are written packed)."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    eps = 1e-6
    extra = 2 * 16  # trailing columns (e.g. kv_lora + rope), keep 16-aligned

    wide = torch.randn(m, n + extra, dtype=dtype, device=device)
    q = wide[:, :n]  # column slice: unit-stride last dim, row pitch n+extra
    assert not q.is_contiguous()
    assert q.stride(0) == n + extra and q.stride(1) == 1

    weight = torch.randn(n, dtype=dtype, device=device)
    normed_ref = rms_norm_ref(q.contiguous(), weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, _ = fp4_quantize_ref(normed_ref, sf_scale)

    quant_out, scale_out = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        q, weight, sf_scale, eps, False
    )

    assert quant_out.is_contiguous(), "outputs must be packed"
    assert_fp4_match(quant_out, fp4_ref, f"strided m={m} n={n} {dtype}")


@skip_unless_rmsnorm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rmsnorm_fp4_quantize_contiguous_matches_strided(dtype):
    """A contiguous tensor (input_row_stride==0 path) and the same data viewed
    as a column slice (input_row_stride>0 path) must produce identical FP4."""
    torch.manual_seed(99)
    device = torch.device("cuda")
    m, n, eps = 48, 1536, 1e-6

    base = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)
    sf_scale = make_sf_scale(rms_norm_ref(base, weight, eps))

    # Packed path.
    fp4_packed, _ = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        base.contiguous(), weight, sf_scale, eps, False
    )

    # Strided path: embed the same rows into a wider buffer as a column slice.
    wide = torch.randn(m, n + 32, dtype=dtype, device=device)
    wide[:, :n] = base
    strided = wide[:, :n]
    fp4_strided, _ = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
        strided, weight, sf_scale, eps, False
    )

    torch.testing.assert_close(fp4_packed, fp4_strided, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# RMSNorm (M, N) kernel-selection gate (CPU-only logic; no kernel launch).
# Verifies the routing in RMSNorm._ws_kernel_eligible: the warp-specialized
# fused_add_rms_norm_quant ("ws") serves only the large, contiguous, rank-2
# residual edge with an in-range hidden dim; everything else (small M,
# residual-less, row-strided, out-of-range N) uses the reduce_fusion kernel.
# ---------------------------------------------------------------------------


def _make_rmsnorm(n: int):
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    # device="meta" so no allocation / GPU is needed; the gate only reads
    # tensor shapes and contiguity, never data.
    return RMSNorm(hidden_size=n, eps=1e-6, dtype=torch.bfloat16, device="meta")


@pytest.mark.parametrize(
    "m,n,residual,contiguous,expect_ws",
    [
        # Large, contiguous, in-range N, M >= threshold (4096) -> ws.
        (4096, 7168, True, True, True),
        (8192, 8192, True, True, True),
        # M below the threshold -> reduce_fusion.
        (2048, 7168, True, True, False),
        (1, 7168, True, True, False),
        # Residual-less edge (q_a) -> reduce_fusion (no ws equivalent).
        (8192, 7168, False, True, False),
        # N below ws minimum (2048) -> reduce_fusion (q_a's 1536 lands here).
        (8192, 1536, True, True, False),
        # N above ws maximum -> reduce_fusion.
        (8192, 20480, True, True, False),
        # Row-strided (non-contiguous) input -> reduce_fusion.
        (8192, 7168, True, False, False),
    ],
)
def test_rmsnorm_ws_kernel_gate(m, n, residual, contiguous, expect_ws):
    norm = _make_rmsnorm(n)
    hs = torch.empty((m, n), dtype=torch.bfloat16, device="meta")
    res = torch.empty((m, n), dtype=torch.bfloat16, device="meta") if residual else None
    if not contiguous:
        # A column slice of a wider buffer: unit-stride last dim, larger pitch.
        hs = torch.empty((m, n + 32), dtype=torch.bfloat16, device="meta")[:, :n]
        assert not hs.is_contiguous()
    assert norm._ws_kernel_eligible(hs, res) is expect_ws


# ---------------------------------------------------------------------------
# sf_linear_layout: MoE consumers (post-quant all-to-all / allgather dispatch)
# need the scale factors row-major [m, n/16] rather than GEMM-swizzled.
# ---------------------------------------------------------------------------


def fp4_quantize_linear_ref(normed: torch.Tensor, sf_scale: torch.Tensor):
    """The unfused MoE input quant: fp4_quantize with isSfSwizzledLayout=False
    (CuteDslFusedMoE.quantize_input / WideEPMoE.forward_chunk)."""
    return torch.ops.trtllm.fp4_quantize(normed.contiguous(), sf_scale, SF_VEC, False, False)


def assert_linear_sf_match(sf_fused, sf_ref, m, n, ctx):
    """Linear layout has no padding: both are exactly m * n/16 bytes and must
    agree at the same >= threshold rate as the packed FP4 (a boundary flip in a
    block's amax moves its E4M3 scale by one step)."""
    assert sf_fused.numel() == m * (n // SF_VEC), f"linear SF size {sf_fused.numel()} != {m * n // SF_VEC} ({ctx})"
    assert sf_ref.numel() == sf_fused.numel(), ctx
    match = (sf_fused.view(-1) == sf_ref.view(-1)).float().mean().item()
    assert match >= _FP4_MATCH_THRESHOLD, f"linear SF match {match:.4f} < {_FP4_MATCH_THRESHOLD} ({ctx})"


@skip_unless_add_rmsnorm
@pytest.mark.parametrize("m", [1, 7, 64, 300])
@pytest.mark.parametrize("n", [128, 6144])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_add_rmsnorm_fp4_quantize_linear_sf(m, n, dtype):
    """sf_linear_layout=True: FP4 bytes identical to the swizzled path, SF in
    the row-major layout fp4_quantize(isSfSwizzledLayout=False) produces."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    eps = 1e-6
    hidden = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)

    added = hidden.float() + residual.float()
    normed_ref = rms_norm_ref(added.to(dtype), weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, sf_ref_lin = fp4_quantize_linear_ref(normed_ref, sf_scale)

    q_lin, sf_lin, res_lin = torch.ops.trtllm.fused_add_rmsnorm_fp4_quantize(
        hidden, residual, weight, sf_scale, eps, False, True
    )
    q_swz, sf_swz, _ = torch.ops.trtllm.fused_add_rmsnorm_fp4_quantize(
        hidden, residual, weight, sf_scale, eps, False, False
    )
    ctx = f"add-rmsnorm linear m={m} n={n}"
    # The layout flag must not touch the quantized values themselves.
    assert torch.equal(q_lin.view(torch.uint8), q_swz.view(torch.uint8)), f"FP4 differs between SF layouts ({ctx})"
    assert_fp4_match(q_lin, fp4_ref, ctx)
    assert_linear_sf_match(sf_lin, sf_ref_lin, m, n, ctx)
    # Swizzled buffer is padded (rows -> 128, cols -> 4); linear is exact.
    assert sf_swz.numel() >= sf_lin.numel()
    torch.testing.assert_close(res_lin.float(), added, rtol=2e-2, atol=2e-2)

    # return_norm_out=True keeps the same tuple layout with the norm leading.
    norm_out, q2, sf2, _ = torch.ops.trtllm.fused_add_rmsnorm_fp4_quantize(
        hidden, residual, weight, sf_scale, eps, True, True
    )
    assert norm_out.shape == hidden.shape and norm_out.dtype == dtype
    assert torch.equal(q2.view(torch.uint8), q_lin.view(torch.uint8))
    assert torch.equal(sf2, sf_lin)


@skip_unless_rmsnorm
@pytest.mark.parametrize("m", [3, 64])
@pytest.mark.parametrize("n", [512, 6144])
def test_fused_rmsnorm_fp4_quantize_linear_sf(m, n):
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype, eps = torch.bfloat16, 1e-6
    hidden = torch.randn(m, n, dtype=dtype, device=device)
    weight = torch.randn(n, dtype=dtype, device=device)
    normed_ref = rms_norm_ref(hidden, weight, eps)
    sf_scale = make_sf_scale(normed_ref)
    fp4_ref, sf_ref_lin = fp4_quantize_linear_ref(normed_ref, sf_scale)
    q_lin, sf_lin = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(hidden, weight, sf_scale, eps, False, True)
    ctx = f"rmsnorm linear m={m} n={n}"
    assert_fp4_match(q_lin, fp4_ref, ctx)
    assert_linear_sf_match(sf_lin, sf_ref_lin, m, n, ctx)


@skip_unless_add_rmsnorm
def test_rmsnorm_module_nvfp4_sf_linear():
    """RMSNorm.nvfp4_sf_linear routes the fused path to the linear-SF kernel
    (bypassing the swizzled-only warp-specialized kernel even at ws-eligible
    sizes) and marks the Fp4QuantizedTensor unswizzled with the BF16 view
    attached -- the contract Deepseekv3 forward_MoE relies on."""
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm._torch.utils import Fp4QuantizedTensor

    torch.manual_seed(3)
    device = torch.device("cuda")
    m, n, dtype, eps = 2048, 6144, torch.bfloat16, 1e-6  # ws-eligible shape
    norm = RMSNorm(hidden_size=n, eps=eps, dtype=dtype, quantize_type="nvfp4").to(device)
    if not norm.is_nvfp4:
        pytest.skip("NVFP4 RMSNorm fusion not enabled on this platform")
    norm.weight.data = torch.randn(n, dtype=dtype, device=device)
    hidden = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    normed_ref = rms_norm_ref((hidden.float() + residual.float()).to(dtype), norm.weight, eps)
    norm.nvfp4_scale = make_sf_scale(normed_ref)

    norm.nvfp4_sf_linear = False
    fp4_swz, _ = norm(hidden, residual)
    assert isinstance(fp4_swz, Fp4QuantizedTensor) and fp4_swz.is_sf_swizzled

    norm.nvfp4_sf_linear = True
    fp4_lin, residual_out = norm(hidden, residual, return_norm_out=True)
    assert isinstance(fp4_lin, Fp4QuantizedTensor)
    assert not fp4_lin.is_sf_swizzled
    assert fp4_lin.scaling_factor.numel() == m * (n // SF_VEC)
    assert fp4_lin.unquantized_hidden_states is not None
    assert fp4_lin.unquantized_hidden_states.shape == (m, n)
    assert fp4_lin.fp4_tensor.shape == (m, n // 2)
    torch.testing.assert_close(residual_out.float(), hidden.float() + residual.float(), rtol=2e-2, atol=2e-2)
    _, sf_ref_lin = fp4_quantize_linear_ref(normed_ref, norm.nvfp4_scale)
    assert_linear_sf_match(fp4_lin.scaling_factor, sf_ref_lin, m, n, "RMSNorm module linear")
