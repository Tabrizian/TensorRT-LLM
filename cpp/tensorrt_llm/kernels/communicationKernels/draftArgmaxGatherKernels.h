/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif

// The NCCL device API (device communicators, symmetric windows, GIN and LSA) ships with NCCL 2.28;
// the signal variants this kernel uses are 2.30.5+.
#if ENABLE_MULTI_DEVICE && defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
#define TLLM_DRAFT_ARGMAX_GATHER_DEVICE_API 1
#include <nccl_device.h>
#else
#define TLLM_DRAFT_ARGMAX_GATHER_DEVICE_API 0
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels::draft_argmax_gather
{

//! Tiny all-gather of one fp32 (index, value) pair per row per rank, written with the NCCL
//! device API: each rank's rows land in a per-rank slot of a parity double buffer that lives in
//! a symmetric NCCL window shared by the whole TP group.
//!
//! Window layout (bytes from the window base, every region 4 KiB aligned):
//!   [staging]   kMaxRows x 2 fp32   this rank's rows, written by the pack kernel (plain local memory)
//!   [recv]      kNumBufs x nRanks x kMaxRows x 2 fp32   rank-major slots, one parity buffer per step
//!   [step]      uint64   device-side step counter (CUDA-graph replays keep counting)
constexpr int kMaxRows = 256;
constexpr int kPairFloats = 2;
constexpr int kNumBufs = 2; // a peer is at most one step ahead of a reader
constexpr size_t kAlign = 4096;

constexpr size_t alignUp(size_t x)
{
    return (x + kAlign - 1) / kAlign * kAlign;
}

constexpr size_t slotBytes()
{
    return size_t(kMaxRows) * kPairFloats * sizeof(float);
}

constexpr size_t bufBytes(int nRanks)
{
    return slotBytes() * size_t(nRanks);
}

constexpr size_t stagingOffset()
{
    return 0;
}

constexpr size_t recvOffset()
{
    return alignUp(slotBytes());
}

constexpr size_t stepOffset(int nRanks)
{
    return recvOffset() + alignUp(bufBytes(nRanks) * kNumBufs);
}

constexpr size_t windowBytes(int nRanks)
{
    return stepOffset(nRanks) + kAlign;
}

enum class Mode : int
{
    kUnavailable = 0,
    kLsa = 1, // every peer is load/store accessible: remote stores + one LSA barrier
    kGin = 2, // GIN puts with a weak signal increment per put
};

#if TLLM_DRAFT_ARGMAX_GATHER_DEVICE_API
struct Params
{
    ncclDevComm devComm;
    ncclWindow_t window;
    int rows;
    int nRanks;
    float* out; // [rows, 2 * nRanks] row-major, rank-major pairs
};

//! Launch one CTA. The kernel reads its parity and expected signal count from the window's step
//! counter and increments it when done, so back-to-back launches and CUDA-graph replays need no
//! host-side state.
void launchDraftArgmaxGather(Params const& params, Mode mode, cudaStream_t stream);
#endif

} // namespace kernels::draft_argmax_gather

TRTLLM_NAMESPACE_END
