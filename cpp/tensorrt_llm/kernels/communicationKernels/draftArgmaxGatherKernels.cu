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

#include "tensorrt_llm/kernels/communicationKernels/draftArgmaxGatherKernels.h"

#if TLLM_DRAFT_ARGMAX_GATHER_DEVICE_API

#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::draft_argmax_gather
{
namespace
{

constexpr int kThreads = 256;

// Rank-major slots [nRanks][kMaxRows][2] -> row-major out [rows][2 * nRanks].
__device__ __forceinline__ void writeOut(float const* recv, float* out, int rows, int nRanks)
{
    int const total = rows * nRanks * kPairFloats;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
    {
        int const row = i / (nRanks * kPairFloats);
        int const rem = i % (nRanks * kPairFloats);
        int const r = rem / kPairFloats;
        int const k = rem % kPairFloats;
        out[i] = recv[(size_t(r) * kMaxRows + row) * kPairFloats + k];
    }
}

// GIN: put this rank's rows into slot `rank` of the current parity buffer on every peer; each put
// bumps signal 0 on the peer once it has landed. The reader waits until (nRanks - 1) * (step + 1)
// puts have arrived in total, which with one signal and monotonic counting is exactly "everyone's
// rows for this step are here". flush() makes the staging buffer reusable by the next pack kernel.
__global__ void __launch_bounds__(kThreads) ginGatherKernel(Params params)
{
    int const nRanks = params.devComm.nRanks;
    int const rank = params.devComm.rank;
    uint64_t* stepPtr = reinterpret_cast<uint64_t*>(ncclGetLocalPointer(params.window, stepOffset(nRanks)));
    uint64_t const step = *stepPtr;
    size_t const bufOff = recvOffset() + size_t(step & 1) * bufBytes(nRanks);
    size_t const bytes = size_t(params.rows) * kPairFloats * sizeof(float);

    float const* staging = reinterpret_cast<float const*>(ncclGetLocalPointer(params.window, stagingOffset()));
    float* recv = reinterpret_cast<float*>(ncclGetLocalPointer(params.window, bufOff));
    for (int i = threadIdx.x; i < params.rows * kPairFloats; i += blockDim.x)
    {
        recv[size_t(rank) * kMaxRows * kPairFloats + i] = staging[i];
    }

    ncclGin gin{params.devComm, /*contextIndex=*/0};
    ncclTeam const world = ncclTeamWorld(params.devComm);
    if (threadIdx.x < nRanks - 1)
    {
        int const peer = (rank + 1 + threadIdx.x) % nRanks;
        gin.put(world, peer, params.window, bufOff + size_t(rank) * slotBytes(), params.window, stagingOffset(), bytes,
            ncclGin_WeakSignalInc{0}, ncclGin_None{}, ncclCoopThread{});
    }
    gin.flush(ncclCoopCta{});
    gin.waitSignal(ncclCoopCta{}, 0, uint64_t(nRanks - 1) * (step + 1));
    __syncthreads();

    writeOut(recv, params.out, params.rows, nRanks);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *stepPtr = step + 1;
    }
}

// LSA: store this rank's rows straight into every peer's slot over the fabric, then one LSA barrier
// (release/acquire) and read the local buffer.
__global__ void __launch_bounds__(kThreads) lsaGatherKernel(Params params)
{
    int const nRanks = params.devComm.lsaSize;
    int const rank = params.devComm.lsaRank;
    uint64_t* stepPtr = reinterpret_cast<uint64_t*>(ncclGetLocalPointer(params.window, stepOffset(nRanks)));
    uint64_t const step = *stepPtr;
    size_t const bufOff = recvOffset() + size_t(step & 1) * bufBytes(nRanks);

    float const* staging = reinterpret_cast<float const*>(ncclGetLocalPointer(params.window, stagingOffset()));
    int const n = params.rows * kPairFloats;
    for (int i = threadIdx.x; i < nRanks * n; i += blockDim.x)
    {
        int const peer = i / n;
        int const e = i % n;
        float* dst
            = reinterpret_cast<float*>(ncclGetLsaPointer(params.window, bufOff + size_t(rank) * slotBytes(), peer));
        dst[e] = staging[e];
    }
    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta{}, params.devComm, ncclTeamTagLsa{}, 0};
    bar.sync(ncclCoopCta{}, cuda::memory_order_acq_rel);

    float const* recv = reinterpret_cast<float const*>(ncclGetLocalPointer(params.window, bufOff));
    writeOut(recv, params.out, params.rows, nRanks);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *stepPtr = step + 1;
    }
}

} // namespace

void launchDraftArgmaxGather(Params const& params, Mode mode, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.rows > 0 && params.rows <= kMaxRows, "rows %d out of [1, %d]", params.rows, kMaxRows);
    TLLM_CHECK_WITH_INFO(params.nRanks - 1 <= kThreads, "nRanks %d exceeds the one-put-per-thread CTA", params.nRanks);
    switch (mode)
    {
    case Mode::kGin: ginGatherKernel<<<1, kThreads, 0, stream>>>(params); break;
    case Mode::kLsa: lsaGatherKernel<<<1, kThreads, 0, stream>>>(params); break;
    default: TLLM_CHECK_WITH_INFO(false, "draft argmax gather: no device-API mode selected");
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels::draft_argmax_gather

TRTLLM_NAMESPACE_END

#endif // TLLM_DRAFT_ARGMAX_GATHER_DEVICE_API
