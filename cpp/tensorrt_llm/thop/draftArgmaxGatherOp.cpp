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

// MTP draft-argmax pair all-gather over the NCCL device API (GIN puts or LSA stores) on a
// symmetric window. One persistent runner per NCCL communicator (= per TP group):
//   draft_argmax_gather_init(group)        collective; allocates the window, builds the device comm,
//                                          returns the mode (0 unavailable, 1 LSA, 2 GIN)
//   draft_argmax_gather_staging(group)     [kMaxRows, 2] fp32 view over the window's staging region;
//                                          the pack kernel writes this rank's rows into it
//   draft_argmax_gather(group, rows)       launches the gather, returns [rows, 2 * tp] fp32
// The runner is torn down with the communicator through NcclCommResourceManager.

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/draftArgmaxGatherKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

#if TLLM_DRAFT_ARGMAX_GATHER_DEVICE_API

namespace
{
namespace dag = tensorrt_llm::kernels::draft_argmax_gather;
using tensorrt_llm::common::nccl_util::NcclCommResourceManager;
using tensorrt_llm::common::nccl_util::NCCLWindowAllocator;
using tensorrt_llm::common::nccl_util::NCCLWindowBuffer;

struct Runner
{
    std::shared_ptr<ncclComm_t> parentComm; // the group's communicator (registry key)
    ncclComm_t ownComm{nullptr};            // dedicated split: a failed device-comm attempt cannot poison the model's
    int nRanks{0};
    NCCLWindowBuffer window;                // symmetric, registered on ownComm; owned by NCCLWindowAllocator's pool
    ncclDevComm devComm{};
    bool devCommValid{false};
    dag::Mode mode{dag::Mode::kUnavailable};
};

std::mutex gRunnersMutex;
std::unordered_map<ncclComm_t, std::shared_ptr<Runner>> gRunners;

std::set<int> toGroup(torch::List<int64_t> const& group_)
{
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    return group;
}

void destroyRunner(Runner& r)
{
    if (r.ownComm != nullptr)
    {
        if (r.devCommValid)
        {
            TLLM_NCCL_CHECK_WARN(ncclDevCommDestroy(r.ownComm, &r.devComm));
            r.devCommValid = false;
        }
        if (r.window.isValid())
        {
            NCCLWindowAllocator::getInstance().releaseBuffer(r.ownComm, r.window.ptr);
            r.window = NCCLWindowBuffer{};
        }
        NcclCommResourceManager::getInstance().cleanupResources(r.ownComm);
        TLLM_NCCL_CHECK_WARN(ncclCommDestroy(r.ownComm));
        r.ownComm = nullptr;
    }
    r.mode = dag::Mode::kUnavailable;
}

// min-reduce of one int over the group (collective).
int agreeMin(ncclComm_t comm, int local, cudaStream_t stream)
{
    int* flag = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&flag, sizeof(int)));
    TLLM_CUDA_CHECK(cudaMemcpyAsync(flag, &local, sizeof(int), cudaMemcpyHostToDevice, stream));
    TLLM_NCCL_CHECK(ncclAllReduce(flag, flag, 1, ncclInt32, ncclMin, comm, stream));
    int agreed = 0;
    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
    TLLM_CUDA_CHECK(cudaMemcpy(&agreed, flag, sizeof(int), cudaMemcpyDeviceToHost));
    TLLM_CUDA_CHECK(cudaFree(flag));
    return agreed;
}

// One attempt at `mode` on a fresh split of the group communicator. Collective. Returns true when
// every rank got a usable device comm; on failure the split comm is destroyed on every rank.
bool tryMode(Runner& runner, dag::Mode mode, int rank, cudaStream_t stream)
{
    ncclComm_t own = nullptr;
    TLLM_NCCL_CHECK(ncclCommSplit(*runner.parentComm, /*color=*/0, /*key=*/rank, &own, nullptr));
    NCCLWindowBuffer window = NCCLWindowAllocator::getInstance().requestBuffer(own, dag::windowBytes(runner.nRanks));
    bool ok = window.isValid();
    ncclDevComm devComm{};
    bool devCommValid = false;
    if (ok)
    {
        TLLM_CUDA_CHECK(cudaMemsetAsync(window.ptr, 0, window.size, stream));
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.lsaBarrierCount = 1;
        if (mode == dag::Mode::kGin)
        {
            reqs.ginSignalCount = 1;
            reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
        }
        ncclResult_t const res = ncclDevCommCreate(own, &reqs, &devComm);
        devCommValid = res == ncclSuccess;
        ok = devCommValid;
        if (ok && mode == dag::Mode::kLsa && devComm.lsaSize != devComm.nRanks)
        {
            ok = false; // a peer outside the load/store domain and no GIN to reach it
        }
        if (ok && mode == dag::Mode::kGin && devComm.ginConnectionCount == 0)
        {
            ok = false;
        }
        if (!devCommValid)
        {
            TLLM_LOG_INFO("[draft_argmax_gather] rank %d: ncclDevCommCreate(mode %d) -> %s", rank, int(mode),
                ncclGetErrorString(res));
        }
    }
    // Agree on the parent comm: the split may be in an error state after a failed create.
    bool const allOk = agreeMin(*runner.parentComm, ok ? 1 : 0, stream) != 0;
    if (allOk)
    {
        runner.ownComm = own;
        runner.window = window;
        runner.devComm = devComm;
        runner.devCommValid = true;
        runner.mode = mode;
        return true;
    }
    if (devCommValid)
    {
        TLLM_NCCL_CHECK_WARN(ncclDevCommDestroy(own, &devComm));
    }
    if (window.isValid())
    {
        NCCLWindowAllocator::getInstance().releaseBuffer(own, window.ptr);
    }
    NcclCommResourceManager::getInstance().cleanupResources(own);
    TLLM_NCCL_CHECK_WARN(ncclCommDestroy(own));
    return false;
}

// Collective over the group: every rank must call it. The mode is identical on every rank because
// each decision is reduced with ncclAllReduce(min).
std::shared_ptr<Runner> getOrCreateRunner(std::set<int> const& group)
{
    auto comm = getComm(group);
    TLLM_CHECK_WITH_INFO(comm && *comm, "draft argmax gather: no NCCL communicator for the group");
    {
        std::lock_guard<std::mutex> lock(gRunnersMutex);
        auto it = gRunners.find(*comm);
        if (it != gRunners.end())
        {
            return it->second;
        }
    }

    auto runner = std::make_shared<Runner>();
    runner->parentComm = comm;
    runner->nRanks = static_cast<int>(group.size());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    bool const supported = ncclCommQueryProperties(*comm, &props) == ncclSuccess && props.deviceApiSupport
        && tensorrt_llm::common::nccl_util::isNcclWindowSupported();
    int const hasGin = supported && props.ginType != NCCL_GIN_TYPE_NONE ? 1 : 0;
    TLLM_LOG_INFO("[draft_argmax_gather] rank %d/%d: deviceApi=%d ginType=%d nLsaTeams=%d windows=%d", props.rank,
        props.nRanks, int(props.deviceApiSupport), int(props.ginType), props.nLsaTeams, int(supported));

    if (agreeMin(*comm, supported ? 1 : 0, stream) != 0)
    {
        // GIN first where every rank has a backend, then LSA; each attempt on its own split comm.
        if (agreeMin(*comm, hasGin, stream) != 0 && !tryMode(*runner, dag::Mode::kGin, props.rank, stream))
        {
            TLLM_LOG_WARNING("[draft_argmax_gather] GIN device comm unavailable; trying LSA");
        }
        if (runner->mode == dag::Mode::kUnavailable && !tryMode(*runner, dag::Mode::kLsa, props.rank, stream))
        {
            TLLM_LOG_WARNING("[draft_argmax_gather] LSA device comm unavailable; AllReduce exchange stays in use");
        }
    }
    TLLM_LOG_INFO("[draft_argmax_gather] rank %d: mode %d (lsaSize %d, ginConnections %d)", props.rank,
        static_cast<int>(runner->mode), runner->devCommValid ? runner->devComm.lsaSize : 0,
        runner->devCommValid ? int(runner->devComm.ginConnectionCount) : 0);

    std::lock_guard<std::mutex> lock(gRunnersMutex);
    gRunners[*comm] = runner;
    NcclCommResourceManager::getInstance().registerResource(
        *comm,
        [c = *comm]()
        {
            std::shared_ptr<Runner> r;
            {
                std::lock_guard<std::mutex> l(gRunnersMutex);
                auto it = gRunners.find(c);
                if (it == gRunners.end())
                {
                    return;
                }
                r = it->second;
                gRunners.erase(it);
            }
            destroyRunner(*r);
        },
        "draft_argmax_gather");
    return runner;
}

std::shared_ptr<Runner> findRunner(std::set<int> const& group)
{
    auto comm = getComm(group);
    std::lock_guard<std::mutex> lock(gRunnersMutex);
    auto it = gRunners.find(*comm);
    TLLM_CHECK_WITH_INFO(it != gRunners.end(), "draft argmax gather: call draft_argmax_gather_init first");
    return it->second;
}

} // namespace

int64_t draft_argmax_gather_init(torch::List<int64_t> group_)
{
    auto runner = getOrCreateRunner(toGroup(group_));
    return static_cast<int64_t>(runner->mode);
}

torch::Tensor draft_argmax_gather_staging(torch::List<int64_t> group_)
{
    auto runner = findRunner(toGroup(group_));
    TLLM_CHECK_WITH_INFO(runner->mode != dag::Mode::kUnavailable, "draft argmax gather: unavailable");
    auto* base = static_cast<char*>(runner->window.ptr) + dag::stagingOffset();
    // A view over window memory; the runner (and the window pool) outlive it.
    return torch::from_blob(base, {dag::kMaxRows, dag::kPairFloats},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, at::cuda::current_device()));
}

torch::Tensor draft_argmax_gather(torch::Tensor staging, int64_t rows, torch::List<int64_t> group_)
{
    auto runner = findRunner(toGroup(group_));
    TLLM_CHECK_WITH_INFO(runner->mode != dag::Mode::kUnavailable, "draft argmax gather: unavailable");
    TLLM_CHECK_WITH_INFO(staging.data_ptr() == static_cast<char*>(runner->window.ptr) + dag::stagingOffset(),
        "draft argmax gather: staging must be the tensor returned by draft_argmax_gather_staging");
    TLLM_CHECK_WITH_INFO(
        rows > 0 && rows <= dag::kMaxRows, "draft argmax gather: rows %ld out of [1, %d]", long(rows), dag::kMaxRows);
    auto out = torch::empty({rows, int64_t(dag::kPairFloats) * runner->nRanks}, staging.options());
    dag::Params params{};
    params.devComm = runner->devComm;
    params.window = runner->window.window;
    params.rows = static_cast<int>(rows);
    params.nRanks = runner->nRanks;
    params.out = out.data_ptr<float>();
    dag::launchDraftArgmaxGather(params, runner->mode, at::cuda::getCurrentCUDAStream(staging.get_device()).stream());
    return out;
}

#else  // !TLLM_DRAFT_ARGMAX_GATHER_DEVICE_API

int64_t draft_argmax_gather_init(torch::List<int64_t>)
{
    return 0;
}

torch::Tensor draft_argmax_gather_staging(torch::List<int64_t>)
{
    TLLM_THROW("draft argmax gather: built without the NCCL device API");
}

torch::Tensor draft_argmax_gather(torch::Tensor, int64_t, torch::List<int64_t>)
{
    TLLM_THROW("draft argmax gather: built without the NCCL device API");
}

#endif // TLLM_DRAFT_ARGMAX_GATHER_DEVICE_API

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("draft_argmax_gather_init(int[] group) -> int");
    m.def("draft_argmax_gather_staging(int[] group) -> Tensor");
    m.def("draft_argmax_gather(Tensor staging, int rows, int[] group) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("draft_argmax_gather_staging", &tensorrt_llm::torch_ext::draft_argmax_gather_staging);
    m.impl("draft_argmax_gather", &tensorrt_llm::torch_ext::draft_argmax_gather);
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("draft_argmax_gather_init", &tensorrt_llm::torch_ext::draft_argmax_gather_init);
}
