/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "CutlassUtils.h"

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void computeMxE4m3SfAndOutputScale(float& outputScale,
                                                     cutlass::float_ue8m0_t& sfOut,
                                                     float amax,
                                                     float const& sfScale) {
  float const amaxPow2 = trunc_abs_float_to_pow2(amax);
  float const sfVal = amaxPow2 * (1.f / 256.f) * sfScale;
  cutlass::Array<float, 1> sfArrayFp32;
  sfArrayFp32[0] = sfVal;
  sfOut = castArray<cutlass::float_ue8m0_t>(sfArrayFp32)[0];
  outputScale = sfVal != 0.f ? scale_rcp_exp_only(sfVal) : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void computeMxE4m3SfAndOutputScale(float& outputScale,
                                                     cutlass::float_ue8m0_t& sfOut,
                                                     float amax,
                                                     float const& sfScale,
                                                     float const& sfScaleInv) {
  float const amaxPow2 = trunc_abs_float_to_pow2(amax);
  float const sfVal = amaxPow2 * (1.f / 256.f) * sfScale;
  cutlass::Array<float, 1> sfArrayFp32;
  sfArrayFp32[0] = sfVal;
  sfOut = castArray<cutlass::float_ue8m0_t>(sfArrayFp32)[0];
  outputScale = sfVal != 0.f ? scale_rcp_exp_only(sfVal * sfScaleInv) : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumEltsPerThread, typename OutT>
inline __device__ void convertFloatToMxE4m3(OutT& out,
                                            cutlass::float_ue8m0_t& sfOut,
                                            float const (&input)[NumEltsPerThread],
                                            float sfScale) {
  // MxE4m3 uses one UE8M0 scale for each group of 32 E4M3 elements.
  int32_t constexpr NumEltsPerSf = 32;
  int32_t constexpr NumThreadsPerVec = NumEltsPerSf / NumEltsPerThread;
  static_assert(NumEltsPerSf % NumEltsPerThread == 0 && NumEltsPerThread % 4 == 0,
                "NumEltsPerThread not supported.");
  static_assert(sizeof(OutT) == NumEltsPerThread,
                "Output type not supported."); // 1 byte per element.

  float localAmax = 0.f;
#pragma unroll
  for (int32_t i = 0; i < NumEltsPerThread; ++i) {
    localAmax = fmaxf(localAmax, fabsf(input[i]));
  }

#pragma unroll
  for (int32_t step = 1; step < NumThreadsPerVec; step *= 2) {
    localAmax = fmaxf(__shfl_xor_sync(uint32_t(-1), localAmax, step), localAmax);
  }

  float outputScale;
  computeMxE4m3SfAndOutputScale(outputScale, sfOut, localAmax, sfScale);

  cutlass::Array<float, NumEltsPerThread> scaled;
#pragma unroll
  for (int32_t i = 0; i < NumEltsPerThread; ++i) {
    scaled[i] = input[i] * outputScale;
  }

  using OutVec = cutlass::Array<cutlass::float_e4m3_t, NumEltsPerThread>;
  reinterpret_cast<OutVec&>(out) = castArray<cutlass::float_e4m3_t>(scaled);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumEltsPerThread, typename OutT>
inline __device__ void convertFp16ToMxE4m3(OutT& out,
                                           cutlass::float_ue8m0_t& sfOut,
                                           cutlass::half_t const (&in)[NumEltsPerThread],
                                           float sfScale) {
  // MxE4m3 uses one UE8M0 scale for each group of 32 E4M3 elements.
  int32_t constexpr NumEltsPerSf = 32;
  int32_t constexpr NumThreadsPerVec = NumEltsPerSf / NumEltsPerThread;
  static_assert(NumEltsPerSf % NumEltsPerThread == 0 && NumEltsPerThread % 4 == 0,
                "NumEltsPerThread not supported.");
  static_assert(sizeof(OutT) == NumEltsPerThread,
                "Output type not supported."); // 1 byte per element.

  auto inH2 = reinterpret_cast<half2 const*>(&in[0]);
  auto localAmaxH2 = __habs2(inH2[0]);
#pragma unroll
  for (int32_t i = 0; i < NumEltsPerThread / 2; ++i) {
    localAmaxH2 = __hmax2(localAmaxH2, __habs2(inH2[i]));
  }

  // Perform warp-level reduction to achieve the amax of the vector of 16 elements.
  if constexpr (NumThreadsPerVec > 1) {
    static_assert(NumThreadsPerVec == 2 || NumThreadsPerVec == 4, "Not supported.");
    for (int32_t step = 1; step < NumThreadsPerVec; step *= 2) {
      localAmaxH2 = __hmax2(__shfl_xor_sync(uint32_t(-1), localAmaxH2, step), localAmaxH2);
    }
  }

  float localAmax = float(__hmax(localAmaxH2.x, localAmaxH2.y));
  float outputScale;
  computeMxE4m3SfAndOutputScale(outputScale, sfOut, localAmax, sfScale);

  cutlass::Array<float, NumEltsPerThread> scaled;
#pragma unroll
  for (int32_t i = 0; i < NumEltsPerThread / 2; ++i) {
    float2 tmp = __half22float2(inH2[i]);
    scaled[i * 2 + 0] = tmp.x * outputScale;
    scaled[i * 2 + 1] = tmp.y * outputScale;
  }

  using OutVec = cutlass::Array<cutlass::float_e4m3_t, NumEltsPerThread>;
  reinterpret_cast<OutVec&>(out) = castArray<cutlass::float_e4m3_t>(scaled);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumVals, int32_t NumPackedRegs, typename OutRegs>
inline __device__ void dsv4Fp8QuantEpilogue(OutRegs& out,
                                            cutlass::Array<float, NumVals> const& vals,
                                            float* scalePtr,
                                            int64_t scaleOffset) {
  static_assert(NumVals == NumPackedRegs * 4, "One packed register stores four E4M3 values.");
  static_assert(NumVals == 128, "DSv4 fused epilogue processes one 1x128 quant group.");

  float amax = 0.f;
#pragma unroll
  for (int32_t ii = 0; ii < NumVals; ++ii) {
    amax = fmaxf(amax, fabsf(vals[ii]));
  }

  float const clampedAmax = fmaxf(amax, 1.0e-12f);
  float const outScale = clampedAmax * (1.f / 448.f);
  float const invScale = 448.f / clampedAmax;
  scalePtr[scaleOffset] = outScale;

#pragma unroll
  for (int32_t regIdx = 0; regIdx < NumPackedRegs; ++regIdx) {
    int32_t const ii = regIdx * 4;
    out[regIdx] = convert_float4_to_e4m3(vals[ii + 0] * invScale,
                                         vals[ii + 1] * invScale,
                                         vals[ii + 2] * invScale,
                                         vals[ii + 3] * invScale);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumVals, int32_t NumPackedRegs, bool IsNeox, typename OutRegs>
inline __device__ void dsv4InvRopeFp8QuantEpilogue(OutRegs& out,
                                                   cutlass::Array<float, NumVals>& vals,
                                                   float* scalePtr,
                                                   int32_t const* positionIds,
                                                   float const* cosSinCache,
                                                   int64_t scaleOffset,
                                                   int32_t tokenIdx,
                                                   int32_t headDimOffset,
                                                   int32_t warpGrpThreadIdx) {
  static_assert(NumVals == NumPackedRegs * 4, "One packed register stores four E4M3 values.");
  static_assert(NumVals == 128, "DSv4 fused epilogue processes one 1x128 quant group.");

  int32_t constexpr ropeStart = 448;
  int32_t constexpr ropeHalf = 32;
  int32_t constexpr ropeChunkStart = ropeStart - ropeHalf * 2;

  (void)warpGrpThreadIdx;
  if (headDimOffset == ropeChunkStart) {
    int32_t const position = positionIds[tokenIdx];
    float const* csRow = cosSinCache + position * 64;
#pragma unroll
    for (int32_t ropeIdx = 0; ropeIdx < ropeHalf; ++ropeIdx) {
      int32_t constexpr ropeOffset = ropeStart - ropeChunkStart;
      float const cosVal = csRow[ropeIdx];
      float const sinVal = csRow[ropeHalf + ropeIdx];
      if constexpr (IsNeox) {
        int32_t const ii = ropeOffset + ropeIdx;
        int32_t const jj = ii + ropeHalf;
        float const firstHalf = vals[ii];
        float const secondHalf = vals[jj];
        vals[ii] = firstHalf * cosVal + secondHalf * sinVal;
        vals[jj] = secondHalf * cosVal - firstHalf * sinVal;
      } else {
        int32_t const ii = ropeOffset + ropeIdx * 2;
        int32_t const jj = ii + 1;
        float const even = vals[ii];
        float const odd = vals[jj];
        vals[ii] = even * cosVal + odd * sinVal;
        vals[jj] = odd * cosVal - even * sinVal;
      }
    }
  }

  float amax = 0.f;
#pragma unroll
  for (int32_t ii = 0; ii < NumVals; ++ii) {
    amax = fmaxf(amax, fabsf(vals[ii]));
  }

  float const clampedAmax = fmaxf(amax, 1.0e-12f);
  float const outScale = clampedAmax * (1.f / 448.f);
  float const invScale = 448.f / clampedAmax;
  scalePtr[scaleOffset] = outScale;

#pragma unroll
  for (int32_t regIdx = 0; regIdx < NumPackedRegs; ++regIdx) {
    int32_t const ii = regIdx * 4;
    out[regIdx] = convert_float4_to_e4m3(vals[ii + 0] * invScale,
                                         vals[ii + 1] * invScale,
                                         vals[ii + 2] * invScale,
                                         vals[ii + 3] * invScale);
  }
}

} // namespace dev
} // namespace trtllm
