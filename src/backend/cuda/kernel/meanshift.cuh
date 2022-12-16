/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <math.hpp>

namespace arrayfire {
namespace cuda {

template<typename AccType, typename T, int channels>
__global__ void meanshift(Param<T> out, CParam<T> in, int radius, float cvar,
                          uint numIters, int nBBS0, int nBBS1) {
    unsigned b2 = blockIdx.x / nBBS0;
    unsigned b3 = blockIdx.y / nBBS1;
    const T* iptr =
        (const T*)in.ptr + (b2 * in.strides[2] + b3 * in.strides[3]);
    T* optr      = (T*)out.ptr + (b2 * out.strides[2] + b3 * out.strides[3]);
    const int gx = blockDim.x * (blockIdx.x - b2 * nBBS0) + threadIdx.x;
    const int gy = blockDim.y * (blockIdx.y - b3 * nBBS1) + threadIdx.y;

    if (gx >= in.dims[0] || gy >= in.dims[1]) return;

    int meanPosI = gx;
    int meanPosJ = gy;

    T currentCenterColors[channels];
    T tempColors[channels];

    AccType currentMeanColors[channels];

#pragma unroll
    for (int ch = 0; ch < channels; ++ch)
        currentCenterColors[ch] = iptr[(
            gx * in.strides[0] + gy * in.strides[1] + ch * in.strides[2])];

    const int dim0LenLmt = in.dims[0] - 1;
    const int dim1LenLmt = in.dims[1] - 1;

    // scope of meanshift iterations begin
    for (uint it = 0; it < numIters; ++it) {
        int oldMeanPosJ = meanPosJ;
        int oldMeanPosI = meanPosI;
        unsigned count  = 0;

        int shift_x = 0;
        int shift_y = 0;

#pragma unroll
        for (int ch = 0; ch < channels; ++ch) currentMeanColors[ch] = 0;

        for (int wj = -radius; wj <= radius; ++wj) {
            int hit_count = 0;
            int tj        = meanPosJ + wj;

            if (tj < 0 || tj > dim1LenLmt) continue;

            for (int wi = -radius; wi <= radius; ++wi) {
                int ti = meanPosI + wi;

                if (ti < 0 || ti > dim0LenLmt) continue;

                AccType norm = 0;
#pragma unroll
                for (int ch = 0; ch < channels; ++ch) {
                    tempColors[ch] =
                        iptr[(ti * in.strides[0] + tj * in.strides[1] +
                              ch * in.strides[2])];
                    AccType diff = (AccType)currentCenterColors[ch] -
                                   (AccType)tempColors[ch];
                    norm += (diff * diff);
                }

                if (norm <= cvar) {
#pragma unroll
                    for (int ch = 0; ch < channels; ++ch)
                        currentMeanColors[ch] += (AccType)tempColors[ch];

                    shift_x += ti;
                    ++hit_count;
                }
            }
            count += hit_count;
            shift_y += tj * hit_count;
        }

        if (count == 0) break;

        const AccType fcount = 1 / (AccType)count;

        meanPosI = __float2int_rz(shift_x * fcount);
        meanPosJ = __float2int_rz(shift_y * fcount);

#pragma unroll
        for (int ch = 0; ch < channels; ++ch)
            currentMeanColors[ch] =
                __float2int_rz(currentMeanColors[ch] * fcount);

        AccType norm = 0;
#pragma unroll
        for (int ch = 0; ch < channels; ++ch) {
            AccType diff =
                (AccType)currentCenterColors[ch] - currentMeanColors[ch];
            norm += (diff * diff);
        }

        bool stop = (meanPosJ == oldMeanPosJ && meanPosI == oldMeanPosI) ||
                    ((abs(oldMeanPosJ - meanPosJ) +
                      abs(oldMeanPosI - meanPosI) + norm) <= 1);

#pragma unroll
        for (int ch = 0; ch < channels; ++ch)
            currentCenterColors[ch] = (T)(currentMeanColors[ch]);

        if (stop) break;
    }  // scope of meanshift iterations end

#pragma unroll
    for (int ch = 0; ch < channels; ++ch)
        optr[(gx * out.strides[0] + gy * out.strides[1] +
              ch * out.strides[2])] = currentCenterColors[ch];
}

}  // namespace cuda
}  // namespace arrayfire
