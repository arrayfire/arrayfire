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
#include <shared.hpp>
#include <types.hpp>

namespace arrayfire {
namespace cuda {

template<typename T, bool isLinear>
__global__ void histogram(Param<uint> out, CParam<T> in, int len, int nbins,
                          float minval, float maxval, int nBBS) {
    SharedMemory<uint> shared;
    uint *shrdMem = shared.getPointer();

    // offset input and output to account for batch ops
    unsigned b2 = blockIdx.x / nBBS;
    const data_t<T> *iptr =
        in.ptr + b2 * in.strides[2] + blockIdx.y * in.strides[3];
    uint *optr = out.ptr + b2 * out.strides[2] + blockIdx.y * out.strides[3];

    int start = (blockIdx.x - b2 * nBBS) * THRD_LOAD * blockDim.x + threadIdx.x;
    int end   = min((start + THRD_LOAD * blockDim.x), len);
    float step = (maxval - minval) / (float)nbins;
    compute_t<T> minvalT(minval);

    // If nbins > max shared memory allocated, then just use atomicAdd on global
    // memory
    bool use_global = nbins > MAX_BINS;

    // Skip initializing shared memory
    if (!use_global) {
        for (int i = threadIdx.x; i < nbins; i += blockDim.x) shrdMem[i] = 0;
        __syncthreads();
    }

    for (int row = start; row < end; row += blockDim.x) {
        int idx =
            isLinear
                ? row
                : ((row % in.dims[0]) + (row / in.dims[0]) * in.strides[1]);
        int bin =
            (int)(static_cast<float>(compute_t<T>(iptr[idx]) - minvalT) / step);
        bin = (bin < 0) ? 0 : bin;
        bin = (bin >= nbins) ? (nbins - 1) : bin;

        if (use_global) {
            atomicAdd((optr + bin), 1);
        } else {
            atomicAdd((shrdMem + bin), 1);
        }
    }

    // No need to write to global if use_global is true
    if (!use_global) {
        __syncthreads();
        for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
            atomicAdd((optr + i), shrdMem[i]);
        }
    }
}

}  // namespace cuda
}  // namespace arrayfire
