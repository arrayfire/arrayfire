/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <backend.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <af/defines.h>

namespace cuda {
namespace kernel {

// Kernel Launch Config Values
static const int THREADS = 128;

template <typename T>
__global__ void moments_kernel(Param<float> out, CParam<T> in,
                               af_moment_type moment, const bool pBatch) {
    const dim_t idw = blockIdx.y / in.dims[2];
    const dim_t idz = blockIdx.y - idw * in.dims[2];

    const dim_t idy = blockIdx.x;
    dim_t idx       = threadIdx.x;

    if (idy >= in.dims[1] || idz >= in.dims[2] || idw >= in.dims[3]) return;

    extern __shared__ float blk_moment_sum[];
    if (threadIdx.x < out.dims[0]) { blk_moment_sum[threadIdx.x] = 0.f; }
    __syncthreads();

    dim_t mId = idy * in.strides[1] + idx;
    if (pBatch) { mId += idw * in.strides[3] + idz * in.strides[2]; }

    for (; idx < in.dims[0]; idx += blockDim.x) {
        dim_t m_off = 0;
        float val   = (float)in.ptr[mId];
        mId += blockDim.x;

        if ((moment & AF_MOMENT_M00) > 0) {
            atomicAdd(blk_moment_sum + m_off++, val);
        }
        if ((moment & AF_MOMENT_M01) > 0) {
            atomicAdd(blk_moment_sum + m_off++, idx * val);
        }
        if ((moment & AF_MOMENT_M10) > 0) {
            atomicAdd(blk_moment_sum + m_off++, idy * val);
        }
        if ((moment & AF_MOMENT_M11) > 0) {
            atomicAdd(blk_moment_sum + m_off, idx * idy * val);
        }
    }

    __syncthreads();

    float *offset = const_cast<float *>(
        out.ptr + (idw * out.strides[3] + idz * out.strides[2]) + threadIdx.x);
    if (threadIdx.x < out.dims[0])
        atomicAdd(offset, blk_moment_sum[threadIdx.x]);
}

// Wrapper functions
template <typename T>
void moments(Param<float> out, CParam<T> in, const af_moment_type moment) {
    dim3 threads(THREADS, 1, 1);
    dim3 blocks(in.dims[1], in.dims[2] * in.dims[3]);

    bool pBatch = !(in.dims[2] == 1 && in.dims[3] == 1);

    CUDA_LAUNCH_SMEM((moments_kernel<T>), blocks, threads,
                     sizeof(float) * out.dims[0], out, in, moment, pBatch);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
