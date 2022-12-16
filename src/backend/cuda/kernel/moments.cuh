/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>

namespace arrayfire {
namespace cuda {

template<typename T>
__global__ void moments(Param<float> out, CParam<T> in, af::momentType moment,
                        const bool pBatch) {
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

}  // namespace cuda
}  // namespace arrayfire
