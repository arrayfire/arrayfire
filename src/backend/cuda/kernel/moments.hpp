/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <backend.hpp>
#include <Param.hpp>
#include <dispatch.hpp>
#include <math.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
namespace kernel
{

    // Kernel Launch Config Values
    static const int THREADS = 128;

    template<typename T>
    __global__
    void moments_kernel(CParam<float> out, CParam<T> in, af_moment_type moment,
                  const dim_t blocksMatX, const bool pBatch)
    {
        const dim_t idw = blockIdx.y / in.dims[2];
        const dim_t idz = blockIdx.y - idw * in.dims[2];

        const dim_t idy = blockIdx.x;
        dim_t idx = threadIdx.x;

        __shared__ float blk_moment_sum[4];
        if(threadIdx.x < 4) {
            blk_moment_sum[threadIdx.x] = 0.f;
        }
        __syncthreads();

        for(unsigned i=0; i<blocksMatX; ++i) {
            dim_t mId = idy * in.strides[1] + idx;
            if(pBatch) {
                mId += idw * in.strides[3] + idz * in.strides[2];
            }

            if (idx >= in.dims[0] || idy >= in.dims[1] ||
                idz >= in.dims[2] || idw >= in.dims[3] )
                break;

            dim_t m_off = 0;
            float val = (float)in.ptr[mId];

            if((moment & AF_MOMENT_M00) > 0) {
                atomicAdd(blk_moment_sum + m_off++, val);
            }
            if((moment & AF_MOMENT_M01) > 0) {
                atomicAdd(blk_moment_sum + m_off++, idx * val);
            }
            if((moment & AF_MOMENT_M10) > 0) {
                atomicAdd(blk_moment_sum + m_off++, idy * val);
            }
            if((moment & AF_MOMENT_M11) > 0) {
                atomicAdd(blk_moment_sum + m_off, idx * idy * val);
            }

            idx += blockDim.x;
        }

        __syncthreads();

        float *offset = const_cast<float *>(out.ptr + (idw * out.strides[3] + idz * out.strides[2]) + threadIdx.x);
        if(threadIdx.x < out.dims[0])
            atomicAdd(offset, blk_moment_sum[threadIdx.x]);
    }

    // Wrapper functions
    template <typename T>
    void moments(Param<float> out, CParam<T> in, const af_moment_type moment) {
        dim3 threads(THREADS, 1, 1);
        dim_t blocksMatX = divup(in.dims[0], threads.x);
        dim3 blocks(in.dims[1], in.dims[2] * in.dims[3]);

        bool pBatch = !(in.dims[2] == 1 && in.dims[3] == 1);

        CUDA_LAUNCH((moments_kernel<T>), blocks, threads,
                     out, in, moment, blocksMatX, pBatch);
        POST_LAUNCH_CHECK();
    }

}
}
