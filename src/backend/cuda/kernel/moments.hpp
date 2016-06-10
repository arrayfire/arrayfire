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

    // Moment functions
    template<typename T>
    __device__ inline static
    T moments_m00(const dim_t mId, const dim_t idx, const dim_t idy, CParam<T> in)
    {
        return in.ptr[mId];
    }

    template<typename T>
    __device__ inline static
    T moments_m01(const dim_t mId, const dim_t idx, const dim_t idy, CParam<T> in)
    {
        return idx * in.ptr[mId];
    }

    template<typename T>
    __device__ inline static
    T moments_m10(const dim_t mId, const dim_t idx, const dim_t idy, CParam<T> in)
    {
        return idy * in.ptr[mId];
    }

    template<typename T>
    __device__ inline static
    T moments_m11(const dim_t mId, const dim_t idx, const dim_t idy, CParam<T> in)
    {

        return idx * idy * in.ptr[mId];
    }

    template<typename T, af_moment_type moment>
    __global__
    void moments_kernel(CParam<float> out, CParam<T> in,
                  const dim_t blocksMatX, const bool pBatch)
    {
        const dim_t idw = blockIdx.y / in.dims[2];
        const dim_t idz = blockIdx.y - idw * in.dims[2];

        const dim_t idy = blockIdx.x / blocksMatX;
        const dim_t blockIdx_x = blockIdx.x - idy * blocksMatX;
        const dim_t idx = blockIdx_x * blockDim.x + threadIdx.x;

        dim_t mId = idy * in.strides[1] + idx;
        if(pBatch) {
            mId += idw * in.strides[3] + idz * in.strides[2];
        }

        if (idx >= in.dims[0] || idy >= in.dims[1] ||
            idz >= in.dims[2] || idw >= in.dims[3] )
            return;

        __shared__ float blk_moment_sum;
        blk_moment_sum = 0.f;
        __syncthreads();

        switch(moment) {
            case AF_MOMENT_M00:
                atomicAdd(&blk_moment_sum, (float)moments_m00(mId, idx, idy, in));
                break;
            case AF_MOMENT_M01:
                atomicAdd(&blk_moment_sum, (float)moments_m01(mId, idx, idy, in));
                break;
            case AF_MOMENT_M10:
                atomicAdd(&blk_moment_sum, (float)moments_m10(mId, idx, idy, in));
                break;
            case AF_MOMENT_M11:
                atomicAdd(&blk_moment_sum, (float)moments_m11(mId, idx, idy, in));
                break;
            default:
                break;
        }
        __syncthreads();

        float *offset = const_cast<float *>(out.ptr + (idw * out.strides[1] + idz));
        if(threadIdx.x == 0)
            atomicAdd(offset, blk_moment_sum);
    }

    // Wrapper functions
    template <typename T, af_moment_type moment>
    void moments(Param<float> out, CParam<T> in) {
        dim3 threads(THREADS, 1, 1);
        dim_t blocksMatX = divup(in.dims[0], threads.x);
        dim3 blocks(blocksMatX * in.dims[1], in.dims[2] * in.dims[3]);

        bool pBatch = !(in.dims[2] == 1 && in.dims[3] == 1);

        CUDA_LAUNCH((moments_kernel<T, moment>), blocks, threads,
                     out, in, blocksMatX, pBatch);
        POST_LAUNCH_CHECK();
    }

}
}
