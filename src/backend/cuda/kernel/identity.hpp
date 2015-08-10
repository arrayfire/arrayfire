/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <platform.hpp>
#include <debug_cuda.hpp>
#include <Param.hpp>
#include <math.hpp>

namespace cuda
{
namespace kernel
{

    template<typename T>
    __global__
    static void identity_kernel(Param<T> out, int blocks_x, int blocks_y)
    {
        unsigned idz = blockIdx.x / blocks_x;
        unsigned idw = blockIdx.y / blocks_y;

        unsigned blockIdx_x = blockIdx.x - idz * blocks_x;
        unsigned blockIdx_y = blockIdx.y - idw * blocks_y;

        unsigned idx = threadIdx.x + blockIdx_x * blockDim.x;
        unsigned idy = threadIdx.y + blockIdx_y * blockDim.y;

        if(idx >= out.dims[0] ||
           idy >= out.dims[1] ||
           idz >= out.dims[2] ||
           idw >= out.dims[3])
            return;

        T *ptr = out.ptr + idz * out.strides[2] + idw * out.strides[3];
        T val = (idx == idy) ? scalar<T>(1) : scalar<T>(0);
        ptr[idx + idy * out.strides[1]] = val;
    }

    template<typename T>
    static void identity(Param<T> out)
    {
        dim3 threads(32, 8);
        int blocks_x = divup(out.dims[0], threads.x);
        int blocks_y = divup(out.dims[1], threads.y);
        dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

        CUDA_LAUNCH((identity_kernel<T>), blocks, threads, out, blocks_x, blocks_y);
        POST_LAUNCH_CHECK();
    }
}
}
