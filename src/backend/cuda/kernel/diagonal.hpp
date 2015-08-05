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
    __global__ static void
    diagCreateKernel(Param<T> out, CParam<T> in, int num, int blocks_x)
    {
        unsigned idz = blockIdx.x / blocks_x;
        unsigned blockIdx_x = blockIdx.x - idz * blocks_x;

        unsigned idx = threadIdx.x + blockIdx_x * blockDim.x;
        unsigned idy = threadIdx.y + blockIdx.y * blockDim.y;

        if (idx >= out.dims[0] ||
            idy >= out.dims[1] ||
            idz >= out.dims[2]) return;


        T *optr = out.ptr + idz * out.strides[2] + idy * out.strides[1] + idx;
        const T *iptr = in.ptr  + idz *  in.strides[1] + ((num > 0) ? idx : idy);

        T val = (idx == (idy - num)) ? *iptr : scalar<T>(0);
        *optr = val;
    }

    template<typename T>
    static void diagCreate(Param<T> out, CParam<T> in, int num)
    {
        dim3 threads(32, 8);
        int blocks_x = divup(out.dims[0], threads.x);
        int blocks_y = divup(out.dims[1], threads.y);
        dim3 blocks(blocks_x * out.dims[2], blocks_y);

        CUDA_LAUNCH((diagCreateKernel<T>), blocks, threads, out, in, num, blocks_x);
        POST_LAUNCH_CHECK();
    }

    template<typename T>
    __global__ static void
    diagExtractKernel(Param<T> out, CParam<T> in, int num, int blocks_z)
    {
        unsigned idw = blockIdx.y / blocks_z;
        unsigned idz = blockIdx.y  - idw * blocks_z;

        unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx >= out.dims[0] ||
            idz >= out.dims[2] ||
            idw >= out.dims[3]) return;

        T *optr = out.ptr + idz * out.strides[2] + idw * out.strides[3] + idx;

        if (idx >= in.dims[0] || idx >= in.dims[1]) *optr = scalar<T>(0);

        int i_off = (num > 0) ? (num * in.strides[1] + idx) : (idx - num);
        const T *iptr = in.ptr  + idz *  in.strides[2] + idw *  in.strides[3] + i_off;
        *optr = iptr[idx * in.strides[1]];
    }

    template<typename T>
    static void diagExtract(Param<T> out, CParam<T> in, int num)
    {
        dim3 threads(256, 1);
        int blocks_x = divup(out.dims[0], threads.x);
        int blocks_z = out.dims[2];
        dim3 blocks(blocks_x, out.dims[3] * blocks_z);

        CUDA_LAUNCH((diagExtractKernel<T>), blocks, threads, out, in, num, blocks_z);
        POST_LAUNCH_CHECK();
    }

}
}
