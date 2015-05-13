/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <backend.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>

namespace cuda
{

namespace kernel
{

    static const dim_type TILE_DIM  = 32;
    static const dim_type THREADS_X = TILE_DIM;
    static const dim_type THREADS_Y = 256 / TILE_DIM;

    template<typename T, bool conjugate>
    __device__ T doOp(T in)
    {
        if (conjugate) return conj(in);
        else return in;
    }

    // Kernel is going access original data in colleased format
    template<typename T, bool conjugate, bool is32Multiple>
    __global__
    void transpose(Param<T> out, CParam<T> in,
                   const dim_type blocksPerMatX, const dim_type blocksPerMatY)
    {
        __shared__ T shrdMem[TILE_DIM][TILE_DIM+1];
        // create variables to hold output dimensions
        const dim_type oDim0 = out.dims[0];
        const dim_type oDim1 = out.dims[1];
        const dim_type iDim0 = in.dims[0];
        const dim_type iDim1 = in.dims[1];

        // calculate strides
        const dim_type oStride1 = out.strides[1];
        const dim_type iStride1 = in.strides[1];

        const dim_type lx = threadIdx.x;
        const dim_type ly = threadIdx.y;

        // batch based block Id
        const dim_type batchId_x = blockIdx.x / blocksPerMatX;
        const dim_type blockIdx_x = (blockIdx.x - batchId_x * blocksPerMatX);

        const dim_type batchId_y = blockIdx.y / blocksPerMatY;
        const dim_type blockIdx_y = (blockIdx.y - batchId_y * blocksPerMatY);

        const dim_type x0 = TILE_DIM * blockIdx_x;
        const dim_type y0 = TILE_DIM * blockIdx_y;

        // calculate global indices
        dim_type gx      = lx + x0;
        dim_type gy      = ly + y0;

        // offset in and out based on batch id
        in.ptr  += batchId_x *  in.strides[2] + batchId_y *  in.strides[3];
        out.ptr += batchId_x * out.strides[2] + batchId_y * out.strides[3];

#pragma unroll
        for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
            dim_type gy_ = gy+repeat;
            if (is32Multiple || (gx<iDim0 && gy_<iDim1))
                shrdMem[ly + repeat][lx] = in.ptr[gy_ * iStride1 + gx];
        }
        __syncthreads();

        gx = lx + y0;
        gy = ly + x0;

#pragma unroll
        for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
            dim_type gy_ = gy+repeat;
            if (is32Multiple || (gx<oDim0 && gy_<oDim1))
                out.ptr[gy_ * oStride1 + gx] = doOp<T, conjugate>(shrdMem[lx][ly + repeat]);
        }
    }

    template<typename T, bool conjugate>
    void transpose(Param<T> out, CParam<T> in, const dim_type ndims)
    {
        // dimensions passed to this function should be input dimensions
        // any necessary transformations and dimension related calculations are
        // carried out here and inside the kernel
        dim3 threads(kernel::THREADS_X,kernel::THREADS_Y);


        dim_type blk_x = divup(in.dims[0],TILE_DIM);
        dim_type blk_y = divup(in.dims[1],TILE_DIM);
        // launch batch * blk_x blocks along x dimension
        dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

        if (in.dims[0] % TILE_DIM == 0 && in.dims[1] % TILE_DIM == 0)
            (transpose<T, conjugate, true >)<<<blocks, threads>>>(out, in, blk_x, blk_y);
        else
            (transpose<T, conjugate, false>)<<<blocks, threads>>>(out, in, blk_x, blk_y);

        POST_LAUNCH_CHECK();
    }
}

}
