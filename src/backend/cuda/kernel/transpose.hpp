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

    static const int TILE_DIM  = 32;
    static const int THREADS_X = TILE_DIM;
    static const int THREADS_Y = 256 / TILE_DIM;

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
                   const int blocksPerMatX, const int blocksPerMatY)
    {
        __shared__ T shrdMem[TILE_DIM][TILE_DIM+1];
        // create variables to hold output dimensions
        const int oDim0 = out.dims[0];
        const int oDim1 = out.dims[1];
        const int iDim0 = in.dims[0];
        const int iDim1 = in.dims[1];

        // calculate strides
        const int oStride1 = out.strides[1];
        const int iStride1 = in.strides[1];

        const int lx = threadIdx.x;
        const int ly = threadIdx.y;

        // batch based block Id
        const int batchId_x = blockIdx.x / blocksPerMatX;
        const int blockIdx_x = (blockIdx.x - batchId_x * blocksPerMatX);

        const int batchId_y = blockIdx.y / blocksPerMatY;
        const int blockIdx_y = (blockIdx.y - batchId_y * blocksPerMatY);

        const int x0 = TILE_DIM * blockIdx_x;
        const int y0 = TILE_DIM * blockIdx_y;

        // calculate global indices
        int gx      = lx + x0;
        int gy      = ly + y0;

        // offset in and out based on batch id
        in.ptr  += batchId_x *  in.strides[2] + batchId_y *  in.strides[3];
        out.ptr += batchId_x * out.strides[2] + batchId_y * out.strides[3];

#pragma unroll
        for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
            int gy_ = gy+repeat;
            if (is32Multiple || (gx<iDim0 && gy_<iDim1))
                shrdMem[ly + repeat][lx] = in.ptr[gy_ * iStride1 + gx];
        }
        __syncthreads();

        gx = lx + y0;
        gy = ly + x0;

#pragma unroll
        for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
            int gy_ = gy+repeat;
            if (is32Multiple || (gx<oDim0 && gy_<oDim1))
                out.ptr[gy_ * oStride1 + gx] = doOp<T, conjugate>(shrdMem[lx][ly + repeat]);
        }
    }

    template<typename T, bool conjugate>
    void transpose(Param<T> out, CParam<T> in, const int ndims)
    {
        // dimensions passed to this function should be input dimensions
        // any necessary transformations and dimension related calculations are
        // carried out here and inside the kernel
        dim3 threads(kernel::THREADS_X,kernel::THREADS_Y);


        int blk_x = divup(in.dims[0],TILE_DIM);
        int blk_y = divup(in.dims[1],TILE_DIM);
        // launch batch * blk_x blocks along x dimension
        dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

        if (in.dims[0] % TILE_DIM == 0 && in.dims[1] % TILE_DIM == 0)
            CUDA_LAUNCH((transpose<T, conjugate, true >), blocks, threads, out, in, blk_x, blk_y);
        else
            CUDA_LAUNCH((transpose<T, conjugate, false>), blocks, threads, out, in, blk_x, blk_y);

        POST_LAUNCH_CHECK();
    }
}

}
