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

    // Hint from txbob
    // https://devtalk.nvidia.com/default/topic/765696/efficient-in-place-transpose-of-multiple-square-float-matrices
    //
    // Kernel is going access original data in colleased format
    template<typename T, bool conjugate, bool is32Multiple>
    __global__
    void transposeIP(Param<T> in, const int blocksPerMatX, const int blocksPerMatY)
    {
        __shared__ T shrdMem_s[TILE_DIM][TILE_DIM+1];
        __shared__ T shrdMem_d[TILE_DIM][TILE_DIM+1];

        // create variables to hold output dimensions
        const int iDim0 = in.dims[0];
        const int iDim1 = in.dims[1];

        // calculate strides
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

        // offset in and out based on batch id
        T *iptr = in.ptr + batchId_x * in.strides[2] + batchId_y * in.strides[3];

        if(blockIdx_y > blockIdx_x) {       // Off diagonal blocks
            // calculate global indices
            int gx      = lx + x0;
            int gy      = ly + y0;
            int dx      = lx + y0;
            int dy      = ly + x0;

            // Copy to shared memory
#pragma unroll
            for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {

                int gy_ = gy + repeat;
                if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
                    shrdMem_s[ly + repeat][lx] = iptr[gy_ * iStride1 + gx];

                int dy_ = dy + repeat;
                if (is32Multiple || (dx < iDim0 && dy_ < iDim1))
                    shrdMem_d[ly + repeat][lx] = iptr[dy_ * iStride1 + dx];
            }

            __syncthreads();

            // Copy from shared to global memory
#pragma unroll
            for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {

                int dy_ = dy + repeat;
                if (is32Multiple || (dx < iDim0 && dy_ < iDim1))
                    iptr[dy_ * iStride1 + dx] = doOp<T, conjugate>(shrdMem_s[lx][ly + repeat]);

                int gy_ = gy + repeat;
                if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
                    iptr[gy_ * iStride1 + gx] = doOp<T, conjugate>(shrdMem_d[lx][ly + repeat]);
            }

        } else if (blockIdx_y == blockIdx_x) {    // Diagonal blocks
            // calculate global indices
            int gx      = lx + x0;
            int gy      = ly + y0;

            // offset in and out based on batch id
            iptr = in.ptr + batchId_x * in.strides[2] + batchId_y * in.strides[3];

            // Copy to shared memory
#pragma unroll
            for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
                int gy_ = gy + repeat;
                if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
                    shrdMem_s[ly + repeat][lx] = iptr[gy_ * iStride1 + gx];
            }

            __syncthreads();

            // Copy from shared to global memory
#pragma unroll
            for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
                int gy_ = gy + repeat;
                if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
                    iptr[gy_ * iStride1 + gx] = doOp<T, conjugate>(shrdMem_s[lx][ly + repeat]);
            }
        }
    }

    template<typename T, bool conjugate>
    void transpose_inplace(Param<T> in)
    {
        // dimensions passed to this function should be input dimensions
        // any necessary transformations and dimension related calculations are
        // carried out here and inside the kernel
        dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);


        int blk_x = divup(in.dims[0],TILE_DIM);
        int blk_y = divup(in.dims[1],TILE_DIM);

        // launch batch * blk_x blocks along x dimension
        dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

        if (in.dims[0] % TILE_DIM == 0 && in.dims[1] % TILE_DIM == 0)
            CUDA_LAUNCH((transposeIP<T, conjugate, true >), blocks, threads, in, blk_x, blk_y);
        else
            CUDA_LAUNCH((transposeIP<T, conjugate, false>), blocks, threads, in, blk_x, blk_y);

        POST_LAUNCH_CHECK();
    }
}

}
