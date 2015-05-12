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

    // Hint from txbob
    // https://devtalk.nvidia.com/default/topic/765696/efficient-in-place-transpose-of-multiple-square-float-matrices
    //
    // Kernel is going access original data in colleased format
    template<typename T, bool conjugate, bool is32Multiple>
    __global__
    void transposeIP(Param<T> in, const dim_type blocksPerMatX, const dim_type blocksPerMatY)
    {
        __shared__ T shrdMem_s[TILE_DIM][TILE_DIM+1];
        __shared__ T shrdMem_d[TILE_DIM][TILE_DIM+1];

        // create variables to hold output dimensions
        const dim_type iDim0 = in.dims[0];
        const dim_type iDim1 = in.dims[1];

        // calculate strides
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

        // offset in and out based on batch id
        T *iptr = in.ptr + batchId_x * in.strides[2] + batchId_y * in.strides[3];

        if(blockIdx_y > blockIdx_x) {       // Off diagonal blocks
            // calculate global indices
            dim_type gx      = lx + x0;
            dim_type gy      = ly + y0;
            dim_type dx      = lx + y0;
            dim_type dy      = ly + x0;

            // Copy to shared memory
#pragma unroll
            for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {

                dim_type gy_ = gy + repeat;
                if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
                    shrdMem_s[ly + repeat][lx] = iptr[gy_ * iStride1 + gx];

                dim_type dy_ = dy + repeat;
                if (is32Multiple || (dx < iDim0 && dy_ < iDim1))
                    shrdMem_d[ly + repeat][lx] = iptr[dy_ * iStride1 + dx];
            }

            __syncthreads();

            // Copy from shared to global memory
#pragma unroll
            for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {

                dim_type dy_ = dy + repeat;
                if (is32Multiple || (dx < iDim0 && dy_ < iDim1))
                    iptr[dy_ * iStride1 + dx] = doOp<T, conjugate>(shrdMem_s[lx][ly + repeat]);

                dim_type gy_ = gy + repeat;
                if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
                    iptr[gy_ * iStride1 + gx] = doOp<T, conjugate>(shrdMem_d[lx][ly + repeat]);
            }

        } else if (blockIdx_y == blockIdx_x) {    // Diagonal blocks
            // calculate global indices
            dim_type gx      = lx + x0;
            dim_type gy      = ly + y0;

            // offset in and out based on batch id
            iptr = in.ptr + batchId_x * in.strides[2] + batchId_y * in.strides[3];

            // Copy to shared memory
#pragma unroll
            for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
                dim_type gy_ = gy + repeat;
                if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
                    shrdMem_s[ly + repeat][lx] = iptr[gy_ * iStride1 + gx];
            }

            __syncthreads();

            // Copy from shared to global memory
#pragma unroll
            for (dim_type repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
                dim_type gy_ = gy + repeat;
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


        dim_type blk_x = divup(in.dims[0],TILE_DIM);
        dim_type blk_y = divup(in.dims[1],TILE_DIM);

        // launch batch * blk_x blocks along x dimension
        dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

        if (in.dims[0] % TILE_DIM == 0 && in.dims[1] % TILE_DIM == 0)
            (transposeIP<T, conjugate, true >)<<<blocks, threads>>>(in, blk_x, blk_y);
        else
            (transposeIP<T, conjugate, false>)<<<blocks, threads>>>(in, blk_x, blk_y);

        POST_LAUNCH_CHECK();
    }
}

}
