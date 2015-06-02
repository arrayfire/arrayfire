/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 32;
        static const unsigned TY = 8;

#define sidx(y, x) scratch[y + 1][x + 1]

        template<typename T>
        __global__
        void gradient_kernel(Param<T> grad0, Param<T> grad1, CParam<T> in,
                             const int blocksPerMatX, const int blocksPerMatY)
        {
            const int idz = blockIdx.x / blocksPerMatX;
            const int idw = blockIdx.y / blocksPerMatY;

            const int blockIdx_x = blockIdx.x - idz * blocksPerMatX;
            const int blockIdx_y = blockIdx.y - idw * blocksPerMatY;

            const int xB = blockIdx_x * blockDim.x;
            const int yB = blockIdx_y * blockDim.y;

            const int idx = threadIdx.x + xB;
            const int idy = threadIdx.y + yB;

            bool cond = (idx >= in.dims[0] || idy >= in.dims[1] ||
                         idz >= in.dims[2] || idw >= in.dims[3]);

            int xmax = (TX > (in.dims[0] - xB)) ? (in.dims[0] - xB) : TX;
            int ymax = (TY > (in.dims[1] - yB)) ? (in.dims[1] - yB) : TY;

            int iIdx = idw * in.strides[3] + idz * in.strides[2]
                          + idy * in.strides[1] + idx;

            int g0dx = idw * grad0.strides[3] + idz * grad0.strides[2]
                          + idy * grad0.strides[1] + idx;

            int g1dx = idw * grad1.strides[3] + idz * grad1.strides[2]
                          + idy * grad1.strides[1] + idx;

            __shared__ T scratch[TY + 2][TX + 2];

            // Multipliers - 0.5 for interior, 1 for edge cases
            float xf = 0.5 * (1 + (idx == 0 || idx >= (in.dims[0] - 1)));
            float yf = 0.5 * (1 + (idy == 0 || idy >= (in.dims[1] - 1)));

            // Copy data to scratch space
            sidx(threadIdx.y, threadIdx.x) = cond ? scalar<T>(0) : in.ptr[iIdx];

            __syncthreads();

            // Copy buffer zone data. Corner (0,0) etc, are not used.
            // Cols
            if(threadIdx.y == 0) {
                // Y-1
                sidx(-1, threadIdx.x) = (cond || idy == 0) ?
                                        sidx(0, threadIdx.x) : in.ptr[iIdx - in.strides[1]];
                sidx(ymax, threadIdx.x) = (cond || idy + ymax >= in.dims[1] - 1) ?
                                        sidx(ymax - 1, threadIdx.x) : in.ptr[iIdx + ymax * in.strides[1]];
            }
            // Rows
            if(threadIdx.x == 0) {
                sidx(threadIdx.y, -1) = (cond || idx == 0) ?
                                        sidx(threadIdx.y, 0) : in.ptr[iIdx - 1];
                sidx(threadIdx.y, xmax) = (cond || idx + xmax >= in.dims[0] - 1) ?
                                        sidx(threadIdx.y, xmax - 1) : in.ptr[iIdx + xmax];
            }

            __syncthreads();

            if (cond) return;

            grad0.ptr[g0dx] = xf * (sidx(threadIdx.y, threadIdx.x + 1)
                                 -  sidx(threadIdx.y, threadIdx.x - 1));
            grad1.ptr[g1dx] = yf * (sidx(threadIdx.y + 1, threadIdx.x)
                                 -  sidx(threadIdx.y - 1, threadIdx.x));
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void gradient(Param<T> grad0, Param<T> grad1, CParam<T> in)
        {
            dim3 threads(TX, TY, 1);

            int blocksPerMatX = divup(in.dims[0], TX);
            int blocksPerMatY = divup(in.dims[1], TY);
            dim3 blocks(blocksPerMatX * in.dims[2],
                        blocksPerMatY * in.dims[3],
                        1);

            gradient_kernel<T><<<blocks, threads>>>(grad0, grad1, in, blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}
