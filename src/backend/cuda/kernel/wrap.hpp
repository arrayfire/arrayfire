/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>
#include "config.hpp"
#include "atomics.hpp"

namespace cuda
{
    namespace kernel
    {

        ///////////////////////////////////////////////////////////////////////////
        // Wrap Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, bool is_column>
        __global__
        void wrap_kernel(Param<T> out, CParam<T> in,
                         const dim_t wx, const dim_t wy,
                         const dim_t sx, const dim_t sy,
                         const dim_t px, const dim_t py,
                         const dim_t nx, const dim_t ny,
                         dim_t blocks_x,
                         dim_t blocks_y)
        {
            dim_t idx2 = blockIdx.x / blocks_x;
            dim_t idx3 = blockIdx.y / blocks_y;

            dim_t blockIdx_x = blockIdx.x - idx2 * blocks_x;
            dim_t blockIdx_y = blockIdx.y - idx3 * blocks_y;

            dim_t oidx0 = threadIdx.x + blockDim.x * blockIdx_x;
            dim_t oidx1 = threadIdx.y + blockDim.y * blockIdx_y;

                  T *optr = out.ptr + idx2 * out.strides[2] + idx3 * out.strides[3];
            const T *iptr =  in.ptr + idx2 *  in.strides[2] + idx3 *  in.strides[3];


            if (oidx0 >= out.dims[0] || oidx1 >= out.dims[1]) return;

            dim_t pidx0 = oidx0 + px;
            dim_t pidx1 = oidx1 + py;

            // The last time a value appears in the unwrapped index is padded_index / stride
            // Each previous index has the value appear "stride" locations earlier
            // We work our way back from the last index

            const dim_t x_end = min(pidx0 / sx, nx - 1);
            const dim_t y_end = min(pidx1 / sy, ny - 1);

            const dim_t x_off = pidx0 - sx * x_end;
            const dim_t y_off = pidx1 - sy * y_end;

            T val = scalar<T>(0);
            dim_t idx = 1;

            for (dim_t y = y_end, yo = y_off; y >= 0 && yo < wy; yo += sy, y--) {
                dim_t win_end_y = yo * wx;
                dim_t dim_end_y = y * nx;

                for (dim_t x = x_end, xo = x_off; x >= 0 && xo < wx; xo += sx, x--) {

                    dim_t win_end = win_end_y + xo;
                    dim_t dim_end = dim_end_y + x;

                    if (is_column) {
                        idx = dim_end * in.strides[1] + win_end;
                    } else {
                        idx = dim_end + win_end * in.strides[1];
                    }

                    val = val + iptr[idx];
                }
            }

            optr[oidx1 * out.strides[1] + oidx0] = val;
        }

        template <typename T>
        void wrap(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy,
                  const dim_t px, const dim_t py,
                  const bool is_column)
        {
            dim_t nx = (out.dims[0] + 2 * px - wx) / sx + 1;
            dim_t ny = (out.dims[1] + 2 * py - wy) / sy + 1;

            dim3 threads(THREADS_X, THREADS_Y);
            dim_t blocks_x = divup(out.dims[0], threads.x);
            dim_t blocks_y = divup(out.dims[1], threads.y);

            dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

            if (is_column) {
                CUDA_LAUNCH((wrap_kernel<T, true >), blocks, threads,
                            out, in, wx, wy, sx, sy, px, py, nx, ny, blocks_x, blocks_y);
            } else {
                CUDA_LAUNCH((wrap_kernel<T, false>), blocks, threads,
                            out, in, wx, wy, sx, sy, px, py, nx, ny, blocks_x, blocks_y);
            }
        }
    }
}
