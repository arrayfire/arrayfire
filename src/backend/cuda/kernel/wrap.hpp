/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include "atomics.hpp"
#include "config.hpp"

namespace cuda {
namespace kernel {

///////////////////////////////////////////////////////////////////////////
// Wrap Kernel
///////////////////////////////////////////////////////////////////////////
template <typename T, bool is_column>
__global__ void wrap_kernel(Param<T> out, CParam<T> in, const int wx,
                            const int wy, const int sx, const int sy,
                            const int px, const int py, const int nx,
                            const int ny, int blocks_x, int blocks_y) {
    int idx2 = blockIdx.x / blocks_x;
    int idx3 = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;

    int blockIdx_x = blockIdx.x - idx2 * blocks_x;
    int blockIdx_y = (blockIdx.y + blockIdx.z * gridDim.y) - idx3 * blocks_y;

    int oidx0 = threadIdx.x + blockDim.x * blockIdx_x;
    int oidx1 = threadIdx.y + blockDim.y * blockIdx_y;

    T *optr       = out.ptr + idx2 * out.strides[2] + idx3 * out.strides[3];
    const T *iptr = in.ptr + idx2 * in.strides[2] + idx3 * in.strides[3];

    if (oidx0 >= out.dims[0] || oidx1 >= out.dims[1] || idx2 >= out.dims[2] ||
        idx3 >= out.dims[3])
        return;

    int pidx0 = oidx0 + px;
    int pidx1 = oidx1 + py;

    // The last time a value appears in the unwrapped index is padded_index /
    // stride Each previous index has the value appear "stride" locations
    // earlier We work our way back from the last index

    const int x_end = min(pidx0 / sx, nx - 1);
    const int y_end = min(pidx1 / sy, ny - 1);

    const int x_off = pidx0 - sx * x_end;
    const int y_off = pidx1 - sy * y_end;

    T val   = scalar<T>(0);
    int idx = 1;

    for (int y = y_end, yo = y_off; y >= 0 && yo < wy; yo += sy, y--) {
        int win_end_y = yo * wx;
        int dim_end_y = y * nx;

        for (int x = x_end, xo = x_off; x >= 0 && xo < wx; xo += sx, x--) {
            int win_end = win_end_y + xo;
            int dim_end = dim_end_y + x;

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
void wrap(Param<T> out, CParam<T> in, const int wx, const int wy, const int sx,
          const int sy, const int px, const int py, const bool is_column) {
    int nx = (out.dims[0] + 2 * px - wx) / sx + 1;
    int ny = (out.dims[1] + 2 * py - wy) / sy + 1;

    dim3 threads(THREADS_X, THREADS_Y);
    int blocks_x = divup(out.dims[0], threads.x);
    int blocks_y = divup(out.dims[1], threads.y);

    dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

    const int maxBlocksY =
        cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
    blocks.z = divup(blocks.y, maxBlocksY);
    blocks.y = divup(blocks.y, blocks.z);

    if (is_column) {
        CUDA_LAUNCH((wrap_kernel<T, true>), blocks, threads, out, in, wx, wy,
                    sx, sy, px, py, nx, ny, blocks_x, blocks_y);
    } else {
        CUDA_LAUNCH((wrap_kernel<T, false>), blocks, threads, out, in, wx, wy,
                    sx, sy, px, py, nx, ny, blocks_x, blocks_y);
    }
}
}  // namespace kernel
}  // namespace cuda
