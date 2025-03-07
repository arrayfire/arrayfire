/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <math.hpp>

namespace arrayfire {
namespace cuda {

template<typename T, bool is_column>
__global__ void wrap(Param<T> out, CParam<T> in, const int wx, const int wy,
                     const int sx, const int sy, const int px, const int py,
                     const int nx, const int ny, int blocks_x, int blocks_y) {
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

template<typename T, bool is_column>
__global__ void wrap_dilated(Param<T> out, CParam<T> in, const int wx,
                             const int wy, const int sx, const int sy,
                             const int px, const int py, const int dx,
                             const int dy, const int nx, const int ny,
                             int blocks_x, int blocks_y) {
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

    int eff_wx = wx + (wx - 1) * (dx - 1);
    int eff_wy = wy + (wy - 1) * (dy - 1);

    int pidx0 = oidx0 + px;
    int pidx1 = oidx1 + py;

    // The last time a value appears in the unwrapped index is padded_index /
    // stride Each previous index has the value appear "stride" locations
    // earlier We work our way back from the last index

    const int x_start = (pidx0 < eff_wx) ? 0 : (pidx0 - eff_wx) / sx + 1;
    const int y_start = (pidx1 < eff_wy) ? 0 : (pidx1 - eff_wy) / sy + 1;

    const int x_end = min(pidx0 / sx + 1, nx);
    const int y_end = min(pidx1 / sy + 1, ny);

    T val   = scalar<T>(0);
    int idx = 1;

    for (int y = y_start; y < y_end; y++) {
        int fy      = (pidx1 - y * sy);
        bool yvalid = (fy % dy == 0) && (y < ny);
        fy /= dy;

        int win_end_y = fy * wx;
        int dim_end_y = y * nx;

        for (int x = x_start; x < x_end; x++) {
            int fx      = (pidx0 - x * sx);
            bool xvalid = (fx % dx == 0) && (x < nx);
            fx /= dx;

            int win_end = win_end_y + fx;
            int dim_end = dim_end_y + x;

            if (is_column) {
                idx = dim_end * in.strides[1] + win_end;
            } else {
                idx = dim_end + win_end * in.strides[1];
            }

            T ival;
            ival = (yvalid && xvalid) ? iptr[idx] : T(0);
            val  = val + ival;
        }
    }

    optr[oidx1 * out.strides[1] + oidx0] = val;
}

}  // namespace cuda
}  // namespace arrayfire
