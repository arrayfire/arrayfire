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
__global__ void unwrap(Param<T> out, CParam<T> in, const int wx, const int wy,
                       const int sx, const int sy, const int px, const int py,
                       const int dx, const int dy, const int nx, int reps) {
    // Compute channel and volume
    const int w = (blockIdx.y + blockIdx.z * gridDim.y) / in.dims[2];
    const int z = (blockIdx.y + blockIdx.z * gridDim.y) % in.dims[2];

    if (w >= in.dims[3] || z >= in.dims[2]) return;

    // Compute offset for channel and volume
    const int cOut = w * out.strides[3] + z * out.strides[2];
    const int cIn  = w * in.strides[3] + z * in.strides[2];

    // Compute the output column index
    const int id = is_column ? (blockIdx.x * blockDim.y + threadIdx.y)
                             : (blockIdx.x * blockDim.x + threadIdx.x);

    if (id >= (is_column ? out.dims[1] : out.dims[0])) return;

    // Compute the starting index of window in x and y of input
    const int startx = (id % nx) * sx;
    const int starty = (id / nx) * sy;

    const int spx = startx - px;
    const int spy = starty - py;

    // Offset the global pointers to the respective starting indices
    T* optr       = out.ptr + cOut + id * (is_column ? out.strides[1] : 1);
    const T* iptr = in.ptr + cIn;

    // Compute output index local to column
    int outIdx        = is_column ? threadIdx.x : threadIdx.y;
    const int oStride = is_column ? blockDim.x : blockDim.y;
    bool cond         = (spx >= 0 && spx + (wx * dx) < in.dims[0] && spy >= 0 &&
                 spy + (wy * dy) < in.dims[1]);

    for (int i = 0; i < reps; i++) {
        if (outIdx >= (is_column ? out.dims[0] : out.dims[1])) return;

        // Compute input index local to window
        const int x = outIdx % wx;
        const int y = outIdx / wx;

        const int xpad = spx + x * dx;
        const int ypad = spy + y * dy;

        // Copy
        T val = scalar<T>(0.0);
        if (cond || (xpad >= 0 && xpad < in.dims[0] && ypad >= 0 &&
                     ypad < in.dims[1])) {
            const int inIdx = ypad * in.strides[1] + xpad * in.strides[0];
            val             = iptr[inIdx];
        }

        if (is_column) {
            optr[outIdx] = val;
        } else {
            optr[outIdx * out.strides[1]] = val;
        }
        outIdx += oStride;
    }
}

}  // namespace cuda
}  // namespace arrayfire
