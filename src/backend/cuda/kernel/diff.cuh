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

template<typename T, bool D>
inline void diff_this(T* out, const T* in, const unsigned oMem,
                      const unsigned iMem0, const unsigned iMem1,
                      const unsigned iMem2) {
    // iMem2 can never be 0
    if (D == 0) {  // Diff1
        out[oMem] = in[iMem1] - in[iMem0];
    } else {  // Diff2
        out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
    }
}

template<typename T, unsigned dim, bool isDiff2>
__global__ void diff(Param<T> out, CParam<T> in, const unsigned oElem,
                     const unsigned blocksPerMatX,
                     const unsigned blocksPerMatY) {
    unsigned idz = blockIdx.x / blocksPerMatX;
    unsigned idw = (blockIdx.y + blockIdx.z * gridDim.y) / blocksPerMatY;

    unsigned blockIdx_x = blockIdx.x - idz * blocksPerMatX;
    unsigned blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - idw * blocksPerMatY;

    unsigned idx = threadIdx.x + blockIdx_x * blockDim.x;
    unsigned idy = threadIdx.y + blockIdx_y * blockDim.y;

    if (idx >= out.dims[0] || idy >= out.dims[1] || idz >= out.dims[2] ||
        idw >= out.dims[3])
        return;

    unsigned iMem0 =
        idw * in.strides[3] + idz * in.strides[2] + idy * in.strides[1] + idx;
    unsigned iMem1 = iMem0 + in.strides[dim];
    unsigned iMem2 = iMem1 + in.strides[dim];

    unsigned oMem = idw * out.strides[3] + idz * out.strides[2] +
                    idy * out.strides[1] + idx;

    iMem2 *= isDiff2;

    diff_this<T, isDiff2>(out.ptr, in.ptr, oMem, iMem0, iMem1, iMem2);
}

}  // namespace cuda
}  // namespace arrayfire
