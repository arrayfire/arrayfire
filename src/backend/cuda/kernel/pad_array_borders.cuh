/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <math.hpp>
#include <utility.hpp>

namespace arrayfire {
namespace cuda {

template<af::borderType BType>
__device__ int idxByndEdge(const int i, const int lb, const int len) {
    uint retVal;
    switch (BType) {
        case AF_PAD_SYM: retVal = trimIndex(i - lb, len); break;
        case AF_PAD_CLAMP_TO_EDGE: retVal = clamp(i - lb, 0, len - 1); break;
        case AF_PAD_PERIODIC: {
            int rem   = (i - lb) % len;
            bool cond = rem < 0;
            retVal    = cond * (rem + len) + (1 - cond) * rem;
        } break;
        default: retVal = 0; break;  // AF_PAD_ZERO
    }
    return retVal;
}

template<typename T, af::borderType BType>
__global__ void padBorders(Param<T> out, CParam<T> in, const int l0,
                           const int l1, const int l2, const int l3,
                           unsigned blk_x, unsigned blk_y) {
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int k  = blockIdx.x / blk_x;
    const int l  = blockIdx.y / blk_y;

    const int blockIdx_x = blockIdx.x - (blk_x)*k;
    const int blockIdx_y = blockIdx.y - (blk_y)*l;
    const int i          = blockIdx_x * blockDim.x + lx;
    const int j          = blockIdx_y * blockDim.y + ly;

    const int d0 = in.dims[0];
    const int d1 = in.dims[1];
    const int d2 = in.dims[2];
    const int d3 = in.dims[3];
    const int s0 = in.strides[0];
    const int s1 = in.strides[1];
    const int s2 = in.strides[2];
    const int s3 = in.strides[3];

    const T* src = in.ptr;
    T* dst       = out.ptr;

    bool isNotPadding =
        (l >= l3 && l < (d3 + l3)) && (k >= l2 && k < (d2 + l2)) &&
        (j >= l1 && j < (d1 + l1)) && (i >= l0 && i < (d0 + l0));
    T value = scalar<T>(0);

    if (isNotPadding) {
        unsigned iLOff = (l - l3) * s3;
        unsigned iKOff = (k - l2) * s2;
        unsigned iJOff = (j - l1) * s1;
        unsigned iIOff = (i - l0) * s0;

        value = src[iLOff + iKOff + iJOff + iIOff];
    } else if (BType != AF_PAD_ZERO) {
        unsigned iLOff = idxByndEdge<BType>(l, l3, d3) * s3;
        unsigned iKOff = idxByndEdge<BType>(k, l2, d2) * s2;
        unsigned iJOff = idxByndEdge<BType>(j, l1, d1) * s1;
        unsigned iIOff = idxByndEdge<BType>(i, l0, d0) * s0;

        value = src[iLOff + iKOff + iJOff + iIOff];
    }

    if (i < out.dims[0] && j < out.dims[1] && k < out.dims[2] &&
        l < out.dims[3]) {
        unsigned off = (l * out.strides[3] + k * out.strides[2] +
                        j * out.strides[1] + i * out.strides[0]);
        dst[off]     = value;
    }
}

}  // namespace cuda
}  // namespace arrayfire
