/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <backend.hpp>
#include <math.hpp>

namespace arrayfire {
namespace cuda {

template<typename T>
__global__ void where(uint *optr, CParam<uint> otmp, CParam<uint> rtmp,
                      CParam<T> in, uint blocks_x, uint blocks_y, uint lim) {
    const uint tidx = threadIdx.x;
    const uint tidy = threadIdx.y;

    const uint zid        = blockIdx.x / blocks_x;
    const uint wid        = (blockIdx.y + blockIdx.z * gridDim.y) / blocks_y;
    const uint blockIdx_x = blockIdx.x - (blocks_x)*zid;
    const uint blockIdx_y =
        (blockIdx.y + blockIdx.z * gridDim.y) - (blocks_y)*wid;
    const uint xid = blockIdx_x * blockDim.x * lim + tidx;
    const uint yid = blockIdx_y * blockDim.y + tidy;

    const uint *otptr = otmp.ptr;
    const uint *rtptr = rtmp.ptr;
    const T *iptr     = in.ptr;

    const uint off =
        wid * otmp.strides[3] + zid * otmp.strides[2] + yid * otmp.strides[1];
    const uint bid = wid * rtmp.strides[3] + zid * rtmp.strides[2] +
                     yid * rtmp.strides[1] + blockIdx_x;

    otptr +=
        wid * otmp.strides[3] + zid * otmp.strides[2] + yid * otmp.strides[1];
    iptr += wid * in.strides[3] + zid * in.strides[2] + yid * in.strides[1];

    bool cond =
        (yid < otmp.dims[1]) && (zid < otmp.dims[2]) && (wid < otmp.dims[3]);
    T zero = scalar<T>(0);

    if (!cond) return;

    uint accum = (bid == 0) ? 0 : rtptr[bid - 1];

    for (uint k = 0, id = xid; k < lim && id < otmp.dims[0];
         k++, id += blockDim.x) {
        uint idx = otptr[id] + accum;
        if (iptr[id] != zero) optr[idx - 1] = (off + id);
    }
}

}  // namespace cuda
}  // namespace arrayfire
