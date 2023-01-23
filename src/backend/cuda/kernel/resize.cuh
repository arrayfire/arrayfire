/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <interp.hpp>

namespace arrayfire {
namespace cuda {

// nearest-neighbor resampling
template<typename T>
__host__ __device__ void resize_n(Param<T> out, CParam<T> in, const int o_off,
                                  const int i_off, const int blockIdx_x,
                                  const int blockIdx_y, const float xf,
                                  const float yf) {
    const int ox = threadIdx.x + blockIdx_x * blockDim.x;
    const int oy = threadIdx.y + blockIdx_y * blockDim.y;

    int ix = round(ox * xf);
    int iy = round(oy * yf);

    if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
    if (ix >= in.dims[0]) { ix = in.dims[0] - 1; }
    if (iy >= in.dims[1]) { iy = in.dims[1] - 1; }

    out.ptr[o_off + ox + oy * out.strides[1]] =
        in.ptr[i_off + ix + iy * in.strides[1]];
}

// bilinear resampling
template<typename T>
__host__ __device__ void resize_b(Param<T> out, CParam<T> in, const int o_off,
                                  const int i_off, const int blockIdx_x,
                                  const int blockIdx_y, const float xf_,
                                  const float yf_) {
    const int ox = threadIdx.x + blockIdx_x * blockDim.x;
    const int oy = threadIdx.y + blockIdx_y * blockDim.y;

    float xf = ox * xf_;
    float yf = oy * yf_;

    int ix = floorf(xf);
    int iy = floorf(yf);

    if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
    if (ix >= in.dims[0]) { ix = in.dims[0] - 1; }
    if (iy >= in.dims[1]) { iy = in.dims[1] - 1; }

    float b = xf - ix;
    float a = yf - iy;

    const int ix2 = ix + 1 < in.dims[0] ? ix + 1 : ix;
    const int iy2 = iy + 1 < in.dims[1] ? iy + 1 : iy;

    typedef typename itype_t<T>::wtype WT;
    typedef typename itype_t<T>::vtype VT;

    const T *iptr = in.ptr + i_off;

    const VT p1 = iptr[ix + in.strides[1] * iy];
    const VT p2 = iptr[ix + in.strides[1] * iy2];
    const VT p3 = iptr[ix2 + in.strides[1] * iy];
    const VT p4 = iptr[ix2 + in.strides[1] * iy2];

    VT val = scalar<WT>((1.0f - a) * (1.0f - b)) * p1 +
             scalar<WT>((a) * (1.0f - b)) * p2 +
             scalar<WT>((1.0f - a) * (b)) * p3 + scalar<WT>((a) * (b)) * p4;

    out.ptr[o_off + ox + oy * out.strides[1]] = val;
}

// lower resampling
template<typename T>
__host__ __device__ void resize_l(Param<T> out, CParam<T> in, const int o_off,
                                  const int i_off, const int blockIdx_x,
                                  const int blockIdx_y, const float xf,
                                  const float yf) {
    const int ox = threadIdx.x + blockIdx_x * blockDim.x;
    const int oy = threadIdx.y + blockIdx_y * blockDim.y;

    int ix = (ox * xf);
    int iy = (oy * yf);

    if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
    if (ix >= in.dims[0]) { ix = in.dims[0] - 1; }
    if (iy >= in.dims[1]) { iy = in.dims[1] - 1; }

    out.ptr[o_off + ox + oy * out.strides[1]] =
        in.ptr[i_off + ix + iy * in.strides[1]];
}

template<typename T, af::interpType method>
__global__ void resize(Param<T> out, CParam<T> in, const int b0, const int b1,
                       const float xf, const float yf) {
    const int bIdx = blockIdx.x / b0;
    const int bIdy = blockIdx.y / b1;
    // channel adjustment
    const int i_off      = bIdx * in.strides[2] + bIdy * in.strides[3];
    const int o_off      = bIdx * out.strides[2] + bIdy * out.strides[3];
    const int blockIdx_x = blockIdx.x - bIdx * b0;
    const int blockIdx_y = blockIdx.y - bIdy * b1;

    // core
    if (method == AF_INTERP_NEAREST) {
        resize_n(out, in, o_off, i_off, blockIdx_x, blockIdx_y, xf, yf);
    } else if (method == AF_INTERP_BILINEAR) {
        resize_b(out, in, o_off, i_off, blockIdx_x, blockIdx_y, xf, yf);
    } else if (method == AF_INTERP_LOWER) {
        resize_l(out, in, o_off, i_off, blockIdx_x, blockIdx_y, xf, yf);
    }
}

}  // namespace cuda
}  // namespace arrayfire
