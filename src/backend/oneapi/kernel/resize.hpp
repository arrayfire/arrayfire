/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/complex.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <traits.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename AT, typename BT>
BT mul(AT a, BT b) {
    return a * b;
}
template<typename AT>
std::complex<double> mul(AT a, std::complex<double> b) {
    return std::complex<double>(a * b.real(), a * b.imag());
}

template<typename T>
using wtype_t = typename std::conditional<std::is_same<T, double>::value,
                                          double, float>::type;

template<typename T>
using vtype_t = typename std::conditional<common::is_complex<T>::value, T,
                                          wtype_t<T>>::type;

////////////////////////////////////////////////////////////////////////////////////
// nearest-neighbor resampling
template<typename T>
void resize_n_(T* d_out, const KParam out, const T* d_in, const KParam in,
               const int blockIdx_x, const int blockIdx_y, const float xf,
               const float yf, sycl::nd_item<2>& it) {
    sycl::group g = it.get_group();
    int const ox  = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
    int const oy  = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

    // int ix = convert_int_rtp(ox * xf);
    // int iy = convert_int_rtp(oy * yf);
    int ix = sycl::round(ox * xf);
    int iy = sycl::round(oy * yf);

    if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
    if (ix >= in.dims[0]) { ix = in.dims[0] - 1; }
    if (iy >= in.dims[1]) { iy = in.dims[1] - 1; }

    d_out[ox + oy * out.strides[1]] = d_in[ix + iy * in.strides[1]];
}

////////////////////////////////////////////////////////////////////////////////////
// bilinear resampling
template<typename T, typename VT>
void resize_b_(T* d_out, const KParam out, const T* d_in, const KParam in,
               const int blockIdx_x, const int blockIdx_y, const float xf_,
               const float yf_, sycl::nd_item<2>& it) {
    sycl::group g = it.get_group();

    int const ox = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
    int const oy = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

    float xf = ox * xf_;
    float yf = oy * yf_;

    int ix = sycl::floor(xf);

    int iy = sycl::floor(yf);

    if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
    if (ix >= in.dims[0]) { ix = in.dims[0] - 1; }
    if (iy >= in.dims[1]) { iy = in.dims[1] - 1; }

    float b = xf - ix;
    float a = yf - iy;

    const int ix2 = (ix + 1) < in.dims[0] ? (ix + 1) : ix;
    const int iy2 = (iy + 1) < in.dims[1] ? (iy + 1) : iy;

    const VT p1 = d_in[ix + in.strides[1] * iy];
    const VT p2 = d_in[ix + in.strides[1] * iy2];
    const VT p3 = d_in[ix2 + in.strides[1] * iy];
    const VT p4 = d_in[ix2 + in.strides[1] * iy2];

    d_out[ox + oy * out.strides[1]] =
        mul(((1.0f - a) * (1.0f - b)), p1) + mul(((a) * (1.0f - b)), p2) +
        mul(((1.0f - a) * (b)), p3) + mul(((a) * (b)), p4);
}

////////////////////////////////////////////////////////////////////////////////////
// lower resampling
template<typename T>
void resize_l_(T* d_out, const KParam out, const T* d_in, const KParam in,
               const int blockIdx_x, const int blockIdx_y, const float xf,
               const float yf, sycl::nd_item<2>& it) {
    sycl::group g = it.get_group();

    int const ox = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
    int const oy = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

    int ix = (ox * xf);
    int iy = (oy * yf);

    if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
    if (ix >= in.dims[0]) { ix = in.dims[0] - 1; }
    if (iy >= in.dims[1]) { iy = in.dims[1] - 1; }

    d_out[ox + oy * out.strides[1]] = d_in[ix + iy * in.strides[1]];
}

template<typename T, int method>
class resizeCreateKernel {
   public:
    resizeCreateKernel(write_accessor<T> d_out, const KParam out,
                       read_accessor<T> d_in, const KParam in, const int b0,
                       const int b1, const float xf, const float yf)
        : d_out_(d_out)
        , out_(out)
        , d_in_(d_in)
        , in_(in)
        , b0_(b0)
        , b1_(b1)
        , xf_(xf)
        , yf_(yf) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        int bIdx = g.get_group_id(0) / b0_;
        int bIdy = g.get_group_id(1) / b1_;
        // batch adjustment
        int i_off = bIdy * in_.strides[3] + bIdx * in_.strides[2] + in_.offset;
        int o_off = bIdy * out_.strides[3] + bIdx * out_.strides[2];
        int blockIdx_x = g.get_group_id(0) - bIdx * b0_;
        int blockIdx_y = g.get_group_id(1) - bIdy * b1_;

        switch (method) {
            case AF_INTERP_NEAREST:
                resize_n_<T>(d_out_.get_pointer() + o_off, out_,
                             d_in_.get_pointer() + i_off, in_, blockIdx_x,
                             blockIdx_y, xf_, yf_, it);
                break;
            case AF_INTERP_BILINEAR:
                resize_b_<T, vtype_t<T>>(d_out_.get_pointer() + o_off, out_,
                                         d_in_.get_pointer() + i_off, in_,
                                         blockIdx_x, blockIdx_y, xf_, yf_, it);
                break;
            case AF_INTERP_LOWER:
                resize_l_<T>(d_out_.get_pointer() + o_off, out_,
                             d_in_.get_pointer() + i_off, in_, blockIdx_x,
                             blockIdx_y, xf_, yf_, it);
                break;
        }
    }

   private:
    write_accessor<T> d_out_;
    const KParam out_;
    read_accessor<T> d_in_;
    const KParam in_;
    const int b0_;
    const int b1_;
    const float xf_;
    const float yf_;
};

template<typename T>
void resize(Param<T> out, const Param<T> in, const af_interp_type method) {
    constexpr int RESIZE_TX = 16;
    constexpr int RESIZE_TY = 16;

    auto local = sycl::range(RESIZE_TX, RESIZE_TY);

    int blocksPerMatX = divup(out.info.dims[0], local[0]);
    int blocksPerMatY = divup(out.info.dims[1], local[1]);
    auto global       = sycl::range(local[0] * blocksPerMatX * in.info.dims[2],
                                    local[1] * blocksPerMatY * in.info.dims[3]);

    double xd = (double)in.info.dims[0] / (double)out.info.dims[0];
    double yd = (double)in.info.dims[1] / (double)out.info.dims[1];

    float xf = (float)xd, yf = (float)yd;

    getQueue().submit([&](auto& h) {
        read_accessor<T> d_in{*in.data, h};
        write_accessor<T> d_out{*out.data, h};
        switch (method) {
            case AF_INTERP_NEAREST:
                h.parallel_for(sycl::nd_range{global, local},
                               resizeCreateKernel<T, AF_INTERP_NEAREST>(
                                   d_out, out.info, d_in, in.info,
                                   blocksPerMatX, blocksPerMatY, xf, yf));
                break;
            case AF_INTERP_BILINEAR:
                h.parallel_for(sycl::nd_range{global, local},
                               resizeCreateKernel<T, AF_INTERP_BILINEAR>(
                                   d_out, out.info, d_in, in.info,
                                   blocksPerMatX, blocksPerMatY, xf, yf));
                break;
            case AF_INTERP_LOWER:
                h.parallel_for(sycl::nd_range{global, local},
                               resizeCreateKernel<T, AF_INTERP_LOWER>(
                                   d_out, out.info, d_in, in.info,
                                   blocksPerMatX, blocksPerMatY, xf, yf));
                break;
            default: break;
        }
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
