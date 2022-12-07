/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_oneapi.hpp>

#include <string>
#include <vector>

namespace oneapi {
namespace kernel {

constexpr sycl::specialization_id<bool> compilingAsDouble;

template<typename T>
class tileCreateKernel {
   public:
    tileCreateKernel(sycl::accessor<T, 1> out, sycl::accessor<T, 1> in,
                     const KParam op, const KParam ip, const int blocksPerMatX,
                     const int blocksPerMatY, sycl::handler &h)
        : out_(out)
        , in_(in)
        , op_(op)
        , ip_(ip)
        , blocksPerMatX_(blocksPerMatX)
        , blocksPerMatY_(blocksPerMatY) , h_(h) {}
    void operator()(sycl::nd_item<2> it) const {
        if constexpr(h_.get_specialization_constant<compilingAsDouble>()) {

          sycl::group g = it.get_group();

          const int oz = g.get_group_id(0) / blocksPerMatX_;
          const int ow = g.get_group_id(1) / blocksPerMatY_;

          const int blockIdx_x = g.get_group_id(0) - oz * blocksPerMatX_;
          const int blockIdx_y = g.get_group_id(1) - ow * blocksPerMatY_;

          const int xx = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
          const int yy = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

          if (xx >= op_.dims[0] || yy >= op_.dims[1] || oz >= op_.dims[2] ||
              ow >= op_.dims[3])
            return;

          const int iz  = oz % ip_.dims[2];
          const int iw  = ow % ip_.dims[3];
          const int izw = iw * ip_.strides[3] + iz * ip_.strides[2];
          const int ozw = ow * op_.strides[3] + oz * op_.strides[2];

          const int incy = blocksPerMatY_ * g.get_local_range(1);
          const int incx = blocksPerMatX_ * g.get_local_range(0);

          for (int oy = yy; oy < op_.dims[1]; oy += incy) {
            const int iy = oy % ip_.dims[1];
            for (int ox = xx; ox < op_.dims[0]; ox += incx) {
                const int ix = ox % ip_.dims[0];

                int iMem = izw + iy * ip_.strides[1] + ix;
                int oMem = ozw + oy * op_.strides[1] + ox;

                const int doo = ip_.offset + iMem;
                out_[oMem]    = 5;

                // out_[oMem] = in_[ip_.offset + iMem];
            }
          }
        }
    }

   private:
    sycl::accessor<T, 1> out_;
    sycl::accessor<T, 1> in_;
    const KParam op_;
    const KParam ip_;
    const int blocksPerMatX_;
    const int blocksPerMatY_;
sycl::handler &h_;
};

template<typename T>
void tile(Param<T> out, const Param<T> in) {
    constexpr int TX    = 32;
    constexpr int TY    = 8;
    constexpr int TILEX = 512;
    constexpr int TILEY = 32;

    auto local = sycl::range(TX, TY);

    int blocksPerMatX = divup(out.info.dims[0], TILEX);
    int blocksPerMatY = divup(out.info.dims[1], TILEY);
    auto global       = sycl::range(local[0] * blocksPerMatX * out.info.dims[2],
                                    local[1] * blocksPerMatY * out.info.dims[3]);

    try {
        getQueue().submit([&](auto &h) {
            sycl::accessor d_out{*out.data, h};
            sycl::accessor d_in{*in.data, h};
            h.set_specialization_constant<compilingAsDouble>(std::is_same<double,T>::value);
              // (isDoubleSupported(getActiveDeviceId())   std::is_same<double,T>::value);
            h.parallel_for(sycl::nd_range{global, local},
                           tileCreateKernel<T>(d_out, d_in, out.info, in.info,
                                               blocksPerMatX, blocksPerMatY, h));
        });
    } catch (const sycl::exception &e) { printf("!!! %s\n", e.what()); }

    ONEAPI_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace oneapi
