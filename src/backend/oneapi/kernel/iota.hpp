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
#include <common/half.hpp>
#include <common/kernel_cache.hpp>
#include <debug_oneapi.hpp>
#include <traits.hpp>
#include <af/dim4.hpp>

#include <string>
#include <vector>

namespace oneapi {
namespace kernel {

template<typename T>
class iotaKernel {
public:
    iotaKernel(sycl::accessor<T> out, KParam oinfo,
               const int s0, const int s1, const int s2, const int s3,
               const int blocksPerMatX, const int blocksPerMatY, 
               sycl::stream debug) :
        out_(out), oinfo_(oinfo),
        s0_(s0), s1_(s1), s2_(s2), s3_(s3),
        blocksPerMatX_(blocksPerMatX), blocksPerMatY_(blocksPerMatY),
        debug_(debug) {}

    void operator() (sycl::nd_item<2> it) const {
        //printf("[%d,%d]\n", it.get_global_id(0), it.get_global_id(1));
        //debug_ << "[" << it.get_global_id(0) << "," << it.get_global_id(1) << "]" << sycl::stream_manipulator::endl;

        sycl::group gg = it.get_group();
        const int oz = gg.get_group_id(0) / blocksPerMatX_;
        const int ow = gg.get_group_id(1) / blocksPerMatY_;

        const int blockIdx_x = gg.get_group_id(0) - oz * blocksPerMatX_;
        const int blockIdx_y = gg.get_group_id(1) - ow * blocksPerMatY_;

        const int xx = it.get_local_id(0) + blockIdx_x * gg.get_local_range(0);
        const int yy = it.get_local_id(1) + blockIdx_y * gg.get_local_range(1);

        if (xx >= oinfo_.dims[0] || yy >= oinfo_.dims[1] || oz >= oinfo_.dims[2] ||
            ow >= oinfo_.dims[3])
            return;

        const int ozw = ow * oinfo_.strides[3] + oz * oinfo_.strides[2];

        T val = static_cast<T>((ow % s3_) * s2_ * s1_ * s0_);
        val +=  static_cast<T>((oz % s2_) * s1_ * s0_);

        const int incy = blocksPerMatY_ * gg.get_local_range(1);
        const int incx = blocksPerMatX_ * gg.get_local_range(0);

        for (int oy = yy; oy < oinfo_.dims[1]; oy += incy) {
            T valY   = val + (oy % s1_) * s0_;
            int oyzw = ozw + oy * oinfo_.strides[1];
            for (int ox = xx; ox < oinfo_.dims[0]; ox += incx) {
                int oidx = oyzw + ox;
                out_[oidx] = valY + (ox % s0_);
            }
        }
    }

protected:
    sycl::accessor<T> out_;
    KParam oinfo_;
    int s0_, s1_, s2_, s3_;
    int blocksPerMatX_, blocksPerMatY_;
    sycl::stream debug_;
};

template<typename T>
void iota(Param<T> out, const af::dim4& sdims) {
    constexpr int IOTA_TX = 32;
    constexpr int IOTA_TY = 8;
    constexpr int TILEX   = 512;
    constexpr int TILEY   = 32;

    sycl::range<2> local(IOTA_TX, IOTA_TY);

    int blocksPerMatX = divup(out.info.dims[0], TILEX);
    int blocksPerMatY = divup(out.info.dims[1], TILEY);
    sycl::range<2> global(local[0] * blocksPerMatX * out.info.dims[2],
                          local[1] * blocksPerMatY * out.info.dims[3]);
    sycl::nd_range<2> ndrange(global, local);

    getQueue().submit([=] (sycl::handler &h) {
        auto out_acc = out.data->get_access(h);

        sycl::stream debug_stream(2048, 128, h);

        h.parallel_for(ndrange, iotaKernel<T>(out_acc, out.info,
            static_cast<int>(sdims[0]), static_cast<int>(sdims[1]),
            static_cast<int>(sdims[2]), static_cast<int>(sdims[3]),
            blocksPerMatX, blocksPerMatY, debug_stream));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
