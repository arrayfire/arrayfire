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

template<typename T>
class reorderCreateKernel {
   public:
    reorderCreateKernel(write_accessor<T> out, read_accessor<T> in,
                        const KParam op, const KParam ip, const int d0,
                        const int d1, const int d2, const int d3,
                        const int blocksPerMatX, const int blocksPerMatY)
        : out_(out)
        , in_(in)
        , op_(op)
        , ip_(ip)
        , d0_(d0)
        , d1_(d1)
        , d2_(d2)
        , d3_(d3)
        , blocksPerMatX_(blocksPerMatX)
        , blocksPerMatY_(blocksPerMatY) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        const int oz = g.get_group_id(0) / blocksPerMatX_;
        const int ow = g.get_group_id(1) / blocksPerMatY_;

        const int blockIdx_x = g.get_group_id(0) - oz * blocksPerMatX_;
        const int blockIdx_y = g.get_group_id(1) - ow * blocksPerMatY_;

        const int xx = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
        const int yy = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

        bool valid = (xx < op_.dims[0] && yy < op_.dims[1] &&
                      oz < op_.dims[2] && ow < op_.dims[3]);

        const int incy = blocksPerMatY_ * g.get_local_range(1);
        const int incx = blocksPerMatX_ * g.get_local_range(0);

        const int o_off    = ow * op_.strides[3] + oz * op_.strides[2];
        const int rdims[4] = {d0_, d1_, d2_, d3_};
        int ids[4]         = {0};

        ids[rdims[3]] = ow;
        ids[rdims[2]] = oz;

        for (int oy = yy; oy < op_.dims[1]; oy += incy) {
            ids[rdims[1]] = oy;
            for (int ox = xx; ox < op_.dims[0]; ox += incx) {
                ids[rdims[0]] = ox;

                const int oIdx = o_off + oy * op_.strides[1] + ox;

                const int iIdx = ids[3] * ip_.strides[3] +
                                 ids[2] * ip_.strides[2] +
                                 ids[1] * ip_.strides[1] + ids[0];

                if (valid) { out_[oIdx] = in_[ip_.offset + iIdx]; }
            }
        }
    }

   private:
    write_accessor<T> out_;
    read_accessor<T> in_;
    const KParam op_;
    const KParam ip_;
    const int d0_;
    const int d1_;
    const int d2_;
    const int d3_;
    const int blocksPerMatX_;
    const int blocksPerMatY_;
};

template<typename T>
void reorder(Param<T> out, const Param<T> in, const dim_t* rdims) {
    constexpr int TX    = 32;
    constexpr int TY    = 8;
    constexpr int TILEX = 512;
    constexpr int TILEY = 32;

    auto local = sycl::range(TX, TY);

    int blocksPerMatX = divup(out.info.dims[0], TILEX);
    int blocksPerMatY = divup(out.info.dims[1], TILEY);
    auto global       = sycl::range(local[0] * blocksPerMatX * out.info.dims[2],
                                    local[1] * blocksPerMatY * out.info.dims[3]);

    getQueue().submit([&](auto& h) {
        read_accessor<T> d_in{*in.data, h};
        write_accessor<T> d_out{*out.data, h};
        h.parallel_for(
            sycl::nd_range{global, local},
            reorderCreateKernel<T>(
                d_out, d_in, out.info, in.info, static_cast<int>(rdims[0]),
                static_cast<int>(rdims[1]), static_cast<int>(rdims[2]),
                static_cast<int>(rdims[3]), blocksPerMatX, blocksPerMatY));
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
