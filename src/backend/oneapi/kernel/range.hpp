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
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <traits.hpp>
#include <af/dim4.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
class rangeOp {
   public:
    rangeOp(write_accessor<T> out, KParam oinfo, const int dim,
            const int blocksPerMatX, const int blocksPerMatY)
        : out_(out)
        , oinfo_(oinfo)
        , dim_(dim)
        , blocksPerMatX_(blocksPerMatX)
        , blocksPerMatY_(blocksPerMatY) {}

    void operator()(sycl::nd_item<2> it) const {
        const int mul0 = (dim_ == 0);
        const int mul1 = (dim_ == 1);
        const int mul2 = (dim_ == 2);
        const int mul3 = (dim_ == 3);

        sycl::group g = it.get_group();
        const int oz  = g.get_group_id(0) / blocksPerMatX_;
        const int ow  = g.get_group_id(1) / blocksPerMatY_;

        const int blockIdx_x = g.get_group_id(0) - oz * blocksPerMatX_;
        const int blockIdx_y = g.get_group_id(1) - ow * blocksPerMatY_;

        const int xx = it.get_local_id(0) + blockIdx_x * it.get_local_range(0);
        const int yy = it.get_local_id(1) + blockIdx_y * it.get_local_range(1);

        const size_t odx = oinfo_.dims[0];
        const size_t ody = oinfo_.dims[1];
        const size_t odz = oinfo_.dims[2];
        const size_t odw = oinfo_.dims[3];

        if (xx < odx && yy < ody && oz < odz && ow < odw) {
            const int ozw = ow * oinfo_.strides[3] + oz * oinfo_.strides[2];

            const int incy = blocksPerMatY_ * g.get_local_range(1);
            const int incx = blocksPerMatX_ * g.get_local_range(0);

            compute_t<T> valZW = (mul3 * ow) + (mul2 * oz);

            T* optr = out_.get_pointer();
            for (int oy = yy; oy < oinfo_.dims[1]; oy += incy) {
                compute_t<T> valYZW = valZW + (mul1 * oy);
                int oyzw            = ozw + oy * oinfo_.strides[1];
                for (int ox = xx; ox < oinfo_.dims[0]; ox += incx) {
                    int oidx         = oyzw + ox;
                    compute_t<T> val = valYZW + (mul0 * ox);

                    optr[oidx] = val;
                }
            }
        }
    }

   protected:
    write_accessor<T> out_;
    KParam oinfo_;
    int dim_;
    int blocksPerMatX_, blocksPerMatY_;
};

template<typename T>
void range(Param<T> out, const int dim) {
    constexpr int RANGE_TX    = 32;
    constexpr int RANGE_TY    = 8;
    constexpr int RANGE_TILEX = 512;
    constexpr int RANGE_TILEY = 32;

    sycl::range<2> local(RANGE_TX, RANGE_TY);

    int blocksPerMatX = divup(out.info.dims[0], RANGE_TILEX);
    int blocksPerMatY = divup(out.info.dims[1], RANGE_TILEY);
    sycl::range<2> global(local[0] * blocksPerMatX * out.info.dims[2],
                          local[1] * blocksPerMatY * out.info.dims[3]);
    sycl::nd_range<2> ndrange(global, local);

    getQueue().submit([&](sycl::handler& h) {
        write_accessor<T> out_acc{*out.data, h};

        h.parallel_for(ndrange, rangeOp<T>(out_acc, out.info, dim,
                                           blocksPerMatX, blocksPerMatY));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
