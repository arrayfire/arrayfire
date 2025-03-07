/*******************************************************
 * Copyright (c) 2022 ArrayFire
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
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <traits.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
class triangleKernel {
   public:
    triangleKernel(write_accessor<T> rAcc, KParam rinfo, read_accessor<T> iAcc,
                   KParam iinfo, const int groups_x, const int groups_y,
                   const bool is_upper, const bool is_unit_diag)
        : rAcc_(rAcc)
        , rinfo_(rinfo)
        , iAcc_(iAcc)
        , iinfo_(iinfo)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , is_upper_(is_upper)
        , is_unit_diag_(is_unit_diag) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        const int oz  = g.get_group_id(0) / groups_x_;
        const int ow  = g.get_group_id(1) / groups_y_;

        const int groupId_0 = g.get_group_id(0) - oz * groups_x_;
        const int groupId_1 = g.get_group_id(1) - ow * groups_y_;

        const int xx = it.get_local_id(0) + groupId_0 * it.get_local_range(0);
        const int yy = it.get_local_id(1) + groupId_1 * it.get_local_range(1);

        const int incy = groups_y_ * it.get_local_range(1);
        const int incx = groups_x_ * it.get_local_range(0);

        T *d_r       = rAcc_.get_pointer();
        const T *d_i = iAcc_.get_pointer() + iinfo_.offset;

        if (oz < rinfo_.dims[2] && ow < rinfo_.dims[3]) {
            d_i = d_i + oz * iinfo_.strides[2] + ow * iinfo_.strides[3];
            d_r = d_r + oz * rinfo_.strides[2] + ow * rinfo_.strides[3];

            for (int oy = yy; oy < rinfo_.dims[1]; oy += incy) {
                const T *Yd_i = d_i + oy * iinfo_.strides[1];
                T *Yd_r       = d_r + oy * rinfo_.strides[1];

                for (int ox = xx; ox < rinfo_.dims[0]; ox += incx) {
                    bool cond         = is_upper_ ? (oy >= ox) : (oy <= ox);
                    bool do_unit_diag = is_unit_diag_ && (oy == ox);
                    if (cond) {
                        Yd_r[ox] = do_unit_diag ? (T)(1) : Yd_i[ox];
                    } else {
                        Yd_r[ox] = (T)(0);
                    }
                }
            }
        }
    }

   private:
    write_accessor<T> rAcc_;
    KParam rinfo_;
    read_accessor<T> iAcc_;
    KParam iinfo_;
    const int groups_x_;
    const int groups_y_;
    const bool is_upper_;
    const bool is_unit_diag_;
};

template<typename T>
void triangle(Param<T> out, const Param<T> in, bool is_upper,
              bool is_unit_diag) {
    constexpr unsigned TX    = 32;
    constexpr unsigned TY    = 8;
    constexpr unsigned TILEX = 128;
    constexpr unsigned TILEY = 32;

    auto local = sycl::range{TX, TY};

    int groups_x = divup(out.info.dims[0], TILEX);
    int groups_y = divup(out.info.dims[1], TILEY);

    auto global = sycl::range{groups_x * out.info.dims[2] * local[0],
                              groups_y * out.info.dims[3] * local[1]};

    getQueue().submit([&](sycl::handler &h) {
        read_accessor<T> iAcc{*in.data, h};
        write_accessor<T> rAcc{*out.data, h};

        h.parallel_for(
            sycl::nd_range{global, local},
            triangleKernel<T>(rAcc, out.info, iAcc, in.info, groups_x, groups_y,
                              is_upper, is_unit_diag));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
