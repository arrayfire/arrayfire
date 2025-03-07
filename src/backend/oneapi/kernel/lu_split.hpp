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

#include <math.hpp>
#include <array>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T, bool same_dims>
class luSplitKernel {
   public:
    luSplitKernel(write_accessor<T> lower, KParam lInfo,
                  write_accessor<T> upper, KParam uInfo, read_accessor<T> in,
                  KParam iInfo, const int groupsPerMatX,
                  const int groupsPerMatY)
        : lower_(lower)
        , lInfo_(lInfo)
        , upper_(upper)
        , uInfo_(uInfo)
        , in_(in)
        , iInfo_(iInfo)
        , groupsPerMatX_(groupsPerMatX)
        , groupsPerMatY_(groupsPerMatY) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        const int oz  = g.get_group_id(0) / groupsPerMatX_;
        const int ow  = g.get_group_id(1) / groupsPerMatY_;

        const int blockIdx_x = g.get_group_id(0) - oz * groupsPerMatX_;
        const int blockIdx_y = g.get_group_id(1) - ow * groupsPerMatY_;

        const int xx = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
        const int yy = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

        const int incy = groupsPerMatY_ * g.get_local_range(1);
        const int incx = groupsPerMatX_ * g.get_local_range(0);

        T *d_l       = lower_.get_pointer();
        T *d_u       = upper_.get_pointer();
        const T *d_i = in_.get_pointer();

        if (oz < iInfo_.dims[2] && ow < iInfo_.dims[3]) {
            d_i = d_i + oz * iInfo_.strides[2] + ow * iInfo_.strides[3];
            d_l = d_l + oz * lInfo_.strides[2] + ow * lInfo_.strides[3];
            d_u = d_u + oz * uInfo_.strides[2] + ow * uInfo_.strides[3];

            for (int oy = yy; oy < iInfo_.dims[1]; oy += incy) {
                const T *Yd_i = d_i + oy * iInfo_.strides[1];
                T *Yd_l       = d_l + oy * lInfo_.strides[1];
                T *Yd_u       = d_u + oy * uInfo_.strides[1];
                for (int ox = xx; ox < iInfo_.dims[0]; ox += incx) {
                    if (ox > oy) {
                        if (same_dims || oy < lInfo_.dims[1])
                            Yd_l[ox] = Yd_i[ox];
                        if (!same_dims || ox < uInfo_.dims[0])
                            Yd_u[ox] = scalar<T>(0);
                    } else if (oy > ox) {
                        if (same_dims || oy < lInfo_.dims[1])
                            Yd_l[ox] = scalar<T>(0);
                        if (!same_dims || ox < uInfo_.dims[0])
                            Yd_u[ox] = Yd_i[ox];
                    } else if (ox == oy) {
                        if (same_dims || oy < lInfo_.dims[1])
                            Yd_l[ox] = scalar<T>(1.0);
                        if (!same_dims || ox < uInfo_.dims[0])
                            Yd_u[ox] = Yd_i[ox];
                    }
                }
            }
        }
    }

   protected:
    write_accessor<T> lower_;
    KParam lInfo_;
    write_accessor<T> upper_;
    KParam uInfo_;
    read_accessor<T> in_;
    KParam iInfo_;
    int groupsPerMatX_;
    int groupsPerMatY_;
};

template<typename T>
void lu_split(Param<T> lower, Param<T> upper, Param<T> in) {
    constexpr unsigned TX    = 32;
    constexpr unsigned TY    = 8;
    constexpr unsigned TILEX = 128;
    constexpr unsigned TILEY = 32;

    const bool sameDims = lower.info.dims[0] == in.info.dims[0] &&
                          lower.info.dims[1] == in.info.dims[1];

    sycl::range<2> local(TX, TY);

    int groupsPerMatX = divup(in.info.dims[0], TILEX);
    int groupsPerMatY = divup(in.info.dims[1], TILEY);
    sycl::range<2> global(groupsPerMatX * in.info.dims[2] * local[0],
                          groupsPerMatY * in.info.dims[3] * local[1]);

    getQueue().submit([&](sycl::handler &h) {
        read_accessor<T> iData{*in.data, h};
        write_accessor<T> lData{*lower.data, h};
        write_accessor<T> uData{*upper.data, h};

        if (sameDims) {
            h.parallel_for(sycl::nd_range{global, local},
                           luSplitKernel<T, true>(
                               lData, lower.info, uData, upper.info, iData,
                               in.info, groupsPerMatX, groupsPerMatY));
        } else {
            h.parallel_for(sycl::nd_range{global, local},
                           luSplitKernel<T, false>(
                               lData, lower.info, uData, upper.info, iData,
                               in.info, groupsPerMatX, groupsPerMatY));
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
