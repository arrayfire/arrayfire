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
#include <types.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
class identityKernel {
   public:
    identityKernel(write_accessor<T> out, KParam oInfo, const int groups_x,
                   const int groups_y)
        : out_(out), oInfo_(oInfo), groups_x_(groups_x), groups_y_(groups_y) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        size_t idz = g.get_group_id(0) / groups_x_;
        size_t idw = g.get_group_id(1) / groups_y_;

        size_t groupId_x = g.get_group_id(0) - idz * groups_x_;
        size_t groupId_y = g.get_group_id(1) - idw * groups_y_;

        size_t idx = it.get_local_id(0) + groupId_x * g.get_local_range(0);
        size_t idy = it.get_local_id(1) + groupId_y * g.get_local_range(1);

        size_t xlim = oInfo_.dims[0];
        size_t ylim = oInfo_.dims[1];
        size_t zlim = oInfo_.dims[2];
        size_t wlim = oInfo_.dims[3];
        if (idx < xlim && idy < ylim && idz < zlim && idw < wlim) {
            const T one  = scalar<T>(1);
            const T zero = scalar<T>(0);

            T *ptr = out_.get_pointer() + idz * oInfo_.strides[2] +
                     idw * oInfo_.strides[3];
            T val                              = (idx == idy) ? one : zero;
            ptr[idx + idy * oInfo_.strides[1]] = val;
        }
    }

   protected:
    write_accessor<T> out_;
    KParam oInfo_;
    int groups_x_;
    int groups_y_;
};

template<typename T>
void identity(Param<T> out) {
    sycl::range<2> local{32, 8};

    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_y = divup(out.info.dims[1], local[1]);
    sycl::range<2> global{groups_x * out.info.dims[2] * local[0],
                          groups_y * out.info.dims[3] * local[1]};

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> oData{*out.data, h};

        h.parallel_for(sycl::nd_range{global, local},
                       identityKernel<T>(oData, out.info, groups_x, groups_y));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
