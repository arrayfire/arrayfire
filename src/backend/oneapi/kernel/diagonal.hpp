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
class diagCreateKernel {
   public:
    diagCreateKernel(write_accessor<T> oData, KParam oInfo,
                     read_accessor<T> iData, KParam iInfo, int num,
                     int groups_x)
        : oData_(oData)
        , oInfo_(oInfo)
        , iData_(iData)
        , iInfo_(iInfo)
        , num_(num)
        , groups_x_(groups_x) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g      = it.get_group();
        unsigned idz       = g.get_group_id(0) / groups_x_;
        unsigned groupId_x = g.get_group_id(0) - idz * groups_x_;

        unsigned idx = it.get_local_id(0) + groupId_x * g.get_local_range(0);
        unsigned idy = it.get_global_id(1);

        if (idx >= oInfo_.dims[0] || idy >= oInfo_.dims[1] ||
            idz >= oInfo_.dims[2])
            return;

        T *optr = oData_.get_pointer();
        optr += idz * oInfo_.strides[2] + idy * oInfo_.strides[1] + idx;

        const T *iptr = iData_.get_pointer();
        iptr +=
            idz * iInfo_.strides[1] + ((num_ > 0) ? idx : idy) + iInfo_.offset;

        T val = (idx == (idy - num_)) ? *iptr : (T)(0);
        *optr = val;
    }

   private:
    write_accessor<T> oData_;
    KParam oInfo_;
    read_accessor<T> iData_;
    KParam iInfo_;
    int num_;
    int groups_x_;
};

template<typename T>
static void diagCreate(Param<T> out, Param<T> in, int num) {
    auto local   = sycl::range{32, 8};
    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_y = divup(out.info.dims[1], local[1]);
    auto global  = sycl::range{groups_x * local[0] * out.info.dims[2],
                              groups_y * local[1]};

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> oData{*out.data, h};
        read_accessor<T> iData{*in.data, h};

        h.parallel_for(sycl::nd_range{global, local},
                       diagCreateKernel<T>(oData, out.info, iData, in.info, num,
                                           groups_x));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
class diagExtractKernel {
   public:
    diagExtractKernel(write_accessor<T> oData, KParam oInfo,
                      read_accessor<T> iData, KParam iInfo, int num,
                      int groups_z)
        : oData_(oData)
        , oInfo_(oInfo)
        , iData_(iData)
        , iInfo_(iInfo)
        , num_(num)
        , groups_z_(groups_z) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        unsigned idw  = g.get_group_id(1) / groups_z_;
        unsigned idz  = g.get_group_id(1) - idw * groups_z_;

        unsigned idx = it.get_global_id(0);

        if (idx >= oInfo_.dims[0] || idz >= oInfo_.dims[2] ||
            idw >= oInfo_.dims[3])
            return;

        T *optr = oData_.get_pointer();
        optr += idz * oInfo_.strides[2] + idw * oInfo_.strides[3] + idx;

        if (idx >= iInfo_.dims[0] || idx >= iInfo_.dims[1]) {
            *optr = (T)(0);
            return;
        }

        int i_off = (num_ > 0) ? (num_ * iInfo_.strides[1] + idx)
                               : (idx - num_) + iInfo_.offset;

        const T *iptr = iData_.get_pointer();
        iptr += idz * iInfo_.strides[2] + idw * iInfo_.strides[3] + i_off;

        *optr = iptr[idx * iInfo_.strides[1]];
    }

   private:
    write_accessor<T> oData_;
    KParam oInfo_;
    read_accessor<T> iData_;
    KParam iInfo_;
    int num_;
    int groups_z_;
};

template<typename T>
static void diagExtract(Param<T> out, Param<T> in, int num) {
    auto local   = sycl::range{256, 1};
    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_z = out.info.dims[2];
    auto global  = sycl::range{groups_x * local[0],
                              groups_z * local[1] * out.info.dims[3]};

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> oData{*out.data, h};
        read_accessor<T> iData{*in.data, h};

        h.parallel_for(sycl::nd_range{global, local},
                       diagExtractKernel<T>(oData, out.info, iData, in.info,
                                            num, groups_z));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
