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
#include <af/defines.h>

#include <sycl/sycl.hpp>

#include <array>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T, int BType>
class padBordersKernel {
   public:
    padBordersKernel(write_accessor<T> out, KParam oInfo, read_accessor<T> in,
                     KParam iInfo, const dim_t l0, const dim_t l1,
                     const dim_t l2, const dim_t l3, const int groups_x,
                     const int groups_y)
        : out_(out)
        , oInfo_(oInfo)
        , in_(in)
        , iInfo_(iInfo)
        , l0_(l0)
        , l1_(l1)
        , l2_(l2)
        , l3_(l3)
        , groups_x_(groups_x)
        , groups_y_(groups_y) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        const int lx  = it.get_local_id(0);
        const int ly  = it.get_local_id(1);
        const int k   = g.get_group_id(0) / groups_x_;
        const int l   = g.get_group_id(1) / groups_y_;

        const int blockIdx_x = g.get_group_id(0) - (groups_x_)*k;
        const int blockIdx_y = g.get_group_id(1) - (groups_y_)*l;
        const int i          = blockIdx_x * g.get_local_range(0) + lx;
        const int j          = blockIdx_y * g.get_local_range(1) + ly;

        const size_t d0 = iInfo_.dims[0];
        const size_t d1 = iInfo_.dims[1];
        const size_t d2 = iInfo_.dims[2];
        const size_t d3 = iInfo_.dims[3];
        const size_t s0 = iInfo_.strides[0];
        const size_t s1 = iInfo_.strides[1];
        const size_t s2 = iInfo_.strides[2];
        const size_t s3 = iInfo_.strides[3];

        const T* src = in_.get_pointer() + iInfo_.offset;
        T* dst       = out_.get_pointer();

        bool isNotPadding =
            (l >= l3_ && l < (d3 + l3_)) && (k >= l2_ && k < (d2 + l2_)) &&
            (j >= l1_ && j < (d1 + l1_)) && (i >= l0_ && i < (d0 + l0_));

        T value = scalar<T>(0);
        if (isNotPadding) {
            unsigned iLOff = (l - l3_) * s3;
            unsigned iKOff = (k - l2_) * s2;
            unsigned iJOff = (j - l1_) * s1;
            unsigned iIOff = (i - l0_) * s0;

            value = src[iLOff + iKOff + iJOff + iIOff];
        } else if (BType != AF_PAD_ZERO) {
            unsigned iLOff =
                padBordersKernel<T, BType>::idxByndEdge(l, l3_, d3) * s3;
            unsigned iKOff =
                padBordersKernel<T, BType>::idxByndEdge(k, l2_, d2) * s2;
            unsigned iJOff =
                padBordersKernel<T, BType>::idxByndEdge(j, l1_, d1) * s1;
            unsigned iIOff =
                padBordersKernel<T, BType>::idxByndEdge(i, l0_, d0) * s0;

            value = src[iLOff + iKOff + iJOff + iIOff];
        }

        size_t xlim = oInfo_.dims[0];
        size_t ylim = oInfo_.dims[1];
        size_t zlim = oInfo_.dims[2];
        size_t wlim = oInfo_.dims[3];

        size_t woStrides = oInfo_.strides[3];
        size_t zoStrides = oInfo_.strides[2];
        size_t yoStrides = oInfo_.strides[1];
        size_t xoStrides = oInfo_.strides[0];

        if (i < xlim && j < ylim && k < zlim && l < wlim) {
            unsigned off =
                (l * woStrides + k * zoStrides + j * yoStrides + i * xoStrides);
            dst[off] = value;
        }
    }

    static int trimIndex(int idx, const int len) {
        int ret_val = idx;
        if (ret_val < 0) {
            int offset = (abs(ret_val) - 1) % len;
            ret_val    = offset;
        } else if (ret_val >= len) {
            int offset = abs(ret_val) % len;
            ret_val    = len - offset - 1;
        }
        return ret_val;
    }

    static int idxByndEdge(const int i, const int lb, const int len) {
        uint retVal;
        switch (BType) {
            case AF_PAD_SYM:
                retVal = padBordersKernel<T, BType>::trimIndex(i - lb, len);
                break;
            case AF_PAD_CLAMP_TO_EDGE:
                retVal = sycl::clamp(i - lb, 0, len - 1);
                break;
            case AF_PAD_PERIODIC: {
                int rem   = (i - lb) % len;
                bool cond = rem < 0;
                retVal    = cond * (rem + len) + (1 - cond) * rem;
            } break;
            default: retVal = 0; break;  // AF_PAD_ZERO
        }
        return retVal;
    }

   protected:
    write_accessor<T> out_;
    KParam oInfo_;
    read_accessor<T> in_;
    KParam iInfo_;
    const dim_t l0_;
    const dim_t l1_;
    const dim_t l2_;
    const dim_t l3_;
    const int groups_x_;
    const int groups_y_;
};

static const int PADB_THREADS_X = 32;
static const int PADB_THREADS_Y = 8;

template<typename T>
void padBorders(Param<T> out, Param<T> in, dim4 const lBoundPadding,
                const af::borderType btype) {
    sycl::range<2> local(PADB_THREADS_X, PADB_THREADS_Y);

    int groups_x = divup(out.info.dims[0], PADB_THREADS_X);
    int groups_y = divup(out.info.dims[1], PADB_THREADS_Y);

    sycl::range<2> global(groups_x * out.info.dims[2] * local[0],
                          groups_y * out.info.dims[3] * local[1]);

    getQueue().submit([&](sycl::handler& h) {
        read_accessor<T> iData{*in.data, h};
        write_accessor<T> oData{*out.data, h};

        switch (btype) {
            case AF_PAD_ZERO:
                h.parallel_for(
                    sycl::nd_range{global, local},
                    padBordersKernel<T, AF_PAD_ZERO>(
                        oData, out.info, iData, in.info, lBoundPadding[0],
                        lBoundPadding[1], lBoundPadding[2], lBoundPadding[3],
                        groups_x, groups_y));
                break;
            case AF_PAD_SYM:
                h.parallel_for(
                    sycl::nd_range{global, local},
                    padBordersKernel<T, AF_PAD_SYM>(
                        oData, out.info, iData, in.info, lBoundPadding[0],
                        lBoundPadding[1], lBoundPadding[2], lBoundPadding[3],
                        groups_x, groups_y));
                break;
            case AF_PAD_CLAMP_TO_EDGE:
                h.parallel_for(
                    sycl::nd_range{global, local},
                    padBordersKernel<T, AF_PAD_CLAMP_TO_EDGE>(
                        oData, out.info, iData, in.info, lBoundPadding[0],
                        lBoundPadding[1], lBoundPadding[2], lBoundPadding[3],
                        groups_x, groups_y));
                break;
            case AF_PAD_PERIODIC:
                h.parallel_for(
                    sycl::nd_range{global, local},
                    padBordersKernel<T, AF_PAD_PERIODIC>(
                        oData, out.info, iData, in.info, lBoundPadding[0],
                        lBoundPadding[1], lBoundPadding[2], lBoundPadding[3],
                        groups_x, groups_y));
                break;
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
