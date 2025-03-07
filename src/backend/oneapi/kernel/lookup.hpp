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
#include <kernel/accessors.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

int trimIndex(int idx, const int len) {
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

template<typename in_t, typename idx_t>
class lookupNDCreateKernel {
   public:
    lookupNDCreateKernel(write_accessor<in_t> out, KParam oInfo,
                         read_accessor<in_t> in, KParam iInfo,
                         read_accessor<idx_t> indices, KParam idxInfo,
                         int nBBS0, int nBBS1, const int DIM)
        : out_(out)
        , oInfo_(oInfo)
        , in_(in)
        , iInfo_(iInfo)
        , indices_(indices)
        , idxInfo_(idxInfo)
        , nBBS0_(nBBS0)
        , nBBS1_(nBBS1)
        , DIM_(DIM) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        int lx = it.get_local_id(0);
        int ly = it.get_local_id(1);

        int gz = g.get_group_id(0) / nBBS0_;
        int gw = g.get_group_id(1) / nBBS1_;

        int gx = g.get_local_range(0) * (g.get_group_id(0) - gz * nBBS0_) + lx;
        int gy = g.get_local_range(1) * (g.get_group_id(1) - gw * nBBS1_) + ly;

        const idx_t *idxPtr = indices_.get_pointer();

        int i = iInfo_.strides[0] *
                (DIM_ == 0 ? trimIndex((int)idxPtr[gx], iInfo_.dims[0]) : gx);
        int j = iInfo_.strides[1] *
                (DIM_ == 1 ? trimIndex((int)idxPtr[gy], iInfo_.dims[1]) : gy);
        int k = iInfo_.strides[2] *
                (DIM_ == 2 ? trimIndex((int)idxPtr[gz], iInfo_.dims[2]) : gz);
        int l = iInfo_.strides[3] *
                (DIM_ == 3 ? trimIndex((int)idxPtr[gw], iInfo_.dims[3]) : gw);

        const in_t *inPtr = in_.get_pointer() + (i + j + k + l) + iInfo_.offset;
        in_t *outPtr =
            out_.get_pointer() +
            (gx * oInfo_.strides[0] + gy * oInfo_.strides[1] +
             gz * oInfo_.strides[2] + gw * oInfo_.strides[3] + oInfo_.offset);

        if (gx < oInfo_.dims[0] && gy < oInfo_.dims[1] && gz < oInfo_.dims[2] &&
            gw < oInfo_.dims[3]) {
            outPtr[0] = inPtr[0];
        }
    }

   private:
    write_accessor<in_t> out_;
    KParam oInfo_;
    read_accessor<in_t> in_;
    KParam iInfo_;
    read_accessor<idx_t> indices_;
    KParam idxInfo_;
    int nBBS0_;
    int nBBS1_;
    const int DIM_;
};

template<typename in_t, typename idx_t>
void lookup(Param<in_t> out, const Param<in_t> in, const Param<idx_t> indices,
            const unsigned dim) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;

    auto local = sycl::range(THREADS_X, THREADS_Y);

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    auto global = sycl::range(blk_x * out.info.dims[2] * THREADS_X,
                              blk_y * out.info.dims[3] * THREADS_Y);

    getQueue().submit([&](auto &h) {
        write_accessor<in_t> d_out{*out.data, h};
        read_accessor<in_t> d_in{*in.data, h};
        read_accessor<idx_t> d_indices{*indices.data, h};
        h.parallel_for(sycl::nd_range{global, local},
                       lookupNDCreateKernel<in_t, idx_t>(
                           d_out, out.info, d_in, in.info, d_indices,
                           indices.info, blk_x, blk_y, dim));
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
