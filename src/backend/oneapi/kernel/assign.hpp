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
#include <debug_oneapi.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace oneapi {
namespace kernel {

typedef struct {
    int offs[4];
    int strds[4];
    char isSeq[4];
} AssignKernelParam_t;

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

template<typename T>
class assignKernel {
   public:
    assignKernel(sycl::accessor<T> out, KParam oInfo, sycl::accessor<T> in,
                 KParam iInfo, AssignKernelParam_t p, sycl::accessor<uint> ptr0,
                 sycl::accessor<uint> ptr1, sycl::accessor<uint> ptr2,
                 sycl::accessor<uint> ptr3, const int nBBS0, const int nBBS1,
                 sycl::stream debug)
        : out_(out)
        , oInfo_(oInfo)
        , in_(in)
        , iInfo_(iInfo)
        , p_(p)
        , ptr0_(ptr0)
        , ptr1_(ptr1)
        , ptr2_(ptr2)
        , ptr3_(ptr3)
        , nBBS0_(nBBS0)
        , nBBS1_(nBBS1)
        , debug_(debug) {}

    void operator()(sycl::nd_item<2> it) const {
        // retrive booleans that tell us which index to use
        const bool s0 = p_.isSeq[0];
        const bool s1 = p_.isSeq[1];
        const bool s2 = p_.isSeq[2];
        const bool s3 = p_.isSeq[3];

        sycl::group g = it.get_group();
        const int gz  = g.get_group_id(0) / nBBS0_;
        const int gw  = g.get_group_id(1) / nBBS1_;
        const int gx =
            g.get_local_range(0) * (g.get_group_id(0) - gz * nBBS0_) +
            it.get_local_id(0);
        const int gy =
            g.get_local_range(1) * (g.get_group_id(1) - gw * nBBS1_) +
            it.get_local_id(1);
        if (gx < iInfo_.dims[0] && gy < iInfo_.dims[1] && gz < iInfo_.dims[2] &&
            gw < iInfo_.dims[3]) {
            // calculate pointer offsets for input
            int i = p_.strds[0] *
                    trimIndex(s0 ? gx + p_.offs[0] : ptr0_[gx], oInfo_.dims[0]);
            int j = p_.strds[1] *
                    trimIndex(s1 ? gy + p_.offs[1] : ptr1_[gy], oInfo_.dims[1]);
            int k = p_.strds[2] *
                    trimIndex(s2 ? gz + p_.offs[2] : ptr2_[gz], oInfo_.dims[2]);
            int l = p_.strds[3] *
                    trimIndex(s3 ? gw + p_.offs[3] : ptr3_[gw], oInfo_.dims[3]);

            T* iptr = in_.get_pointer();
            // offset input and output pointers
            const T* src =
                iptr + (gx * iInfo_.strides[0] + gy * iInfo_.strides[1] +
                        gz * iInfo_.strides[2] + gw * iInfo_.strides[3] +
                        iInfo_.offset);

            T* optr = out_.get_pointer();
            T* dst  = optr + (i + j + k + l) + oInfo_.offset;
            // set the output
            dst[0] = src[0];
        }
    }

   protected:
    sycl::accessor<T> out_, in_;
    KParam oInfo_, iInfo_;
    AssignKernelParam_t p_;
    sycl::accessor<uint> ptr0_, ptr1_, ptr2_, ptr3_;
    const int nBBS0_, nBBS1_;
    sycl::stream debug_;
};

template<typename T>
void assign(Param<T> out, const Param<T> in, const AssignKernelParam_t& p,
            sycl::buffer<uint>* bPtr[4]) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;

    sycl::range<2> local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    sycl::range<2> global(blk_x * in.info.dims[2] * THREADS_X,
                          blk_y * in.info.dims[3] * THREADS_Y);

    getQueue().submit([=](sycl::handler& h) {
        auto out_acc = out.data->get_access(h);
        auto in_acc  = in.data->get_access(h);

        auto bptr0 = bPtr[0]->get_access(h);
        auto bptr1 = bPtr[1]->get_access(h);
        auto bptr2 = bPtr[2]->get_access(h);
        auto bptr3 = bPtr[3]->get_access(h);

        sycl::stream debug_stream(2048, 128, h);

        h.parallel_for(
            sycl::nd_range<2>(global, local),
            assignKernel<T>(out_acc, out.info, in_acc, in.info, p, bptr0, bptr1,
                            bptr2, bptr3, blk_x, blk_y, debug_stream));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace oneapi
