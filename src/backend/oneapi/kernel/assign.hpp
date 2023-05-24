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
#include <kernel/accessors.hpp>
#include <kernel/assign_kernel_param.hpp>
#include <traits.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

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
    assignKernel(write_accessor<T> out, KParam oInfo, read_accessor<T> in,
                 KParam iInfo, AssignKernelParam p, const int nBBS0,
                 const int nBBS1)
        : out_(out)
        , in_(in)
        , oInfo_(oInfo)
        , iInfo_(iInfo)
        , p_(p)
        , nBBS0_(nBBS0)
        , nBBS1_(nBBS1) {}

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

        size_t idims0 = iInfo_.dims[0];
        size_t idims1 = iInfo_.dims[1];
        size_t idims2 = iInfo_.dims[2];
        size_t idims3 = iInfo_.dims[3];

        if (gx < idims0 && gy < idims1 && gz < idims2 && gw < idims3) {
            // calculate pointer offsets for input
            int i =
                p_.strds[0] *
                trimIndex(s0 ? gx + p_.offs[0] : p_.ptr[0][gx], oInfo_.dims[0]);
            int j =
                p_.strds[1] *
                trimIndex(s1 ? gy + p_.offs[1] : p_.ptr[1][gy], oInfo_.dims[1]);
            int k =
                p_.strds[2] *
                trimIndex(s2 ? gz + p_.offs[2] : p_.ptr[2][gz], oInfo_.dims[2]);
            int l =
                p_.strds[3] *
                trimIndex(s3 ? gw + p_.offs[3] : p_.ptr[3][gw], oInfo_.dims[3]);

            const T* iptr = in_.get_pointer();
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
    write_accessor<T> out_;
    read_accessor<T> in_;
    KParam oInfo_, iInfo_;
    AssignKernelParam p_;
    const int nBBS0_, nBBS1_;
};

template<typename T>
void assign(Param<T> out, const Param<T> in, const AssignKernelParam& p,
            sycl::buffer<uint>* bPtr[4]) {
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;
    using sycl::access_mode;

    sycl::range<2> local(THREADS_X, THREADS_Y);

    int blk_x = divup(in.info.dims[0], THREADS_X);
    int blk_y = divup(in.info.dims[1], THREADS_Y);

    sycl::range<2> global(blk_x * in.info.dims[2] * THREADS_X,
                          blk_y * in.info.dims[3] * THREADS_Y);

    getQueue().submit([&](sycl::handler& h) {
        auto pp = p;
        write_accessor<T> out_acc{*out.data, h};
        read_accessor<T> in_acc{*in.data, h};

        pp.ptr[0] = bPtr[0]->template get_access<access_mode::read>(h);
        pp.ptr[1] = bPtr[1]->template get_access<access_mode::read>(h);
        pp.ptr[2] = bPtr[2]->template get_access<access_mode::read>(h);
        pp.ptr[3] = bPtr[3]->template get_access<access_mode::read>(h);

        h.parallel_for(sycl::nd_range<2>(global, local),
                       assignKernel<T>(out_acc, out.info, in_acc, in.info, pp,
                                       blk_x, blk_y));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
