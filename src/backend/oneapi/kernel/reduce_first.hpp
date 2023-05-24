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
#include <backend.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/reduce_config.hpp>
#include <math.hpp>
#include <memory.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <climits>
#include <complex>
#include <iostream>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename Ti, typename To, af_op_t op, uint DIMX>
class reduceFirstKernelSMEM {
   public:
    reduceFirstKernelSMEM(write_accessor<To> out, KParam oInfo,
                          read_accessor<Ti> in, KParam iInfo, uint groups_x,
                          uint groups_y, uint repeat, bool change_nan,
                          To nanval,
                          sycl::local_accessor<compute_t<To>, 1> s_val)
        : out_(out)
        , oInfo_(oInfo)
        , iInfo_(iInfo)
        , in_(in)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , repeat_(repeat)
        , change_nan_(change_nan)
        , nanval_(nanval)
        , s_val_(s_val) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g   = it.get_group();
        const uint lidx = it.get_local_id(0);
        const uint lidy = it.get_local_id(1);
        const uint lid  = lidy * g.get_local_range(0) + lidx;

        const uint zid       = g.get_group_id(0) / groups_x_;
        const uint wid       = g.get_group_id(1) / groups_y_;
        const uint groupId_x = g.get_group_id(0) - (groups_x_)*zid;
        const uint groupId_y = g.get_group_id(1) - (groups_y_)*wid;
        const uint xid = groupId_x * g.get_local_range(0) * repeat_ + lidx;
        const uint yid = groupId_y * g.get_local_range(1) + lidy;

        common::Binary<compute_t<To>, op> reduce;
        common::Transform<Ti, compute_t<To>, op> transform;

        const Ti *iptr = in_.get_pointer() + wid * iInfo_.strides[3] +
                         zid * iInfo_.strides[2] + yid * iInfo_.strides[1] +
                         iInfo_.offset;

        auto optr = out_.get_pointer() + wid * oInfo_.strides[3] +
                    zid * oInfo_.strides[2] + yid * oInfo_.strides[1];

        bool cond = (yid < iInfo_.dims[1]) && (zid < iInfo_.dims[2]) &&
                    (wid < iInfo_.dims[3]);

        dim_t last = (xid + repeat_ * DIMX);
        int lim    = sycl::min(last, iInfo_.dims[0]);

        compute_t<To> out_val = common::Binary<compute_t<To>, op>::init();
        for (int id = xid; cond && id < lim; id += DIMX) {
            compute_t<To> in_val = transform(iptr[id]);
            if (change_nan_)
                in_val = !IS_NAN(in_val) ? in_val
                                         : static_cast<compute_t<To>>(nanval_);
            out_val = reduce(in_val, out_val);
        }

        s_val_[lid] = out_val;

        it.barrier();
        compute_t<To> *s_ptr = s_val_.get_pointer() + lidy * DIMX;

        if (DIMX == 256) {
            if (lidx < 128)
                s_ptr[lidx] = reduce(s_ptr[lidx], s_ptr[lidx + 128]);
            it.barrier();
        }

        if (DIMX >= 128) {
            if (lidx < 64) s_ptr[lidx] = reduce(s_ptr[lidx], s_ptr[lidx + 64]);
            it.barrier();
        }

        if (DIMX >= 64) {
            if (lidx < 32) s_ptr[lidx] = reduce(s_ptr[lidx], s_ptr[lidx + 32]);
            it.barrier();
        }

        // TODO: replace with subgroup operations in optimized kernels
        if (lidx < 16) s_ptr[lidx] = reduce(s_ptr[lidx], s_ptr[lidx + 16]);
        it.barrier();

        if (lidx < 8) s_ptr[lidx] = reduce(s_ptr[lidx], s_ptr[lidx + 8]);
        it.barrier();

        if (lidx < 4) s_ptr[lidx] = reduce(s_ptr[lidx], s_ptr[lidx + 4]);
        it.barrier();

        if (lidx < 2) s_ptr[lidx] = reduce(s_ptr[lidx], s_ptr[lidx + 2]);
        it.barrier();

        if (lidx < 1) s_ptr[lidx] = reduce(s_ptr[lidx], s_ptr[lidx + 1]);
        it.barrier();

        if (cond && lidx == 0) optr[groupId_x] = data_t<To>(s_ptr[lidx]);
    }

   protected:
    write_accessor<To> out_;
    KParam oInfo_, iInfo_;
    read_accessor<Ti> in_;
    uint groups_x_, groups_y_, repeat_;
    bool change_nan_;
    To nanval_;
    sycl::local_accessor<compute_t<To>, 1> s_val_;
};

template<typename Ti, typename To, af_op_t op>
void reduce_first_launcher_default(Param<To> out, Param<Ti> in,
                                   const uint groups_x, const uint groups_y,
                                   const uint threads_x, bool change_nan,
                                   double nanval) {
    sycl::range<2> local(threads_x, creduce::THREADS_PER_BLOCK / threads_x);
    sycl::range<2> global(groups_x * in.info.dims[2] * local[0],
                          groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (groups_x * threads_x));

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<To> out_acc{*out.data, h};
        read_accessor<Ti> in_acc{*in.data, h};

        auto shrdMem = sycl::local_accessor<compute_t<To>, 1>(
            creduce::THREADS_PER_BLOCK, h);

        switch (threads_x) {
            case 32:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    reduceFirstKernelSMEM<Ti, To, op, 32>(
                        out_acc, out.info, in_acc, in.info, groups_x, groups_y,
                        repeat, change_nan, scalar<To>(nanval), shrdMem));
                break;
            case 64:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    reduceFirstKernelSMEM<Ti, To, op, 64>(
                        out_acc, out.info, in_acc, in.info, groups_x, groups_y,
                        repeat, change_nan, scalar<To>(nanval), shrdMem));
                break;
            case 128:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    reduceFirstKernelSMEM<Ti, To, op, 128>(
                        out_acc, out.info, in_acc, in.info, groups_x, groups_y,
                        repeat, change_nan, scalar<To>(nanval), shrdMem));
                break;
            case 256:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    reduceFirstKernelSMEM<Ti, To, op, 256>(
                        out_acc, out.info, in_acc, in.info, groups_x, groups_y,
                        repeat, change_nan, scalar<To>(nanval), shrdMem));
                break;
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op>
void reduce_first_default(Param<To> out, Param<Ti> in, bool change_nan,
                          double nanval) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, creduce::THREADS_PER_BLOCK);
    uint threads_y = creduce::THREADS_PER_BLOCK / threads_x;

    uint blocks_x = divup(in.info.dims[0], threads_x * creduce::REPEAT);
    uint blocks_y = divup(in.info.dims[1], threads_y);

    Param<To> tmp = out;
    bufptr<To> tmp_alloc;
    if (blocks_x > 1) {
        tmp_alloc = memAlloc<To>(blocks_x * in.info.dims[1] * in.info.dims[2] *
                                 in.info.dims[3]);
        tmp.data  = tmp_alloc.get();

        tmp.info.dims[0] = blocks_x;
        for (int k = 1; k < 4; k++) tmp.info.strides[k] *= blocks_x;
    }

    reduce_first_launcher_default<Ti, To, op>(tmp, in, blocks_x, blocks_y,
                                              threads_x, change_nan, nanval);

    if (blocks_x > 1) {
        // FIXME: Is there an alternative to the if condition?
        if (op == af_notzero_t) {
            reduce_first_launcher_default<To, To, af_add_t>(
                out, tmp, 1, blocks_y, threads_x, change_nan, nanval);
        } else {
            reduce_first_launcher_default<To, To, op>(
                out, tmp, 1, blocks_y, threads_x, change_nan, nanval);
        }
    }
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
