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

template<typename T>
using global_atomic_ref =
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>;

template<typename Ti, typename To, af_op_t op>
class reduceAllKernelSMEM {
   public:
    reduceAllKernelSMEM(write_accessor<To> out, KParam oInfo,
                        sycl::accessor<unsigned> retCount,
                        sycl::accessor<To> tmp, KParam tmpInfo,
                        read_accessor<Ti> in, KParam iInfo, uint DIMX,
                        uint groups_x, uint groups_y, uint repeat,
                        bool change_nan, To nanval,
                        sycl::local_accessor<compute_t<To>, 1> s_ptr,
                        sycl::local_accessor<bool, 1> amLast)
        : out_(out)
        , retCount_(retCount)
        , tmp_(tmp)
        , in_(in)
        , oInfo_(oInfo)
        , tmpInfo_(tmpInfo)
        , iInfo_(iInfo)
        , DIMX_(DIMX)
        , repeat_(repeat)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , change_nan_(change_nan)
        , nanval_(nanval)
        , s_ptr_(s_ptr)
        , amLast_(amLast) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g   = it.get_group();
        const uint lidx = it.get_local_id(0);
        const uint lidy = it.get_local_id(1);
        const uint lid  = lidy * DIMX_ + lidx;

        const uint zid       = g.get_group_id(0) / groups_x_;
        const uint wid       = g.get_group_id(1) / groups_y_;
        const uint groupId_x = g.get_group_id(0) - (groups_x_)*zid;
        const uint groupId_y = g.get_group_id(1) - (groups_y_)*wid;
        const uint xid = groupId_x * g.get_local_range(0) * repeat_ + lidx;
        const uint yid = groupId_y * g.get_local_range(1) + lidy;

        common::Binary<compute_t<To>, op> reduce;
        common::Transform<Ti, compute_t<To>, op> transform;

        auto iptr = in_.get_pointer() + wid * iInfo_.strides[3] +
                    zid * iInfo_.strides[2] + yid * iInfo_.strides[1] +
                    iInfo_.offset;

        bool cond = (yid < iInfo_.dims[1]) && (zid < iInfo_.dims[2]) &&
                    (wid < iInfo_.dims[3]);

        dim_t last = (xid + repeat_ * DIMX_);
        int lim    = min(last, iInfo_.dims[0]);

        compute_t<To> out_val = common::Binary<compute_t<To>, op>::init();
        for (int id = xid; cond && id < lim; id += DIMX_) {
            compute_t<To> in_val = transform(iptr[id]);
            if (change_nan_)
                in_val = !IS_NAN(in_val) ? in_val
                                         : static_cast<compute_t<To>>(nanval_);
            out_val = reduce(in_val, out_val);
        }

        s_ptr_[lid] = out_val;

        group_barrier(g);

        if (creduce::THREADS_PER_BLOCK == 256) {
            if (lid < 128) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 128]);
            group_barrier(g);
        }

        if (creduce::THREADS_PER_BLOCK >= 128) {
            if (lid < 64) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 64]);
            group_barrier(g);
        }

        if (creduce::THREADS_PER_BLOCK >= 64) {
            if (lid < 32) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 32]);
            group_barrier(g);
        }

        // TODO: replace with subgroup operations in optimized kernels
        if (lid < 16) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 16]);
        group_barrier(g);

        if (lid < 8) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 8]);
        group_barrier(g);

        if (lid < 4) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 4]);
        group_barrier(g);

        if (lid < 2) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 2]);
        group_barrier(g);

        if (lid < 1) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 1]);
        group_barrier(g);

        const unsigned total_blocks =
            (g.get_group_range(0) * g.get_group_range(1));
        const int uubidx =
            (g.get_group_range(0) * g.get_group_id(1)) + g.get_group_id(0);
        if (cond && lid == 0) {
            if (total_blocks != 1) {
                tmp_[uubidx] = s_ptr_[0];
            } else {
                out_[0] = s_ptr_[0];
            }
        }

        // Last block to perform final reduction
        if (total_blocks > 1) {
            sycl::atomic_fence(sycl::memory_order::seq_cst,
                               sycl::memory_scope::device);

            // thread 0 takes a ticket
            if (lid == 0) {
                unsigned int ticket = global_atomic_ref<uint>(retCount_[0])++;
                // If the ticket ID == number of blocks, we are the last block
                amLast_[0] = (ticket == (total_blocks - 1));
            }
            group_barrier(g);

            if (amLast_[0]) {
                int i   = lid;
                out_val = common::Binary<compute_t<To>, op>::init();

                while (i < total_blocks) {
                    compute_t<To> in_val = compute_t<To>(tmp_[i]);
                    out_val              = reduce(in_val, out_val);
                    i += creduce::THREADS_PER_BLOCK;
                }

                s_ptr_[lid] = out_val;
                group_barrier(g);

                // reduce final block
                if (creduce::THREADS_PER_BLOCK == 256) {
                    if (lid < 128)
                        s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 128]);
                    group_barrier(g);
                }

                if (creduce::THREADS_PER_BLOCK >= 128) {
                    if (lid < 64)
                        s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 64]);
                    group_barrier(g);
                }

                if (creduce::THREADS_PER_BLOCK >= 64) {
                    if (lid < 32)
                        s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 32]);
                    group_barrier(g);
                }

                if (lid < 16)
                    s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 16]);
                group_barrier(g);

                if (lid < 8) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 8]);
                group_barrier(g);

                if (lid < 4) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 4]);
                group_barrier(g);

                if (lid < 2) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 2]);
                group_barrier(g);

                if (lid < 1) s_ptr_[lid] = reduce(s_ptr_[lid], s_ptr_[lid + 1]);
                group_barrier(g);

                if (lid == 0) {
                    out_[0] = s_ptr_[0];

                    // reset retirement count so that next run succeeds
                    retCount_[0] = 0;
                }
            }
        }
    }

   protected:
    write_accessor<To> out_;
    sycl::accessor<unsigned> retCount_;
    sycl::accessor<To> tmp_;
    read_accessor<Ti> in_;
    KParam oInfo_, tmpInfo_, iInfo_;
    uint DIMX_, repeat_;
    uint groups_x_, groups_y_;
    bool change_nan_;
    To nanval_;
    sycl::local_accessor<compute_t<To>, 1> s_ptr_;
    sycl::local_accessor<bool, 1> amLast_;
};

template<typename Ti, typename To, af_op_t op>
void reduce_all_launcher_default(Param<To> out, Param<Ti> in,
                                 const uint groups_x, const uint groups_y,
                                 const uint threads_x, bool change_nan,
                                 double nanval) {
    sycl::range<2> local(threads_x, creduce::THREADS_PER_BLOCK / threads_x);
    sycl::range<2> global(groups_x * in.info.dims[2] * local[0],
                          groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (groups_x * threads_x));

    long tmp_elements = groups_x * in.info.dims[2] * groups_y * in.info.dims[3];
    if (tmp_elements > UINT_MAX) {
        AF_ERROR(
            "Too many blocks requested (typeof(retirementCount) == unsigned)",
            AF_ERR_RUNTIME);
    }

    Array<To> tmp = createEmptyArray<To>(tmp_elements);
    auto tmp_get = tmp.get();
    
    Array<unsigned> retirementCount = createValueArray<unsigned>(1, 0);
    auto ret_get = retirementCount.get();

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<To> out_acc{*out.data, h};
        auto retCount_acc = ret_get->get_access(h);
        auto tmp_acc      = tmp_get->get_access(h);
        read_accessor<Ti> in_acc{*in.data, h};

        auto shrdMem = sycl::local_accessor<compute_t<To>, 1>(
            creduce::THREADS_PER_BLOCK, h);
        auto amLast = sycl::local_accessor<bool, 1>(1, h);
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            reduceAllKernelSMEM<Ti, To, op>(
                out_acc, out.info, retCount_acc, tmp_acc, (KParam)tmp, in_acc,
                in.info, threads_x, groups_x, groups_y, repeat, change_nan,
                scalar<To>(nanval), shrdMem, amLast));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
