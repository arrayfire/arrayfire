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

#include <algorithm>
#include <climits>
#include <complex>
#include <iostream>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename Ti, typename To, af_op_t op, uint dim, uint DIMY>
class reduceDimKernelSMEM {
   public:
    reduceDimKernelSMEM(Param<To> out, Param<Ti> in, uint groups_x,
                        uint groups_y, uint offset_dim, bool change_nan,
                        To nanval, sycl::local_accessor<compute_t<To>, 1> s_val,
                        sycl::handler &h)
        : out_(out.template get_accessor<sycl::access::mode::write>(h))
        , in_(in.template get_accessor<sycl::access::mode::read>(h))
        , oInfo_(out.info)
        , iInfo_(in.info)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , offset_dim_(offset_dim)
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
        const uint xid       = groupId_x * g.get_local_range(0) + lidx;
        const uint yid       = groupId_y;

        uint ids[4] = {xid, yid, zid, wid};
        using sycl::global_ptr;

        data_t<To> *optr = out_.get_pointer() + ids[3] * oInfo_.strides[3] +
                           ids[2] * oInfo_.strides[2] +
                           ids[1] * oInfo_.strides[1] + ids[0];

        const uint groupIdx_dim = ids[dim];
        ids[dim]                = ids[dim] * g.get_local_range(1) + lidy;

        const data_t<Ti> *iptr =
            in_.get_pointer() + ids[3] * iInfo_.strides[3] +
            ids[2] * iInfo_.strides[2] + ids[1] * iInfo_.strides[1] + ids[0];

        const uint id_dim_in   = ids[dim];
        const uint istride_dim = iInfo_.strides[dim];
        bool is_valid          = (ids[0] < iInfo_.dims[0]) &&
                        (ids[1] < iInfo_.dims[1]) &&
                        (ids[2] < iInfo_.dims[2]) && (ids[3] < iInfo_.dims[3]);

        common::Binary<compute_t<To>, op> reduce;
        common::Transform<data_t<Ti>, compute_t<To>, op> transform;

        compute_t<To> out_val = common::Binary<compute_t<To>, op>::init();
        for (int id = id_dim_in; is_valid && (id < iInfo_.dims[dim]);
             id += offset_dim_ * g.get_local_range(1)) {
            compute_t<To> in_val = transform(*iptr);
            if (change_nan_) {
                in_val = !IS_NAN(in_val) ? in_val
                                         : static_cast<compute_t<To>>(nanval_);
            }
            out_val = reduce(in_val, out_val);
            iptr += offset_dim_ * g.get_local_range(1) * istride_dim;
        }

        s_val_[lid] = out_val;

        it.barrier();
        compute_t<To> *s_ptr = s_val_.get_pointer() + lid;

        if (DIMY == 8) {
            if (lidy < 4)
                *s_ptr = reduce(*s_ptr, s_ptr[creduce::THREADS_X * 4]);
            it.barrier();
        }

        if (DIMY >= 4) {
            if (lidy < 2)
                *s_ptr = reduce(*s_ptr, s_ptr[creduce::THREADS_X * 2]);
            it.barrier();
        }

        if (DIMY >= 2) {
            if (lidy < 1)
                *s_ptr = reduce(*s_ptr, s_ptr[creduce::THREADS_X * 1]);
            it.barrier();
        }

        if (lidy == 0 && is_valid && (groupIdx_dim < oInfo_.dims[dim])) {
            *optr = data_t<To>(*s_ptr);
        }
    }

   protected:
    write_accessor<data_t<To>> out_;
    read_accessor<data_t<Ti>> in_;
    KParam oInfo_, iInfo_;
    uint groups_x_, groups_y_, offset_dim_;
    bool change_nan_;
    To nanval_;
    sycl::local_accessor<compute_t<To>, 1> s_val_;
};

template<typename Ti, typename To, af_op_t op, uint dim>
void reduce_dim_launcher_default(Param<To> out, Param<Ti> in,
                                 const uint threads_y,
                                 const dim_t blocks_dim[4], bool change_nan,
                                 double nanval) {
    sycl::range<2> local(creduce::THREADS_X, threads_y);
    sycl::range<2> global(blocks_dim[0] * blocks_dim[2] * local[0],
                          blocks_dim[1] * blocks_dim[3] * local[1]);

    getQueue().submit([&](sycl::handler &h) {
        auto shrdMem = sycl::local_accessor<compute_t<To>, 1>(
            creduce::THREADS_X * threads_y, h);

        switch (threads_y) {
            case 8:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    reduceDimKernelSMEM<Ti, To, op, dim, 8>(
                        out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                        change_nan, scalar<To>(nanval), shrdMem, h));
                break;
            case 4:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    reduceDimKernelSMEM<Ti, To, op, dim, 4>(
                        out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                        change_nan, scalar<To>(nanval), shrdMem, h));
                break;
            case 2:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    reduceDimKernelSMEM<Ti, To, op, dim, 2>(
                        out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                        change_nan, scalar<To>(nanval), shrdMem, h));
                break;
            case 1:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    reduceDimKernelSMEM<Ti, To, op, dim, 1>(
                        out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                        change_nan, scalar<To>(nanval), shrdMem, h));
                break;
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op, int dim>
void reduce_dim_default(Param<To> out, Param<Ti> in, bool change_nan,
                        double nanval) {
    uint threads_y = std::min(creduce::THREADS_Y, nextpow2(in.info.dims[dim]));
    uint threads_x = creduce::THREADS_X;

    dim_t blocks_dim[] = {divup(in.info.dims[0], threads_x), in.info.dims[1],
                          in.info.dims[2], in.info.dims[3]};
    blocks_dim[dim]    = divup(in.info.dims[dim], threads_y * creduce::REPEAT);

    Param<To> tmp = out;
    bufptr<To> tmp_alloc;
    if (blocks_dim[dim] > 1) {
        tmp.info.dims[dim] = blocks_dim[dim];
        int tmp_elements   = tmp.info.dims[0] * tmp.info.dims[1] *
                           tmp.info.dims[2] * tmp.info.dims[3];

        tmp_alloc = memAlloc<To>(tmp_elements);
        tmp.data  = tmp_alloc.get();

        tmp.info.dims[dim] = blocks_dim[dim];
        for (int k = dim + 1; k < 4; k++)
            tmp.info.strides[k] *= blocks_dim[dim];
    }

    reduce_dim_launcher_default<Ti, To, op, dim>(tmp, in, threads_y, blocks_dim,
                                                 change_nan, nanval);

    if (blocks_dim[dim] > 1) {
        blocks_dim[dim] = 1;

        if (op == af_notzero_t) {
            reduce_dim_launcher_default<To, To, af_add_t, dim>(
                out, tmp, threads_y, blocks_dim, change_nan, nanval);
        } else {
            reduce_dim_launcher_default<To, To, op, dim>(
                out, tmp, threads_y, blocks_dim, change_nan, nanval);
        }
    }
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
