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
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/reduce_config.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <minmax_op.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <climits>
#include <complex>
#include <iostream>
#include <memory>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T, af_op_t op, uint dim, bool is_first, uint DIMY>
class ireduceDimKernelSMEM {
   public:
    ireduceDimKernelSMEM(write_accessor<T> out, KParam oInfo,
                         write_accessor<uint> oloc, KParam olocInfo,
                         read_accessor<T> in, KParam iInfo,
                         read_accessor<uint> iloc, KParam ilocInfo,
                         uint groups_x, uint groups_y, uint groups_dim,
                         bool rlenValid, read_accessor<uint> rlen,
                         KParam rlenInfo,
                         sycl::local_accessor<compute_t<T>, 1> s_val,
                         sycl::local_accessor<uint, 1> s_idx)
        : out_(out)
        , oInfo_(oInfo)
        , oloc_(oloc)
        , olocInfo_(olocInfo)
        , in_(in)
        , iInfo_(iInfo)
        , iloc_(iloc)
        , ilocInfo_(ilocInfo)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , groups_dim_(groups_dim)
        , rlenValid_(rlenValid)
        , rlen_(rlen)
        , rlenInfo_(rlenInfo)
        , s_val_(s_val)
        , s_idx_(s_idx) {}

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
        T *optr     = out_.get_pointer() + ids[3] * oInfo_.strides[3] +
                  ids[2] * oInfo_.strides[2] + ids[1] * oInfo_.strides[1] +
                  ids[0] + oInfo_.offset;

        uint *olptr = oloc_.get_pointer() + ids[3] * oInfo_.strides[3] +
                      ids[2] * oInfo_.strides[2] + ids[1] * oInfo_.strides[1] +
                      ids[0] + oInfo_.offset;

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before
        // offsetting in
        const bool rlen_valid =
            (ids[0] < rlenInfo_.dims[0]) && (ids[1] < rlenInfo_.dims[1]) &&
            (ids[2] < rlenInfo_.dims[2]) && (ids[3] < rlenInfo_.dims[3]);
        const bool rlen_nonnull = rlenValid_;
        const uint *rlenptr =
            (rlen_nonnull && rlen_valid)
                ? rlen_.get_pointer() + ids[3] * rlenInfo_.strides[3] +
                      ids[2] * rlenInfo_.strides[2] +
                      ids[1] * rlenInfo_.strides[1] + ids[0] + rlenInfo_.offset
                : nullptr;

        const uint groupIdx_dim = ids[dim];

        // add thread offset for reduced dim for inputs
        ids[dim] = ids[dim] * g.get_local_range(1) + lidy;

        const T *iptr = in_.get_pointer() + ids[3] * iInfo_.strides[3] +
                        ids[2] * iInfo_.strides[2] +
                        ids[1] * iInfo_.strides[1] + ids[0] + iInfo_.offset;
        const uint *ilptr;
        if (!is_first) {
            ilptr = iloc_.get_pointer() + ids[3] * iInfo_.strides[3] +
                    ids[2] * iInfo_.strides[2] + ids[1] * iInfo_.strides[1] +
                    ids[0] + iInfo_.offset;
        }

        const uint id_dim_in   = ids[dim];
        const uint istride_dim = iInfo_.strides[dim];

        size_t xlim   = iInfo_.dims[0];
        size_t ylim   = iInfo_.dims[1];
        size_t zlim   = iInfo_.dims[2];
        size_t wlim   = iInfo_.dims[3];
        bool is_valid = (ids[0] < xlim) && (ids[1] < ylim) && (ids[2] < zlim) &&
                        (ids[3] < wlim);

        compute_t<T> out_val = common::Binary<compute_t<T>, op>::init();
        uint out_idx         = id_dim_in;

        uint lim = rlenptr ? *rlenptr : iInfo_.dims[0];
        lim      = is_first ? sycl::min((uint)iInfo_.dims[dim], lim) : lim;

        bool within_ragged_bounds =
            (is_first) ? (out_idx < lim)
                       : ((rlenptr) ? ((is_valid) && (*ilptr < lim)) : true);
        if (is_valid && id_dim_in < iInfo_.dims[dim] && within_ragged_bounds) {
            out_val = *iptr;
            if (!is_first) out_idx = *ilptr;
        }

        MinMaxOp<op, compute_t<T>> Op(out_val, out_idx);

        const uint id_dim_in_start =
            id_dim_in + groups_dim_ * g.get_local_range(1);
        for (int id = id_dim_in_start; is_valid && (id < lim);
             id += groups_dim_ * g.get_local_range(1)) {
            iptr = iptr + groups_dim_ * g.get_local_range(1) * istride_dim;
            if (!is_first) {
                ilptr =
                    ilptr + groups_dim_ * g.get_local_range(1) * istride_dim;
                Op(*iptr, *ilptr);
            } else {
                Op(*iptr, id);
            }
        }

        s_val_[lid] = Op.m_val;
        s_idx_[lid] = Op.m_idx;
        it.barrier();

        compute_t<T> *s_vptr = s_val_.get_pointer() + lid;
        uint *s_iptr         = s_idx_.get_pointer() + lid;

        if (DIMY == 8) {
            if (lidy < 4) {
                Op(s_vptr[g.get_local_range(0) * 4],
                   s_iptr[g.get_local_range(0) * 4]);
                *s_vptr = Op.m_val;
                *s_iptr = Op.m_idx;
            }
            it.barrier();
        }
        if (DIMY >= 4) {
            if (lidy < 2) {
                Op(s_vptr[g.get_local_range(0) * 2],
                   s_iptr[g.get_local_range(0) * 2]);
                *s_vptr = Op.m_val;
                *s_iptr = Op.m_idx;
            }
            it.barrier();
        }
        if (DIMY >= 2) {
            if (lidy < 1) {
                Op(s_vptr[g.get_local_range(0) * 1],
                   s_iptr[g.get_local_range(0) * 1]);
                *s_vptr = Op.m_val;
                *s_iptr = Op.m_idx;
            }
            it.barrier();
        }
        if (is_valid && lidy == 0 && (groupIdx_dim < oInfo_.dims[dim])) {
            *optr  = data_t<T>(s_vptr[0]);
            *olptr = s_iptr[0];
        }
    }

   protected:
    write_accessor<T> out_;
    KParam oInfo_;
    write_accessor<uint> oloc_;
    KParam olocInfo_;
    read_accessor<T> in_;
    KParam iInfo_;
    read_accessor<uint> iloc_;
    KParam ilocInfo_;
    uint groups_x_, groups_y_, groups_dim_;
    bool rlenValid_;
    read_accessor<uint> rlen_;
    KParam rlenInfo_;
    sycl::local_accessor<compute_t<T>, 1> s_val_;
    sycl::local_accessor<uint, 1> s_idx_;
};

template<typename T, af_op_t op, int dim, bool is_first>
void ireduce_dim_launcher(Param<T> out, Param<uint> oloc, Param<T> in,
                          Param<uint> iloc, const uint threads_y,
                          const dim_t groups_dim[4], Param<uint> rlen) {
    sycl::range<2> local(creduce::THREADS_X, threads_y);
    sycl::range<2> global(groups_dim[0] * groups_dim[2] * local[0],
                          groups_dim[1] * groups_dim[3] * local[1]);

    auto iempty = memAlloc<uint>(1);
    auto rempty = memAlloc<uint>(1);
    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> out_acc{*out.data, h};
        write_accessor<uint> oloc_acc{*oloc.data, h};
        read_accessor<T> in_acc{*in.data, h};

        read_accessor<uint> iloc_acc{*iempty, h};
        if (iloc.info.dims[0] * iloc.info.dims[1] * iloc.info.dims[2] *
                iloc.info.dims[3] >
            0) {
            iloc_acc = read_accessor<uint>{*iloc.data, h};
        }

        read_accessor<uint> rlen_acc{*rempty, h};
        bool rlenValid = (rlen.info.dims[0] * rlen.info.dims[1] *
                              rlen.info.dims[2] * rlen.info.dims[3] >
                          0);
        if (rlenValid) { rlen_acc = read_accessor<uint>{*rlen.data, h}; }

        auto shrdVal = sycl::local_accessor<compute_t<T>, 1>(
            creduce::THREADS_PER_BLOCK, h);
        auto shrdLoc =
            sycl::local_accessor<uint, 1>(creduce::THREADS_PER_BLOCK, h);

        switch (threads_y) {
            case 8:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    ireduceDimKernelSMEM<T, op, dim, is_first, 8>(
                        out_acc, out.info, oloc_acc, oloc.info, in_acc, in.info,
                        iloc_acc, iloc.info, groups_dim[0], groups_dim[1],
                        groups_dim[dim], rlenValid, rlen_acc, rlen.info,
                        shrdVal, shrdLoc));
                break;
            case 4:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    ireduceDimKernelSMEM<T, op, dim, is_first, 8>(
                        out_acc, out.info, oloc_acc, oloc.info, in_acc, in.info,
                        iloc_acc, iloc.info, groups_dim[0], groups_dim[1],
                        groups_dim[dim], rlenValid, rlen_acc, rlen.info,
                        shrdVal, shrdLoc));
                break;
            case 2:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    ireduceDimKernelSMEM<T, op, dim, is_first, 8>(
                        out_acc, out.info, oloc_acc, oloc.info, in_acc, in.info,
                        iloc_acc, iloc.info, groups_dim[0], groups_dim[1],
                        groups_dim[dim], rlenValid, rlen_acc, rlen.info,
                        shrdVal, shrdLoc));
                break;
            case 1:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    ireduceDimKernelSMEM<T, op, dim, is_first, 8>(
                        out_acc, out.info, oloc_acc, oloc.info, in_acc, in.info,
                        iloc_acc, iloc.info, groups_dim[0], groups_dim[1],
                        groups_dim[dim], rlenValid, rlen_acc, rlen.info,
                        shrdVal, shrdLoc));
                break;
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op, int dim>
void ireduce_dim(Param<T> out, Param<uint> oloc, Param<T> in,
                 Param<uint> rlen) {
    uint threads_y = std::min(creduce::THREADS_Y, nextpow2(in.info.dims[dim]));
    uint threads_x = creduce::THREADS_X;

    dim_t blocks_dim[] = {divup(in.info.dims[0], threads_x), in.info.dims[1],
                          in.info.dims[2], in.info.dims[3]};

    blocks_dim[dim] = divup(in.info.dims[dim], threads_y * creduce::REPEAT);

    Param<T> tmp      = out;
    Param<uint> tlptr = oloc;
    bufptr<T> tmp_alloc;
    bufptr<uint> tlptr_alloc;

    if (blocks_dim[dim] > 1) {
        int tmp_elements   = 1;
        tmp.info.dims[dim] = blocks_dim[dim];

        for (int k = 0; k < 4; k++) tmp_elements *= tmp.info.dims[k];
        tmp_alloc   = memAlloc<T>(tmp_elements);
        tlptr_alloc = memAlloc<uint>(tmp_elements);
        tmp.data    = tmp_alloc.get();
        tlptr.data  = tlptr_alloc.get();

        for (int k = dim + 1; k < 4; k++)
            tmp.info.strides[k] *= blocks_dim[dim];
    }

    Param<uint> nullparam;
    ireduce_dim_launcher<T, op, dim, true>(tmp, tlptr, in, nullparam, threads_y,
                                           blocks_dim, rlen);

    if (blocks_dim[dim] > 1) {
        blocks_dim[dim] = 1;

        ireduce_dim_launcher<T, op, dim, false>(out, oloc, tmp, tlptr,
                                                threads_y, blocks_dim, rlen);
    }
}

template<typename T, af_op_t op, bool is_first, uint DIMX>
class ireduceFirstKernelSMEM {
   public:
    ireduceFirstKernelSMEM(write_accessor<T> out, KParam oInfo,
                           write_accessor<uint> oloc, KParam olocInfo,
                           read_accessor<T> in, KParam iInfo,
                           read_accessor<uint> iloc, KParam ilocInfo,
                           uint groups_x, uint groups_y, uint repeat,
                           bool rlenValid, read_accessor<uint> rlen,
                           KParam rlenInfo,
                           sycl::local_accessor<compute_t<T>, 1> s_val,
                           sycl::local_accessor<uint, 1> s_idx)
        : out_(out)
        , oInfo_(oInfo)
        , oloc_(oloc)
        , olocInfo_(olocInfo)
        , in_(in)
        , iInfo_(iInfo)
        , iloc_(iloc)
        , ilocInfo_(ilocInfo)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , repeat_(repeat)
        , rlenValid_(rlenValid)
        , rlen_(rlen)
        , rlenInfo_(rlenInfo)
        , s_val_(s_val)
        , s_idx_(s_idx) {}

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

        const T *iptr = in_.get_pointer() + wid * iInfo_.strides[3] +
                        zid * iInfo_.strides[2] + yid * iInfo_.strides[1] +
                        iInfo_.offset;

        T *optr = out_.get_pointer() + wid * oInfo_.strides[3] +
                  zid * oInfo_.strides[2] + yid * oInfo_.strides[1] +
                  oInfo_.offset;

        const uint *rlenptr =
            (rlenValid_) ? rlen_.get_pointer() + wid * rlenInfo_.strides[3] +
                               zid * rlenInfo_.strides[2] +
                               yid * rlenInfo_.strides[1] + rlenInfo_.offset
                         : nullptr;

        const uint *ilptr;
        if (!is_first) {
            ilptr = iloc_.get_pointer() + wid * iInfo_.strides[3] +
                    zid * iInfo_.strides[2] + yid * iInfo_.strides[1] +
                    iInfo_.offset;
        }
        uint *olptr = oloc_.get_pointer() + wid * oInfo_.strides[3] +
                      zid * oInfo_.strides[2] + yid * oInfo_.strides[1] +
                      oInfo_.offset;

        size_t ylim   = iInfo_.dims[1];
        size_t zlim   = iInfo_.dims[2];
        size_t wlim   = iInfo_.dims[3];
        bool is_valid = (yid < ylim) && (zid < zlim) && (wid < wlim);
        // bool is_valid = (yid < iInfo_.dims[1]) && (zid < iInfo_.dims[2]) &&
        //(wid < iInfo_.dims[3]);

        int minlen = rlenptr ? sycl::min(*rlenptr, (uint)iInfo_.dims[0])
                             : iInfo_.dims[0];
        int lim    = sycl::min((int)(xid + repeat_ * DIMX), minlen);

        compute_t<T> out_val = common::Binary<compute_t<T>, op>::init();
        uint idx             = xid;

        if (xid < lim && is_valid) {
            out_val = static_cast<compute_t<T>>(iptr[xid]);
            if (!is_first) idx = ilptr[xid];
        }

        MinMaxOp<op, compute_t<T>> Op(out_val, idx);
        for (int id = xid; is_valid && id < lim; id += DIMX) {
            Op(static_cast<compute_t<T>>(iptr[id]),
               (!is_first) ? ilptr[id] : id);
        }

        s_val_[lid] = Op.m_val;
        s_idx_[lid] = Op.m_idx;
        it.barrier();

        compute_t<T> *s_vptr = s_val_.get_pointer() + lidy * DIMX;
        uint *s_iptr         = s_idx_.get_pointer() + lidy * DIMX;

        if (DIMX == 256) {
            if (lidx < 128) {
                Op(s_vptr[lidx + 128], s_iptr[lidx + 128]);
                s_vptr[lidx] = Op.m_val;
                s_iptr[lidx] = Op.m_idx;
            }
            it.barrier();
        }

        if (DIMX >= 128) {
            if (lidx < 64) {
                Op(s_vptr[lidx + 64], s_iptr[lidx + 64]);
                s_vptr[lidx] = Op.m_val;
                s_iptr[lidx] = Op.m_idx;
            }
            it.barrier();
        }

        if (DIMX >= 64) {
            if (lidx < 32) {
                Op(s_vptr[lidx + 32], s_iptr[lidx + 32]);
                s_vptr[lidx] = Op.m_val;
                s_iptr[lidx] = Op.m_idx;
            }
            it.barrier();
        }

        // TODO: replace with subgroup operations in optimized kernels
        if (lidx < 16) {
            Op(s_vptr[lidx + 16], s_iptr[lidx + 16]);
            s_vptr[lidx] = Op.m_val;
            s_iptr[lidx] = Op.m_idx;
        }
        it.barrier();

        if (lidx < 8) {
            Op(s_vptr[lidx + 8], s_iptr[lidx + 8]);
            s_vptr[lidx] = Op.m_val;
            s_iptr[lidx] = Op.m_idx;
        }
        it.barrier();

        if (lidx < 4) {
            Op(s_vptr[lidx + 4], s_iptr[lidx + 4]);
            s_vptr[lidx] = Op.m_val;
            s_iptr[lidx] = Op.m_idx;
        }
        it.barrier();

        if (lidx < 2) {
            Op(s_vptr[lidx + 2], s_iptr[lidx + 2]);
            s_vptr[lidx] = Op.m_val;
            s_iptr[lidx] = Op.m_idx;
        }
        it.barrier();

        if (lidx < 1) {
            Op(s_vptr[lidx + 1], s_iptr[lidx + 1]);
            s_vptr[lidx] = Op.m_val;
            s_iptr[lidx] = Op.m_idx;
        }
        it.barrier();

        if (is_valid && lidx == 0) {
            optr[groupId_x]  = data_t<T>(s_vptr[0]);
            olptr[groupId_x] = s_iptr[0];
        }
    }

   protected:
    write_accessor<T> out_;
    KParam oInfo_;
    write_accessor<uint> oloc_;
    KParam olocInfo_;
    read_accessor<T> in_;
    KParam iInfo_;
    read_accessor<uint> iloc_;
    KParam ilocInfo_;
    uint groups_x_, groups_y_, repeat_;
    bool rlenValid_;
    read_accessor<uint> rlen_;
    KParam rlenInfo_;
    sycl::local_accessor<compute_t<T>, 1> s_val_;
    sycl::local_accessor<uint, 1> s_idx_;
};

template<typename T, af_op_t op, bool is_first>
void ireduce_first_launcher(Param<T> out, Param<uint> oloc, Param<T> in,
                            Param<uint> iloc, const uint groups_x,
                            const uint groups_y, const uint threads_x,
                            Param<uint> rlen) {
    sycl::range<2> local(threads_x, creduce::THREADS_PER_BLOCK / threads_x);
    sycl::range<2> global(groups_x * in.info.dims[2] * local[0],
                          groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (groups_x * threads_x));

    auto iempty = memAlloc<uint>(1);
    auto rempty = memAlloc<uint>(1);
    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> out_acc{*out.data, h};
        write_accessor<uint> oloc_acc{*oloc.data, h};
        read_accessor<T> in_acc{*in.data, h};

        read_accessor<uint> iloc_acc{*iempty, h};
        if (iloc.info.dims[0] * iloc.info.dims[1] * iloc.info.dims[2] *
                iloc.info.dims[3] >
            0) {
            iloc_acc = read_accessor<uint>{*iloc.data, h};
        }

        read_accessor<uint> rlen_acc{*rempty, h};
        bool rlenValid = (rlen.info.dims[0] * rlen.info.dims[1] *
                              rlen.info.dims[2] * rlen.info.dims[3] >
                          0);
        if (rlenValid) { rlen_acc = read_accessor<uint>{*rlen.data, h}; }

        auto shrdVal = sycl::local_accessor<compute_t<T>, 1>(
            creduce::THREADS_PER_BLOCK, h);
        auto shrdLoc =
            sycl::local_accessor<uint, 1>(creduce::THREADS_PER_BLOCK, h);

        switch (threads_x) {
            case 32:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    ireduceFirstKernelSMEM<T, op, is_first, 32>(
                        out_acc, out.info, oloc_acc, oloc.info, in_acc, in.info,
                        iloc_acc, iloc.info, groups_x, groups_y, repeat,
                        rlenValid, rlen_acc, rlen.info, shrdVal, shrdLoc));
                break;
            case 64:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    ireduceFirstKernelSMEM<T, op, is_first, 64>(
                        out_acc, out.info, oloc_acc, oloc.info, in_acc, in.info,
                        iloc_acc, iloc.info, groups_x, groups_y, repeat,
                        rlenValid, rlen_acc, rlen.info, shrdVal, shrdLoc));
                break;
            case 128:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    ireduceFirstKernelSMEM<T, op, is_first, 128>(
                        out_acc, out.info, oloc_acc, oloc.info, in_acc, in.info,
                        iloc_acc, iloc.info, groups_x, groups_y, repeat,
                        rlenValid, rlen_acc, rlen.info, shrdVal, shrdLoc));
                break;
            case 256:
                h.parallel_for(
                    sycl::nd_range<2>(global, local),
                    ireduceFirstKernelSMEM<T, op, is_first, 256>(
                        out_acc, out.info, oloc_acc, oloc.info, in_acc, in.info,
                        iloc_acc, iloc.info, groups_x, groups_y, repeat,
                        rlenValid, rlen_acc, rlen.info, shrdVal, shrdLoc));
                break;
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
void ireduce_first(Param<T> out, Param<uint> oloc, Param<T> in,
                   Param<uint> rlen) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, creduce::THREADS_PER_BLOCK);
    uint threads_y = creduce::THREADS_PER_BLOCK / threads_x;

    uint blocks_x = divup(in.info.dims[0], threads_x * creduce::REPEAT);
    uint blocks_y = divup(in.info.dims[1], threads_y);

    Param<T> tmp      = out;
    Param<uint> tlptr = oloc;
    bufptr<T> tmp_alloc;
    bufptr<uint> tlptr_alloc;
    if (blocks_x > 1) {
        auto elements =
            blocks_x * in.info.dims[1] * in.info.dims[2] * in.info.dims[3];
        tmp_alloc   = memAlloc<T>(elements);
        tlptr_alloc = memAlloc<uint>(elements);
        tmp.data    = tmp_alloc.get();
        tlptr.data  = tlptr_alloc.get();

        tmp.info.dims[0] = blocks_x;
        for (int k = 1; k < 4; k++) tmp.info.strides[k] *= blocks_x;
    }

    Param<uint> nullparam;
    ireduce_first_launcher<T, op, true>(tmp, tlptr, in, nullparam, blocks_x,
                                        blocks_y, threads_x, rlen);

    if (blocks_x > 1) {
        ireduce_first_launcher<T, op, false>(out, oloc, tmp, tlptr, 1, blocks_y,
                                             threads_x, rlen);
    }
}

template<typename T, af_op_t op>
void ireduce(Param<T> out, Param<uint> oloc, Param<T> in, int dim,
             Param<uint> rlen) {
    switch (dim) {
        case 0: return ireduce_first<T, op>(out, oloc, in, rlen);
        case 1: return ireduce_dim<T, op, 1>(out, oloc, in, rlen);
        case 2: return ireduce_dim<T, op, 2>(out, oloc, in, rlen);
        case 3: return ireduce_dim<T, op, 3>(out, oloc, in, rlen);
    }
}

template<typename T, af_op_t op>
T ireduce_all(uint *idx, Param<T> in) {
    int in_elements =
        in.info.dims[0] * in.info.dims[1] * in.info.dims[2] * in.info.dims[3];

    bool is_linear = (in.info.strides[0] == 1);
    for (int k = 1; k < 4; k++) {
        is_linear &= (in.info.strides[k] ==
                      (in.info.strides[k - 1] * in.info.dims[k - 1]));
    }

    if (is_linear) {
        in.info.dims[0] = in_elements;
        for (int k = 1; k < 4; k++) {
            in.info.dims[k]    = 1;
            in.info.strides[k] = in_elements;
        }
    }

    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, creduce::THREADS_PER_BLOCK);
    uint threads_y = creduce::THREADS_PER_BLOCK / threads_x;

    // TODO: perf REPEAT, consider removing or runtime eval
    // max problem size < SM resident threads, don't use REPEAT
    uint groups_x = divup(in.info.dims[0], threads_x * creduce::REPEAT);
    uint groups_y = divup(in.info.dims[1], threads_y);

    Array<T> tmp = createEmptyArray<T>(
        {groups_x, in.info.dims[1], in.info.dims[2], in.info.dims[3]});

    int tmp_elements  = tmp.elements();
    Array<uint> tlptr = createEmptyArray<uint>({tmp_elements, 1, 1, 1});

    Param<uint> nullparam;
    Array<uint> rlen = createEmptyArray<uint>(af::dim4(0));
    ireduce_first_launcher<T, op, true>(tmp, tlptr, in, nullparam, groups_x,
                                        groups_y, threads_x, rlen);

    sycl::host_accessor h_ptr_raw{*tmp.get()};
    sycl::host_accessor h_lptr_raw{*tlptr.get()};
    if (!is_linear) {
        // Converting n-d index into a linear index
        // in is of size   [   dims0, dims1, dims2, dims3]
        // tidx is of size [blocks_x, dims1, dims2, dims3]
        // i / blocks_x gives you the batch number "N"
        // "N * dims0 + i" gives the linear index
        for (int i = 0; i < tmp_elements; i++) {
            h_lptr_raw[i] += (i / groups_x) * in.info.dims[0];
        }
    }

    MinMaxOp<op, T> Op(h_ptr_raw[0], h_lptr_raw[0]);

    for (int i = 1; i < tmp_elements; i++) { Op(h_ptr_raw[i], h_lptr_raw[i]); }

    *idx = Op.m_idx;
    return Op.m_val;
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
