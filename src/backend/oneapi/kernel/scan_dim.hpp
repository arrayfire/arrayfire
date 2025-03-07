/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <backend.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/default_config.hpp>
#include <memory.hpp>

#include <sycl/sycl.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename Ti, typename To, af_op_t op, int dim>
class scanDimKernel {
   public:
    scanDimKernel(write_accessor<To> out_acc, KParam oInfo,
                  write_accessor<To> tmp_acc, KParam tInfo,
                  read_accessor<Ti> in_acc, KParam iInfo, const uint groups_x,
                  const uint groups_y, const uint blocks_dim, const uint lim,
                  const bool isFinalPass, const uint DIMY,
                  const bool inclusive_scan, sycl::local_accessor<To, 1> s_val,
                  sycl::local_accessor<To, 1> s_tmp)
        : out_acc_(out_acc)
        , tmp_acc_(tmp_acc)
        , in_acc_(in_acc)
        , oInfo_(oInfo)
        , tInfo_(tInfo)
        , iInfo_(iInfo)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , blocks_dim_(blocks_dim)
        , lim_(lim)
        , DIMY_(DIMY)
        , isFinalPass_(isFinalPass)
        , inclusive_scan_(inclusive_scan)
        , s_val_(s_val)
        , s_tmp_(s_tmp) {}

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

        const Ti *iptr = in_acc_.get_pointer();
        To *optr       = out_acc_.get_pointer();
        To *tptr       = tmp_acc_.get_pointer();

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before
        // offsetting in
        tptr += ids[3] * tInfo_.strides[3] + ids[2] * tInfo_.strides[2] +
                ids[1] * tInfo_.strides[1] + ids[0];

        const int groupIdx_dim = ids[dim];
        ids[dim]               = ids[dim] * g.get_local_range(1) * lim_ + lidy;

        optr += ids[3] * oInfo_.strides[3] + ids[2] * oInfo_.strides[2] +
                ids[1] * oInfo_.strides[1] + ids[0];
        iptr += ids[3] * iInfo_.strides[3] + ids[2] * iInfo_.strides[2] +
                ids[1] * iInfo_.strides[1] + ids[0];
        int id_dim        = ids[dim];
        const int out_dim = oInfo_.dims[dim];

        bool is_valid = (ids[0] < oInfo_.dims[0]) &&
                        (ids[1] < oInfo_.dims[1]) &&
                        (ids[2] < oInfo_.dims[2]) && (ids[3] < oInfo_.dims[3]);

        const int ostride_dim = oInfo_.strides[dim];
        const int istride_dim = iInfo_.strides[dim];

        To *sptr = s_val_.get_pointer() + lid;

        common::Transform<Ti, To, op> transform;
        common::Binary<To, op> binop;

        const To init = common::Binary<To, op>::init();
        To val        = init;

        const bool isLast = (lidy == (DIMY_ - 1));

        for (int k = 0; k < lim_; k++) {
            if (isLast) s_tmp_[lidx] = val;

            bool cond = (is_valid) && (id_dim < out_dim);
            val       = cond ? transform(*iptr) : init;
            *sptr     = val;
            group_barrier(g);

            int start = 0;
#pragma unroll
            for (int off = 1; off < DIMY_; off *= 2) {
                if (lidy >= off)
                    val = binop(val, sptr[(start - off) * (int)THREADS_X]);
                start                   = DIMY_ - start;
                sptr[start * THREADS_X] = val;

                group_barrier(g);
            }

            val = binop(val, s_tmp_[lidx]);
            if (inclusive_scan_) {
                if (cond) { *optr = val; }
            } else if (is_valid) {
                if (id_dim == (out_dim - 1)) {
                    *(optr - (id_dim * ostride_dim)) = init;
                } else if (id_dim < (out_dim - 1)) {
                    *(optr + ostride_dim) = val;
                }
            }
            id_dim += g.get_local_range(1);
            iptr += g.get_local_range(1) * istride_dim;
            optr += g.get_local_range(1) * ostride_dim;
            group_barrier(g);
        }

        if (!isFinalPass_ && is_valid && (groupIdx_dim < tInfo_.dims[dim]) &&
            isLast) {
            *tptr = val;
        }
    }

   protected:
    write_accessor<To> out_acc_;
    write_accessor<To> tmp_acc_;
    read_accessor<Ti> in_acc_;
    KParam oInfo_, tInfo_, iInfo_;
    const uint groups_x_, groups_y_, blocks_dim_, lim_, DIMY_;
    const bool isFinalPass_, inclusive_scan_;
    sycl::local_accessor<To, 1> s_val_;
    sycl::local_accessor<To, 1> s_tmp_;
};

template<typename To, af_op_t op, int dim>
class scanDimBcastKernel {
   public:
    scanDimBcastKernel(write_accessor<To> out_acc, KParam oInfo,
                       read_accessor<To> tmp_acc, KParam tInfo,
                       const uint groups_x, const uint groups_y,
                       const uint groups_dim, const uint lim,
                       const bool inclusive_scan)
        : out_acc_(out_acc)
        , tmp_acc_(tmp_acc)
        , oInfo_(oInfo)
        , tInfo_(tInfo)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , groups_dim_(groups_dim)
        , lim_(lim)
        , inclusive_scan_(inclusive_scan) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g   = it.get_group();
        const uint lidx = it.get_local_id(0);
        const uint lidy = it.get_local_id(1);

        const uint zid       = g.get_group_id(0) / groups_x_;
        const uint wid       = g.get_group_id(1) / groups_y_;
        const uint groupId_x = g.get_group_id(0) - (groups_x_)*zid;
        const uint groupId_y = g.get_group_id(1) - (groups_y_)*wid;
        const uint xid       = groupId_x * g.get_local_range(0) + lidx;
        const uint yid       = groupId_y;

        uint ids[4] = {xid, yid, zid, wid};

        const To *tptr = tmp_acc_.get_pointer();
        To *optr       = out_acc_.get_pointer();

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before
        // offsetting in
        tptr += ids[3] * tInfo_.strides[3] + ids[2] * tInfo_.strides[2] +
                ids[1] * tInfo_.strides[1] + ids[0];

        const int groupIdx_dim = ids[dim];
        ids[dim]               = ids[dim] * g.get_local_range(1) * lim_ + lidy;

        optr += ids[3] * oInfo_.strides[3] + ids[2] * oInfo_.strides[2] +
                ids[1] * oInfo_.strides[1] + ids[0];
        const int id_dim  = ids[dim];
        const int out_dim = oInfo_.dims[dim];

        // Shift broadcast one step to the right for exclusive scan (#2366)
        int offset = inclusive_scan_ ? 0 : oInfo_.strides[dim];
        optr += offset;

        bool is_valid = (ids[0] < oInfo_.dims[0]) &&
                        (ids[1] < oInfo_.dims[1]) &&
                        (ids[2] < oInfo_.dims[2]) && (ids[3] < oInfo_.dims[3]);

        if (!is_valid) return;
        if (groupIdx_dim == 0) return;

        To accum = *(tptr - tInfo_.strides[dim]);

        common::Binary<To, op> binop;
        const int ostride_dim = oInfo_.strides[dim];

        for (int k = 0, id = id_dim; is_valid && k < lim_ && (id < out_dim);
             k++, id += g.get_local_range(1)) {
            *optr = binop(*optr, accum);
            optr += g.get_local_range(1) * ostride_dim;
        }
    }

   protected:
    write_accessor<To> out_acc_;
    read_accessor<To> tmp_acc_;
    KParam oInfo_, tInfo_;
    const uint groups_x_, groups_y_, groups_dim_, lim_;
    const bool inclusive_scan_;
};

template<typename Ti, typename To, af_op_t op, int dim>
static void scan_dim_launcher(Param<To> out, Param<To> tmp, Param<Ti> in,
                              const uint threads_y, const dim_t blocks_all[4],
                              bool isFinalPass, bool inclusive_scan) {
    sycl::range<2> local(THREADS_X, threads_y);
    sycl::range<2> global(blocks_all[0] * blocks_all[2] * local[0],
                          blocks_all[1] * blocks_all[3] * local[1]);

    uint lim = divup(out.info.dims[dim], (threads_y * blocks_all[dim]));

    getQueue().submit([&](sycl::handler &h) {
        // TODO: specify access modes in all kernels
        write_accessor<To> out_acc{*out.data, h};
        write_accessor<To> tmp_acc{*tmp.data, h};
        read_accessor<Ti> in_acc{*in.data, h};

        auto s_val = sycl::local_accessor<compute_t<To>, 1>(
            THREADS_X * threads_y * 2, h);
        auto s_tmp = sycl::local_accessor<compute_t<To>, 1>(THREADS_X, h);

        h.parallel_for(
            sycl::nd_range<2>(global, local),
            scanDimKernel<Ti, To, op, dim>(
                out_acc, out.info, tmp_acc, tmp.info, in_acc, in.info,
                blocks_all[0], blocks_all[1], blocks_all[dim], lim, isFinalPass,
                threads_y, inclusive_scan, s_val, s_tmp));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename To, af_op_t op, int dim>
static void bcast_dim_launcher(Param<To> out, Param<To> tmp,
                               const uint threads_y, const dim_t blocks_all[4],
                               bool inclusive_scan) {
    sycl::range<2> local(THREADS_X, threads_y);
    sycl::range<2> global(blocks_all[0] * blocks_all[2] * local[0],
                          blocks_all[1] * blocks_all[3] * local[1]);

    uint lim = divup(out.info.dims[dim], (threads_y * blocks_all[dim]));

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<To> out_acc{*out.data, h};
        read_accessor<To> tmp_acc{*tmp.data, h};

        h.parallel_for(
            sycl::nd_range<2>(global, local),
            scanDimBcastKernel<To, op, dim>(
                out_acc, out.info, tmp_acc, tmp.info, blocks_all[0],
                blocks_all[1], blocks_all[dim], lim, inclusive_scan));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op, int dim>
static void scan_dim(Param<To> out, Param<Ti> in, bool inclusive_scan) {
    uint threads_y = std::min(THREADS_Y, nextpow2(out.info.dims[dim]));
    uint threads_x = THREADS_X;

    dim_t blocks_all[] = {divup(out.info.dims[0], threads_x), out.info.dims[1],
                          out.info.dims[2], out.info.dims[3]};

    blocks_all[dim] = divup(out.info.dims[dim], threads_y * REPEAT);

    if (blocks_all[dim] == 1) {
        scan_dim_launcher<Ti, To, op, dim>(out, out, in, threads_y, blocks_all,
                                           true, inclusive_scan);
    } else {
        Param<To> tmp = out;

        tmp.info.dims[dim]  = blocks_all[dim];
        tmp.info.strides[0] = 1;
        for (int k = 1; k < 4; k++)
            tmp.info.strides[k] =
                tmp.info.strides[k - 1] * tmp.info.dims[k - 1];

        int tmp_elements = tmp.info.strides[3] * tmp.info.dims[3];
        auto tmp_alloc   = memAlloc<To>(tmp_elements);
        tmp.data         = tmp_alloc.get();

        scan_dim_launcher<Ti, To, op, dim>(out, tmp, in, threads_y, blocks_all,
                                           false, inclusive_scan);

        int bdim        = blocks_all[dim];
        blocks_all[dim] = 1;

        // FIXME: Is there an alternative to the if condition ?
        if (op == af_notzero_t) {
            scan_dim_launcher<To, To, af_add_t, dim>(tmp, tmp, tmp, threads_y,
                                                     blocks_all, true, true);
        } else {
            scan_dim_launcher<To, To, op, dim>(tmp, tmp, tmp, threads_y,
                                               blocks_all, true, true);
        }

        blocks_all[dim] = bdim;
        bcast_dim_launcher<To, op, dim>(out, tmp, threads_y, blocks_all,
                                        inclusive_scan);
    }
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
