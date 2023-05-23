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

template<typename Ti, typename To, af_op_t op>
class scanFirstKernel {
   public:
    scanFirstKernel(write_accessor<To> out_acc, KParam oInfo,
                    write_accessor<To> tmp_acc, KParam tInfo,
                    read_accessor<Ti> in_acc, KParam iInfo, const uint groups_x,
                    const uint groups_y, const uint lim, const bool isFinalPass,
                    const uint DIMX, const bool inclusive_scan,
                    sycl::local_accessor<To, 1> s_val,
                    sycl::local_accessor<To, 1> s_tmp)
        : out_acc_(out_acc)
        , tmp_acc_(tmp_acc)
        , in_acc_(in_acc)
        , oInfo_(oInfo)
        , tInfo_(tInfo)
        , iInfo_(iInfo)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , lim_(lim)
        , DIMX_(DIMX)
        , isFinalPass_(isFinalPass)
        , inclusive_scan_(inclusive_scan)
        , s_val_(s_val)
        , s_tmp_(s_tmp) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g   = it.get_group();
        const uint lidx = it.get_local_id(0);
        const uint lidy = it.get_local_id(1);

        const uint zid       = g.get_group_id(0) / groups_x_;
        const uint wid       = g.get_group_id(1) / groups_y_;
        const uint groupId_x = g.get_group_id(0) - (groups_x_)*zid;
        const uint groupId_y = g.get_group_id(1) - (groups_y_)*wid;
        const uint xid       = groupId_x * g.get_local_range(0) * lim_ + lidx;
        const uint yid       = groupId_y * g.get_local_range(1) + lidy;

        bool cond_yzw = (yid < oInfo_.dims[1]) && (zid < oInfo_.dims[2]) &&
                        (wid < oInfo_.dims[3]);

        // if (!cond_yzw) return;  // retire warps early TODO: move

        const Ti *iptr = in_acc_.get_pointer();
        To *optr       = out_acc_.get_pointer();
        To *tptr       = tmp_acc_.get_pointer();

        iptr += wid * iInfo_.strides[3] + zid * iInfo_.strides[2] +
                yid * iInfo_.strides[1];
        optr += wid * oInfo_.strides[3] + zid * oInfo_.strides[2] +
                yid * oInfo_.strides[1];
        tptr += wid * tInfo_.strides[3] + zid * tInfo_.strides[2] +
                yid * tInfo_.strides[1];

        To *sptr = s_val_.get_pointer() + lidy * (2 * DIMX_ + 1);

        common::Transform<Ti, To, op> transform;
        common::Binary<To, op> binop;

        const To init = common::Binary<To, op>::init();
        int id        = xid;
        To val        = init;

        const bool isLast = (lidx == (DIMX_ - 1));
        for (int k = 0; k < lim_; k++) {
            if (isLast) s_tmp_[lidy] = val;

            bool cond  = (id < oInfo_.dims[0]) && cond_yzw;
            val        = cond ? transform(iptr[id]) : init;
            sptr[lidx] = val;
            group_barrier(g);

            int start = 0;
            for (int off = 1; off < DIMX_; off *= 2) {
                if (lidx >= off) val = binop(val, sptr[(start - off) + lidx]);
                start              = DIMX_ - start;
                sptr[start + lidx] = val;

                group_barrier(g);
            }

            val = binop(val, s_tmp_[lidy]);

            if (inclusive_scan_) {
                if (cond) { optr[id] = val; }
            } else {
                if (cond_yzw && id == (oInfo_.dims[0] - 1)) {
                    optr[0] = init;
                } else if (cond_yzw && id < (oInfo_.dims[0] - 1)) {
                    optr[id + 1] = val;
                }
            }
            id += g.get_local_range(0);
            group_barrier(g);
        }

        if (!isFinalPass_ && isLast && cond_yzw) { tptr[groupId_x] = val; }
    }

   protected:
    write_accessor<To> out_acc_;
    write_accessor<To> tmp_acc_;
    read_accessor<Ti> in_acc_;
    KParam oInfo_, tInfo_, iInfo_;
    const uint groups_x_, groups_y_, lim_, DIMX_;
    const bool isFinalPass_, inclusive_scan_;
    sycl::local_accessor<To, 1> s_val_;
    sycl::local_accessor<To, 1> s_tmp_;
};

template<typename To, af_op_t op>
class scanFirstBcastKernel {
   public:
    scanFirstBcastKernel(write_accessor<To> out_acc, KParam oInfo,
                         read_accessor<To> tmp_acc, KParam tInfo,
                         const uint groups_x, const uint groups_y,
                         const uint lim, const bool inclusive_scan)
        : out_acc_(out_acc)
        , tmp_acc_(tmp_acc)
        , oInfo_(oInfo)
        , tInfo_(tInfo)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
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
        const uint xid       = groupId_x * g.get_local_range(0) * lim_ + lidx;
        const uint yid       = groupId_y * g.get_local_range(1) + lidy;

        if (groupId_x == 0) return;

        bool cond = (yid < oInfo_.dims[1]) && (zid < oInfo_.dims[2]) &&
                    (wid < oInfo_.dims[3]);
        if (!cond) return;

        To *optr       = out_acc_.get_pointer();
        const To *tptr = tmp_acc_.get_pointer();

        optr += wid * oInfo_.strides[3] + zid * oInfo_.strides[2] +
                yid * oInfo_.strides[1];
        tptr += wid * tInfo_.strides[3] + zid * tInfo_.strides[2] +
                yid * tInfo_.strides[1];

        common::Binary<To, op> binop;
        To accum = tptr[groupId_x - 1];

        // Shift broadcast one step to the right for exclusive scan (#2366)
        int offset = !inclusive_scan_;
        for (int k = 0, id = xid + offset; k < lim_ && id < oInfo_.dims[0];
             k++, id += g.get_local_range(0)) {
            optr[id] = binop(accum, optr[id]);
        }
    }

   protected:
    write_accessor<To> out_acc_;
    read_accessor<To> tmp_acc_;
    KParam oInfo_, tInfo_;
    const uint groups_x_, groups_y_, lim_;
    const bool inclusive_scan_;
};

template<typename Ti, typename To, af_op_t op>
static void scan_first_launcher(Param<To> out, Param<To> tmp, Param<Ti> in,
                                const uint groups_x, const uint groups_y,
                                const uint threads_x, bool isFinalPass,
                                bool inclusive_scan) {
    sycl::range<2> local(threads_x, THREADS_PER_BLOCK / threads_x);
    sycl::range<2> global(groups_x * out.info.dims[2] * local[0],
                          groups_y * out.info.dims[3] * local[1]);
    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<To> out_acc{*out.data, h};
        write_accessor<To> tmp_acc{*tmp.data, h};
        read_accessor<Ti> in_acc{*in.data, h};

        const int DIMY            = THREADS_PER_BLOCK / threads_x;
        const int SHARED_MEM_SIZE = (2 * threads_x + 1) * (DIMY);
        auto s_val = sycl::local_accessor<compute_t<To>, 1>(SHARED_MEM_SIZE, h);
        auto s_tmp = sycl::local_accessor<compute_t<To>, 1>(DIMY, h);

        // TODO threads_x as template arg for #pragma unroll?
        h.parallel_for(sycl::nd_range<2>(global, local),
                       scanFirstKernel<Ti, To, op>(
                           out_acc, out.info, tmp_acc, tmp.info, in_acc,
                           in.info, groups_x, groups_y, lim, isFinalPass,
                           threads_x, inclusive_scan, s_val, s_tmp));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename To, af_op_t op>
static void bcast_first_launcher(Param<To> out, Param<To> tmp,
                                 const uint groups_x, const uint groups_y,
                                 const uint threads_x, bool inclusive_scan) {
    sycl::range<2> local(threads_x, THREADS_PER_BLOCK / threads_x);
    sycl::range<2> global(groups_x * out.info.dims[2] * local[0],
                          groups_y * out.info.dims[3] * local[1]);
    uint lim = divup(out.info.dims[0], (threads_x * groups_x));

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<To> out_acc{*out.data, h};
        read_accessor<To> tmp_acc{*tmp.data, h};

        h.parallel_for(sycl::nd_range<2>(global, local),
                       scanFirstBcastKernel<To, op>(
                           out_acc, out.info, tmp_acc, tmp.info, groups_x,
                           groups_y, lim, inclusive_scan));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op>
static void scan_first(Param<To> out, Param<Ti> in, bool inclusive_scan) {
    uint threads_x = nextpow2(std::max(32u, (uint)out.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_BLOCK);
    uint threads_y = THREADS_PER_BLOCK / threads_x;

    uint groups_x = divup(out.info.dims[0], threads_x * REPEAT);
    uint groups_y = divup(out.info.dims[1], threads_y);

    if (groups_x == 1) {
        scan_first_launcher<Ti, To, op>(out, out, in, groups_x, groups_y,
                                        threads_x, true, inclusive_scan);
    } else {
        Param<To> tmp = out;

        tmp.info.dims[0]    = groups_x;
        tmp.info.strides[0] = 1;
        for (int k = 1; k < 4; k++)
            tmp.info.strides[k] =
                tmp.info.strides[k - 1] * tmp.info.dims[k - 1];

        int tmp_elements = tmp.info.strides[3] * tmp.info.dims[3];
        auto tmp_alloc   = memAlloc<To>(tmp_elements);
        tmp.data         = tmp_alloc.get();

        scan_first_launcher<Ti, To, op>(out, tmp, in, groups_x, groups_y,
                                        threads_x, false, inclusive_scan);

        // FIXME: Is there an alternative to the if condition ?
        if (op == af_notzero_t) {
            scan_first_launcher<To, To, af_add_t>(tmp, tmp, tmp, 1, groups_y,
                                                  threads_x, true, true);
        } else {
            scan_first_launcher<To, To, op>(tmp, tmp, tmp, 1, groups_y,
                                            threads_x, true, true);
        }

        bcast_first_launcher<To, op>(out, tmp, groups_x, groups_y, threads_x,
                                     inclusive_scan);
    }
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
