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
#include <kernel/accessors.hpp>
#include <math.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

constexpr uint DIMX  = 32;
constexpr uint DIMY  = 8;
constexpr int REPEAT = 64;

int getOffset(const dim_t *dims, const dim_t *strides, const dim_t *refdims,
              int ids[4]) {
    int off = 0;
    off += ids[3] * (dims[3] == refdims[3]) * strides[3];
    off += ids[2] * (dims[2] == refdims[2]) * strides[2];
    off += ids[1] * (dims[1] == refdims[1]) * strides[1];
    return off;
}

template<typename T>
class selectKernelCreateKernel {
   public:
    selectKernelCreateKernel(write_accessor<T> optr, KParam oinfo,
                             read_accessor<char> cptr_, KParam cinfo,
                             read_accessor<T> aptr_, KParam ainfo,
                             read_accessor<T> bptr_, KParam binfo, int groups_0,
                             int groups_1, const bool is_same)
        : optr_(optr)
        , oinfo_(oinfo)
        , cptr__(cptr_)
        , cinfo_(cinfo)
        , aptr__(aptr_)
        , ainfo_(ainfo)
        , bptr__(bptr_)
        , binfo_(binfo)
        , groups_0_(groups_0)
        , groups_1_(groups_1)
        , is_same_(is_same) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        const char *cptr = cptr__.get_pointer() + cinfo_.offset;
        const T *aptr    = aptr__.get_pointer() + ainfo_.offset;
        const T *bptr    = bptr__.get_pointer() + binfo_.offset;

        const int idz = g.get_group_id(0) / groups_0_;
        const int idw = g.get_group_id(1) / groups_1_;

        const int group_id_0 = g.get_group_id(0) - idz * groups_0_;
        const int group_id_1 = g.get_group_id(1) - idw * groups_1_;

        const int idx0 = group_id_0 * g.get_local_range(0) + it.get_local_id(0);
        const int idy  = group_id_1 * g.get_local_range(1) + it.get_local_id(1);

        const int off = idw * oinfo_.strides[3] + idz * oinfo_.strides[2] +
                        idy * oinfo_.strides[1];

        const bool valid = (idw < oinfo_.dims[3] && idz < oinfo_.dims[2] &&
                            idy < oinfo_.dims[1]);

        int ids[] = {idx0, idy, idz, idw};

        T *optr_pointer = optr_.get_pointer();
        optr_pointer += off;
        aptr += getOffset(ainfo_.dims, ainfo_.strides, oinfo_.dims, ids);
        bptr += getOffset(binfo_.dims, binfo_.strides, oinfo_.dims, ids);
        cptr += getOffset(cinfo_.dims, cinfo_.strides, oinfo_.dims, ids);

        if (is_same_) {
            for (int idx = idx0; idx < oinfo_.dims[0];
                 idx += g.get_local_range(0) * groups_0_) {
                if (valid)
                    optr_pointer[idx] = (cptr[idx]) ? aptr[idx] : bptr[idx];
            }
        } else {
            bool csame = cinfo_.dims[0] == oinfo_.dims[0];
            bool asame = ainfo_.dims[0] == oinfo_.dims[0];
            bool bsame = binfo_.dims[0] == oinfo_.dims[0];
            for (int idx = idx0; idx < oinfo_.dims[0];
                 idx += g.get_local_range(0) * groups_0_) {
                if (valid)
                    optr_pointer[idx] = (cptr[csame * idx]) ? aptr[asame * idx]
                                                            : bptr[bsame * idx];
            }
        }
    }

   private:
    write_accessor<T> optr_;
    KParam oinfo_;
    read_accessor<char> cptr__;
    KParam cinfo_;
    read_accessor<T> aptr__;
    KParam ainfo_;
    read_accessor<T> bptr__;
    KParam binfo_;
    int groups_0_;
    int groups_1_;
    const bool is_same_;
};

template<typename T>
void selectLauncher(Param<T> out, Param<char> cond, Param<T> a, Param<T> b,
                    const int ndims, const bool is_same) {
    int threads[] = {DIMX, DIMY};

    if (ndims == 1) {
        threads[0] *= threads[1];
        threads[1] = 1;
    }

    auto local = sycl::range(threads[0], threads[1]);

    int groups_0 = divup(out.info.dims[0], REPEAT * local[0]);
    int groups_1 = divup(out.info.dims[1], local[1]);

    auto global = sycl::range(groups_0 * out.info.dims[2] * local[0],
                              groups_1 * out.info.dims[3] * local[1]);

    getQueue().submit([&](auto &h) {
        write_accessor<T> d_out{*out.data, h};
        read_accessor<char> d_cond{*cond.data, h};
        read_accessor<T> d_a{*a.data, h};
        read_accessor<T> d_b{*b.data, h};
        h.parallel_for(sycl::nd_range{global, local},
                       selectKernelCreateKernel<T>(
                           d_out, out.info, d_cond, cond.info, d_a, a.info, d_b,
                           b.info, groups_0, groups_1, is_same));
    });
}

template<typename T>
class selectScalarCreateKernel {
   public:
    selectScalarCreateKernel(write_accessor<T> optr, KParam oinfo,
                             read_accessor<char> cptr_, KParam cinfo,
                             read_accessor<T> aptr_, KParam ainfo, T b,
                             int groups_0, int groups_1, const bool flip)
        : optr_(optr)
        , oinfo_(oinfo)
        , cptr__(cptr_)
        , cinfo_(cinfo)
        , aptr__(aptr_)
        , ainfo_(ainfo)
        , b_(b)
        , groups_0_(groups_0)
        , groups_1_(groups_1)
        , flip_(flip) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        const char *cptr = cptr__.get_pointer() + cinfo_.offset;
        const T *aptr    = aptr__.get_pointer() + ainfo_.offset;

        const int idz = g.get_group_id(0) / groups_0_;
        const int idw = g.get_group_id(1) / groups_1_;

        const int group_id_0 = g.get_group_id(0) - idz * groups_0_;
        const int group_id_1 = g.get_group_id(1) - idw * groups_1_;

        const int idx0 = group_id_0 * g.get_local_range(0) + it.get_local_id(0);
        const int idy  = group_id_1 * g.get_local_range(1) + it.get_local_id(1);

        const int off = idw * oinfo_.strides[3] + idz * oinfo_.strides[2] +
                        idy * oinfo_.strides[1];

        int ids[] = {idx0, idy, idz, idw};
        T *optr   = optr_.get_pointer();
        optr += off;
        aptr += getOffset(ainfo_.dims, ainfo_.strides, oinfo_.dims, ids);
        cptr += getOffset(cinfo_.dims, cinfo_.strides, oinfo_.dims, ids);

        if (idw >= oinfo_.dims[3] || idz >= oinfo_.dims[2] ||
            idy >= oinfo_.dims[1]) {
            return;
        }

        for (int idx = idx0; idx < oinfo_.dims[0];
             idx += g.get_local_range(0) * groups_0_) {
            optr[idx] = (cptr[idx] ^ flip_) ? aptr[idx] : b_;
        }
    }

   private:
    write_accessor<T> optr_;
    KParam oinfo_;
    read_accessor<char> cptr__;
    KParam cinfo_;
    read_accessor<T> aptr__;
    KParam ainfo_;
    T b_;
    int groups_0_;
    int groups_1_;
    const bool flip_;
};

template<typename T>
void select(Param<T> out, Param<char> cond, Param<T> a, Param<T> b, int ndims) {
    bool is_same = true;
    for (int i = 0; i < 4; i++) {
        is_same &= (a.info.dims[i] == b.info.dims[i]);
    }
    selectLauncher<T>(out, cond, a, b, ndims, is_same);
}

template<typename T>
void select_scalar(Param<T> out, Param<char> cond, Param<T> a, const T b,
                   const int ndims, const bool flip) {
    int threads[] = {DIMX, DIMY};

    if (ndims == 1) {
        threads[0] *= threads[1];
        threads[1] = 1;
    }

    auto local = sycl::range(threads[0], threads[1]);

    int groups_0 = divup(out.info.dims[0], REPEAT * local[0]);
    int groups_1 = divup(out.info.dims[1], local[1]);

    auto global = sycl::range(groups_0 * out.info.dims[2] * local[0],
                              groups_1 * out.info.dims[3] * local[1]);

    getQueue().submit([&](auto &h) {
        write_accessor<T> d_out{*out.data, h};
        read_accessor<char> d_cond{*cond.data, h};
        read_accessor<T> d_a{*a.data, h};
        h.parallel_for(
            sycl::nd_range{global, local},
            selectScalarCreateKernel<T>(d_out, out.info, d_cond, cond.info, d_a,
                                        a.info, b, groups_0, groups_1, flip));
    });
}
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
