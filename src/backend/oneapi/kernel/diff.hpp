/*******************************************************
 * Copyright (c) 2022 ArrayFire
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
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <traits.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
class diffKernel {
   public:
    diffKernel(write_accessor<T> outAcc, const read_accessor<T> inAcc,
               const KParam op, const KParam ip, const int oElem,
               const int blocksPerMatX, const int blocksPerMatY,
               const bool isDiff2, const unsigned DIM)
        : outAcc_(outAcc)
        , inAcc_(inAcc)
        , op_(op)
        , ip_(ip)
        , oElem_(oElem)
        , blocksPerMatX_(blocksPerMatX)
        , blocksPerMatY_(blocksPerMatY)
        , isDiff2_(isDiff2)
        , DIM_(DIM) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        const int idz = g.get_group_id(0) / blocksPerMatX_;
        const int idw = g.get_group_id(1) / blocksPerMatY_;

        const int blockIdx_x = g.get_group_id(0) - idz * blocksPerMatX_;
        const int blockIdx_y = g.get_group_id(1) - idw * blocksPerMatY_;

        const int idx = it.get_local_id(0) + blockIdx_x * g.get_local_range(0);
        const int idy = it.get_local_id(1) + blockIdx_y * g.get_local_range(1);

        if (idx >= op_.dims[0] || idy >= op_.dims[1] || idz >= op_.dims[2] ||
            idw >= op_.dims[3])
            return;

        int iMem0 = idw * ip_.strides[3] + idz * ip_.strides[2] +
                    idy * ip_.strides[1] + idx;
        int iMem1 = iMem0 + ip_.strides[DIM_];
        int iMem2 = iMem1 + ip_.strides[DIM_];

        int oMem = idw * op_.strides[3] + idz * op_.strides[2] +
                   idy * op_.strides[1] + idx;

        iMem2 *= isDiff2_;

        T *out      = outAcc_.get_pointer();
        const T *in = inAcc_.get_pointer() + ip_.offset;
        if (isDiff2_ == 0) {
            out[oMem] = in[iMem1] - in[iMem0];
        } else {
            out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
        }

        // diff_this(out, in + ip.offset, oMem, iMem0, iMem1, iMem2);
    }

   private:
    write_accessor<T> outAcc_;
    const read_accessor<T> inAcc_;
    const KParam op_;
    const KParam ip_;
    const int oElem_;
    const int blocksPerMatX_;
    const int blocksPerMatY_;
    const bool isDiff2_;
    const unsigned DIM_;
};

template<typename T>
void diff(Param<T> out, const Param<T> in, const unsigned indims,
          const unsigned dim, const bool isDiff2) {
    constexpr int TX = 16;
    constexpr int TY = 16;

    auto local = sycl::range{TX, TY};
    if (dim == 0 && indims == 1) { local = sycl::range{TX * TY, 1}; }

    int blocksPerMatX = divup(out.info.dims[0], local[0]);
    int blocksPerMatY = divup(out.info.dims[1], local[1]);
    auto global       = sycl::range{local[0] * blocksPerMatX * out.info.dims[2],
                              local[1] * blocksPerMatY * out.info.dims[3]};

    const int oElem = out.info.dims[0] * out.info.dims[1] * out.info.dims[2] *
                      out.info.dims[3];

    getQueue().submit([&](sycl::handler &h) {
        read_accessor<T> inAcc   = {*in.data, h};
        write_accessor<T> outAcc = {*out.data, h};

        h.parallel_for(
            sycl::nd_range{global, local},
            diffKernel<T>(outAcc, inAcc, out.info, in.info, oElem,
                          blocksPerMatX, blocksPerMatY, isDiff2, dim));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
