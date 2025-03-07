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
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/default_config.hpp>

#include <sycl/sycl.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

#define sidx(y, x) scratch_[((y + 1) * (TX + 2)) + (x + 1)]

template<typename T, int TX, int TY>
class gradientCreateKernel {
   public:
    gradientCreateKernel(write_accessor<T> d_grad0, const KParam grad0,
                         write_accessor<T> d_grad1, const KParam grad1,
                         read_accessor<T> d_in, const KParam in,
                         const int blocksPerMatX, const int blocksPerMatY,
                         sycl::local_accessor<T> scratch)
        : d_grad0_(d_grad0)
        , grad0_(grad0)
        , d_grad1_(d_grad1)
        , grad1_(grad1)
        , d_in_(d_in)
        , in_(in)
        , blocksPerMatX_(blocksPerMatX)
        , blocksPerMatY_(blocksPerMatY)
        , scratch_(scratch) {}
    void operator()(sycl::nd_item<2> it) const {
        auto g = it.get_group();

        const int idz = g.get_group_id(0) / blocksPerMatX_;
        const int idw = g.get_group_id(1) / blocksPerMatY_;

        const int blockIdx_x = g.get_group_id(0) - idz * blocksPerMatX_;
        const int blockIdx_y = g.get_group_id(1) - idw * blocksPerMatY_;

        const int xB = blockIdx_x * g.get_local_range(0);
        const int yB = blockIdx_y * g.get_local_range(1);

        const int tx = it.get_local_id(0);
        const int ty = it.get_local_id(1);

        const int idx = tx + xB;
        const int idy = ty + yB;

        const bool cond = (idx >= in_.dims[0] || idy >= in_.dims[1] ||
                           idz >= in_.dims[2] || idw >= in_.dims[3]);

        int xmax = (TX > (in_.dims[0] - xB)) ? (in_.dims[0] - xB) : TX;
        int ymax = (TY > (in_.dims[1] - yB)) ? (in_.dims[1] - yB) : TY;

        int iIdx = in_.offset + idw * in_.strides[3] + idz * in_.strides[2] +
                   idy * in_.strides[1] + idx;

        int g0dx = idw * grad0_.strides[3] + idz * grad0_.strides[2] +
                   idy * grad0_.strides[1] + idx;

        int g1dx = idw * grad1_.strides[3] + idz * grad1_.strides[2] +
                   idy * grad1_.strides[1] + idx;

        // Multipliers - 0.5 for interior, 1 for edge cases
        typename std::conditional<std::is_same<T, std::complex<double>>::value,
                                  double, float>::type
            xf = 0.5 * (1 + (idx == 0 || idx >= (in_.dims[0] - 1))),
            yf = 0.5 * (1 + (idy == 0 || idy >= (in_.dims[1] - 1)));

        // Copy data to scratch space
        T zero = (T)(0);
        if (cond) {
            sidx(ty, tx) = zero;
        } else {
            sidx(ty, tx) = d_in_[iIdx];
        }

        it.barrier();

        // Copy buffer zone data. Corner (0,0) etc, are not used.
        // Cols
        if (ty == 0) {
            // Y-1
            sidx(-1, tx) =
                (cond || idy == 0) ? sidx(0, tx) : d_in_[iIdx - in_.strides[1]];
            sidx(ymax, tx) = (cond || (idy + ymax) >= in_.dims[1])
                                 ? sidx(ymax - 1, tx)
                                 : d_in_[iIdx + ymax * in_.strides[1]];
        }
        // Rows
        if (tx == 0) {
            sidx(ty, -1)   = (cond || idx == 0) ? sidx(ty, 0) : d_in_[iIdx - 1];
            sidx(ty, xmax) = (cond || (idx + xmax) >= in_.dims[0])
                                 ? sidx(ty, xmax - 1)
                                 : d_in_[iIdx + xmax];
        }

        it.barrier();

        if (cond) return;

        d_grad0_[g0dx] = xf * (sidx(ty, tx + 1) - sidx(ty, tx - 1));
        d_grad1_[g1dx] = yf * (sidx(ty + 1, tx) - sidx(ty - 1, tx));
    }

   private:
    write_accessor<T> d_grad0_;
    const KParam grad0_;
    write_accessor<T> d_grad1_;
    const KParam grad1_;
    read_accessor<T> d_in_;
    const KParam in_;
    const int blocksPerMatX_;
    const int blocksPerMatY_;
    sycl::local_accessor<T> scratch_;
};

template<typename T>
void gradient(Param<T> grad0, Param<T> grad1, const Param<T> in) {
    constexpr int TX = 32;
    constexpr int TY = 8;

    auto local = sycl::range{TX, TY};

    int blocksPerMatX = divup(in.info.dims[0], TX);
    int blocksPerMatY = divup(in.info.dims[1], TY);
    auto global       = sycl::range{local[0] * blocksPerMatX * in.info.dims[2],
                              local[1] * blocksPerMatY * in.info.dims[3]};

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> grad0Acc{*grad0.data, h};
        write_accessor<T> grad1Acc{*grad1.data, h};
        read_accessor<T> inAcc{*in.data, h};
        auto scratch = sycl::local_accessor<T>((TY + 2) * (TX + 2), h);
        h.parallel_for(sycl::nd_range{global, local},
                       gradientCreateKernel<T, TX, TY>(
                           grad0Acc, grad0.info, grad1Acc, grad1.info, inAcc,
                           in.info, blocksPerMatX, blocksPerMatY, scratch));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
