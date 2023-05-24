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
#include <math.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
class wrapCreateKernel {
   public:
    wrapCreateKernel(write_accessor<T> optrAcc, KParam out,
                     read_accessor<T> iptrAcc, KParam in, const int wx,
                     const int wy, const int sx, const int sy, const int px,
                     const int py, const int nx, const int ny, int groups_x,
                     int groups_y, const bool is_column)
        : optrAcc_(optrAcc)
        , out_(out)
        , iptrAcc_(iptrAcc)
        , in_(in)
        , wx_(wx)
        , wy_(wy)
        , sx_(sx)
        , sy_(sy)
        , px_(px)
        , py_(py)
        , nx_(nx)
        , ny_(ny)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , is_column_(is_column) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        int idx2 = g.get_group_id(0) / groups_x_;
        int idx3 = g.get_group_id(1) / groups_y_;

        int groupId_x = g.get_group_id(0) - idx2 * groups_x_;
        int groupId_y = g.get_group_id(1) - idx3 * groups_y_;

        int oidx0 = it.get_local_id(0) + g.get_local_range(0) * groupId_x;
        int oidx1 = it.get_local_id(1) + g.get_local_range(1) * groupId_y;

        T *optr = optrAcc_.get_pointer() + idx2 * out_.strides[2] +
                  idx3 * out_.strides[3] + out_.offset;
        const T *iptr = iptrAcc_.get_pointer() + idx2 * in_.strides[2] +
                        idx3 * in_.strides[3] + in_.offset;

        if (oidx0 >= out_.dims[0] || oidx1 >= out_.dims[1]) return;

        int pidx0 = oidx0 + px_;
        int pidx1 = oidx1 + py_;

        // The last time a value appears in_ the unwrapped index is padded_index
        // / stride Each previous index has the value appear "stride" locations
        // earlier We work our way back from the last index

        const int x_end = sycl::min(pidx0 / sx_, nx_ - 1);
        const int y_end = sycl::min(pidx1 / sy_, ny_ - 1);

        const int x_off = pidx0 - sx_ * x_end;
        const int y_off = pidx1 - sy_ * y_end;

        T val   = (T)0;
        int idx = 1;

        for (int y = y_end, yo = y_off; y >= 0 && yo < wy_; yo += sy_, y--) {
            int win_end_y = yo * wx_;
            int dim_end_y = y * nx_;

            for (int x = x_end, xo = x_off; x >= 0 && xo < wx_;
                 xo += sx_, x--) {
                int win_end = win_end_y + xo;
                int dim_end = dim_end_y + x;

                if (is_column_) {
                    idx = dim_end * in_.strides[1] + win_end;
                } else {
                    idx = dim_end + win_end * in_.strides[1];
                }

                // No need to include anything special for complex
                // Add for complex numbers is just vector add of reals
                // Might need to change if we generalize add to more binary ops
                val = val + iptr[idx];
            }
        }

        optr[oidx1 * out_.strides[1] + oidx0] = val;
    }

   private:
    write_accessor<T> optrAcc_;
    KParam out_;
    read_accessor<T> iptrAcc_;
    KParam in_;
    const int wx_;
    const int wy_;
    const int sx_;
    const int sy_;
    const int px_;
    const int py_;
    const int nx_;
    const int ny_;
    int groups_x_;
    int groups_y_;
    const bool is_column_;
};

template<typename T>
void wrap(Param<T> out, const Param<T> in, const dim_t wx, const dim_t wy,
          const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
          const bool is_column) {
    dim_t nx = (out.info.dims[0] + 2 * px - wx) / sx + 1;
    dim_t ny = (out.info.dims[1] + 2 * py - wy) / sy + 1;

    auto local = sycl::range{THREADS_X, THREADS_Y};

    dim_t groups_x = divup(out.info.dims[0], local[0]);
    dim_t groups_y = divup(out.info.dims[1], local[1]);

    auto global = sycl::range{groups_x * local[0] * out.info.dims[2],
                              groups_y * local[1]};

    auto Q = getQueue();
    Q.submit([&](sycl::handler &h) {
        sycl::accessor outAcc{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor inAcc{*in.data, h, sycl::read_only};
        h.parallel_for(sycl::nd_range{global, local},
                       wrapCreateKernel<T>(outAcc, out.info, inAcc, in.info, wx,
                                           wy, sx, sy, px, py, nx, ny, groups_x,
                                           groups_y, is_column));
    });
    ONEAPI_DEBUG_FINISH(Q);
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
