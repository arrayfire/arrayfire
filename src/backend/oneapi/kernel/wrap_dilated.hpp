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
class wrapDilatedCreateKernel {
   public:
    wrapDilatedCreateKernel(write_accessor<data_t<T>> optrAcc, KParam out,
                            read_accessor<data_t<T>> iptrAcc, KParam in,
                            const int wx, const int wy, const int sx,
                            const int sy, const int px, const int py,
                            const int dx, const int dy, const int nx,
                            const int ny, int groups_x, int groups_y,
                            const bool is_column)
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
        , dx_(dx)
        , dy_(dy)
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

        data_t<T> *optr = optrAcc_.get_pointer() + idx2 * out_.strides[2] +
                          idx3 * out_.strides[3];
        const data_t<T> *iptr = iptrAcc_.get_pointer() + idx2 * in_.strides[2] +
                                idx3 * in_.strides[3] + in_.offset;

        if (oidx0 >= out_.dims[0] || oidx1 >= out_.dims[1]) return;

        int eff_wx = wx_ + (wx_ - 1) * (dx_ - 1);
        int eff_wy = wy_ + (wy_ - 1) * (dy_ - 1);

        int pidx0 = oidx0 + px_;
        int pidx1 = oidx1 + py_;

        // The last time a value appears in_ the unwrapped index is padded_index
        // / stride Each previous index has the value appear "stride" locations
        // earlier We work our way back from the last index

        const int y_start = (pidx1 < eff_wy) ? 0 : (pidx1 - eff_wy) / sy_ + 1;
        const int y_end   = sycl::min(pidx1 / sy_ + 1, ny_);

        const int x_start = (pidx0 < eff_wx) ? 0 : (pidx0 - eff_wx) / sx_ + 1;
        const int x_end   = sycl::min(pidx0 / sx_ + 1, nx_);

        compute_t<T> val(0);
        int idx = 1;

        for (int y = y_start; y < y_end; y++) {
            int fy      = (pidx1 - y * sy_);
            bool yvalid = (fy % dy_ == 0) && (y < ny_);
            fy /= dy_;

            int win_end_y = fy * wx_;
            int dim_end_y = y * nx_;

            for (int x = x_start; x < x_end; x++) {
                int fx      = (pidx0 - x * sx_);
                bool xvalid = (fx % dx_ == 0) && (x < nx_);
                fx /= dx_;

                int win_end = win_end_y + fx;
                int dim_end = dim_end_y + x;

                if (is_column_) {
                    idx = dim_end * in_.strides[1] + win_end;
                } else {
                    idx = dim_end + win_end * in_.strides[1];
                }

                compute_t<T> ival;
                ival = (yvalid && xvalid) ? iptr[idx] : compute_t<T>(0);
                val  = val + ival;
            }
        }

        optr[oidx1 * out_.strides[1] + oidx0] = val;
    }

   private:
    write_accessor<data_t<T>> optrAcc_;
    KParam out_;
    read_accessor<data_t<T>> iptrAcc_;
    KParam in_;
    const int wx_;
    const int wy_;
    const int sx_;
    const int sy_;
    const int px_;
    const int py_;
    const int dx_;
    const int dy_;
    const int nx_;
    const int ny_;
    int groups_x_;
    int groups_y_;
    const bool is_column_;
};

template<typename T>
void wrap_dilated(Param<T> out, const Param<T> in, const dim_t wx,
                  const dim_t wy, const dim_t sx, const dim_t sy,
                  const dim_t px, const dim_t py, const dim_t dx,
                  const dim_t dy, const bool is_column) {
    dim_t nx = 1 + (out.info.dims[0] + 2 * px - (((wx - 1) * dx) + 1)) / sx;
    dim_t ny = 1 + (out.info.dims[1] + 2 * py - (((wy - 1) * dy) + 1)) / sy;

    auto local = sycl::range{THREADS_X, THREADS_Y};

    dim_t groups_x = divup(out.info.dims[0], local[0]);
    dim_t groups_y = divup(out.info.dims[1], local[1]);

    auto global = sycl::range{local[0] * groups_x * out.info.dims[2],
                              local[1] * groups_y * out.info.dims[3]};

    auto Q = getQueue();
    Q.submit([&](sycl::handler &h) {
        write_accessor<data_t<T>> outAcc =
            out.template get_accessor<sycl::access_mode::write>(h);
        read_accessor<data_t<T>> inAcc =
            in.template get_accessor<sycl::access_mode::read>(h);
        h.parallel_for(sycl::nd_range{global, local},
                       wrapDilatedCreateKernel<T>(
                           outAcc, out.info, inAcc, in.info, wx, wy, sx, sy, px,
                           py, dx, dy, nx, ny, groups_x, groups_y, is_column));
    });
    ONEAPI_DEBUG_FINISH(Q);
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
