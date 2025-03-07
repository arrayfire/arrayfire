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
#include <kernel/default_config.hpp>

#include <sycl/sycl.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
class unwrapCreateKernel {
   public:
    unwrapCreateKernel(sycl::accessor<T, 1, sycl::access::mode::write> d_out,
                       const KParam out,
                       sycl::accessor<T, 1, sycl::access::mode::read> d_in,
                       const KParam in, const int wx, const int wy,
                       const int sx, const int sy, const int px, const int py,
                       const int dx, const int dy, const int nx, const int reps,
                       const bool IS_COLUMN)
        : d_out_(d_out)
        , out_(out)
        , d_in_(d_in)
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
        , reps_(reps)
        , IS_COLUMN_(IS_COLUMN) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        // Compute channel and volume
        const int w = g.get_group_id(1) / in_.dims[2];
        const int z = g.get_group_id(1) - w * in_.dims[2];

        if (w >= in_.dims[3] || z >= in_.dims[2]) return;

        // Compute offset for channel and volume
        const int cOut = w * out_.strides[3] + z * out_.strides[2];
        const int cIn  = w * in_.strides[3] + z * in_.strides[2];

        // Compute the output column index
        const int id = IS_COLUMN_ ? (g.get_group_id(0) * g.get_local_range(1) +
                                     it.get_local_id(1))
                                  : it.get_global_id(0);

        if (id >= (IS_COLUMN_ ? out_.dims[1] : out_.dims[0])) return;

        // Compute the starting index of window in_ x and y of input
        const int startx = (id % nx_) * sx_;
        const int starty = (id / nx_) * sy_;

        const int spx = startx - px_;
        const int spy = starty - py_;

        // Offset the global pointers to the respective starting indices
        T *optr = d_out_.get_pointer() + cOut +
                  id * (IS_COLUMN_ ? out_.strides[1] : 1);
        const T *iptr = d_in_.get_pointer() + cIn + in_.offset;

        bool cond = (spx >= 0 && spx + (wx_ * dx_) < in_.dims[0] && spy >= 0 &&
                     spy + (wy_ * dy_) < in_.dims[1]);

        // Compute output index local to column
        int outIdx = IS_COLUMN_ ? it.get_local_id(0) : it.get_local_id(1);
        const int oStride =
            IS_COLUMN_ ? it.get_local_range(0) : it.get_local_range(1);

        for (int i = 0; i < reps_; i++) {
            if (outIdx >= (IS_COLUMN_ ? out_.dims[0] : out_.dims[1])) return;

            // Compute input index local to window
            const int y = outIdx / wx_;
            const int x = outIdx % wx_;

            const int xpad = spx + x * dx_;
            const int ypad = spy + y * dy_;

            // Copy
            T val = (T)0;
            if (cond || (xpad >= 0 && xpad < in_.dims[0] && ypad >= 0 &&
                         ypad < in_.dims[1])) {
                const int inIdx = ypad * in_.strides[1] + xpad * in_.strides[0];
                val             = iptr[inIdx];
            }

            if (IS_COLUMN_) {
                optr[outIdx] = val;
            } else {
                optr[outIdx * out_.strides[1]] = val;
            }

            outIdx += oStride;
        }
    }

   private:
    sycl::accessor<T, 1, sycl::access::mode::write> d_out_;
    const KParam out_;
    sycl::accessor<T, 1, sycl::access::mode::read> d_in_;
    const KParam in_;
    const int wx_;
    const int wy_;
    const int sx_;
    const int sy_;
    const int px_;
    const int py_;
    const int dx_;
    const int dy_;
    const int nx_;
    const int reps_;
    const bool IS_COLUMN_;
};

template<typename T>
void unwrap(Param<T> out, const Param<T> in, const dim_t wx, const dim_t wy,
            const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
            const dim_t dx, const dim_t dy, const dim_t nx,
            const bool IS_COLUMN) {
    dim_t TX = 1, TY = 1;
    dim_t BX       = 1;
    const dim_t BY = out.info.dims[2] * out.info.dims[3];
    int reps       = 1;

    if (IS_COLUMN) {
        TX   = std::min(THREADS_PER_BLOCK, nextpow2(out.info.dims[0]));
        TY   = THREADS_PER_BLOCK / TX;
        BX   = divup(out.info.dims[1], TY);
        reps = divup((wx * wy), TX);
    } else {
        TX   = THREADS_X;
        TY   = THREADS_Y;
        BX   = divup(out.info.dims[0], TX);
        reps = divup((wx * wy), TY);
    }

    auto local  = sycl::range(TX, TY);
    auto global = sycl::range(local[0] * BX, local[1] * BY);

    getQueue().submit([&](auto &h) {
        sycl::accessor d_out{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor d_in{*in.data, h, sycl::read_only};
        h.parallel_for(
            sycl::nd_range{global, local},
            unwrapCreateKernel<T>(d_out, out.info, d_in, in.info, wx, wy, sx,
                                  sy, px, py, dx, dy, nx, reps, IS_COLUMN));
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
