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
#include <kernel/config.hpp>

#include <string>
#include <vector>

namespace oneapi {
namespace kernel {

template<typename T>
void unwrap(Param<T> out, const Param<T> in, const dim_t wx, const dim_t wy,
            const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
            const dim_t dx, const dim_t dy, const dim_t nx,
            const bool is_column) {
    const bool IS_COLUMN = is_column;

    dim_t TX = 1, TY = 1;
    dim_t BX       = 1;
    const dim_t BY = out.info.dims[2] * out.info.dims[3];
    int reps       = 1;

    if (is_column) {
        TX   = std::min(THREADS_PER_GROUP, nextpow2(out.info.dims[0]));
        TY   = THREADS_PER_GROUP / TX;
        BX   = divup(out.info.dims[1], TY);
        reps = divup((wx * wy), TX);
    } else {
        TX   = THREADS_X;
        TY   = THREADS_X;
        BX   = divup(out.info.dims[0], TX);
        reps = divup((wx * wy), TY);
    }

    auto local  = sycl::range(TX, TY);
    auto global = sycl::range(local[0] * BX, local[1] * BY);

    getQueue().submit([&](auto &h) {
        sycl::accessor outAcc{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor inAcc{*in.data, h, sycl::read_only};
        h.parallel_for(sycl::nd_range{global, local}, [=](auto it) {
            sycl::group g = it.get_group();

            // Compute channel and volume
            const int w = g.get_group_id(1) / in.info.dims[2];
            const int z = g.get_group_id(1) - w * in.info.dims[2];

            if (w >= in.info.dims[3] || z >= in.info.dims[2]) return;

            // Compute offset for channel and volume
            const int cOut = w * out.info.strides[3] + z * out.info.strides[2];
            const int cIn  = w * in.info.strides[3] + z * in.info.strides[2];

            // Compute the output column index
            const int id = IS_COLUMN
                               ? (g.get_group_id(0) * g.get_local_range(1) +
                                  it.get_local_id(1))
                               : it.get_global_id(0);

            if (id >= (IS_COLUMN ? out.info.dims[1] : out.info.dims[0])) return;

            // Compute the starting index of window in x and y of input
            const int startx = (id % nx) * sx;
            const int starty = (id / nx) * sy;

            const int spx = startx - px;
            const int spy = starty - py;

            // Offset the global pointers to the respective starting indices
            T *d_out = outAcc.get_pointer();
            T *optr = d_out + cOut + id * (IS_COLUMN ? out.info.strides[1] : 1);
            T *d_in = inAcc.get_pointer();
            const T *iptr = d_in + cIn + in.info.offset;

            bool cond = (spx >= 0 && spx + (wx * dx) < in.info.dims[0] &&
                         spy >= 0 && spy + (wy * dy) < in.info.dims[1]);

            // Compute output index local to column
            int outIdx = IS_COLUMN ? it.get_local_id(0) : it.get_local_id(1);
            const int oStride =
                IS_COLUMN ? it.get_local_range(0) : it.get_local_range(1);

            for (int i = 0; i < reps; i++) {
                if (outIdx >= (IS_COLUMN ? out.info.dims[0] : out.info.dims[1]))
                    return;

                // Compute input index local to window
                const int y = outIdx / wx;
                const int x = outIdx % wx;

                const int xpad = spx + x * dx;
                const int ypad = spy + y * dy;

                // Copy
                T val = (T)0;
                if (cond || (xpad >= 0 && xpad < in.info.dims[0] && ypad >= 0 &&
                             ypad < in.info.dims[1])) {
                    const int inIdx =
                        ypad * in.info.strides[1] + xpad * in.info.strides[0];
                    val = iptr[inIdx];
                }

                if (IS_COLUMN) {
                    optr[outIdx] = val;
                } else {
                    optr[outIdx * out.info.strides[1]] = val;
                }

                outIdx += oStride;
            }
        });
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
