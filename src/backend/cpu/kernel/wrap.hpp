/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <err_cpu.hpp>
#include <math.hpp>

#include <algorithm>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T, int d>
void wrap_dim(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy,
              const dim_t sx, const dim_t sy, const dim_t px, const dim_t py) {
    const T *inPtr = in.get();
    T *outPtr      = out.get();

    af::dim4 idims    = in.dims();
    af::dim4 odims    = out.dims();
    af::dim4 istrides = in.strides();
    af::dim4 ostrides = out.strides();

    dim_t nx = (odims[0] + 2 * px - wx) / sx + 1;

    for (dim_t w = 0; w < idims[3]; w++) {
        for (dim_t z = 0; z < idims[2]; z++) {
            dim_t cIn      = w * istrides[3] + z * istrides[2];
            dim_t cOut     = w * ostrides[3] + z * ostrides[2];
            const T *iptr_ = inPtr + cIn;
            T *optr        = outPtr + cOut;

            for (dim_t col = 0; col < idims[d]; col++) {
                // Offset output ptr
                const T *iptr = iptr_ + col * istrides[d];

                // Calculate input window index
                dim_t winy = (col / nx);
                dim_t winx = (col % nx);

                dim_t startx = winx * sx;
                dim_t starty = winy * sy;

                dim_t spx = startx - px;
                dim_t spy = starty - py;

                // Short cut condition ensuring all values within input
                // dimensions
                bool cond = (spx >= 0 && spx + wx < odims[0] && spy >= 0 &&
                             spy + wy < odims[1]);

                for (dim_t y = 0; y < wy; y++) {
                    for (dim_t x = 0; x < wx; x++) {
                        dim_t xpad = spx + x;
                        dim_t ypad = spy + y;

                        dim_t iloc = (y * wx + x);
                        if (d == 0) iloc *= istrides[1];

                        if (cond || (xpad >= 0 && xpad < odims[0] &&
                                     ypad >= 0 && ypad < odims[1])) {
                            dim_t oloc =
                                (ypad * ostrides[1] + xpad * ostrides[0]);
                            // FIXME: When using threads, atomize this
                            optr[oloc] += iptr[iloc];
                        }
                    }
                }
            }
        }
    }
}

template<typename T>
void wrap_dim_dilated(Param<T> out, CParam<T> in, const dim_t wx,
                      const dim_t wy, const dim_t sx, const dim_t sy,
                      const dim_t px, const dim_t py, const dim_t dx,
                      const dim_t dy, const int d) {
    const T *inPtr = in.get();
    T *outPtr      = out.get();

    af::dim4 idims    = in.dims();
    af::dim4 odims    = out.dims();
    af::dim4 istrides = in.strides();
    af::dim4 ostrides = out.strides();

    dim_t nx = 1 + (odims[0] + 2 * px - (((wx - 1) * dx) + 1)) / sx;

    for (dim_t w = 0; w < idims[3]; w++) {
        for (dim_t z = 0; z < idims[2]; z++) {
            dim_t cIn              = w * istrides[3] + z * istrides[2];
            dim_t cOut             = w * ostrides[3] + z * ostrides[2];
            const data_t<T> *iptr_ = inPtr + cIn;
            data_t<T> *optr        = outPtr + cOut;

            for (dim_t col = 0; col < idims[d]; col++) {
                // Offset output ptr
                const data_t<T> *iptr = iptr_ + col * istrides[d];

                // Calculate input window index
                dim_t winy = (col / nx);
                dim_t winx = (col % nx);

                dim_t startx = winx * sx;
                dim_t starty = winy * sy;

                dim_t spx = startx - px;
                dim_t spy = starty - py;

                // Short cut condition ensuring all values within input
                // dimensions
                bool cond = (spx >= 0 && spx + (wx * dx) < odims[0] &&
                             spy >= 0 && spy + (wy * dy) < odims[1]);

                for (dim_t y = 0; y < wy; y++) {
                    dim_t ypad = spy + y * dy;
                    for (dim_t x = 0; x < wx; x++) {
                        dim_t xpad = spx + x * dx;

                        dim_t iloc = (y * wx + x);
                        if (d == 0) iloc *= istrides[1];

                        if (cond || (xpad >= 0 && xpad < odims[0] &&
                                     ypad >= 0 && ypad < odims[1])) {
                            dim_t oloc =
                                (ypad * ostrides[1] + xpad * ostrides[0]);
                            // FIXME: When using threads, atomize this
                            optr[oloc] = static_cast<compute_t<T>>(optr[oloc]) +
                                         static_cast<compute_t<T>>(iptr[iloc]);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
