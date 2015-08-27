/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <wrap.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <dispatch.hpp>
#include <math.hpp>

namespace cpu
{

    template<typename T, int d>
    void wrap_dim(T *outPtr, const T *inPtr,
                  const af::dim4 &odims, const af::dim4 &idims,
                  const af::dim4 &ostrides, const af::dim4 &istrides,
                  const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy,
                  const dim_t px, const dim_t py)
    {
        dim_t nx = (odims[0] + 2 * px - wx) / sx + 1;

        for(dim_t w = 0; w < idims[3]; w++) {
            for(dim_t z = 0; z < idims[2]; z++) {

                dim_t cIn  = w * istrides[3] + z * istrides[2];
                dim_t cOut = w * ostrides[3] + z * ostrides[2];
                const T* iptr_ = inPtr  + cIn;
                T* optr= outPtr + cOut;

                for(dim_t col = 0; col < idims[d]; col++) {
                    // Offset output ptr
                    const T* iptr = iptr_ + col * istrides[d];

                    // Calculate input window index
                    dim_t winy = (col / nx);
                    dim_t winx = (col % nx);

                    dim_t startx = winx * sx;
                    dim_t starty = winy * sy;

                    dim_t spx = startx - px;
                    dim_t spy = starty - py;

                    // Short cut condition ensuring all values within input dimensions
                    bool cond = (spx >= 0 && spx + wx < odims[0] && spy >= 0 && spy + wy < odims[1]);

                    for(dim_t y = 0; y < wy; y++) {
                        for(dim_t x = 0; x < wx; x++) {
                            dim_t xpad = spx + x;
                            dim_t ypad = spy + y;

                            dim_t iloc = (y * wx + x);
                            if (d == 0) iloc *= istrides[1];

                            if(cond || (xpad >= 0 && xpad < odims[0] && ypad >= 0 && ypad < odims[1])) {
                                dim_t oloc = (ypad * ostrides[1] + xpad * ostrides[0]);
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
    Array<T> wrap(const Array<T> &in,
                  const dim_t ox, const dim_t oy,
                  const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy,
                  const dim_t px, const dim_t py,
                  const bool is_column)
    {
        af::dim4 idims = in.dims();
        af::dim4 odims(ox, oy, idims[2], idims[3]);
        Array<T> out = createValueArray<T>(odims, scalar<T>(0));

        const T *inPtr = in.get();
        T *outPtr = out.get();

        af::dim4 istrides = in.strides();
        af::dim4 ostrides = out.strides();

        if (is_column) {
            wrap_dim<T, true >(outPtr, inPtr, odims, idims, ostrides, istrides, wx, wy, sx, sy, px, py);
        } else {
            wrap_dim<T, false>(outPtr, inPtr, odims, idims, ostrides, istrides, wx, wy, sx, sy, px, py);
        }

        return out;
    }


#define INSTANTIATE(T)                                          \
    template Array<T> wrap<T> (const Array<T> &in,              \
                               const dim_t ox, const dim_t oy,  \
                               const dim_t wx, const dim_t wy,  \
                               const dim_t sx, const dim_t sy,  \
                               const dim_t px, const dim_t py,  \
                               const bool is_column);


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
