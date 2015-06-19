/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <unwrap.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <dispatch.hpp>
#include <math.hpp>

namespace cpu
{
    template<typename T>
    void unwrap_(T *outPtr, const T *inPtr, const af::dim4 &odims, const af::dim4 &idims,
                 const af::dim4 &ostrides, const af::dim4 &istrides,
                 const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy,
                 const dim_t px, const dim_t py)
    {
        dim_t nx = (idims[0] + 2 * px - wx) / sx + 1;

        for(dim_t w = 0; w < odims[3]; w++) {
            for(dim_t z = 0; z < odims[2]; z++) {

                dim_t cOut = w * ostrides[3] + z * ostrides[2];
                dim_t cIn  = w * istrides[3] + z * istrides[2];
                const T* iptr = inPtr  + cIn;
                      T* optr_= outPtr + cOut;

                for(dim_t col = 0; col < odims[1]; col++) {
                    // Offset output ptr
                    T* optr = optr_ + col * ostrides[1];

                    // Calculate input window index
                    dim_t winy = (col / nx);
                    dim_t winx = (col % nx);

                    dim_t startx = winx * sx;
                    dim_t starty = winy * sy;

                    dim_t spx = startx - px;
                    dim_t spy = starty - py;

                    // Short cut condition ensuring all values within input dimensions
                    bool cond = false;
                    if(spx >= 0 && spx + wx < idims[0] && spy >= 0 && spy + wy < idims[1])
                        cond = true;

                    for(dim_t y = 0; y < wy; y++) {
                        for(dim_t x = 0; x < wx; x++) {
                            dim_t xpad = spx + x;
                            dim_t ypad = spy + y;

                            dim_t oloc = (y * wx + x) * ostrides[0];
                            if(cond || (xpad >= 0 && xpad < idims[0] && ypad >= 0 && ypad < idims[1])) {
                                dim_t iloc = (ypad * istrides[1] + xpad * istrides[0]);
                                optr[oloc] = iptr[iloc];
                            } else {
                                optr[oloc] = scalar<T>(0.0);
                            }
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    Array<T> unwrap(const Array<T> &in, const dim_t wx, const dim_t wy,
                    const dim_t sx, const dim_t sy, const dim_t px, const dim_t py)
    {
        af::dim4 idims = in.dims();

        dim_t nx = (idims[0] + 2 * px - wx) / sx + 1;
        dim_t ny = (idims[1] + 2 * py - wy) / sx + 1;

        af::dim4 odims(wx * wy, nx * ny, idims[2], idims[3]);

        // Create output placeholder
        Array<T> outArray = createEmptyArray<T>(odims);

        // Get pointers to raw data
        const T *inPtr = in.get();
              T *outPtr = outArray.get();

        af::dim4 ostrides = outArray.strides();
        af::dim4 istrides = in.strides();

        unwrap_(outPtr, inPtr, odims, idims, ostrides, istrides, wx, wy, sx, sy, px, py);

        return outArray;
    }


#define INSTANTIATE(T)                                                                  \
    template Array<T> unwrap<T> (const Array<T> &in, const dim_t wx, const dim_t wy,    \
                    const dim_t sx, const dim_t sy, const dim_t px, const dim_t py);


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

