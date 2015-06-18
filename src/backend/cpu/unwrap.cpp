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
                 const dim_t wx, const dim_t wy, const dim_t sx, const dim_t sy)
    {
        dim_t nx = divup(idims[0] - wx, sx) + (sx >= idims[0] ? 0 : 1);

        for(dim_t w = 0; w < odims[3]; w++) {
            for(dim_t z = 0; z < odims[2]; z++) {
                dim_t cOut = w * ostrides[3] + z * ostrides[2];
                dim_t cIn  = w * istrides[3] + z * istrides[2];
                for(dim_t col = 0; col < odims[1]; col++) {
                    // Calculate input window index
                    dim_t winy = (col / nx);
                    dim_t winx = (col % nx);

                    dim_t startx = winx * sx;
                    dim_t starty = winy * sy;

                          T* optr = outPtr + cOut + col * ostrides[1];
                    const T* iptr = inPtr  + cIn  + starty * istrides[1] + startx;

                    // Condition shortcuts
                    bool cond = true;
                    if((startx + wx >= idims[0]) || (starty + wy >= idims[1]))
                        cond = false;

                    for(dim_t y = 0; y < wy; y++) {
                        for(dim_t x = 0; x < wx; x++) {
                            dim_t oloc = (y * wx + x) * ostrides[0];
                            dim_t iloc = (y * istrides[1] + x * istrides[0]);
                            if(cond || (startx + x < idims[0] && starty + y < idims[1]))
                                optr[oloc] = iptr[iloc];
                            else
                                optr[oloc] = scalar<T>(0.0);
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    Array<T> unwrap(const Array<T> &in, const dim_t wx, const dim_t wy,
                    const dim_t sx, const dim_t sy)
    {
        af::dim4 idims = in.dims();

        dim_t nx = divup(idims[0] - wx, sx) + (sx >= idims[0] ? 0 : 1);
        dim_t ny = divup(idims[1] - wy, sy) + (sy >= idims[1] ? 0 : 1);

        af::dim4 odims(wx * wy, nx * ny, idims[2], idims[3]);

        // Create output placeholder
        Array<T> outArray = createEmptyArray<T>(odims);

        // Get pointers to raw data
        const T *inPtr = in.get();
              T *outPtr = outArray.get();

        af::dim4 ostrides = outArray.strides();
        af::dim4 istrides = in.strides();

        unwrap_(outPtr, inPtr, odims, idims, ostrides, istrides, wx, wy, sx, sy);

        return outArray;
    }


#define INSTANTIATE(T)                                                                  \
    template Array<T> unwrap<T> (const Array<T> &in, const dim_t wx, const dim_t wy,    \
                                 const dim_t sx, const dim_t sy);


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

