/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <Array.hpp>
#include <triangle.hpp>
#include <math.hpp>

namespace cpu
{

template<typename T, bool is_upper>
void triangle(Array<T> &out, const Array<T> &in)
{
    T *o = out.get();
    const T *i = in.get();

    dim4 odm = out.dims();

    dim4 ost = out.strides();
    dim4 ist = in.strides();

    for(dim_type ow = 0; ow < odm[3]; ow++) {
        const dim_type oW = ow * ost[3];
        const dim_type iW = ow * ist[3];

        for(dim_type oz = 0; oz < odm[2]; oz++) {
            const dim_type oZW = oW + oz * ost[2];
            const dim_type iZW = iW + oz * ist[2];

            for(dim_type oy = 0; oy < odm[1]; oy++) {
                const dim_type oYZW = oZW + oy * ost[1];
                const dim_type iYZW = iZW + oy * ist[1];

                for(dim_type ox = 0; ox < odm[0]; ox++) {
                    const dim_type oMem = oYZW + ox;
                    const dim_type iMem = iYZW + ox;

                    bool cond = is_upper ? (oy >= ox) : (oy <= ox);
                    if(cond) {
                        o[oMem] = i[iMem];
                    } else {
                        o[oMem] = scalar<T>(0);
                    }

                }
            }
        }
    }
}

#define INSTANTIATE(T)                                          \
    template void triangle<T, true >(Array<T> &out, const Array<T> &in);  \
    template void triangle<T, false>(Array<T> &out, const Array<T> &in);  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)

}
