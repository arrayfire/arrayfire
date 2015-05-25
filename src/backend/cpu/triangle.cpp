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

template<typename T, bool is_upper, bool is_unit_diag>
void triangle(Array<T> &out, const Array<T> &in)
{
    T *o = out.get();
    const T *i = in.get();

    dim4 odm = out.dims();

    dim4 ost = out.strides();
    dim4 ist = in.strides();

    for(dim_t ow = 0; ow < odm[3]; ow++) {
        const dim_t oW = ow * ost[3];
        const dim_t iW = ow * ist[3];

        for(dim_t oz = 0; oz < odm[2]; oz++) {
            const dim_t oZW = oW + oz * ost[2];
            const dim_t iZW = iW + oz * ist[2];

            for(dim_t oy = 0; oy < odm[1]; oy++) {
                const dim_t oYZW = oZW + oy * ost[1];
                const dim_t iYZW = iZW + oy * ist[1];

                for(dim_t ox = 0; ox < odm[0]; ox++) {
                    const dim_t oMem = oYZW + ox;
                    const dim_t iMem = iYZW + ox;

                    bool cond = is_upper ? (oy >= ox) : (oy <= ox);
                    bool do_unit_diag = (is_unit_diag && ox == oy);
                    if(cond) {
                        o[oMem] = do_unit_diag ? scalar<T>(1) : i[iMem];
                    } else {
                        o[oMem] = scalar<T>(0);
                    }

                }
            }
        }
    }
}

template<typename T, bool is_upper, bool is_unit_diag>
Array<T> triangle(const Array<T> &in)
{
    Array<T> out = createEmptyArray<T>(in.dims());
    triangle<T, is_upper, is_unit_diag>(out, in);
    return out;
}

#define INSTANTIATE(T)                                                  \
    template void triangle<T, true ,  true>(Array<T> &out, const Array<T> &in); \
    template void triangle<T, false,  true>(Array<T> &out, const Array<T> &in); \
    template void triangle<T, true , false>(Array<T> &out, const Array<T> &in); \
    template void triangle<T, false, false>(Array<T> &out, const Array<T> &in); \
    template Array<T> triangle<T, true ,  true>(const Array<T> &in);    \
    template Array<T> triangle<T, false,  true>(const Array<T> &in);    \
    template Array<T> triangle<T, true , false>(const Array<T> &in);    \
    template Array<T> triangle<T, false, false>(const Array<T> &in);    \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(char)
    INSTANTIATE(uchar)

}
