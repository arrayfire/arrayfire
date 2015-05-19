/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <reorder.hpp>
#include <stdexcept>
#include <err_cpu.hpp>

namespace cpu
{
    template<typename T>
    Array<T> reorder(const Array<T> &in, const af::dim4 &rdims)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims(0);
        for(int i = 0; i < 4; i++)
            oDims[i] = iDims[rdims[i]];

        Array<T> out = createEmptyArray<T>(oDims);

        T* outPtr = out.get();
        const T* inPtr = in.get();

        const af::dim4 ist = in.strides();
        const af::dim4 ost = out.strides();


        dim_t ids[4]  = {0};
        for(dim_t ow = 0; ow < oDims[3]; ow++) {
            const dim_t oW = ow * ost[3];
            ids[rdims[3]] = ow;
            for(dim_t oz = 0; oz < oDims[2]; oz++) {
                const dim_t oZW = oW + oz * ost[2];
                ids[rdims[2]] = oz;
                for(dim_t oy = 0; oy < oDims[1]; oy++) {
                    const dim_t oYZW = oZW + oy * ost[1];
                    ids[rdims[1]] = oy;
                    for(dim_t ox = 0; ox < oDims[0]; ox++) {
                        const dim_t oIdx = oYZW + ox;

                        ids[rdims[0]] = ox;
                        const dim_t iIdx = ids[3] * ist[3] + ids[2] * ist[2] +
                                              ids[1] * ist[1] + ids[0];

                        outPtr[oIdx] = inPtr[iIdx];
                    }
                }
            }
        }

        return out;
    }

#define INSTANTIATE(T)                                                         \
    template Array<T> reorder<T>(const Array<T> &in, const af::dim4 &rdims);  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)


}
