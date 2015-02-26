/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <tile.hpp>
#include <stdexcept>
#include <err_cpu.hpp>

namespace cpu
{
    template<typename T>
    Array<T> tile(const Array<T> &in, const af::dim4 &tileDims)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;
        oDims *= tileDims;

        if(iDims.elements() == 0 || oDims.elements() == 0) {
            throw std::runtime_error("Elements are 0");
        }

        Array<T> out = createEmptyArray<T>(oDims);

        T* outPtr = out.get();
        const T* inPtr = in.get();

        const af::dim4 ist = in.strides();
        const af::dim4 ost = out.strides();

        for(dim_type ow = 0; ow < oDims[3]; ow++) {
            const dim_type iw = ow % iDims[3];
            const dim_type iW = iw * ist[3];
            const dim_type oW = ow * ost[3];
            for(dim_type oz = 0; oz < oDims[2]; oz++) {
                const dim_type iz = oz % iDims[2];
                const dim_type iZW = iW + iz * ist[2];
                const dim_type oZW = oW + oz * ost[2];
                for(dim_type oy = 0; oy < oDims[1]; oy++) {
                    const dim_type iy = oy % iDims[1];
                    const dim_type iYZW = iZW + iy * ist[1];
                    const dim_type oYZW = oZW + oy * ost[1];
                    for(dim_type ox = 0; ox < oDims[0]; ox++) {
                        const dim_type ix = ox % iDims[0];
                        const dim_type iMem = iYZW + ix;
                        const dim_type oMem = oYZW + ox;
                        outPtr[oMem] = inPtr[iMem];
                    }
                }
            }
        }

        return out;
    }

#define INSTANTIATE(T)                                                         \
    template Array<T> tile<T>(const Array<T> &in, const af::dim4 &tileDims);  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

}
