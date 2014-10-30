/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <shift.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <cassert>

namespace cpu
{
    static inline dim_type simple_mod(const dim_type i, const dim_type dim)
    {
        return (i < dim) ? i : (i - dim);
    }

    template<typename T>
    Array<T> *shift(const Array<T> &in, const af::dim4 &sdims)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;

        Array<T> *out = createEmptyArray<T>(oDims);

        T* outPtr = out->get();
        const T* inPtr = in.get();

        const af::dim4 ist = in.strides();
        const af::dim4 ost = out->strides();

        dim_type sdims_[4];
        // Need to do this because we are mapping output to input in the kernel
        for(int i = 0; i < 4; i++) {
            // sdims_[i] will always be positive and always [0, oDims[i]].
            // Negative shifts are converted to position by going the other way round
            sdims_[i] = -(sdims[i] % oDims[i]) + oDims[i] * (sdims[i] > 0);
            assert(sdims_[i] >= 0 && sdims_[i] <= oDims[i]);
        }

        for(dim_type ow = 0; ow < oDims[3]; ow++) {
            const dim_type oW = ow * ost[3];
            const dim_type iw = simple_mod((ow + sdims_[3]), oDims[3]);
            const dim_type iW = iw * ist[3];
            for(dim_type oz = 0; oz < oDims[2]; oz++) {
                const dim_type oZW = oW + oz * ost[2];
                const dim_type iz = simple_mod((oz + sdims_[2]), oDims[2]);
                const dim_type iZW = iW + iz * ist[2];
                for(dim_type oy = 0; oy < oDims[1]; oy++) {
                    const dim_type oYZW = oZW + oy * ost[1];
                    const dim_type iy = simple_mod((oy + sdims_[1]), oDims[1]);
                    const dim_type iYZW = iZW + iy * ist[1];
                    for(dim_type ox = 0; ox < oDims[0]; ox++) {
                        const dim_type oIdx = oYZW + ox;
                        const dim_type ix = simple_mod((ox + sdims_[0]), oDims[0]);
                        const dim_type iIdx = iYZW + ix;

                        outPtr[oIdx] = inPtr[iIdx];
                    }
                }
            }
        }

        return out;
    }

#define INSTANTIATE(T)                                                          \
    template Array<T>* shift<T>(const Array<T> &in, const af::dim4 &sdims);     \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

}
