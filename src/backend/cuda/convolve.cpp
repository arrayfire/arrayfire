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
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <convolve.hpp>
#include <kernel/convolve.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T, typename accT, dim_t baseDim, bool expand>
Array<T> convolve(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind)
{
    const dim4 sDims    = signal.dims();
    const dim4 fDims    = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for(dim_t d=0; d<4; ++d) {
            if (kind==CONVOLVE_BATCH_NONE || kind==CONVOLVE_BATCH_KERNEL) {
                oDims[d] = sDims[d]+fDims[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sDims[d]+fDims[d]-1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind==CONVOLVE_BATCH_KERNEL) {
            for (dim_t i=baseDim; i<4; ++i)
                oDims[i] = fDims[i];
        }
    }

    Array<T> out   = createEmptyArray<T>(oDims);

    kernel::convolve_nd<T, accT, baseDim, expand>(out, signal, filter, kind);

    return out;
}

template<typename T, typename accT, bool expand>
Array<T> convolve2(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter)
{
    const dim4 cfDims   = c_filter.dims();
    const dim4 rfDims   = r_filter.dims();

    const dim_t cfLen= cfDims.elements();
    const dim_t rfLen= rfDims.elements();

    const dim4 sDims = signal.dims();
    dim4 tDims = sDims;
    dim4 oDims = sDims;

    if (expand) {
        tDims[0] += cfLen - 1;
        oDims[0] += cfLen - 1;
        oDims[1] += rfLen - 1;
    }

    Array<T> temp= createEmptyArray<T>(tDims);
    Array<T> out = createEmptyArray<T>(oDims);

    kernel::convolve2<T, accT, 0, expand>(temp, signal, c_filter);
    kernel::convolve2<T, accT, 1, expand>(out, temp, r_filter);

    return out;
}

#define INSTANTIATE(T, accT)                                            \
    template Array<T> convolve <T, accT, 1, true >(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 1, false>(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 2, true >(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 2, false>(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 3, true >(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 3, false>(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve2<T, accT, true >(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter); \
    template Array<T> convolve2<T, accT, false>(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}
