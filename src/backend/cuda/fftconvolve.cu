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
#include <fftconvolve.hpp>
#include <kernel/fftconvolve.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T, typename convT, bool isDouble, bool roundOut, dim_type baseDim>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind)
{
    const dim4 sDims = signal.dims();
    const dim4 fDims = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for(dim_type d=0; d<4; ++d) {
            if (kind==ONE2ONE || kind==ONE2MANY) {
                oDims[d] = sDims[d]+fDims[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sDims[d]+fDims[d]-1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind==ONE2MANY) {
            for (dim_type i=baseDim; i<4; ++i)
                oDims[i] = fDims[i];
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    if (expand)
        kernel::fftconvolve<T, convT, isDouble, roundOut, baseDim, true >(out, signal, filter, kind);
    else
        kernel::fftconvolve<T, convT, isDouble, roundOut, baseDim, false>(out, signal, filter, kind);

    return out;
}

#define INSTANTIATE(T, convT, isDouble, roundOut)                                                       \
    template Array<T> fftconvolve <T, convT, isDouble, roundOut, 1>                                     \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind);    \
    template Array<T> fftconvolve <T, convT, isDouble, roundOut, 2>                                     \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind);    \
    template Array<T> fftconvolve <T, convT, isDouble, roundOut, 3>                                     \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind);

INSTANTIATE(double, double, true , false)
INSTANTIATE(float , float,  false, false)
INSTANTIATE(uint  , float,  false, true)
INSTANTIATE(int   , float,  false, true)
INSTANTIATE(uchar , float,  false, true)
INSTANTIATE(char  , float,  false, true)

}
