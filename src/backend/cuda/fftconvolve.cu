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

#include <fft.hpp>

using af::dim4;

namespace cuda
{

template<typename T>
static const dim4 calcPackedSize(Array<T> const& i1,
                                 Array<T> const& i2,
                                 const dim_t baseDim)
{
    const dim4 i1d = i1.dims();
    const dim4 i2d = i2.dims();

    dim_t pd[4] = {1, 1, 1, 1};


    dim_t max_d0 = (i1d[0] > i2d[0]) ? i1d[0] : i2d[0];
    dim_t min_d0 = (i1d[0] < i2d[0]) ? i1d[0] : i2d[0];
    pd[0]  = nextpow2((unsigned)((int)ceil(max_d0 / 2.f) + min_d0 - 1));

    for (dim_t k = 1; k < 4; k++) {
        if (k < baseDim) {
            pd[k] = nextpow2((unsigned)(i1d[k] + i2d[k] - 1));
        } else {
            pd[k] = i1d[k];
        }
    }

    return dim4(pd[0], pd[1], pd[2], pd[3]);
}

template<typename T, typename convT, typename cT, bool isDouble, bool roundOut, dim_t baseDim>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind)
{
    const dim4 sDims = signal.dims();
    const dim4 fDims = filter.dims();

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

    const dim4 spDims = calcPackedSize<T>(signal, filter, baseDim);
    const dim4 fpDims = calcPackedSize<T>(filter, signal, baseDim);
    Array<cT> signal_packed = createEmptyArray<cT>(spDims);
    Array<cT> filter_packed = createEmptyArray<cT>(fpDims);

    kernel::packDataHelper<cT, T>(signal_packed, filter_packed, signal, filter, baseDim);

    fft_inplace<cT, baseDim, true>(signal_packed);
    fft_inplace<cT, baseDim, true>(filter_packed);

    Array<T> out = createEmptyArray<T>(oDims);

    if (expand)
        kernel::complexMultiplyHelper<T, cT>(out, signal_packed, filter_packed, signal, filter, kind);
    else
        kernel::complexMultiplyHelper<T, cT>(out, signal_packed, filter_packed, signal, filter, kind);

    if (kind == CONVOLVE_BATCH_KERNEL) {
        fft_inplace<cT, baseDim, false>(filter_packed);
        if (expand)
            kernel::reorderOutputHelper<T, cT, roundOut, baseDim, true >(out, filter_packed, signal, filter, kind);
        else
            kernel::reorderOutputHelper<T, cT, roundOut, baseDim, false>(out, filter_packed, signal, filter, kind);
    } else {
        fft_inplace<cT, baseDim, false>(signal_packed);
        if (expand)
            kernel::reorderOutputHelper<T, cT, roundOut, baseDim, true >(out, signal_packed, signal, filter, kind);
        else
            kernel::reorderOutputHelper<T, cT, roundOut, baseDim, false>(out, signal_packed, signal, filter, kind);
    }

    return out;
}

#define INSTANTIATE(T, convT, cT, isDouble, roundOut)                                                   \
    template Array<T> fftconvolve <T, convT, cT, isDouble, roundOut, 1>                                 \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind);    \
    template Array<T> fftconvolve <T, convT, cT, isDouble, roundOut, 2>                                 \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind);    \
    template Array<T> fftconvolve <T, convT, cT, isDouble, roundOut, 3>                                 \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind);

INSTANTIATE(double, double, cdouble, true , false)
INSTANTIATE(float , float,  cfloat,  false, false)
INSTANTIATE(uint  , float,  cfloat,  false, true)
INSTANTIATE(int   , float,  cfloat,  false, true)
INSTANTIATE(uchar , float,  cfloat,  false, true)
INSTANTIATE(char  , float,  cfloat,  false, true)

}
