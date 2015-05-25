/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <err_opencl.hpp>
#include <fft.hpp>

using af::dim4;

namespace opencl
{

template<typename T>
static const dim4 calcPackedSize(Array<T> const& i1,
                                 Array<T> const& i2,
                                 const dim_t baseDim)
{
    const dim4 i1d = i1.dims();
    const dim4 i2d = i2.dims();

    dim_t pd[4];

    // Pack both signal and filter on same memory array, this will ensure
    // better use of batched cuFFT capabilities
    for (dim_t k = 0; k < 4; k++) {
        if (k == 0)
            pd[k] = nextpow2((unsigned)(i1d[k] + i2d[k] - 1)) / 2;
        else if (k < baseDim)
            pd[k] = nextpow2((unsigned)(i1d[k] + i2d[k] - 1));
        else if (k == baseDim)
            pd[k] = i1d[k] + i2d[k];
        else
            pd[k] = 1;
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
            if (kind==ONE2ONE || kind==ONE2MANY) {
                oDims[d] = sDims[d]+fDims[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sDims[d]+fDims[d]-1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind==ONE2MANY) {
            for (dim_t i=baseDim; i<4; ++i)
                oDims[i] = fDims[i];
        }
    }

    const dim4 pDims = calcPackedSize<T>(signal, filter, baseDim);
    Array<cT> packed = createEmptyArray<cT>(pDims);

    kernel::packDataHelper<cT, T, isDouble, convT>(packed, signal, filter, baseDim, kind);

    fft_common<cT, baseDim, true>(packed, packed);

    kernel::complexMultiplyHelper<cT, T, isDouble, convT>(packed, signal, filter, baseDim, kind);

    // Compute inverse FFT only on complex-multiplied data
    if (kind == ONE2MANY) {
        std::vector<af_seq> seqs;
        for (dim_t k = 0; k < 4; k++) {
            if (k < baseDim)
                seqs.push_back(af_make_seq(0, pDims[k]-1, 1));
            else if (k == baseDim)
                seqs.push_back(af_make_seq(1, pDims[k]-1, 1));
            else
                seqs.push_back(af_make_seq(0, 0, 1));
        }

        Array<cT> subPacked = createSubArray<cT>(packed, seqs);
        fft_common<cT, baseDim, false>(subPacked, subPacked);
    }
    else {
        std::vector<af_seq> seqs;
        for (dim_t k = 0; k < 4; k++) {
            if (k < baseDim)
                seqs.push_back(af_make_seq(0, pDims[k]-1, 1));
            else if (k == baseDim)
                seqs.push_back(af_make_seq(0, sDims[k]-1, 1));
            else
                seqs.push_back(af_make_seq(0, 0, 1));
        }

        Array<cT> subPacked = createSubArray<cT>(packed, seqs);
        fft_common<cT, baseDim, false>(subPacked, subPacked);
    }

    Array<T> out = createEmptyArray<T>(oDims);

    if (expand)
        kernel::reorderOutputHelper<T, cT, isDouble, roundOut, true , convT>(out, packed, signal, filter, baseDim, kind);
    else
        kernel::reorderOutputHelper<T, cT, isDouble, roundOut, false, convT>(out, packed, signal, filter, baseDim, kind);

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
