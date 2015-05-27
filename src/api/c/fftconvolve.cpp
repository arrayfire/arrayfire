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
#include <af/signal.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <arith.hpp>
#include <fft.hpp>
#include <fftconvolve.hpp>
#include <convolve_common.hpp>
#include <dispatch.hpp>

using af::dim4;
using namespace detail;

template<typename T, typename convT, typename cT, bool isDouble, bool roundOut, dim_t baseDim>
inline static af_array fftconvolve(const af_array &s, const af_array &f, const bool expand, ConvolveBatchKind kind)
{
    return getHandle(fftconvolve<T, convT, cT, isDouble, roundOut, baseDim>(getArray<T>(s), castArray<T>(f), expand, kind));
}

template<dim_t baseDim>
ConvolveBatchKind identifyBatchKind(const dim4 &sDims, const dim4 &fDims)
{
    dim_t sn = sDims.ndims();
    dim_t fn = fDims.ndims();

    if (sn==baseDim && fn==baseDim)
        return ONE2ONE;
    else if (sn==baseDim && (fn>baseDim && fn<=4))
        return ONE2MANY;
    else if ((sn>baseDim && sn<=4) && fn==baseDim)
        return MANY2ONE;
    else if ((sn>baseDim && sn<=4) && (fn>baseDim && fn<=4)) {
        bool doesDimensionsMatch = true;
        for (dim_t i=baseDim; i<4; i++) {
            if (sDims[i]!=fDims[i]) {
                doesDimensionsMatch = false;
                break;
            }
        }
        return (doesDimensionsMatch ? MANY2MANY : CONVOLVE_UNSUPPORTED_BATCH_MODE);
    }
    else
        return CONVOLVE_UNSUPPORTED_BATCH_MODE;
}

template<typename T, int baseDim>
static inline
af_array fftconvcplx(const af_array signal, const af_array filter, bool expand,
                     ConvolveBatchKind kind)
{
    const Array<T> S = getArray<T>(signal);
    const Array<T> F = castArray<T>(filter);
    const dim4 sdims = S.dims();
    const dim4 fdims = F.dims();
    dim4 tdims(1, 1, 1, 1);
    std::vector<af_seq> index(4);

    int count = 1;
    for (int i = 0; i < baseDim; i++) {
        dim_t tdim_i = sdims[i] + fdims[i] - 1;

        // Pad temporary buffers to power of 2 for performance
        tdims[i] = nextpow2(tdim_i);

        // The normalization factor
        count *= tdims[i];

        // Get the indexing params for output
        if (expand) {
            index[i].begin = 0;
            index[i].end = tdim_i - 1;
        } else {
            index[i].begin = fdims[i] / 2;
            index[i].end = index[i].begin + sdims[i] - 1;
        }
        index[i].step = 1;
    }

    for (int i = baseDim; i < 4; i++) {
        tdims[i] = std::max(sdims[i], fdims[i]);
        index[i] = af_span;
    }

    // fft(signal)
    Array<T> T1 = fft<T, T, baseDim, false>(S, 1.0, baseDim, tdims.get());

    // fft(filter)
    Array<T> T2 = fft<T, T, baseDim, false>(F, 1.0, baseDim, tdims.get());

    // fft(signal) * fft(filter)
    T1 = arithOp<T, af_mul_t>(T1, T2, tdims);

    // ifft(ffit(signal) * fft(filter))
    T1 = ifft<T, baseDim>(T1, 1.0/(double)count, baseDim, tdims.get());

    // Index to proper offsets
    T1 = createSubArray<T>(T1, index);
    return getHandle(T1);
}

template<dim_t baseDim>
af_err fft_convolve(af_array *out, const af_array signal, const af_array filter, const bool expand)
{
    try {
        ArrayInfo sInfo = getInfo(signal);
        ArrayInfo fInfo = getInfo(filter);

        af_dtype stype  = sInfo.getType();

        dim4 sdims = sInfo.dims();
        dim4 fdims = fInfo.dims();

        ConvolveBatchKind convBT = identifyBatchKind<baseDim>(sdims, fdims);

        ARG_ASSERT(1, (convBT != CONVOLVE_UNSUPPORTED_BATCH_MODE));

        af_array output;
        switch(stype) {
            case f64: output = fftconvolve<double, double, cdouble, true , false, baseDim>(signal, filter, expand, convBT); break;
            case f32: output = fftconvolve<float , float,  cfloat,  false, false, baseDim>(signal, filter, expand, convBT); break;
            case u32: output = fftconvolve<uint  , float,  cfloat,  false, true,  baseDim>(signal, filter, expand, convBT); break;
            case s32: output = fftconvolve<int   , float,  cfloat,  false, true,  baseDim>(signal, filter, expand, convBT); break;
            case u8:  output = fftconvolve<uchar , float,  cfloat,  false, true,  baseDim>(signal, filter, expand, convBT); break;
            case b8:  output = fftconvolve<char  , float,  cfloat,  false, true,  baseDim>(signal, filter, expand, convBT); break;
            case c32: output = fftconvcplx<cfloat , baseDim>(signal, filter, expand, convBT); break;
            case c64: output = fftconvcplx<cdouble, baseDim>(signal, filter, expand, convBT); break;
            default: TYPE_ERROR(1, stype);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft_convolve1(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode)
{
    return fft_convolve<1>(out, signal, filter, mode == AF_CONV_EXPAND);
}

af_err af_fft_convolve2(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode)
{
    return fft_convolve<2>(out, signal, filter, mode == AF_CONV_EXPAND);
}

af_err af_fft_convolve3(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode)
{
    return fft_convolve<3>(out, signal, filter, mode == AF_CONV_EXPAND);
}
