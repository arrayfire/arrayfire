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
#include <fftconvolve.hpp>
#include <convolve_common.hpp>
#include <dispatch.hpp>
#include <complex.hpp>
#include <fft_common.hpp>

using af::dim4;
using namespace detail;

template<typename T, typename convT, typename cT, int baseDim>
static inline
af_array fftconvolve_fallback(const af_array signal, const af_array filter, bool expand)
{
    const Array<cT> S = castArray<cT>(signal);
    const Array<cT> F = castArray<cT>(filter);
    const dim4 sdims = S.dims();
    const dim4 fdims = F.dims();
    dim4 odims(1, 1, 1, 1);
    dim4 psdims(1, 1, 1, 1);
    dim4 pfdims(1, 1, 1, 1);

    std::vector<af_seq> index(4);

    int count = 1;
    for (int i = 0; i < baseDim; i++) {
        dim_t tdim_i = sdims[i] + fdims[i] - 1;

        // Pad temporary buffers to power of 2 for performance
        odims[i] = nextpow2(tdim_i);
        psdims[i] = nextpow2(tdim_i);
        pfdims[i] = nextpow2(tdim_i);

        // The normalization factor
        count *= odims[i];

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
        odims[i] = std::max(sdims[i], fdims[i]);
        psdims[i] = sdims[i];
        pfdims[i] = fdims[i];
        index[i] = af_span;
    }

    // fft(signal)
    Array<cT> T1 = fft<cT, cT, baseDim, true>(S, 1.0, baseDim, psdims.get());

    // fft(filter)
    Array<cT> T2 = fft<cT, cT, baseDim, true>(F, 1.0, baseDim, pfdims.get());

    // fft(signal) * fft(filter)
    T1 = arithOp<cT, af_mul_t>(T1, T2, odims);

    // ifft(ffit(signal) * fft(filter))
    T1 = fft<cT, cT, baseDim, false>(T1, 1.0/(double)count, baseDim, odims.get());

    // Index to proper offsets
    T1 = createSubArray<cT>(T1, index);

    if (getInfo(signal).isComplex() || getInfo(filter).isComplex()) {
        return getHandle(cast<T>(T1));
    } else {
        return getHandle(cast<T>(real<convT>(T1)));
    }
}

template<typename T, typename convT, typename cT, bool isDouble, bool roundOut, dim_t baseDim>
inline static af_array fftconvolve(const af_array &s, const af_array &f, const bool expand, ConvolveBatchKind kind)
{
    if (kind == CONVOLVE_BATCH_DIFF) return fftconvolve_fallback<T, convT, cT, baseDim>(s, f, expand);
    else return getHandle(fftconvolve<T, convT, cT, isDouble, roundOut, baseDim>(getArray<T>(s), castArray<T>(f), expand, kind));
}

template<dim_t baseDim>
ConvolveBatchKind identifyBatchKind(const dim4 &sDims, const dim4 &fDims)
{
    dim_t sn = sDims.ndims();
    dim_t fn = fDims.ndims();

    if (sn==baseDim && fn==baseDim)
        return CONVOLVE_BATCH_NONE;
    else if (sn==baseDim && (fn>baseDim && fn<=4))
        return CONVOLVE_BATCH_KERNEL;
    else if ((sn>baseDim && sn<=4) && fn==baseDim)
        return CONVOLVE_BATCH_SIGNAL;
    else if ((sn>baseDim && sn<=4) && (fn>baseDim && fn<=4)) {
       bool doesDimensionsMatch = true;
        bool isInterleaved = true;
        for (dim_t i=baseDim; i<4; i++) {
            doesDimensionsMatch &= (sDims[i] == fDims[i]);
            isInterleaved &= (sDims[i] == 1 || fDims[i] == 1 || sDims[i] == fDims[i]);
        }
        if (doesDimensionsMatch) return CONVOLVE_BATCH_SAME;
        return (isInterleaved ? CONVOLVE_BATCH_DIFF : CONVOLVE_BATCH_UNSUPPORTED);
    }
    else
        return CONVOLVE_BATCH_UNSUPPORTED;
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

        ARG_ASSERT(1, (convBT != CONVOLVE_BATCH_UNSUPPORTED));

        af_array output;
        switch(stype) {
            case f64: output = fftconvolve<double, double, cdouble, true , false, baseDim>(signal, filter, expand, convBT); break;
            case f32: output = fftconvolve<float , float,  cfloat,  false, false, baseDim>(signal, filter, expand, convBT); break;
            case u32: output = fftconvolve<uint  , float,  cfloat,  false, true,  baseDim>(signal, filter, expand, convBT); break;
            case s32: output = fftconvolve<int   , float,  cfloat,  false, true,  baseDim>(signal, filter, expand, convBT); break;
            case u8:  output = fftconvolve<uchar , float,  cfloat,  false, true,  baseDim>(signal, filter, expand, convBT); break;
            case b8:  output = fftconvolve<char  , float,  cfloat,  false, true,  baseDim>(signal, filter, expand, convBT); break;
            case c32: output = fftconvolve_fallback<cfloat , cfloat , cfloat , baseDim>(signal, filter, expand); break;
            case c64: output = fftconvolve_fallback<cdouble, cdouble, cdouble, baseDim>(signal, filter, expand); break;
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
