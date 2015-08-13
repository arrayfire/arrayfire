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
#include <convolve.hpp>
#include <fftconvolve.hpp>
#include <convolve_common.hpp>

#include <cstdio>

using af::dim4;
using namespace detail;

template<typename T, typename accT, dim_t baseDim, bool expand>
inline static af_array convolve(const af_array &s, const af_array &f, ConvolveBatchKind kind)
{
    return getHandle(convolve<T, accT, baseDim, expand>(getArray<T>(s), castArray<accT>(f), kind));
}

template<typename T, typename accT, bool expand>
inline static af_array convolve2(const af_array &s, const af_array &c_f, const af_array &r_f)
{
    return getHandle(convolve2<T, accT, expand>(getArray<T>(s),
                                                castArray<accT>(c_f),
                                                castArray<accT>(r_f)));
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

template<dim_t baseDim, bool expand>
af_err convolve(af_array *out, const af_array signal, const af_array filter)
{
    try {
        ArrayInfo sInfo = getInfo(signal);
        ArrayInfo fInfo = getInfo(filter);

        af_dtype stype  = sInfo.getType();

        dim4 sdims = sInfo.dims();
        dim4 fdims = fInfo.dims();

        ConvolveBatchKind convBT = identifyBatchKind<baseDim>(sdims, fdims);

        ARG_ASSERT(1, (convBT != CONVOLVE_BATCH_UNSUPPORTED && convBT != CONVOLVE_BATCH_DIFF));

        af_array output;
        switch(stype) {
            case c32: output = convolve<cfloat ,  cfloat, baseDim, expand>(signal, filter, convBT); break;
            case c64: output = convolve<cdouble, cdouble, baseDim, expand>(signal, filter, convBT); break;
            case f32: output = convolve<float  ,   float, baseDim, expand>(signal, filter, convBT); break;
            case f64: output = convolve<double ,  double, baseDim, expand>(signal, filter, convBT); break;
            case u32: output = convolve<uint   ,   float, baseDim, expand>(signal, filter, convBT); break;
            case s32: output = convolve<int    ,   float, baseDim, expand>(signal, filter, convBT); break;
            case u8:  output = convolve<uchar  ,   float, baseDim, expand>(signal, filter, convBT); break;
            case b8:  output = convolve<char   ,   float, baseDim, expand>(signal, filter, convBT); break;
            default: TYPE_ERROR(1, stype);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<bool expand>
af_err convolve2_sep(af_array *out, af_array col_filter, af_array row_filter, const af_array signal)
{
    try {
        ArrayInfo sInfo = getInfo(signal);
        ArrayInfo cfInfo= getInfo(col_filter);
        ArrayInfo rfInfo= getInfo(row_filter);

        af_dtype signalType  = sInfo.getType();

        dim4 signalDims = sInfo.dims();

        ARG_ASSERT(1, (signalDims.ndims()>=2));
        ARG_ASSERT(2, cfInfo.isVector());
        ARG_ASSERT(3, rfInfo.isVector());

        af_array output;
        switch(signalType) {
            case c32: output = convolve2<cfloat ,  cfloat, expand>(signal, col_filter, row_filter); break;
            case c64: output = convolve2<cdouble, cdouble, expand>(signal, col_filter, row_filter); break;
            case f32: output = convolve2<float  ,   float, expand>(signal, col_filter, row_filter); break;
            case f64: output = convolve2<double ,  double, expand>(signal, col_filter, row_filter); break;
            case u32: output = convolve2<uint   ,   float, expand>(signal, col_filter, row_filter); break;
            case s32: output = convolve2<int    ,   float, expand>(signal, col_filter, row_filter); break;
            case u8:  output = convolve2<uchar  ,   float, expand>(signal, col_filter, row_filter); break;
            case b8:  output = convolve2<char   ,   float, expand>(signal, col_filter, row_filter); break;
            default: TYPE_ERROR(1, signalType);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}


template<int baseDim>
bool isFreqDomain(const af_array &signal, const af_array filter, af_conv_domain domain)
{
    if (domain == AF_CONV_FREQ) return true;
    if (domain != AF_CONV_AUTO) return false;

    ArrayInfo sInfo = getInfo(signal);
    ArrayInfo fInfo = getInfo(filter);

    dim4 sdims = sInfo.dims();
    dim4 fdims = fInfo.dims();

    if (identifyBatchKind<baseDim>(sdims, fdims) == CONVOLVE_BATCH_DIFF) return true;

    int batch = 1;
    for(int i = 3; i >= baseDim; i--) {
        batch *= std::max(fdims[i], sdims[i]);
    }

    if (batch >= 10) return true;

    if (baseDim == 1) {
        if (fdims[0] > 128) return true;
    }

    if (baseDim == 2) {
        // maximum supported size in 2D domain
        if (fdims[0] > 17 || fdims[1] > 17) return true;

        // Maximum supported non square size
        if (fdims[0] != fdims[1] && fdims[0] > 5) return true;
    }

    if (baseDim == 3) {
        if (fdims[0] > 5 || fdims[1] > 5 || fdims[2] > 5) return true;
    }

    return false;
}

af_err af_convolve1(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, af_conv_domain domain)
{
    try {
        if (isFreqDomain<1>(signal, filter, domain))
            return af_fft_convolve1(out, signal, filter, mode);
    } CATCHALL;

    if (mode == AF_CONV_EXPAND)
            return convolve<1, true >(out, signal, filter);
    else
        return convolve<1, false>(out, signal, filter);
}

af_err af_convolve2(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, af_conv_domain domain)
{
    try {
        if (isFreqDomain<2>(signal, filter, domain))
            return af_fft_convolve2(out, signal, filter, mode);
    } CATCHALL;

    if (mode == AF_CONV_EXPAND)
        return convolve<2, true >(out, signal, filter);
    else
        return convolve<2, false>(out, signal, filter);
}

af_err af_convolve3(af_array *out, const af_array signal, const af_array filter, const af_conv_mode mode, af_conv_domain domain)
{
    try {
        if (isFreqDomain<3>(signal, filter, domain))
            return af_fft_convolve3(out, signal, filter, mode);
    } CATCHALL;

    if (mode == AF_CONV_EXPAND)
        return convolve<3, true >(out, signal, filter);
    else
        return convolve<3, false>(out, signal, filter);
}

af_err af_convolve2_sep(af_array *out, const af_array signal, const af_array col_filter, const af_array row_filter, const af_conv_mode mode)
{
    if (mode == AF_CONV_EXPAND)
        return convolve2_sep<true >(out, signal, col_filter, row_filter);
    else
        return convolve2_sep<false>(out, signal, col_filter, row_filter);
}
